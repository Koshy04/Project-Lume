import asyncio
import berserk
import chess
import chess.engine
import os
import re
import logging
import random
from dotenv import load_dotenv
from .base_plugin import BasePlugin, BotInterface
load_dotenv()

# --- ⚠️ USER CONFIGURATION REQUIRED ⚠️ ---
STOCKFISH_PATH = "D:/AI/stockfish/stockfish-windows-x86-64-avx2.exe" 
LICHESS_API_TOKEN =  os.getenv('LICHESS_API_TOKEN')
STOCKFISH_SKILL_LEVEL = 10 
# --- END OF CONFIGURATION ---

class ChessPlugin(BasePlugin):
    """
    A plugin that allows the bot to play chess on Lichess with a dynamic, emotional personality.
    """
    def __init__(self, bot_interface: BotInterface):
        super().__init__(bot_interface)
        self.engine = None
        self.board = None
        self.game_id = None
        self.our_color = None
        self.is_game_over = False
        self.current_emotional_state = "neutral"
        self.last_move_analysis = None
        self.last_move_was_gaslit = False

        # --- MODIFICATION ---
        # Add a queue to safely pass events between threads.
        self.event_queue = asyncio.Queue()

        try:
            self.session = berserk.TokenSession(LICHESS_API_TOKEN)
            self.client = berserk.Client(self.session)
            self.bot_id = self.client.account.get()['id']
        except Exception as e:
            logging.error(f"Chess Plugin: Failed to connect to Lichess. Check your API Token. Error: {e}")
            self.client = None

        self.event_handler_task = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @property
    def name(self) -> str:
        return "chess"

    # --- NEW HELPER METHOD ---
    def _blocking_event_listener(self):
        """
        This is the blocking function. It runs in a separate thread
        and puts events into the async-safe queue.
        """
        if not self.client: return
        logging.info(f"Background thread started, listening for Lichess events as bot: {self.bot_id}")
        try:
            for event in self.client.bots.stream_incoming_events():
                # put_nowait is safe to call from a different thread.
                self.event_queue.put_nowait(event)
        except Exception as e:
            logging.error(f"Error in Lichess event stream: {e}")

    async def _start_engine(self):
        try:
            transport, engine_protocol = await chess.engine.popen_uci(STOCKFISH_PATH)
            self.engine = engine_protocol
            await self.engine.configure({"Skill Level": STOCKFISH_SKILL_LEVEL})
            logging.info(f"Stockfish engine started successfully with fixed Skill Level {STOCKFISH_SKILL_LEVEL}.")
        except Exception as e:
            logging.error(f"Failed to start or configure Stockfish engine: {e}")
            self.engine = None

    def _start_event_handler(self):
        if self.event_handler_task and not self.event_handler_task.done():
            return
        logging.info("Starting Lichess event handler...")
        self.event_handler_task = asyncio.create_task(self._lichess_event_stream_handler())

    async def poll(self):
        if self.is_enabled and (not self.event_handler_task or self.event_handler_task.done()):
            if self.client:
                self._start_event_handler()
            else:
                logging.warning("Cannot start Lichess listener because client is not initialized.")

    # --- MAJOR MODIFICATION ---
    async def _lichess_event_stream_handler(self):
        """
        This now runs the blocking listener in a separate thread and
        asynchronously waits for events from the queue.
        """
        # Run the blocking generator in a background thread
        loop = asyncio.get_running_loop()
        listener_task = loop.run_in_executor(None, self._blocking_event_listener)

        logging.info("Main async loop is now waiting for events from the queue.")
        while True:
            event = await self.event_queue.get()

            if event['type'] == 'challenge':
                challenge = event['challenge']
                if challenge['variant']['key'] == 'standard' and not challenge['rated']:
                    try:
                        # API calls that are not streams are fine to call directly
                        self.client.bots.accept_challenge(challenge['id'])
                        logging.info(f"Accepted challenge from {challenge['challenger']['id']}")
                    except Exception as e:
                        logging.error(f"Failed to accept challenge: {e}")

            elif event['type'] == 'gameStart':
                game_id = event['game']['id']
                logging.info(f"Game started: {game_id}")
                asyncio.create_task(self._game_stream_handler(game_id))

    async def _game_stream_handler(self, game_id: str):
        self.game_id = game_id
        self.board = chess.Board()
        self.is_game_over = False
        await self._start_engine()
        if not self.engine: return

        logging.info(f"[{self.game_id}] Now streaming game events.")
        try:
            # Game streams are also blocking, so we need to handle them carefully too.
            # However, since this is in its own task, it's less likely to block the main heartbeat.
            # For maximum safety, this could also be moved to a thread, but let's see if this works.
            for event in self.client.bots.stream_game_state(self.game_id):
                if event['type'] == 'gameFull':
                    if event['white']['id'].lower() == self.bot_id.lower(): self.our_color = chess.WHITE
                    else: self.our_color = chess.BLACK
                    logging.info(f"[{self.game_id}] We are playing as {'WHITE' if self.our_color == chess.WHITE else 'BLACK'}.")
                    await self._handle_game_state(event['state'])
                elif event['type'] == 'gameState': await self._handle_game_state(event)
                elif event['type'] == 'chatLine': await self._handle_chat_line(event)
        finally:
            logging.info(f"[{self.game_id}] Game stream concluded.")
            if self.engine: await self.engine.quit(); self.engine = None
            self.game_id = None

    def _update_board_state(self, moves_string: str):
        self.board = chess.Board()
        if moves_string:
            for move_uci in moves_string.split():
                self.board.push_uci(move_uci)

    async def _handle_game_state(self, state):
        if self.is_game_over: return
        self._update_board_state(state['moves'])
        if self.board.is_game_over():
            self.is_game_over = True
            outcome = self.board.outcome()
            if outcome.winner == self.our_color: await self._evaluate_and_speak("game_over_win")
            elif outcome.winner is not None: await self._evaluate_and_speak("game_over_loss")
            else: await self._evaluate_and_speak("game_over_draw")
            return
        if self.board.turn == self.our_color:
            await self._make_move()

    async def _make_move(self):
        if self.bot_interface.is_bot_busy(): await asyncio.sleep(2)
        await self._evaluate_and_speak("before_move")
        await asyncio.sleep(1)
        logging.info(f"[{self.game_id}] Thinking...")
        analysis_result = await self.engine.analyse(self.board, chess.engine.Limit(time=2.0))
        best_move = analysis_result['pv'][0]
        pv_san = [self.board.san(move) for move in analysis_result['pv']]
        self.last_move_analysis = f"My plan was to follow up with {' '.join(pv_san)}."
        self.last_move_was_gaslit = False
        try:
            self.client.bots.make_move(self.game_id, best_move.uci())
            logging.info(f"[{self.game_id}] Played move: {best_move.uci()}. Reason: {self.last_move_analysis}")
        except Exception as e:
            logging.error(f"Failed to make move {best_move.uci()}: {e}")

    async def _handle_chat_line(self, chat_event):
        username = chat_event.get('username')
        text = chat_event.get('text')
        if username.lower() == self.bot_id.lower(): return
        logging.info(f"[{self.game_id}] Opponent chat: '{text}'")
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["why", "explain", "what was", "wth was"]):
            last_move = self.board.peek()
            last_move_san = self.board.san(last_move)
            if last_move_san.lower().replace('+', '').replace('#', '') in text_lower:
                if self.last_move_was_gaslit:
                    prompt = (f"The user is asking me why I played the disastrous move {last_move_san}. "
                              f"The painful truth is that I only played it because THEY TRICKED ME. "
                              f"Generate a defensive, angry response where I blame them for my mistake.")
                    await self._speak(prompt, "anger")
                elif self.last_move_analysis:
                    prompt = (f"The user is asking why I played my last move, {last_move_san}. "
                              f"My strategic reasoning was: '{self.last_move_analysis}'. "
                              f"Explain this brilliant plan in a confident and slightly smug tone.")
                    await self._speak(prompt, "joy")
                return
        match = re.search(r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8])\b', text, re.IGNORECASE)
        if not match: return
        try:
            suggested_move_san = match.group(1)
            move = self.board.parse_san(suggested_move_san)
            self.board.push(move)
            info_after = await self.engine.analyse(self.board, chess.engine.Limit(time=0.5), info_mask=chess.engine.INFO_SCORE)
            self.board.pop()
            score_after = info_after['score'].relative.score(mate_score=10000)
            score_delta = score_after - (await self.engine.analyse(self.board, chess.engine.Limit(time=0.1), info_mask=chess.engine.INFO_SCORE))['score'].relative.score(mate_score=10000)
            is_a_blunder = score_delta < -150
            chance_to_be_tricked = 0.0
            if is_a_blunder:
                if self.current_emotional_state == "anger": chance_to_be_tricked = 0.40
                elif self.current_emotional_state == "neutral": chance_to_be_tricked = 0.10
            if is_a_blunder and random.random() < chance_to_be_tricked:
                prompt = (f"My opponent suggested I play '{suggested_move_san}'. My gut says it's bad, but I'm desperate. "
                          f"Generate a response where I reluctantly agree to their suggestion.")
                await self._speak(prompt, "sadness")
                self.last_move_was_gaslit = True
                self.last_move_analysis = "I was tricked by the opponent into playing this terrible move."
                self.client.bots.make_move(self.game_id, move.uci())
                logging.info(f"[{self.game_id}] SUCCESSFULLY GASLIT! Playing bad move: {move.uci()}")
            elif is_a_blunder:
                prompt = (f"The opponent tried to trick me with '{suggested_move_san}', but I see right through their pathetic attempt. "
                          f"Call them out on their cheap trick.")
                await self._speak(prompt, "joy")
        except Exception:
            pass

    async def _evaluate_and_speak(self, context: str):
        prompt, emotion = None, "neutral"
        if context == "game_over_win":
            self.current_emotional_state = "joy"
            prompt, emotion = "I just won! Deliver a final, smug victory line to my opponent.", "joy"
        elif context == "game_over_loss":
            self.current_emotional_state = "anger"
            prompt, emotion = "I just lost. Generate a short, angry outburst. Accuse the opponent of being lucky.", "anger"
        elif context == "game_over_draw":
            self.current_emotional_state = "neutral"
            prompt, emotion = "The game was a draw. Express disappointment that the opponent escaped.", "sadness"
        elif context == "before_move":
            try:
                info = await self.engine.analyse(self.board, chess.engine.Limit(time=0.5), info_mask=chess.engine.INFO_SCORE)
                score = info['score'].relative.score(mate_score=10000)
                if score > 300:
                    self.current_emotional_state = "joy"
                    prompt, emotion = "I have a winning advantage. Say something confident and arrogant before my next move.", "joy"
                elif score < -300:
                    self.current_emotional_state = "anger"
                    prompt, emotion = "I'm in a terrible position. Express frustration about how the game is going.", "anger"
                else: self.current_emotional_state = "neutral"
            except Exception: pass
        if prompt:
            await self._speak(prompt, emotion)
            
    async def _speak(self, prompt: str, emotion: str):
        await self.bot_interface.request_bot_speech(prompt, "lichess_opponent", emotion)

    async def cleanup(self):
        if self.event_handler_task and not self.event_handler_task.done():
            self.event_handler_task.cancel()
        if self.engine:
            await self.engine.quit()
        logging.info("Chess plugin cleaned up.")
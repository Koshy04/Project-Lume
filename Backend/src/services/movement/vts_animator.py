import asyncio
import websockets
import json
import math
import random
import uuid
import os
import time
from src.log.custom_logger import logger
import config

class LipSyncManager:
    """Handles Rhubarb data processing and calculates lipsync parameter values."""
    def __init__(self):
        self.viseme_map = self._load_viseme_map()
        self.is_running = False
        self.lipsync_task = None
        self.easer = AnimationMixer._resolve_easing("sineInOut")
        self.current_vals = {}

    def _load_viseme_map(self) -> dict:
        try:
            script_dir = os.path.dirname(__file__)
            config_path = os.path.join(script_dir, 'config', 'visemes.json')
            with open(config_path, 'r') as f:
                mapping = json.load(f)
            logger.info(f"Successfully loaded viseme map from {config_path}")
            return mapping
        except Exception as e:
            logger.error(f"Failed to load viseme map: {e}. Lipsync will be disabled.")
            return {}

    async def _run_rhubarb(self, audio_path: str) -> dict | None:
        if not os.path.exists(config.RHUBARB_EXECUTABLE_PATH):
            logger.error(f"Rhubarb executable not found at: {config.RHUBARB_EXECUTABLE_PATH}")
            return None
        try:
            command = [
                config.RHUBARB_EXECUTABLE_PATH, '-r', 'phonetic', '--extendedShapes', 'GHX', '-f', 'json', audio_path
            ]
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"Rhubarb failed with error: {stderr.decode()}")
                return None
            return json.loads(stdout)
        except Exception as e:
            logger.error(f"An error occurred while running Rhubarb: {e}")
            return None

    def _get_neutral_params(self) -> dict:
        neutral_params = {}
        for params in self.viseme_map.values():
            for param_name in params.keys():
                neutral_params[param_name] = 0.0
        return neutral_params

    async def _update_viseme_transition(self, from_params: dict, to_params: dict, duration: float):
        """Calculates and updates the desired parameter values over time."""
        if duration <= 0:
            return

        fps = 60; dt = 1.0 / fps; steps = max(1, int(fps * duration))
        all_keys = set(from_params.keys()) | set(to_params.keys())
        
        for i in range(steps + 1):
            t = i / steps; progress = self.easer(t)
            
            frame_vals = {}
            for key in all_keys:
                start_val = from_params.get(key, 0.0)
                target_val = to_params.get(key, 0.0)
                frame_vals[key] = start_val + (target_val - start_val) * progress
            self.current_vals = frame_vals
            
            await asyncio.sleep(dt)

    async def get_lipsync_data(self, audio_path: str) -> list | None:
        if not self.viseme_map: return None
        rhubarb_data = await self._run_rhubarb(audio_path)
        if not rhubarb_data or 'mouthCues' not in rhubarb_data or not rhubarb_data['mouthCues']:
            logger.warning("Rhubarb did not produce valid mouth cues.")
            return None
        return rhubarb_data['mouthCues']

    async def animate_from_data(self, cues: list, start_signal: asyncio.Event):
        """Animates lipsync from pre-processed data, waiting for a start signal."""
        self.is_running = True
        
        await start_signal.wait()
        start_time = time.monotonic()

        neutral_params = self._get_neutral_params()
        previous_params = neutral_params.copy()

        try:
            first_cue = cues[0]
            first_cue_params = self.viseme_map.get(first_cue['value'], {})
            await self._update_viseme_transition(previous_params, first_cue_params, first_cue['start'])
            previous_params = first_cue_params

            for i in range(len(cues) - 1):
                current_cue = cues[i]; next_cue = cues[i+1]
                now = time.monotonic()
                target_time = start_time + current_cue['start']
                sleep_duration = target_time - now
                if sleep_duration > 0: await asyncio.sleep(sleep_duration)

                current_params = self.viseme_map.get(current_cue['value'], {})
                next_params = self.viseme_map.get(next_cue['value'], {})
                duration = next_cue['start'] - current_cue['start']
                
                await self._update_viseme_transition(current_params, next_params, duration)
                previous_params = next_params

            last_cue = cues[-1]
            hold_duration = last_cue['end'] - last_cue['start']
            await asyncio.sleep(hold_duration)

        except asyncio.CancelledError:
            logger.info("Lipsync animation was cancelled.")
        finally:
            await self._update_viseme_transition(previous_params, neutral_params, 0.1)
            self.current_vals = {}
            self.is_running = False


class VTubeStudioAPI:
    def __init__(self, host="localhost", port=8001):
        self.uri = f"ws://{host}:{port}"; self.plugin_name = "Advanced Animation Engine"; self.plugin_developer = "AI VTuber"; self.auth_token = None; self.ws = None; self.api_lock = asyncio.Lock()
    def _load_token(self):
        token_path = config.VTS_TOKEN_PATH
        if os.path.exists(token_path):
            try:
                with open(token_path, 'r') as f:
                    self.auth_token = f.read().strip()
                    if self.auth_token: logger.info(f"Loaded VTube Studio authentication token from {token_path}."); return True
            except Exception as e: logger.error(f"Failed to load VTS token from {token_path}: {e}")
        logger.info("No VTS token file found. Will request a new one."); return False
    def _save_token(self):
        token_path = config.VTS_TOKEN_PATH
        try:
            with open(token_path, 'w') as f: f.write(self.auth_token)
            logger.info(f"Saved new VTube Studio authentication token to {token_path}.")
        except Exception as e: logger.error(f"Failed to save VTS token to {token_path}: {e}")
    async def connect(self):
        try: self.ws = await websockets.connect(self.uri); logger.info("Connected to VTube Studio WebSocket."); return True
        except Exception as e: logger.error(f"Failed to connect to VTube Studio: {e}"); self.ws = None; return False
    async def send_request(self, message_type: str, data: dict = None):
        if not self.ws: logger.warning("WebSocket is not connected. Cannot send request."); return {}
        async with self.api_lock:
            try:
                request_payload = {"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0", "requestID": str(uuid.uuid4()), "messageType": message_type}
                if data: request_payload["data"] = data
                await self.ws.send(json.dumps(request_payload)); response_str = await self.ws.recv(); return json.loads(response_str)
            except websockets.exceptions.ConnectionClosed as e: logger.error(f"VTube Studio connection lost: {e}"); self.ws = None; return {}
            except Exception as e: logger.error(f"An error occurred during VTS send/receive: {e}"); return {}
    async def authenticate(self):
        self._load_token()
        if self.auth_token:
            auth_payload = {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_developer, "authenticationToken": self.auth_token}
            auth_response = await self.send_request("AuthenticationRequest", auth_payload)
            if auth_response and auth_response.get("data", {}).get("authenticated"): logger.info("Successfully authenticated with VTube Studio using existing token!"); return True
            else: logger.warning("Authentication with existing token failed. Requesting a new one."); self.auth_token = None
        logger.info("Requesting new authentication token from VTube Studio.")
        token_payload = {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_developer}
        response = await self.send_request("AuthenticationTokenRequest", token_payload)
        if response and response.get("data", {}).get("authenticationToken"):
            self.auth_token = response["data"]["authenticationToken"]
            logger.info("New auth token received. Please check VTube Studio and click 'Allow'. A popup should appear.")
            auth_payload = {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_developer, "authenticationToken": self.auth_token}
            auth_response = await self.send_request("AuthenticationRequest", auth_payload)
            if auth_response and auth_response.get("data", {}).get("authenticated"): logger.info("Successfully authenticated with new token!"); self._save_token(); return True
        logger.error("Authentication with VTube Studio failed. Check if VTube Studio is running and if the API is enabled."); return False
    async def set_parameters(self, parameter_values):
        if not parameter_values: return
        payload = {"faceFound": True, "mode": "set", "parameterValues": parameter_values}
        return await self.send_request("InjectParameterDataRequest", payload)
    async def read_param_value(self, param_id: str):
        payload = {"name": param_id}; response = await self.send_request("ParameterValueRequest", payload); return response or {}
    
    async def trigger_hotkey(self, hotkey_id: str):
        """Triggers a hotkey by ID in VTube Studio"""
        payload = {"hotkeyID": hotkey_id}
        response = await self.send_request("HotkeyTriggerRequest", payload)
        if response and response.get("data"):
            logger.info(f"Successfully triggered hotkey: {hotkey_id}")
            return True
        logger.warning(f"Failed to trigger hotkey: {hotkey_id}")
        return False
        
    async def get_hotkeys_in_current_model(self):
        """Gets all available hotkeys for the current model"""
        response = await self.send_request("HotkeysInCurrentModelRequest")
        if response and response.get("data"):
            hotkeys = response["data"].get("availableHotkeys", [])
            logger.info(f"Available hotkeys in model:")
            for hk in hotkeys:
                logger.info(f"  - {hk['name']} (ID: {hk['hotkeyID']})")
            return hotkeys
        return []

    async def close(self):
        if self.ws: await self.ws.close(); self.ws = None; logger.info("VTube Studio connection closed")

class AnimationMixer:
    """A sophisticated engine for managing animations, idle motion, and lipsync blending."""
    def __init__(self, vts_api, presets, param_ranges, idle_settings, hotkey_mappings, lipsync_manager: LipSyncManager):
        self.vts = vts_api; self.presets = presets; self.param_ranges = param_ranges; self.idle_settings = idle_settings
        self.hotkey_mappings = hotkey_mappings
        self.lipsync_manager = lipsync_manager
        self._param_owner = {}; self._owner_lock = asyncio.Lock(); self._blend_lock = asyncio.Lock()
        self._blend_active_vals = {}; self._blend_idle_vals = {}; self.idle_blend_weight = 0.75
        self.is_running = False; self._blender_task = None; self._idle_tasks = {}; self._anim_tasks = set()
        self._fade_tasks = set()

    async def start(self):
        if self.is_running: return
        self.is_running = True; self._blender_task = asyncio.create_task(self._run_blender()); await self._restart_idle_tasks()
        logger.info("Animation Mixer started with Idle Motion and Blending.")
        
    async def stop(self):
        self.is_running = False
        tasks = [self._blender_task] + list(self._idle_tasks.values()) + list(self._anim_tasks) + list(self._fade_tasks)
        for task in tasks:
            if task and not task.done(): task.cancel()
        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True); logger.info("Animation Mixer stopped.")

    async def _run_blender(self):
        """The single compositor loop that blends idle, gestures, and lipsync."""
        fps = 60; dt = 1.0 / fps
        while self.is_running:
            try:
                if not self.vts.ws: await asyncio.sleep(1); continue
                
                async with self._blend_lock:
                    active, idle = self._blend_active_vals.copy(), self._blend_idle_vals.copy()
                
                lipsync = self.lipsync_manager.current_vals if self.lipsync_manager else {}

                output_params = idle
                output_params.update(active)
                output_params.update(lipsync)
                
                if output_params: 
                    await self.vts.set_parameters([{"id": k, "value": v} for k, v in output_params.items()])
                
                await asyncio.sleep(dt)
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Blender Error: {e}", exc_info=True)
            
    def play(self, animation_name, is_looping=False):
        if animation_name not in self.presets: logger.warning(f"Mixer: Animation '{animation_name}' not found."); return
        anim_id = f"anim_{uuid.uuid4().hex[:8]}"; task = asyncio.create_task(self._execute_animation(anim_id, animation_name, is_looping))
        self._anim_tasks.add(task); task.add_done_callback(self._anim_tasks.discard); logger.info(f"Started animation: {animation_name} (ID: {anim_id})")
    
    @staticmethod
    def _resolve_easing(spec: str = "sineInOut"):
        s = (spec or "sineInOut").strip().lower()
        if s == "linear": return lambda t: t;
        if s == "easein": return lambda t: t * t;
        if s == "easeout": return lambda t: 1 - (1 - t) * (1 - t);
        if s == "sinein": return lambda t: 1 - math.cos(t * math.pi / 2);
        if s == "sineout": return lambda t: math.sin(t * math.pi / 2)
        return lambda t: -(math.cos(math.pi * t) - 1) / 2
    
    async def _trigger_animation_hotkeys(self, animation_name, timing="start"):
        """Triggers hotkeys associated with an animation"""
        if animation_name not in self.hotkey_mappings:
            return
        
        mapping = self.hotkey_mappings[animation_name]
        hotkeys_to_trigger = []
        
        if timing == "start" and "on_start" in mapping:
            hotkeys_to_trigger = mapping["on_start"]
        elif timing == "end" and "on_end" in mapping:
            hotkeys_to_trigger = mapping["on_end"]
        
        for hotkey_id in hotkeys_to_trigger:
            await self.vts.trigger_hotkey(hotkey_id)
            if len(hotkeys_to_trigger) > 1:
                await asyncio.sleep(0.1) # Small delay between multiple hotkeys

    async def _execute_animation(self, anim_id, name, is_looping):
        preset_data = self.presets.get(name, []); stages = preset_data if isinstance(preset_data, list) else [preset_data]
        all_params = set(p for stage in stages for p in stage if p not in ["Length", "KeepTime", "Easing", "visemes"])

        await self._trigger_animation_hotkeys(name, "start")

        while self.is_running:
            if not self.vts.ws: logger.warning(f"Cannot execute animation '{name}', VTS is not connected."); break
            
            initial_data = await asyncio.gather(*[self.vts.read_param_value(p) for p in all_params])
            initials = {p: res.get('data', {}).get('value', 0) for p, res in zip(all_params, initial_data) if res}
            
            async with self._owner_lock:
                for p in all_params: self._param_owner[p] = anim_id
            
            current_vals = initials.copy()
            try:
                for stage in stages:
                    targets = {k: v for k, v in stage.items() if k not in ["Length", "KeepTime", "Easing", "visemes"]}
                    length = stage.get("Length", 0.1); keep_time = stage.get("KeepTime", 0.0); easer = self._resolve_easing(stage.get("Easing"))
                    await self._animate_step(anim_id, current_vals, targets, length, easer)
                    current_vals.update(targets)
                    if keep_time > 0: await asyncio.sleep(keep_time)
            except asyncio.CancelledError: break
            finally:
                async with self._owner_lock:
                    for p in all_params:
                        if self._param_owner.get(p) == anim_id: self._param_owner.pop(p, None)
                await self._compositor_clear_active(list(all_params))
            
            if not is_looping: break

        await self._trigger_animation_hotkeys(name, "end")
        logger.debug(f"Animation finished: {name} (ID: {anim_id})")

    async def _animate_step(self, anim_id, from_vals, to_vals, seconds, easer):
        fps = 60; dt = 1.0 / fps; steps = max(1, int(fps * seconds))
        for i in range(steps + 1):
            t = i / steps; progress = easer(t); current_frame_vals = {}
            for pid, target_val in to_vals.items():
                start_val = from_vals.get(pid, 0)
                current_frame_vals[pid] = start_val + (target_val - start_val) * progress
            await self._compositor_set_active(anim_id, current_frame_vals); await asyncio.sleep(dt)
    
    async def _compositor_set_active(self, anim_id, values):
        async with self._blend_lock, self._owner_lock:
            for pid, v in values.items():
                if self._param_owner.get(pid) == anim_id: self._blend_active_vals[pid] = v
    
    async def _compositor_clear_active(self, pids):
        async with self._blend_lock:
            for pid in pids:
                if pid in self._blend_active_vals:
                    task = asyncio.create_task(self._fade_out_parameter(pid))
                    self._fade_tasks.add(task)
                    task.add_done_callback(self._fade_tasks.discard)
    
    async def _fade_out_parameter(self, pid: str, duration: float = 0.25):
        try:
            async with self._blend_lock:
                start_val = self._blend_active_vals.get(pid, 0)
            if abs(start_val) < 0.01:
                async with self._blend_lock: self._blend_active_vals.pop(pid, None)
                return
            easer = self._resolve_easing("easeOut"); steps = max(1, int(60 * duration))
            for i in range(steps + 1):
                t = i / steps; progress = easer(t); current_val = start_val * (1.0 - progress)
                async with self._blend_lock, self._owner_lock:
                    if self._param_owner.get(pid) is not None: return
                    self._blend_active_vals[pid] = current_val
                await asyncio.sleep(1.0 / 60.0)
        finally:
            async with self._blend_lock, self._owner_lock:
                if self._param_owner.get(pid) is None: self._blend_active_vals.pop(pid, None)
    
    async def _restart_idle_tasks(self):
        for task in self._idle_tasks.values(): task.cancel()
        self._idle_tasks.clear()
        for pid, settings in self.idle_settings.items():
            if settings.get("magnitude", 0) > 0:
                self._idle_tasks[pid] = asyncio.create_task(self._run_idle_param_loop(pid, settings))
    
    async def _run_idle_param_loop(self, pid, settings):
        magnitude, speed = settings.get("magnitude", 0.1), settings.get("speed", 5.0)
        p_min, p_max = self.param_ranges.get(pid, {"min": -1, "max": 1}).values()
        center = 0.0 if p_min < 0 else p_min; start_val = center
        while self.is_running:
            try:
                if not self.vts.ws: await asyncio.sleep(1); continue
                half_span = (p_max - p_min) * magnitude / 2.0
                target = max(p_min, min(p_max, random.uniform(center - half_span, center + half_span)))
                fps = 30; dt = 1.0 / fps; steps = max(1, int(fps * speed * random.uniform(0.8, 1.2)))
                easer = self._resolve_easing("sineInOut")
                for i in range(steps + 1):
                    if pid in self._blend_active_vals: start_val = self._blend_idle_vals.get(pid, center); await asyncio.sleep(0.5); break
                    val = start_val + (target - start_val) * easer(i / steps)
                    async with self._blend_lock: self._blend_idle_vals[pid] = val
                    await asyncio.sleep(dt)
                start_val = target
            except asyncio.CancelledError: break
            except Exception as e: logger.error(f"Idle Error ({pid}): {e}", exc_info=True)
            
class VTSAnimator:
    """Main controller for the VTube Studio animation system."""
    def __init__(self):
        self.vts_api = VTubeStudioAPI()
        self.mixer: AnimationMixer | None = None
        self.lipsync_manager: LipSyncManager | None = None
        self.is_running = False

    async def start(self) -> bool:
        logger.info("--- Initializing VTube Studio Animation System ---")
        if not await self.vts_api.connect(): return False
        if not await self.vts_api.authenticate(): await self.vts_api.close(); return False
        
        # You can optionally list all available hotkeys on startup
        await self.vts_api.get_hotkeys_in_current_model()
        
        presets, params, idle, hotkeys = self._load_assets()
        if not presets: await self.vts_api.close(); return False
        
        self.lipsync_manager = LipSyncManager()
        if not self.lipsync_manager.viseme_map:
             self.lipsync_manager = None
        
        self.mixer = AnimationMixer(self.vts_api, presets, params, idle, hotkeys, self.lipsync_manager)
        await self.mixer.start()
        self.is_running = True
        logger.info("--- VTube Studio Animation System Ready ---")
        return True

    async def stop(self):
        if self.lipsync_manager and self.lipsync_manager.lipsync_task and not self.lipsync_manager.lipsync_task.done():
            self.lipsync_manager.lipsync_task.cancel()
        if self.mixer: await self.mixer.stop()
        await self.vts_api.close()
        self.is_running = False
        logger.info("VTube Studio Animation System shut down.")

    async def get_lipsync_data(self, audio_path: str) -> list | None:
        if self.lipsync_manager and self.is_running:
            return await self.lipsync_manager.get_lipsync_data(audio_path)
        return None

    def start_lipsync_animation(self, cues: list, start_signal: asyncio.Event) -> asyncio.Task | None:
        if self.lipsync_manager and self.is_running:
            if self.lipsync_manager.lipsync_task and not self.lipsync_manager.lipsync_task.done():
                self.lipsync_manager.lipsync_task.cancel()
            
            task = asyncio.create_task(self.lipsync_manager.animate_from_data(cues, start_signal))
            self.lipsync_manager.lipsync_task = task
            return task
        else:
            if not start_signal.is_set(): start_signal.set()
            return None
            
    def play_animation(self, name: str, is_looping: bool = False):
        if self.mixer and self.is_running: self.mixer.play(name, is_looping)
        else: logger.warning("Attempted to play animation, but animator is not running.")
    
    def get_available_animations(self) -> list:
        return list(self.mixer.presets.keys()) if self.mixer else []

    def get_animation_duration(self, animation_name: str) -> float:
        if not self.mixer or animation_name not in self.mixer.presets: return 0.0
        total_duration = 0.0
        preset_data = self.mixer.presets[animation_name]
        stages = preset_data if isinstance(preset_data, list) else [preset_data]
        for stage in stages:
            total_duration += stage.get("Length", 0.0)
            total_duration += stage.get("KeepTime", 0.0)
        return total_duration + 0.1 

    def _load_assets(self):
        presets, params, idle, hotkeys = {}, {}, {}, {}
        try:
            script_dir = os.path.dirname(__file__); config_dir = os.path.join(script_dir, 'config')
            with open(os.path.join(config_dir, "presets.json"), 'r') as f: presets = json.load(f)
            with open(os.path.join(config_dir, "parameters.json"), 'r') as f: params = json.load(f)
            with open(os.path.join(config_dir, "idle_settings.json"), 'r') as f: idle = json.load(f)
            logger.info(f"Loaded {len(presets)} presets, {len(params)} parameters, and {len(idle)} settings from {config_dir}.")

            # Load hotkey mappings (optional file)
            try:
                with open(os.path.join(config_dir, "hotkey_mappings.json"), 'r') as f:
                    hotkeys = json.load(f)
                logger.info(f"Loaded {len(hotkeys)} hotkey mappings.")
            except FileNotFoundError:
                logger.info("No hotkey_mappings.json found. Animations will run without hotkeys.")
            
            return presets, params, idle, hotkeys
        except FileNotFoundError as e:
            logger.error(f"Asset file not found: {e}. Check 'src/services/movement/config/' directory."); return None, None, None, None
        except Exception as e:
            logger.error(f"Error loading asset files: {e}"); return None, None, None, None
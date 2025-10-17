import pyvts
import websockets
import traceback
from config import VTS_PLUGIN_INFO, EMOTION_TO_VTS_ANIMATION
vts_client = None

async def initialize_vts():
    """Establishes and authenticates the connection to VTube Studio."""
    global vts_client
    if vts_client and vts_client.is_connected() and vts_client.is_authenticated():
        print("VTS client already connected and authenticated.")
        return True

    print("Initializing VTube Studio connection...")
    try:
        vts_client = pyvts.vts(plugin_info=VTS_PLUGIN_INFO)
        await vts_client.connect()
        await vts_client.request_authenticate()

        if vts_client.is_authenticated():
            print("VTS Plugin authenticated successfully.")
            return True
        else:
            print("VTS Plugin authentication FAILED. Check VTube Studio to allow the plugin.")
            await vts_client.close()
            vts_client = None
            return False
            
    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        print(f"VTS connection failed: {e}. Is VTS running with the API enabled?")
        vts_client = None
        return False
    except Exception as e:
        print(f"An unexpected error occurred during VTS initialization: {e}")
        traceback.print_exc()
        if vts_client:
            await vts_client.close()
        vts_client = None
        return False

async def trigger_vts_animation(emotion: str):
    """Triggers a VTS hotkey based on the detected emotion."""
    if not vts_client or not vts_client.is_authenticated():
        print("Cannot trigger VTS animation: client not ready.")
        return

    hotkey_name = EMOTION_TO_VTS_ANIMATION.get(emotion.lower())
    if not hotkey_name:
        # print(f"No VTS animation mapped for emotion: '{emotion}'")
        return

    try:
        print(f"Attempting to trigger VTS animation '{hotkey_name}' for emotion '{emotion}'.")
        hotkey_list_request = vts_client.vts_request.requestHotKeyList()
        response = await vts_client.request(hotkey_list_request)

        hotkey_id = None
        for hotkey in response['data']['availableHotkeys']:
            if hotkey['name'] == hotkey_name:
                hotkey_id = hotkey['hotkeyID']
                break

        if hotkey_id:
            trigger_request = vts_client.vts_request.requestTriggerHotKey(hotkey_id)
            await vts_client.request(trigger_request)
            print(f"Successfully triggered VTS Hotkey '{hotkey_name}'.")
        else:
            print(f"Error: Could not find a VTS hotkey named '{hotkey_name}'.")

    except Exception as e:
        print(f"Exception while triggering VTS animation '{hotkey_name}': {e}")
        traceback.print_exc()

async def close_vts_connection():
    """Closes the VTS websocket connection if it's open."""
    global vts_client
    if vts_client and vts_client.is_connected():
        print("Disconnecting from VTube Studio...")
        await vts_client.close()
        vts_client = None
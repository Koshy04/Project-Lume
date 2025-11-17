import asyncio
from src.services.movement.vts_client import VTSClient

async def main():
    client = VTSClient(
        ws_url="ws://localhost:8001",
        token_file="./vts_auth_token.txt",
        plugin_name="PyVTS Test",
        plugin_developer="You"
    )
    await client.connect()
    await client.authenticate()

    # Test reading parameters
    params = await client.read_param_values(["MouthOpen", "MouthSmile"])
    print(params)

    # Test setting parameter
    await client.set_params({"MouthSmile": 0.8})
    await asyncio.sleep(1)
    await client.close()

asyncio.run(main())
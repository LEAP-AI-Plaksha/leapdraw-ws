import asyncio
import websockets
import json
import sys

async def connect_to_server(username):
    uri = f"ws://localhost:8000/ws/test_room/{username}"
    async with websockets.connect(uri) as websocket:
        print(f"✅ Connected to WebSocket server as {username}.")
        
        # Listening for messages
        async def listen():
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    print(f"💬 {data.get('username', 'Server')}: {data.get('message', str(data))}")
                except websockets.ConnectionClosed:
                    print("❌ Disconnected from server.")
                    break

        # Start the listener in the background
        asyncio.create_task(listen())

        # Sending messages from terminal
        while True:
            message = input("Type a message: ")
            if message.lower() == "exit":
                print("👋 Exiting...")
                break
            await websocket.send(json.dumps({"type": "chat_send", "roomId": "test_room", "message": message}))

if __name__ == "__main__":
    username = sys.argv[1] if len(sys.argv) > 1 else "anonymous"
    asyncio.run(connect_to_server(username))

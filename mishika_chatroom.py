from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI()

# Store active chatroom connections
game_rooms = {}

# Message handler map
message_handlers = {}

def message_handler(message_type):
    def decorator(func):
        message_handlers[message_type] = func
        return func
    return decorator

### 🔹 Optimized Broadcast Logic
async def broadcast_message(room_id, message):
    if room_id in game_rooms:
        connections = game_rooms[room_id].get('clients', [])
        for connection in connections.copy():
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"❗ Failed to send message to {connection}: {e}")
                game_rooms[room_id]['clients'].remove(connection)  # Clean inactive connections

### 🔹 Room Creation Logic
class RoomCreateRequest(BaseModel):
    room_id: str

@app.post("/create_room/")
async def create_room(request: RoomCreateRequest):
    room_id = request.room_id
    game_rooms[room_id] = {
        'clients': [],
    }
    print("🚀 Room Created:", game_rooms)
    return {"status": "room created", "room_id": room_id}

### 🔹 Chat Handling Logic
@message_handler("chat_send")
async def handle_chat(room_id, username, data):
    await broadcast_message(room_id, {"type": "chat_receive", "username": username, "message": data.get("message")})

### 🔹 WebSocket Logic
@app.websocket("/ws/{room_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    await websocket.accept()

    # Ensure room exists
    if room_id not in game_rooms:
        await websocket.send_json({"type": "error", "message": "Room does not exist."})
        await websocket.close()
        return

    # Add client to room
    game_rooms[room_id]['clients'].append(websocket)

    await broadcast_message(room_id, {"type": "user_joined", "username": username})

    # Handle incoming messages
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            handler = message_handlers.get(message_type)
            if handler:
                await handler(room_id, username, data)
    except WebSocketDisconnect:
        if websocket in game_rooms[room_id]['clients']:
            game_rooms[room_id]['clients'].remove(websocket)
        
        # Cleanup empty room
        if not game_rooms[room_id]['clients']:
            del game_rooms[room_id]

        await broadcast_message(room_id, {"type": "user_left", "username": username})

### 🔹 Debug Endpoint
@app.get("/debug/game_rooms")
async def debug_game_rooms():
    return {"game_rooms": game_rooms}

if __name__ == "__main__":
    pass

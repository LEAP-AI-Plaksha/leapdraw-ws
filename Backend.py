from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import os
import random
import asyncio
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from supabase import create_client, Client
import os

# Load from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load environment variables from .env file
load_dotenv()

async def test_supabase():
    response = supabase.table("leaderboard").select("*").limit(5).execute()
    return response.data  # Should return some records if everything is connected

# Load all possible prompts from a file
with open('categories.txt', 'r') as f:
    ALL_PROMPTS = [line.strip() for line in f.readlines()]

app = FastAPI()

@app.get("/test_supabase")
async def test_supabase():
    response = supabase.table("leaderboard").select("*").limit(5).execute()
    return response.data  # Should return some records if everything is connected

# Store active game rooms and connections
game_rooms = {}
ROUND_DURATION = 30  # 30 seconds per round

def generate_room_id():
    return ''.join(random.choices("0123456789", k=6))  # 6-digit numeric room ID

class RoomCreateRequest(BaseModel):
    room_id: str = None

@app.post("/create_room/")
async def create_room(request: RoomCreateRequest): #handles room creation and ensures uniqueness.
    """
    Creates a new game room with a unique ID.
    """
    room_id = request.room_id or generate_room_id()
    while room_id in game_rooms:
        room_id = generate_room_id()
    
    game_rooms[room_id] = {
        'clients': set(),
        'scores': {},
        'round': 0,
        'drawer': None,
        'prompts': [],
        'game_started': False,  # Track game state
        'host': None  # Track host
    }
    print(f"🚀 Room {room_id} Created")
    return {"status": "room created", "room_id": room_id}

@app.get("/join_room/{room_id}")
async def join_room(room_id: str): 
    """
    Allows a user to join an existing room if it exists and hasn't started yet.
    """
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="Room not found") 
    
    if game_rooms[room_id]['game_started']:
        raise HTTPException(status_code=403, detail="Game already started")
    
    return {"status": "Room exists", "room_id": room_id}

@app.post("/start_game/{room_id}")
async def start_game(room_id: str):
    """
    API endpoint to start the game manually.
    """
    if room_id not in game_rooms:
        print(f"❌ Start Game Failed: Room {room_id} not found")
        raise HTTPException(status_code=404, detail="Room not found")
    
    if game_rooms[room_id]['game_started']:
        print(f"⚠️ Start Game Failed: Room {room_id} already started")
        raise HTTPException(status_code=400, detail="Game already started")
    
    print(f"✅ Game started in Room {room_id}")
    game_rooms[room_id]['game_started'] = True
    await next_round(room_id)

    return {"status": "Game started"}  # Ensure a valid JSON response

def assign_drawer(room_id):
    room = game_rooms.get(room_id)
    if room and room['clients']:
        room['drawer'] = random.choice(list(room['clients']))
        print(f"✏️ New Drawer Assigned: {room['drawer']}")

async def next_round(room_id):
    """
    Starts the next round and assigns a new drawer.
    """
    if room_id in game_rooms:
        game_rooms[room_id]['round'] += 1
        assign_drawer(room_id)  # Reassign drawer
        print(f"🔄 Starting round {game_rooms[room_id]['round']} in Room {room_id}")
        await broadcast_message(room_id, {"type": "new_round", "drawer": game_rooms[room_id]['drawer']})


@app.websocket("/ws/{room_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    """
    Handles WebSocket connections for real-time interactions.
    """
    await websocket.accept()
    
    if room_id not in game_rooms:
        await websocket.close()
        return

    game_rooms[room_id]['clients'].add(websocket)
    game_rooms[room_id]['scores'].setdefault(username, 0)
    
    if game_rooms[room_id]['host'] is None:
        game_rooms[room_id]['host'] = username  # Assign first player as host

    # Broadcast to all existing clients about the new player
    await broadcast_message(room_id, {"type": "player_joined", "username": username})
    
    # Send updated player list to all clients
    player_list = [client for client in game_rooms[room_id]['scores'].keys()]
    await broadcast_message(room_id, {"type": "player_list_update", "players": player_list})

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "update_score":
                new_score = data.get("score", 0)
                game_rooms[room_id]['scores'][username] += new_score  # Increment score
                await update_leaderboard(username, new_score, room_id)
                await broadcast_leaderboard(room_id)
    except WebSocketDisconnect:
        await handle_player_exit(room_id, username)
        
        # Assign new host if the current host leaves
        if game_rooms[room_id]['host'] == username and game_rooms[room_id]['clients']:
            game_rooms[room_id]['host'] = next(iter(game_rooms[room_id]['clients']))
            await broadcast_message(room_id, {"type": "new_host", "username": game_rooms[room_id]['host']})
    except Exception as e:
        print(f"❗ Error in WebSocket for {username} in {room_id}: {e}")
        game_rooms[room_id]['clients'].discard(websocket)

async def update_leaderboard(username, new_score, room_id):
    """
    Updates the leaderboard in the database.
    """
    print(f"🏆 Updating leaderboard: {username} gained {new_score} points in room {room_id}")

    # Check if the user already exists in the leaderboard
    response = supabase.table("leaderboard").select("id", "score").eq("username", username).eq("room_id", room_id).execute()
    if response.data:
        # User exists, update score
        entry_id = response.data[0]["id"]
        current_score = response.data[0]["score"]
        supabase.table("leaderboard").update({"score": current_score + new_score}).eq("id", entry_id).execute()
    else:
        # New user entry
        supabase.table("leaderboard").insert({"username": username, "score": new_score, "room_id": room_id}).execute()

async def broadcast_leaderboard(room_id):
    """
    Fetches leaderboard from Supabase and sends it to all clients in the room.
    """
    response = supabase.table("leaderboard").select("username", "score").eq("room_id", room_id).order("score", desc=True).execute()
    leaderboard = response.data  # List of players sorted by score

    await broadcast_message(room_id, {"type": "leaderboard_update", "leaderboard": leaderboard})

async def handle_player_exit(room_id, username):
    """
    Handles when a player exits the game.
    """
    if room_id in game_rooms:
        game_rooms[room_id]['clients'].discard(username)
        await broadcast_message(room_id, {"type": "player_left", "username": username})

        # Clean up empty rooms
        if not game_rooms[room_id]['clients']:
            del game_rooms[room_id]

async def broadcast_message(room_id, message):
    """
    Sends a message to all connected clients in a room.
    """
    if room_id in game_rooms:
        clients_copy = list(game_rooms[room_id]['clients'])  # Copy to avoid modification during iteration
        for client in clients_copy:
            try:
                await client.send_json(message)  # Send message
            except Exception as e:
                print(f"❌ Failed to send message to a client: {e}")
                game_rooms[room_id]['clients'].discard(client)  # Remove the client if disconnected


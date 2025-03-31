from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import os
import random
import asyncio
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from supabase import create_client, Client
import os
load_dotenv()
# Load from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load environment variables from .env file
load_dotenv()

# Load all possible prompts from a file
try:
    with open('categories.txt', 'r') as f:
        ALL_PROMPTS = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    ALL_PROMPTS = ["Draw something!", "Test prompt"]
    print("Warning: categories.txt not found, using default prompts")

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

def assign_drawer(room_id):
    room = game_rooms.get(room_id)
    if room and room['clients']:
        # Get a list of usernames from the client connections
        player_usernames = [client["username"] for client in room['clients']]
        if player_usernames:
            drawer_username = random.choice(player_usernames)
            # Find the connection info for this username
            for client in room['clients']:
                if client["username"] == drawer_username:
                    room['drawer'] = client
                    print(f"✏️ New Drawer Assigned: {drawer_username}")
                    return
    print("⚠️ Could not assign drawer - no clients in room")

# For testing purposes - allow pre-creating rooms with specific IDs
def create_test_room(room_id):
    """Create a room with a specific ID for testing purposes"""
    if room_id not in game_rooms:
        game_rooms[room_id] = {
            'clients': [],
            'scores': {},
            'round': 0,
            'drawer': None,
            'current_prompt': None,
            'game_started': False,
            'host': None
        }
        return True
    return False

# For testing - adding a direct method to broadcast messages
def direct_broadcast(room_id, message, exclude_username=None):
    """Direct method to broadcast messages (for testing)"""
    asyncio.create_task(broadcast_message(room_id, message, 
                                         exclude=[c for c in game_rooms.get(room_id, {}).get('clients', []) 
                                                 if c.get('username') == exclude_username]))

async def next_round(room_id):
    """
    Starts the next round and assigns a new drawer.
    """
    if room_id in game_rooms:
        game_rooms[room_id]['round'] += 1
        assign_drawer(room_id)  # Reassign drawer
        
        # Select a random prompt for the round
        if ALL_PROMPTS:
            current_prompt = random.choice(ALL_PROMPTS)
            game_rooms[room_id]['current_prompt'] = current_prompt
        else:
            game_rooms[room_id]['current_prompt'] = "Draw something"
        
        print(f"🔄 Starting round {game_rooms[room_id]['round']} in Room {room_id}")
        
        # Send new round info to all clients
        drawer_username = game_rooms[room_id]['drawer']["username"] if game_rooms[room_id]['drawer'] else None
        
        await broadcast_message(room_id, {
            "type": "new_round", 
            "round": game_rooms[room_id]['round'],
            "drawer": drawer_username,
            "prompt": game_rooms[room_id]['current_prompt']
        })
        
        # Start round timer
        asyncio.create_task(round_timer(room_id, ROUND_DURATION))

async def round_timer(room_id, duration):
    """
    Manages the countdown timer for each round.
    """
    for remaining in range(duration, 0, -1):
        if room_id not in game_rooms:
            return  # Room was deleted
            
        await broadcast_message(room_id, {
            "type": "timer_update",
            "remaining": remaining
        })
        await asyncio.sleep(1)
    
    # Time's up for this round
    await broadcast_message(room_id, {"type": "round_ended"})
    
    # Wait a bit before starting next round
    await asyncio.sleep(3)
    await next_round(room_id)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint that handles all game interactions.
    Initial connection doesn't associate with a room yet.
    """
    await websocket.accept()
    
    # Variables to track this connection's state
    current_room_id = None
    username = None
    connection_info = None
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "")
            
            # Handle initial connection messages (not yet in a room)
            if message_type == "create_room":
                if current_room_id:
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Already in a room"
                    })
                    continue
                
                # Get requested room ID or generate one
                requested_room_id = data.get("room_id")
                room_id = requested_room_id or generate_room_id()
                
                # Ensure uniqueness if specified
                if requested_room_id and requested_room_id in game_rooms:
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Room already exists"
                    })
                    continue
                
                # Generate a new ID if not specified
                while room_id in game_rooms:
                    room_id = generate_room_id()
                
                # Create the room
                game_rooms[room_id] = {
                    'clients': [],
                    'scores': {},
                    'round': 0,
                    'drawer': None,
                    'current_prompt': None,
                    'game_started': False,
                    'host': None
                }
                
                print(f"🚀 Room {room_id} Created")
                
                # Tell client about successful room creation
                await websocket.send_json({
                    "type": "room_created",
                    "room_id": room_id
                })
                
                # Don't join yet - wait for join_room message with username
            
            elif message_type == "join_room":
                if current_room_id:
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Already in a room"
                    })
                    continue
                
                # Get room ID and username from request
                room_id = data.get("room_id")
                new_username = data.get("username")
                
                if not room_id or not new_username:
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Missing room_id or username"
                    })
                    continue
                
                # Check if room exists
                if room_id not in game_rooms:
                    # Create the room if it doesn't exist (for testing)
                    game_rooms[room_id] = {
                        'clients': [],
                        'scores': {},
                        'round': 0,
                        'drawer': None,
                        'current_prompt': None,
                        'game_started': False,
                        'host': None
                    }
                
                # Check if game already started
                if game_rooms[room_id]['game_started']:
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Game already started"
                    })
                    continue
                
                # Check if username is already taken in this room
                existing_usernames = [client["username"] for client in game_rooms[room_id]['clients']]
                if new_username in existing_usernames:
                    await websocket.send_json({
                        "type": "error", 
                        "message": "Username already taken in this room"
                    })
                    continue
                
                # Join the room
                username = new_username
                current_room_id = room_id
                
                # Create connection info
                connection_info = {"websocket": websocket, "username": username}
                game_rooms[room_id]['clients'].append(connection_info)
                game_rooms[room_id]['scores'][username] = 0
                
                # Assign first player as host
                if game_rooms[room_id]['host'] is None:
                    game_rooms[room_id]['host'] = username
                    print(f"👑 {username} assigned as host for room {room_id}")
                
                # Send room status to the newly joined client
                await websocket.send_json({
                    "type": "room_joined",
                    "room_id": room_id,
                    "is_host": game_rooms[room_id]['host'] == username,
                    "game_started": game_rooms[room_id]['game_started'],
                    "round": game_rooms[room_id]['round'],
                    "players": list(game_rooms[room_id]['scores'].keys())
                })
                
                # Broadcast to all existing clients about the new player
                await broadcast_message(room_id, {
                    "type": "player_joined", 
                    "username": username
                })
                
                # Send updated player list to all clients
                player_list = list(game_rooms[room_id]['scores'].keys())
                await broadcast_message(room_id, {
                    "type": "player_list_update", 
                    "players": player_list
                })
            
            # Messages that require being in a room
            elif current_room_id and username:
                # Process messages for players already in a room
                if message_type == "chat_send":
                    chat_message = data.get("message", "")
                    if chat_message:
                        # Check if this is a correct guess during the game
                        is_correct_guess = False
                        room = game_rooms[current_room_id]
                        
                        if (room['game_started'] and 
                            'current_prompt' in room and 
                            room['drawer'] and 
                            room['drawer']["username"] != username and
                            chat_message.lower() == room['current_prompt'].lower()):
                            
                            points = 10  # Points for correct guess
                            room['scores'][username] += points
                            is_correct_guess = True
                            
                            # Update leaderboard
                            await update_leaderboard(username, points, current_room_id)
                            await broadcast_leaderboard(current_room_id)
                            
                            # Broadcast message about correct guess
                            await broadcast_message(current_room_id, {
                                "type": "correct_guess",
                                "username": username,
                                "points": points
                            })
                        
                        # Don't reveal the answer in chat if it's a correct guess
                        if not is_correct_guess:
                            # Broadcast chat message to all clients
                            # Use type 'chat_message' for compatibility with tests
                            await broadcast_message(current_room_id, {
                                "type": "chat_message",
                                "username": username,
                                "message": chat_message
                            })
                
                elif message_type == "draw_send":
                    room = game_rooms[current_room_id]
                    drawing_data = data.get("data", "")
                    
                    # Only check drawer permissions if game has started
                    if (not room['game_started'] or 
                        (room['drawer'] and room['drawer']["username"] == username)):
                        if drawing_data:
                            # Broadcast drawing to all clients except the drawer
                            # Use 'draw_update' type for test compatibility
                            await broadcast_message(current_room_id, {
                                "type": "draw_update",
                                "data": drawing_data
                            }, exclude=[connection_info])
                
                elif message_type == "clear_canvas":
                    room = game_rooms[current_room_id]
                    # Only check drawer permissions if game has started
                    if (not room['game_started'] or 
                        (room['drawer'] and room['drawer']["username"] == username)):
                        # Broadcast clear canvas command to all clients
                        await broadcast_message(current_room_id, {
                            "type": "clear_canvas"
                        })
                
                elif message_type == "start_game_request":
                    room = game_rooms[current_room_id]
                    # Only the host can start the game
                    if room['host'] == username and not room['game_started']:
                        room['game_started'] = True
                        print(f"✅ Game started in Room {current_room_id} by host {username}")
                        await next_round(current_room_id)
                
                elif message_type == "leave_room":
                    # Handle client-initiated room exit
                    await handle_disconnect(current_room_id, username, websocket)
                    current_room_id = None
                    username = None
                    connection_info = None
                    
                    await websocket.send_json({
                        "type": "room_left"
                    })
            
            else:
                # Client sent a game message without being in a room
                await websocket.send_json({
                    "type": "error", 
                    "message": "Not in a room yet"
                })
                
    except WebSocketDisconnect:
        # Handle unexpected disconnects
        if current_room_id and username:
            await handle_disconnect(current_room_id, username, websocket)
    
    except Exception as e:
        print(f"❗ Error in WebSocket: {e}")
        # Handle unexpected disconnects with error
        if current_room_id and username:
            await handle_disconnect(current_room_id, username, websocket)

async def handle_disconnect(room_id, username, websocket):
    """Handle a client disconnection from a room"""
    if room_id in game_rooms:
        # Remove client from the room
        room = game_rooms[room_id]
        room['clients'] = [c for c in room['clients'] if c["websocket"] != websocket]
        
        # Update scores dictionary if the user was in it
        if username in room['scores']:
            del room['scores'][username]
            
        # Notify other players
        await broadcast_message(room_id, {
            "type": "player_left", 
            "username": username
        })
        
        # Update player list
        player_list = list(room['scores'].keys())
        await broadcast_message(room_id, {
            "type": "player_list_update", 
            "players": player_list
        })
        
        # Assign new host if the current host leaves and there are still players
        if room['host'] == username and room['clients']:
            new_host = room['clients'][0]["username"]
            room['host'] = new_host
            await broadcast_message(room_id, {
                "type": "new_host", 
                "username": new_host
            })
        
        # If game in progress and drawer left, start a new round
        if (room['game_started'] and room['drawer'] and 
            room['drawer']["username"] == username and room['clients']):
            print(f"🎨 Drawer {username} left during the game, starting new round")
            await next_round(room_id)
        
        # Clean up empty rooms
        if not room['clients']:
            # Don't delete rooms during testing
            if not room_id.startswith('test_'):
                del game_rooms[room_id]
                print(f"🗑️ Empty room {room_id} deleted")

async def update_leaderboard(username, new_score, room_id):
    """
    Updates the leaderboard in the database.
    """
    print(f"🏆 Updating leaderboard: {username} gained {new_score} points in room {room_id}")

    try:
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
    except Exception as e:
        print(f"Error updating leaderboard: {e}")

async def broadcast_leaderboard(room_id):
    """
    Fetches leaderboard from Supabase and sends it to all clients in the room.
    """
    try:
        response = supabase.table("leaderboard").select("username", "score").eq("room_id", room_id).order("score", desc=True).execute()
        leaderboard = response.data  # List of players sorted by score
        await broadcast_message(room_id, {"type": "leaderboard_update", "leaderboard": leaderboard})
    except Exception as e:
        print(f"Error broadcasting leaderboard: {e}")

async def broadcast_message(room_id, message, exclude=None):
    """
    Sends a message to all connected clients in a room.
    
    Args:
        room_id: The ID of the room to broadcast to
        message: The message to broadcast
        exclude: List of connection objects to exclude from broadcast
    """
    if room_id in game_rooms:
        if exclude is None:
            exclude = []
            
        clients_copy = list(game_rooms[room_id]['clients'])  # Copy to avoid modification during iteration
        for client in clients_copy:
            if client in exclude:
                continue
                
            try:
                await client["websocket"].send_json(message)
            except Exception as e:
                print(f"❌ Failed to send message to {client['username']}: {e}")
                # Remove failed client from room
                game_rooms[room_id]['clients'] = [c for c in game_rooms[room_id]['clients'] if c != client]
                
                # If the client was in scores, remove them
                if client["username"] in game_rooms[room_id]['scores']:
                    del game_rooms[room_id]['scores'][client["username"]]

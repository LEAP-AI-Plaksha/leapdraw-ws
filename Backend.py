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

# AI player constants
AI_PLAYER_NAME = "AI"  # Default name for the AI player
AI_GUESS_DELAY_MIN = 3  # Minimum seconds before AI makes a guess
AI_GUESS_DELAY_MAX = 10  # Maximum seconds before AI makes a guess

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
MAX_ROUNDS = 2  # Maximum number of round sets (where each player draws once)

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
        # Add AI player to test rooms
        add_ai_player(room_id)
        return True
    return False

# For testing - adding a direct method to broadcast messages
def direct_broadcast(room_id, message, exclude_username=None):
    """Direct method to broadcast messages (for testing)"""
    asyncio.create_task(broadcast_message(room_id, message, 
                                         exclude=[c for c in game_rooms.get(room_id, {}).get('clients', []) 
                                                 if c.get('username') == exclude_username]))

async def save_final_scores(room_id):
    """
    Saves the final scores to Supabase at the end of the game.
    Increments existing totals for each player.
    """
    if room_id not in game_rooms:
        return
        
    room = game_rooms[room_id]
    
    for username, score in room['scores'].items():
        if score <= 0:
            continue  # Skip users with no points
            
        try:
            # Check if user exists in user_stats table
            response = supabase.table("user_stats").select("id", "total_score", "games_played").eq("username", username).execute()
            
            if response.data:
                # User exists, update their stats
                user_id = response.data[0]["id"]
                current_score = response.data[0]["total_score"]
                games_played = response.data[0]["games_played"]
                
                # Update with incremented values
                supabase.table("user_stats").update({
                    "total_score": current_score + score,
                    "games_played": games_played + 1
                }).eq("id", user_id).execute()
                
                print(f"📊 Updated stats for {username}: +{score} points, now has {current_score + score} total")
            else:
                # New user, create entry
                supabase.table("user_stats").insert({
                    "username": username,
                    "total_score": score,
                    "games_played": 1
                }).execute()
                
                print(f"📊 Created stats for new user {username}: {score} points")
        except Exception as e:
            print(f"❌ Error saving final score for {username}: {e}")

async def next_round(room_id):
    """
    Starts the next round and assigns a new drawer.
    """
    if room_id in game_rooms:
        room = game_rooms[room_id]
        room['round'] += 1
        
        # If we've gone through all players, increment the round set counter
        if room['round_players_queue'] is None or not room['round_players_queue']:
            # Increment the round set counter
            room['round_set'] += 1
            
            # Check if we've reached the maximum number of rounds
            if room['round_set'] > MAX_ROUNDS:
                # Save final scores to database before ending the game
                await save_final_scores(room_id)
                
                # Game is over, send game end message
                await broadcast_message(room_id, {
                    "type": "game_ended",
                    "final_scores": room['scores']
                })
                
                # Reset game state but keep room open
                room['game_started'] = False
                room['drawer'] = None
                room['current_prompt'] = None
                room['round'] = 0
                room['round_set'] = 0
                
                print(f"🏁 Game ended in Room {room_id} after {MAX_ROUNDS} rounds")
                return
            
            # Copy all usernames to create a queue of players for this round set
            room['round_players_queue'] = [client["username"] for client in room['clients'] 
                                          if not client.get("is_ai", False)]  # Exclude AI player from drawing
            random.shuffle(room['round_players_queue'])
            print(f"📋 New drawing queue for round set {room['round_set']} in room {room_id}: {room['round_players_queue']}")
        
        # Get the next drawer from the queue
        if room['round_players_queue']:
            next_drawer_username = room['round_players_queue'].pop(0)
            # Find the connection info for this username
            for client in room['clients']:
                if client["username"] == next_drawer_username:
                    room['drawer'] = client
                    print(f"✏️ New Drawer Assigned: {next_drawer_username}")
                    break
            else:
                # If player no longer in room, recursively call to get next player
                return await next_round(room_id)
        else:
            # Fallback to random assignment if queue is empty
            assign_drawer(room_id)
        
        # Select a random prompt for the round
        if ALL_PROMPTS:
            current_prompt = random.choice(ALL_PROMPTS)
            room['current_prompt'] = current_prompt
        else:
            room['current_prompt'] = "Draw something"
        
        # Reset round state
        room['prompt_guessed'] = False
        room['correct_guessers'] = set()  # Initialize empty set to track correct guessers
        room['round_time_remaining'] = ROUND_DURATION
        
        print(f"🔄 Starting round {room['round']} (set {room['round_set']}/{MAX_ROUNDS}) in Room {room_id}")
        
        # Send new round info to all clients
        drawer_username = room['drawer']["username"] if room['drawer'] else None
        
        await broadcast_message(room_id, {
            "type": "new_round", 
            "round": room['round'],
            "round_set": room['round_set'],
            "max_rounds": MAX_ROUNDS,
            "drawer": drawer_username,
            "prompt": room['current_prompt'] if drawer_username and drawer_username == room['drawer']["username"] else "???",
            "time": ROUND_DURATION
        })
        
        # Start round timer
        asyncio.create_task(round_timer(room_id, ROUND_DURATION))

async def round_timer(room_id, duration):
    """
    Manages the countdown timer for each round.
    """
    if room_id not in game_rooms:
        return
        
    room = game_rooms[room_id]
    
    for remaining in range(duration, 0, -1):
        # Check if room still exists
        if room_id not in game_rooms:
            return
        
        # Check if everyone except drawer has guessed correctly
        non_drawer_count = len(room['clients']) - 1  # everyone except drawer
        if non_drawer_count > 0 and len(room.get('correct_guessers', set())) >= non_drawer_count:
            print(f"🎯 Everyone guessed correctly in room {room_id}! Ending round early.")
            break
        
        room['round_time_remaining'] = remaining
            
        await broadcast_message(room_id, {
            "type": "timer_update",
            "remaining": remaining
        })
        await asyncio.sleep(1)
    
    # Time's up for this round or everyone guessed correctly
    
    # Award points to drawer based on how many players guessed correctly
    if room_id in game_rooms and 'drawer' in room and room['drawer']:
        drawer_username = room['drawer']["username"]
        correct_guessers = room.get('correct_guessers', set())
        
        # Award 5 points per correct guesser to the drawer
        drawer_points = len(correct_guessers) * 5
        
        if drawer_points > 0 and drawer_username in room['scores']:
            room['scores'][drawer_username] += drawer_points
            
            # Update leaderboard for drawer
            await update_leaderboard(drawer_username, drawer_points, room_id)
            
            # Broadcast drawer points earned
            await broadcast_message(room_id, {
                "type": "drawer_points",
                "username": drawer_username,
                "points": drawer_points,
                "num_correct_guesses": len(correct_guessers)
            })
    
    # Send round ended message
    if room_id in game_rooms:
        room = game_rooms[room_id]
        await broadcast_message(room_id, { 
            "type": "round_ended",
            "reason": "time_up" if room['round_time_remaining'] <= 0 else "all_guessed",
            "correct_answer": room['current_prompt'],
            "correct_guessers": list(room.get('correct_guessers', set())),
            "round_set": room['round_set'],
            "max_rounds": MAX_ROUNDS
        })
    
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
                    'round_set': 0,  # Track which set of rounds we're on (1-3)
                    'drawer': None,
                    'current_prompt': None,
                    'prompt_guessed': False,
                    'correct_guessers': set(),  # Track who has guessed correctly
                    'round_players_queue': None,
                    'round_time_remaining': ROUND_DURATION,
                    'game_started': False,
                    'host': None
                }
                
                print(f"🚀 Room {room_id} Created")
                
                # Add AI player to the room
                add_ai_player(room_id)
                
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
                        'round_set': 0,  # Track which set of rounds we're on (1-3)
                        'drawer': None,
                        'current_prompt': None,
                        'prompt_guessed': False,
                        'correct_guessers': set(),  # Track who has guessed correctly
                        'game_started': False,
                        'host': None
                    }
                    
                    # Add AI player to the room
                    add_ai_player(room_id)
                
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
                    "round_set": game_rooms[room_id].get('round_set', 0),
                    "max_rounds": MAX_ROUNDS,
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
                        
                        # Check if it's a correct guess and user hasn't already guessed
                        if (room['game_started'] and 
                            'current_prompt' in room and 
                            room['drawer'] and 
                            room['drawer']["username"] != username and
                            username not in room.get('correct_guessers', set()) and
                            chat_message.lower() == room['current_prompt'].lower()):
                            
                            points = 10  # Points for correct guess
                            room['scores'][username] += points
                            is_correct_guess = True
                            
                            # Add to set of correct guessers rather than ending round
                            if 'correct_guessers' not in room:
                                room['correct_guessers'] = set()
                            room['correct_guessers'].add(username)
                            
                            # Update leaderboard
                            await update_leaderboard(username, points, current_room_id)
                            await broadcast_leaderboard(current_room_id)
                            
                            # Broadcast message about correct guess
                            await broadcast_message(current_room_id, {
                                "type": "correct_guess",
                                "username": username,
                                "points": points,
                                "total_correct": len(room['correct_guessers'])
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
                    
                    # Only the drawer can send drawing data during an active game
                    # and only if the prompt hasn't been guessed yet
                    if (room['game_started'] and 
                        room['drawer'] and 
                        room['drawer']["username"] == username and
                        not room.get('prompt_guessed', False)):
                        
                        if drawing_data:
                            # Broadcast drawing to all clients except the drawer
                            await broadcast_message(current_room_id, {
                                "type": "draw_update",
                                "data": drawing_data
                            }, exclude=[connection_info])
                            
                            # Trigger AI player to make a guess
                            asyncio.create_task(ai_make_guess(current_room_id))
                
                elif message_type == "clear_canvas":
                    room = game_rooms[current_room_id]
                    # Only the drawer can clear the canvas during an active game
                    # and only if the prompt hasn't been guessed yet
                    if (room['game_started'] and 
                        room['drawer'] and 
                        room['drawer']["username"] == username and
                        not room.get('prompt_guessed', False)):
                        
                        # Broadcast clear canvas command to all clients
                        await broadcast_message(current_room_id, {
                            "type": "clear_canvas"
                        })
                
                elif message_type == "start_game_request":
                    room = game_rooms[current_room_id]
                    # Only the host can start the game
                    if room['host'] == username and not room['game_started']:
                        # Need at least 2 players to start
                        if len(room['clients']) < 2:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Need at least 2 players to start"
                            })
                            continue
                        
                        room['game_started'] = True
                        # Initialize empty queue - will be populated in next_round
                        room['round_players_queue'] = []
                        # Reset round set counter
                        room['round_set'] = 1  # Start with round set 1
                        room['round'] = 0  # Will be incremented in next_round
                        
                        print(f"✅ Game started in Room {current_room_id} by host {username}")
                        
                        # Broadcast game start to all clients
                        await broadcast_message(current_room_id, {
                            "type": "game_started",
                            "max_rounds": MAX_ROUNDS
                        })
                        
                        # Start the first round
                        await next_round(current_room_id)
                
                elif message_type == "restart_game":
                    room = game_rooms[current_room_id]
                    # Only the host can restart the game, and only if the game is not in progress
                    if room['host'] == username and not room['game_started']:
                        # Need at least 2 players to start
                        if len(room['clients']) < 2:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Need at least 2 players to start"
                            })
                            continue
                        
                        # Reset scores
                        for player in room['scores']:
                            room['scores'][player] = 0
                        
                        # Start a new game
                        room['game_started'] = True
                        room['round_players_queue'] = []
                        room['round_set'] = 1  # Start with round set 1
                        room['round'] = 0  # Will be incremented in next_round
                        
                        print(f"🔄 Game restarted in Room {current_room_id} by host {username}")
                        
                        # Broadcast game restart to all clients
                        await broadcast_message(current_room_id, {
                            "type": "game_restarted",
                            "max_rounds": MAX_ROUNDS
                        })
                        
                        # Start the first round
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
        
        # Remove user from correct guessers if they were in it
        if 'correct_guessers' in room and username in room['correct_guessers']:
            room['correct_guessers'].remove(username)
            
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
        
        # If game in progress and drawer left, end the round and go to next drawer
        if (room['game_started'] and room['drawer'] and 
            room['drawer']["username"] == username and room['clients']):
            
            # Send message that drawer left
            await broadcast_message(room_id, {
                "type": "drawer_left",
                "username": username
            })
            
            print(f"🎨 Drawer {username} left during the game, starting new round")
            
            # Wait a moment before starting new round
            await asyncio.sleep(2)
            await next_round(room_id)
        
        # If player was in the queue, remove them
        if 'round_players_queue' in room and username in room['round_players_queue']:
            room['round_players_queue'].remove(username)
        
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
                
            # Skip AI players as they don't have websockets
            if client.get("is_ai", False):
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

def add_ai_player(room_id):
    """
    Adds an AI player to the specified room
    """
    if room_id in game_rooms:
        room = game_rooms[room_id]
        
        # Check if AI player already exists in the room
        if any(client.get("username") == AI_PLAYER_NAME for client in room['clients']):
            # AI player already in the room
            return
        
        # Create AI player connection info (without a websocket)
        ai_connection = {"username": AI_PLAYER_NAME, "is_ai": True}
        
        # Add AI player to room
        room['clients'].append(ai_connection)
        room['scores'][AI_PLAYER_NAME] = 0
        
        print(f"🤖 AI player added to room {room_id}")
        
        # Broadcast to all clients about the new AI player
        asyncio.create_task(broadcast_message(room_id, {
            "type": "player_joined", 
            "username": AI_PLAYER_NAME,
            "is_ai": True
        }))
        
        # Send updated player list to all clients
        player_list = list(room['scores'].keys())
        asyncio.create_task(broadcast_message(room_id, {
            "type": "player_list_update", 
            "players": player_list
        }))

async def ai_make_guess(room_id):
    """
    Have the AI player make a guess based on the drawing
    For now, it always guesses "cat"
    """
    if room_id not in game_rooms:
        return
        
    room = game_rooms[room_id]
    
    # Only guess if AI player is in the room, not the drawer, and hasn't guessed correctly yet
    if (AI_PLAYER_NAME not in [c.get("username") for c in room['clients']] or 
        (room['drawer'] and room['drawer'].get("username") == AI_PLAYER_NAME) or
        AI_PLAYER_NAME in room.get('correct_guessers', set())):
        return
    
    # Random delay to make the AI seem more human-like
    await asyncio.sleep(random.uniform(AI_GUESS_DELAY_MIN, AI_GUESS_DELAY_MAX))
    
    # If the room no longer exists or AI has already guessed correctly, abort
    if (room_id not in game_rooms or 
        AI_PLAYER_NAME in game_rooms[room_id].get('correct_guessers', set())):
        return
        
    room = game_rooms[room_id]
    
    # AI always guesses "cat" for now - this will be replaced with model integration
    ai_guess = "cat"
    
    # Check if the guess is correct
    is_correct = ai_guess.lower() == room['current_prompt'].lower()
    
    # Broadcast AI's guess to all clients
    await broadcast_message(room_id, {
        "type": "chat_message",
        "username": AI_PLAYER_NAME,
        "message": ai_guess,
        "is_ai": True
    })
    
    # If guess is correct, handle scoring
    if is_correct:
        points = 10  # Points for correct guess
        room['scores'][AI_PLAYER_NAME] += points
        
        # Add to set of correct guessers
        if 'correct_guessers' not in room:
            room['correct_guessers'] = set()
        room['correct_guessers'].add(AI_PLAYER_NAME)
        
        # Update leaderboard
        await update_leaderboard(AI_PLAYER_NAME, points, room_id)
        await broadcast_leaderboard(room_id)
        
        # Broadcast message about correct guess
        await broadcast_message(room_id, {
            "type": "correct_guess",
            "username": AI_PLAYER_NAME,
            "points": points,
            "total_correct": len(room['correct_guessers']),
            "is_ai": True
        })

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
import random
import asyncio
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Set, Optional

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Load from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load all possible prompts from a file
with open('categories.txt', 'r') as f:
    ALL_PROMPTS = [line.strip() for line in f.readlines()]

# Store active game rooms and their state
game_rooms: Dict[str, Dict] = {}
ROUND_DURATION = 30  # 30 seconds per round
NUMBER_OF_ROUNDS = 5

async def get_leaderboard_from_supabase(room_id: str) -> List[Dict]:
    """Fetches the leaderboard for a given room from Supabase."""
    response = supabase.table("leaderboard").select("username, score").eq("room_id", room_id).order("score", desc=True).execute()
    return response.data


async def update_leaderboard_supabase(room_id: str, username: str, score: int):
    """Updates the leaderboard for a user in a room in Supabase."""
    try:
        table = supabase.table("leaderboard")
        select_query = table.select("id").eq("room_id", room_id).eq("username", username)
        response = select_query.execute()
        data = response.data
        if data:
            entry_id = data[0]["id"]
            update_query = table.update({"score": score}).eq("id", entry_id)
            updated_response = update_query.execute()
            print(f"Supabase update response: {updated_response}") # For debugging
        else:
            insert_query = table.insert({"room_id": room_id, "username": username, "score": score})
            inserted_response = insert_query.execute()
            print(f"Supabase insert response: {inserted_response}") # For debugging
    except Exception as e:
        print(f"Supabase error: {e}")

async def handle_connection(websocket: WebSocket, room_id: str, username: str):
    await websocket.accept()
    player = {"websocket": websocket, "username": username}

    if room_id not in game_rooms:
        game_rooms[room_id] = {
            'clients': set(),
            'scores': {},
            'round': 0,
            'drawer': None,
            'prompts': [],
            'game_started': False,
            'current_prompt': None,
            'correct_guesses': set(),
            'drawing_start_time': None,
            'drawing_data': []
        }

    room = game_rooms[room_id]
    room['clients'].add((websocket, username))  # Store as tuple
    room['scores'].setdefault(username, 0)
    await update_leaderboard_supabase(room_id, username, room['scores'][username])
    await broadcast_room_message(room_id, {"type": "user_joined", "username": username})
    await send_leaderboard(room_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("type")

            if action == "startGame":
                if not room['game_started']:
                    room['game_started'] = True
                    room['prompts'] = random.sample(ALL_PROMPTS, NUMBER_OF_ROUNDS)
                    room['round'] = 0
                    print(f"Starting game in room {room_id}")  # Debug
                    await next_round(room_id)
                else:
                    await websocket.send_json({"type": "error", "message": "Game already started."})

            elif action == "startDrawing":
                
                if any(ws == player['websocket'] and name == player['username'] for ws, name in room['clients']) and player['username'] == room['drawer'][1]:

                    room['drawing_start_time'] = asyncio.get_event_loop().time()
                    room['drawing_data'] = []
                    await broadcast_room_message(room_id, {"type": "drawingStarted", "drawer": username})
                else:
                    await websocket.send_json({"type": "error", "message": "You are not the drawer."})

            elif action == "drawData":
                if any(ws == player['websocket'] and name == player['username'] for ws, name in room['clients']) and player == room['drawer'] and room['drawing_start_time'] is not None and (asyncio.get_event_loop().time() - room['drawing_start_time'] <= ROUND_DURATION):
                    room['drawing_data'].append(data.get("data"))
                    await broadcast_room_message(room_id, {"type": "drawData", "data": data.get("data")}, exclude_self=True, sender_websocket=websocket)

            elif action == "guess":
                if any(ws == player['websocket'] and name == player['username'] for ws, name in room['clients']) and player != room['drawer'] and username not in room['correct_guesses'] and room['current_prompt']:
                    guess = data.get("guess", "").strip().lower()
                    if guess == room['current_prompt'].lower():
                        room['correct_guesses'].add(username)
                        points = 0
                        num_correct_guesses = len(room['correct_guesses'])
                        num_players = len(room['clients'])

                        if num_players >= 4:
                            if num_correct_guesses == 1:
                                points = 10
                            elif num_correct_guesses == 2:
                                points = 7
                            elif num_correct_guesses == 3:
                                points = 4
                        elif num_players < 4:
                            if num_correct_guesses == 1:
                                points = 10
                            elif num_correct_guesses == 2:
                                points = 7

                        room['scores'][username] += points
                        await update_leaderboard_supabase(room_id, username, room['scores'][username])
                        await broadcast_room_message(room_id, {"type": "correctGuess", "username": username, "guess": guess, "points": points})
                        await send_leaderboard(room_id)

                        if (num_players >= 4 and num_correct_guesses >= 3) or (num_players < 4 and num_correct_guesses >= 2):
                            await end_round(room_id)

            elif action == "getDrawingData":
                if any(ws == player['websocket'] and name == player['username'] for ws, name in room['clients']) and player != room['drawer']:
                    await websocket.send_json({"type": "drawingData", "data": room['drawing_data']})

            elif action == "getLeaderboard":
                await send_leaderboard(room_id, websocket)

    except WebSocketDisconnect:
        await handle_player_exit(room_id, player)
    except Exception as e:
        print(f"WebSocket error for user {username} in room {room_id}: {e}")
        await handle_player_exit(room_id, player)

async def next_round(room_id: str):
    room = game_rooms.get(room_id)
    if not room:
        print(f"Room {room_id} does not exist!")  # Debug
        return

    # Check if the game has reached its round limit BEFORE incrementing
    if room['round'] >= NUMBER_OF_ROUNDS:
        print(f"Game in room {room_id} has ended.")  # Debug
        await end_game(room_id)
        return

    # Proceed to the next round
    room['round'] += 1
    room['correct_guesses'] = set()
    room['drawing_start_time'] = None
    room['drawing_data'] = []

    # Select a drawer from the connected clients
    if room['clients']:
        room['drawer'] = random.choice(list(room['clients'])) 

    else:
        room['drawer'] = None

    # Assign the current prompt if available
    if len(room['prompts']) >= room['round']:
        room['current_prompt'] = room['prompts'][room['round'] - 1]
    else:
        room['current_prompt'] = None

    print(f"New round {room['round']} in room {room_id}, drawer: {room['drawer']}, prompt: {room['current_prompt']}")  # Debug

    # Broadcast the new round details
    await broadcast_room_message(room_id, {
        "type": "newRound",
        "round": room['round'],
        "drawer": room['drawer'][1] if room['drawer'] else None
    })

    # Send the prompt privately to the drawer
    if room['drawer']:
        await room['drawer'][0].send_json({
            "type": "yourPrompt",
            "prompt": room['current_prompt']
        })

    # Start round timer
    asyncio.create_task(round_timer(room_id))


async def round_timer(room_id: str):
    room = game_rooms.get(room_id)
    if not room:
        return

    await asyncio.sleep(ROUND_DURATION)
    if room['current_prompt'] and (len(room['correct_guesses']) < 3 if len(room['clients']) >= 4 else len(room['correct_guesses']) < 2):
        await broadcast_room_message(room_id, {"type": "roundEnded", "prompt": room['current_prompt']})
    if room['round'] <= NUMBER_OF_ROUNDS:
        await next_round(room_id)

async def end_round(room_id: str):
    room = game_rooms.get(room_id)
    if not room:
        return
    await broadcast_room_message(room_id, {"type": "roundEnded", "prompt": room['current_prompt']})
    if room['round'] < NUMBER_OF_ROUNDS:
        await asyncio.sleep(2) # Small delay before next round
        await next_round(room_id)
    elif room['round'] == NUMBER_OF_ROUNDS:
        await end_game(room_id)

async def end_game(room_id: str):
    room = game_rooms.get(room_id)
    if not room:
        return
    await broadcast_room_message(room_id, {"type": "gameOver", "scores": room['scores']})
    # Optionally, you might want to persist final scores or game history in Supabase here
    # For simplicity, we'll keep the room active for potential rematch

async def broadcast_room_message(room_id: str, message: Dict, exclude_self: bool = False, sender_websocket: Optional[WebSocket] = None):
    room = game_rooms.get(room_id)
    if room:
        print(f"Broadcasting {message} to room {room_id}")
        for client_websocket, client_username in list(room['clients']):
            try:
                if not exclude_self or client_websocket != sender_websocket:
                    print(f"Sending to {client_username}")
                    await client_websocket.send_json(message)
            except Exception as e:
                print(f"Error sending message to {client_username}: {e}")


async def send_leaderboard(room_id: str, websocket: Optional[WebSocket] = None):
    leaderboard_data = await get_leaderboard_from_supabase(room_id)
    message = {"type": "leaderboardUpdate", "leaderboard": leaderboard_data}
    if websocket:
        await websocket.send_json(message)
    else:
        await broadcast_room_message(room_id, message)

async def handle_player_exit(room_id: str, player: Dict):
    room = game_rooms.get(room_id)
    if room:
        room['clients'].discard((player['websocket'], player['username']))
        if player['username'] in room['scores']:
            del room['scores'][player['username']]
            # Optionally, you could remove the player's score from Supabase as well
        await broadcast_room_message(room_id, {"type": "user_left", "username": player['username']})
        if not room['clients']:
            del game_rooms[room_id]
            print(f"Room {room_id} is empty and has been deleted.")

@app.websocket("/ws/{room_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    await handle_connection(websocket, room_id, username)

# You might still want a simple endpoint to create a room initially if you want more control over the room ID
@app.post("/create_room/")
async def create_room():
    room_id = ''.join(random.choices("0123456789", k=6))
    while room_id in game_rooms:
        room_id = ''.join(random.choices("0123456789", k=6))
    game_rooms[room_id] = {'clients': set(), 'scores': {}, 'round': 0, 'drawer': None, 'prompts': [], 'game_started': False, 'current_prompt': None, 'correct_guesses': set(), 'drawing_start_time': None, 'drawing_data': []}
    print(f"Room {room_id} created via POST")
    return {"status": "room created", "room_id": room_id}

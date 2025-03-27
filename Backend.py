from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import os
import random
import asyncio
import json
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Load all possible prompts from a file
with open('categories.txt', 'r') as f:
    ALL_PROMPTS = [line.strip() for line in f.readlines()]

app = FastAPI()

# Store active game rooms and connections
game_rooms = {}
ROUND_DURATION = 30  # 30 seconds per round
TOTAL_ROUNDS = 5  # Default number of rounds per game


class RoomCreateRequest(BaseModel):
    room_id: str


@app.post("/create_room/")
async def create_room(request: RoomCreateRequest):
    room_id = request.room_id

    # Prevent duplicate room creation
    if room_id in game_rooms:
        raise HTTPException(status_code=400, detail="Room ID already exists")

    game_rooms[room_id] = {
        'clients': set(),
        'scores': {},
        'round': 0,
        'drawer': None,
        'prompts': [],
    }
    print(f"🚀 Room {room_id} Created")
    return {"status": "room created", "room_id": room_id}


class UpdateLeaderboardRequest(BaseModel):
    username: str
    score: int
    room_id: str


async def update_leaderboard(username, score, room_id):
    if room_id not in game_rooms:
        return

    # Increment the existing score instead of replacing it
    game_rooms[room_id]['scores'][username] = game_rooms[room_id]['scores'].get(username, 0) + score

    # Save leaderboard to file
    leaderboard_file = f"leaderboard_{room_id}.json"
    with open(leaderboard_file, "w") as file:
        json.dump(game_rooms[room_id]['scores'], file, indent=4)

    print(f"✅ Leaderboard updated for {room_id}: {game_rooms[room_id]['scores']}")


@app.post("/update_leaderboard/")
async def update_leaderboard_api(request: UpdateLeaderboardRequest):
    room_id, username, score = request.room_id, request.username, request.score

    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="Room not found")

    await update_leaderboard(username, score, room_id)
    await broadcast_leaderboard(room_id)

    return {"status": "Leaderboard updated"}


@app.websocket("/ws/{room_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    try:
        await websocket.accept()

        if room_id not in game_rooms:
            await websocket.close()
            return

        game_rooms[room_id]['clients'].add(websocket)
        game_rooms[room_id]['scores'].setdefault(username, 0)

        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "update_score":
                new_score = data.get("score", 0)
                await update_leaderboard(username, new_score, room_id)
                await broadcast_leaderboard(room_id)

    except WebSocketDisconnect:
        game_rooms[room_id]['clients'].discard(websocket)
        print(f"❌ {username} disconnected from {room_id}")
    except Exception as e:
        print(f"❗ Error in WebSocket for {username} in {room_id}: {e}")
        game_rooms[room_id]['clients'].discard(websocket)


async def broadcast_message(room_id, message):
    for connection in game_rooms[room_id]['clients'].copy():
        try:
            await connection.send_json(message)
        except:
            game_rooms[room_id]['clients'].remove(connection)


async def next_round(room_id):
    if room_id not in game_rooms:
        return

    game_rooms[room_id]['round'] += 1

    if game_rooms[room_id]['round'] > TOTAL_ROUNDS:
        await broadcast_message(room_id, {"type": "game_over", "scores": game_rooms[room_id]['scores']})
        del game_rooms[room_id]
        return

    drawer = random.choice(list(game_rooms[room_id]['scores'].keys()))
    game_rooms[room_id]['drawer'] = drawer
    prompt = random.choice(ALL_PROMPTS)
    game_rooms[room_id]['prompts'].append(prompt)

    await broadcast_message(room_id, {
        "type": "new_round",
        "round": game_rooms[room_id]['round'],
        "drawer": drawer,
        "prompt": prompt
    })

    await asyncio.sleep(ROUND_DURATION)
    await next_round(room_id)


async def broadcast_leaderboard(room_id):
    leaderboard = sorted(game_rooms[room_id]['scores'].items(), key=lambda x: x[1], reverse=True)
    leaderboard_data = [{"username": user, "score": score} for user, score in leaderboard]

    await broadcast_message(room_id, {"type": "leaderboard_update", "leaderboard": leaderboard_data})


@app.post("/next_round/{room_id}")
async def start_next_round(room_id: str):
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="Room not found")

    await next_round(room_id)
    return {"status": "Next round started"}

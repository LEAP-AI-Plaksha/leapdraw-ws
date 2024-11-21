import asyncio
import websockets
import json
import base64
from PIL import Image
import numpy as np
from io import BytesIO
from inference import predict_from_image, pad_image_to_square  # Import your ML model inference function
import random
from PIL import Image, ImageOps


# Load all possible prompts
with open('categories.txt', 'r') as f:
    ALL_PROMPTS = [line.strip() for line in f.readlines()]

# Dictionary to store rooms and their game state
rooms = {}


async def handle_connection(websocket, path):
    print("New client connected")
    
    try:
        async for message in websocket:
            try:
                # Parse the incoming message
                data = json.loads(message)
                action = data.get("action")
                room_id = data.get("roomId")

                if action == "createRoom" and room_id:
                    # Handle room creation
                    if room_id not in rooms:
                        # Select 3 unique prompts for this game
                        prompts = random.sample(ALL_PROMPTS, 3)
                        # Initialize game state
                        rooms[room_id] = {
                            "clients": [websocket],
                            "prompts": prompts,
                            "current_level": 0,
                            "scores": {"ai": 0, "human": 0}
                        }
                        await websocket.send(json.dumps({
                            "status": "success",
                            "message": f"Room {room_id} created successfully"
                        }))
                        print(f"Room {room_id} created with prompts: {prompts}")
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": f"Room {room_id} already exists"
                        }))
                
                elif action == "joinRoom" and room_id:
                    # Handle room joining
                    if room_id in rooms:
                        rooms[room_id]["clients"].append(websocket)
                        # Notify all clients in the room about the new player
                        for client in rooms[room_id]["clients"]:
                            if client != websocket:
                                await client.send(json.dumps({
                                    "status": "playerJoined",
                                    "roomId": room_id
                                }))
                        await websocket.send(json.dumps({
                            "status": "success",
                            "message": f"Joined room {room_id} successfully"
                        }))
                        print(f"Client joined room {room_id}")
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": f"Room {room_id} does not exist"
                        }))

                elif action == "getPrompt" and room_id:
                    # Send the current prompt to the /draw page
                    room = rooms.get(room_id)
                    if room:
                        current_level = room["current_level"]
                        if current_level < len(room["prompts"]):
                            prompt = room["prompts"][current_level]
                            await websocket.send(json.dumps({
                                "action": "prompt",
                                "prompt": prompt
                            }))
                        else:
                            await websocket.send(json.dumps({
                                "action": "gameOver",
                                "message": "All levels completed"
                            }))
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "Room does not exist"
                        }))

                elif action == "imageReceive" and room_id:
                    print("Received image")
                    image_data = data.get("imageData")
                    if room_id in rooms and image_data:
                        room = rooms[room_id]
                        # Decode the base64 image
                        img_bytes = base64.b64decode(image_data.split(",")[1])
                        img = Image.open(BytesIO(img_bytes)).convert('RGBA')  # Match Streamlit app

                        # Remove alpha channel
                        img = img.convert('RGB')

                        # Pad the image to make it square
                        img = pad_image_to_square(img)

                        # Ensure that all pixels that are non-black become white
                        img_array = np.array(img)
                        mask = (img_array != [0, 0, 0]).any(axis=2)
                        img_array[mask] = [255, 255, 255]
                        img = Image.fromarray(img_array)

                        # Save the processed image for debugging
                        img.save("processed_image.png")

                        # Run the ML model inference
                        try:
                            model_path = "quickdraw_resnet_model.tflite"
                            categories_file = "categories.txt"
                            predicted_category, confidence = predict_from_image(
                                model_path, img, categories_file=categories_file
                            )
                            print(f"AI Prediction: {predicted_category} with confidence {confidence:.2f}")

                            # Rest of your code...
                        except Exception as e:
                            print(f"Model inference error: {e}")
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": "Model inference failed"
                            }))
                        # Broadcast the image to other clients in the room
                        for client in room["clients"]:
                            if client != websocket:  # Don't send back to the sender
                                await client.send(json.dumps({
                                    "action": "imageDeliver",
                                    "roomId": room_id,
                                    "imageData": image_data
                                }))
                        print(f"Image broadcasted to room {room_id}")
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "Room does not exist or invalid image data"
                        }))

                elif action == "humanGuess" and room_id:
                    # Handle human guesses
                    guess = data.get("guess")
                    if room_id in rooms and guess:
                        room = rooms[room_id]
                        current_prompt = room["prompts"][room["current_level"]]

                        # Check if human guessed correctly
                        if guess.lower() == current_prompt.lower():
                            room["scores"]["human"] += 1
                            # Notify both clients to move to the next level
                            for client in room["clients"]:
                                await client.send(json.dumps({
                                    "action": "levelComplete",
                                    "winner": "Human",
                                    "aiScore": room["scores"]["ai"],
                                    "humanScore": room["scores"]["human"]
                                }))
                            # Advance to the next level
                            room["current_level"] += 1
                        else:
                            # Optionally, you can send feedback to the guesser
                            await websocket.send(json.dumps({
                                "action": "guessFeedback",
                                "correct": False
                            }))
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "Room does not exist or invalid guess"
                        }))

                elif action == "nextLevel" and room_id:
                    # Handle moving to the next level
                    room = rooms.get(room_id)
                    if room:
                        if room["current_level"] >= len(room["prompts"]):
                            # Game over, determine winner
                            ai_score = room["scores"]["ai"]
                            human_score = room["scores"]["human"]
                            if ai_score > human_score:
                                winner = "AI"
                            elif human_score > ai_score:
                                winner = "Human"
                            else:
                                winner = "Tie"

                            # Notify both clients of the game result
                            for client in room["clients"]:
                                await client.send(json.dumps({
                                    "action": "gameOver",
                                    "winner": winner,
                                    "aiScore": ai_score,
                                    "humanScore": human_score
                                }))
                            # Remove the room from the server
                            del rooms[room_id]
                            print(f"Game over. Room {room_id} deleted.")
                        else:
                            # Send a message to clients to start the next level
                            for client in room["clients"]:
                                await client.send(json.dumps({
                                    "action": "startLevel",
                                    "level": room["current_level"] + 1
                                }))
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "Room does not exist"
                        }))

                else:
                    # Handle invalid actions or missing room ID
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid action or missing roomId"
                    }))
            except json.JSONDecodeError:
                # Handle invalid JSON
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid JSON format"
                }))
    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected")
        # Clean up the client from rooms
        for room_id, room in list(rooms.items()):
            if websocket in room["clients"]:
                room["clients"].remove(websocket)
                if not room["clients"]:  # If room is empty, delete it
                    del rooms[room_id]
                    print(f"Room {room_id} deleted")
    except Exception as e:
        print("Error:", e)

# Start the WebSocket server
start_server = websockets.serve(handle_connection, "localhost", 8080)

print("WebSocket server is running on ws://localhost:8080")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

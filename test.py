import asyncio
import json
import websockets
import pytest # type: ignore
from httpx import AsyncClient

# Assuming your FastAPI app is running on http://localhost:8000
SERVER_URL = "ws://localhost:8000/ws"
HTTP_URL = "http://localhost:8000"

async def create_room_http():
    async with AsyncClient() as client:
        response = await client.post(f"{HTTP_URL}/create_room/")
        return response.json().get("room_id")

async def test_create_room():
    async with AsyncClient() as client:
        response = await client.post(f"{HTTP_URL}/create_room/")
        assert response.status_code == 200
        assert "room_id" in response.json()
        assert isinstance(response.json()["room_id"], str)
        print("Test Passed: Room creation successful")

async def test_websocket_connection():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/test_user") as websocket:
        assert websocket.open
        print("Test Passed: WebSocket connection successful")

async def test_join_room():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/user1") as ws1, \
               websockets.connect(f"{SERVER_URL}/{room_id}/user2") as ws2:
        await asyncio.sleep(1)  # Allow time for message propagation

        messages_user1 = []
        try:
            while True:  # Collect all messages user1 gets
                msg = await asyncio.wait_for(ws1.recv(), timeout=1)
                messages_user1.append(json.loads(msg))
        except asyncio.TimeoutError:
            pass  # No more messages

        print("Messages received by user1:", messages_user1)

        # Check if user1 received the "user_joined" message for user2
        assert any(msg.get("type") == "user_joined" and msg.get("username") == "user2" for msg in messages_user1), \
            "User1 did not receive 'user_joined' for user2!"

        print("Test Passed: Joining room successful")


async def test_start_game():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/host_user") as websocket:
        await websocket.send(json.dumps({"type": "startGame"}))
        message = json.loads(await websocket.recv())
        message = await websocket.receive()
        print("Received message:", message)

        assert message.get("type") == "newRound"
        assert message.get("drawer") == "host_user"
        assert "prompt" in message
        print("Test Passed: Starting game successful")

async def test_drawing():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/drawer_user") as drawer_ws, \
               websockets.connect(f"{SERVER_URL}/{room_id}/guesser_user") as guesser_ws:
        # Start the game to assign a drawer
        await drawer_ws.send(json.dumps({"type": "startGame"}))
        await asyncio.sleep(0.1)
        start_msg = json.loads(await drawer_ws.recv())
        assert start_msg.get("type") == "newRound"
        if start_msg.get("drawer") == "drawer_user":
            await drawer_ws.send(json.dumps({"type": "startDrawing"}))
            await drawer_ws.send(json.dumps({"type": "drawData", "data": [1, 2, 3]}))
            draw_guesser_msg = json.loads(await guesser_ws.recv())
            assert draw_guesser_msg.get("type") == "drawingStarted"
            draw_data_msg = json.loads(await guesser_ws.recv())
            assert draw_data_msg.get("type") == "drawData"
            assert draw_data_msg.get("data") == [1, 2, 3]
            print("Test Passed: Drawing functionality successful")
        else:
            print("Skipping drawing test as drawer_user was not assigned as drawer.")

async def test_guessing_correct():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/drawer") as drawer_ws, \
               websockets.connect(f"{SERVER_URL}/{room_id}/guesser") as guesser_ws:
        await drawer_ws.send(json.dumps({"type": "startGame"}))
        await asyncio.sleep(0.1)
        new_round_msg = json.loads(await drawer_ws.recv())
        if new_round_msg.get("drawer") == "drawer":
            # Let the drawer start drawing (though we don't fully simulate drawing here)
            await drawer_ws.send(json.dumps({"type": "startDrawing"}))
            # Guesser makes a correct guess (assuming the prompt is "test")
            await guesser_ws.send(json.dumps({"type": "guess", "guess": "test"}))
            correct_guess_msg = json.loads(await guesser_ws.recv())
            assert correct_guess_msg.get("type") == "correctGuess"
            assert correct_guess_msg.get("username") == "guesser"
            print("Test Passed: Correct guessing successful")
        else:
            print("Skipping correct guessing test as 'drawer' was not assigned as drawer.")

async def test_guessing_incorrect():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/drawer") as drawer_ws, \
               websockets.connect(f"{SERVER_URL}/{room_id}/guesser") as guesser_ws:
        await drawer_ws.send(json.dumps({"type": "startGame"}))
        await asyncio.sleep(0.1)
        new_round_msg = json.loads(await drawer_ws.recv())
        if new_round_msg.get("drawer") == "drawer":
            await drawer_ws.send(json.dumps({"type": "startDrawing"}))
            await guesser_ws.send(json.dumps({"type": "guess", "guess": "wrong guess"}))
            # We don't expect a "correctGuess" message
            try:
                await asyncio.wait_for(guesser_ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                print("Test Passed: Incorrect guessing handled (no 'correctGuess' message)")
            else:
                pytest.fail("Incorrect guess should not trigger a 'correctGuess' message.")
        else:
            print("Skipping incorrect guessing test as 'drawer' was not assigned as drawer.")

async def test_leaderboard_update():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/drawer") as drawer_ws, \
               websockets.connect(f"{SERVER_URL}/{room_id}/guesser") as guesser_ws:
        await drawer_ws.send(json.dumps({"type": "startGame"}))
        await asyncio.sleep(0.1)
        new_round_msg = json.loads(await drawer_ws.recv())
        if new_round_msg.get("drawer") == "drawer":
            await drawer_ws.send(json.dumps({"type": "startDrawing"}))
            await guesser_ws.send(json.dumps({"type": "guess", "guess": "test"})) # Assuming "test" is the prompt
            await guesser_ws.recv() # Consume correct guess message
            leaderboard_msg_guesser = json.loads(await guesser_ws.recv())
            assert leaderboard_msg_guesser.get("type") == "leaderboardUpdate"
            assert any(entry.get("username") == "guesser" and entry.get("score") > 0 for entry in leaderboard_msg_guesser.get("leaderboard", []))
            print("Test Passed: Leaderboard update successful")
        else:
            print("Skipping leaderboard update test as 'drawer' was not assigned as drawer.")

async def test_round_progression():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/user1") as ws1:
        await ws1.send(json.dumps({"type": "startGame"}))
        round_1_msg = json.loads(await ws1.recv())
        assert round_1_msg.get("type") == "newRound"
        assert round_1_msg.get("round") == 1
        # We'd need to simulate enough correct guesses or wait for the timer to advance rounds fully
        # For a basic test, we just check the first round start
        print("Test Passed: Round progression (initial round) successful")

async def test_player_disconnect():
    room_id = await create_room_http()
    async with websockets.connect(f"{SERVER_URL}/{room_id}/user1") as ws1, \
               websockets.connect(f"{SERVER_URL}/{room_id}/user2") as ws2:
        await asyncio.sleep(0.1)
        # Simulate user2 disconnecting by simply closing the connection
        await ws2.close()
        await asyncio.sleep(0.1)
        try:
            message_user1 = json.loads(await ws1.recv())
            assert message_user1.get("type") == "user_left"
            assert message_user1.get("username") == "user2"
            print("Test Passed: Player disconnect handled")
        except websockets.exceptions.ConnectionClosedOK:
            print("Test Passed: Player disconnect handled (connection closed)")
        except Exception as e:
            pytest.fail(f"Error during disconnect test: {e}")

async def main():
    await test_create_room()
    await test_websocket_connection()
    await test_join_room()
    await test_start_game()
    await test_drawing()
    await test_guessing_correct()
    await test_guessing_incorrect()
    await test_leaderboard_update()
    await test_round_progression()
    await test_player_disconnect()

if __name__ == "__main__":
    asyncio.run(main())

import serial
import json
import asyncio
import websockets
from threading import Thread

# Bluetooth COM port
BLUETOOTH_PORT = "COM5"  # Your port
BAUD_RATE = 115200

# WebSocket server
WEBSOCKET_HOST = "127.0.0.1"
WEBSOCKET_PORT = 5005

# Store connected clients
clients = set()

async def send_to_clients(data):
    """Send data to all connected WebSocket clients"""
    if clients:
        message = json.dumps(data) + "\n"
        disconnected = set()
        for client in clients:
            try:
                await client.send(message)
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            clients.discard(client)

async def handle_client(websocket):
    """Handle new WebSocket client"""
    clients.add(websocket)
    print(f"✓ Client connected: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            pass  # Just receive, don't need to do anything
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        print(f"✗ Client disconnected: {websocket.remote_address}")

def bluetooth_listener():
    """Listen to Bluetooth serial data"""
    try:
        ser = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=1)
        print(f"✓ Connected to {BLUETOOTH_PORT}\n")
        
        while True:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8').strip()
                    
                    if line and line != "READY":
                        values = [float(x) for x in line.split(',')]
                        data = {
                            'ax': values[0],
                            'ay': values[1],
                            'az': values[2],
                            'gx': values[3],
                            'gy': values[4],
                            'gz': values[5],
                            'temp': values[6]
                        }
                        
                        # Send to all WebSocket clients
                        asyncio.run_coroutine_threadsafe(
                            send_to_clients(data),
                            loop
                        )
                        
                        print(f"Accel: ({data['ax']:.2f}, {data['ay']:.2f}, {data['az']:.2f}) | "
                              f"Gyro: ({data['gx']:.2f}, {data['gy']:.2f}, {data['gz']:.2f}) | "
                              f"Temp: {data['temp']:.1f}°C")
            except ValueError:
                pass
            except Exception as e:
                print(f"Error: {e}")
                
    except serial.SerialException as es:
        print(f"Error: {es}")
        print(f"✗ Cannot open port {BLUETOOTH_PORT}")
        
        print("Available ports: Check Device Manager")

async def start_websocket_server():
    """Start WebSocket server"""
    print(f"✓ WebSocket server listening on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}\n")
    async with websockets.serve(handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    print("=" * 70)
    print("  ESP32 IMU Bluetooth to WebSocket Bridge")
    print("=" * 70)
    print(f"Bluetooth Port: {BLUETOOTH_PORT}")
    print(f"WebSocket: ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    print("=" * 70)
    
    # Get asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start Bluetooth listener in separate thread
    bt_thread = Thread(target=bluetooth_listener, daemon=True)
    bt_thread.start()
    
    # Start WebSocket server
    try:
        loop.run_until_complete(start_websocket_server())
    except KeyboardInterrupt:
        print("\n✓ Shutdown...")
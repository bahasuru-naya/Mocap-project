import asyncio
import json
import websockets
from bleak import BleakClient, BleakScanner

# --- Configuration ---
DEVICE_NAME = "ESP32_IMU_1"
CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# WebSocket server
WEBSOCKET_HOST = "127.0.0.1"
WEBSOCKET_PORT = 5005

# Store connected clients
clients = set()

# --- WebSocket Logic ---
async def handle_client(websocket):
    """Handle new WebSocket client connections."""
    clients.add(websocket)
    print(f"✓ WebSocket client connected: {websocket.remote_address}")
    try:
        # Keep the connection open and wait for it to close
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        print(f"✗ WebSocket client disconnected: {websocket.remote_address}")

async def broadcast_data(data):
    """Send data to all connected WebSocket clients."""
    if not clients:
        return
    message = json.dumps(data)
    # Broadcast to all connected clients safely
    websockets.broadcast(clients, message)

# --- BLE Logic ---
def notification_handler(sender, data: bytearray):
    """Callback function whenever the ESP32 sends a BLE notification."""
    try:
        # Decode the byte array sent by the ESP32 to a string
        line = data.decode('utf-8').strip()
        values = [float(x) for x in line.split(',')]
        
        # Ensure we received the 6 expected values from your ESP32 code
        if len(values) >= 6:
            sensor_data = {
                'ax': values[0],
                'ay': values[1],
                'az': values[2],
                'gx': values[3],
                'gy': values[4],
                'gz': values[5]
            }
            
            # Print to console (formatted nicely)
            print(f"BLE Received -> Accel: ({sensor_data['ax']:.2f}, {sensor_data['ay']:.2f}, {sensor_data['az']:.2f}) | "
                  f"Gyro: ({sensor_data['gx']:.2f}, {sensor_data['gy']:.2f}, {sensor_data['gz']:.2f})")
            
            # Broadcast over WebSockets (create a task since this callback is synchronous)
            asyncio.create_task(broadcast_data(sensor_data))
            
    except Exception as e:
        # Ignore malformed packets that might happen during connection
        pass

async def ble_task():
    """Scan, connect, and listen to the ESP32 via BLE."""
    print(f"\nScanning for BLE device '{DEVICE_NAME}'...")
    
    # Scan for 10 seconds to find the specific device name
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and d.name == DEVICE_NAME,
        timeout=10.0
    )
    
    if not device:
        print(f"✗ Could not find device '{DEVICE_NAME}'. Is it powered on?")
        return

    print(f"✓ Found '{DEVICE_NAME}' ({device.address}). Connecting...")
    
    try:
        async with BleakClient(device) as client:
            print("✓ Connected to BLE Device!")
            
            # Subscribe to the TX characteristic
            await client.start_notify(CHARACTERISTIC_UUID_TX, notification_handler)
            print("✓ Listening for IMU data... (Press Ctrl+C to stop)\n")
            
            # Keep the BLE connection alive forever
            while client.is_connected:
                await asyncio.sleep(1.0)
                
    except Exception as e:
        print(f"\nBLE Connection error: {e}")

# --- Main Application Loop ---
async def main():
    print("=" * 70)
    print("  ESP32 IMU WIRELESS BLE to WebSocket Bridge")
    print("=" * 70)
    print(f"WebSocket listening on: ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    print("=" * 70)
    
    # Start WebSocket server in the background
    async with websockets.serve(handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
        # Run the BLE connection task in the foreground
        await ble_task()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Shutdown complete.")
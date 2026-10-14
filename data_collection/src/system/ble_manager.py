import asyncio
import threading
import time
from bleak import BleakClient, BleakScanner

class BLEThread(threading.Thread):
    def __init__(self, target_devices, data_queue, status_queue, command_queue):
        super().__init__()
        self.target_devices = target_devices
        self.data_queue = data_queue
        self.status_queue = status_queue
        self.command_queue = command_queue
        self.running = True
        self.clients = {}
        self.loop = None
        self.connect_lock = None
        self.daemon = True
        self.CHARACTERISTIC_UUID_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        self.CHARACTERISTIC_UUID_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
        
    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.main_task())
        except Exception as e:
            print(f"BLE Event Loop Error: {e}")
        
    async def main_task(self):
        # Serialize scan/connect so one device reconnecting does not disturb the others' streams
        self.connect_lock = asyncio.Lock()
        # Create connection task for each device
        tasks = [self.connect_to_device(name) for name in self.target_devices]
        
        # Add the command processor task
        cmd_task = asyncio.create_task(self.process_commands())
        
        # Run all concurrently
        await asyncio.gather(*tasks, cmd_task)

    async def process_commands(self):
        while self.running:
            while not self.command_queue.empty():
                cmd = self.command_queue.get()
                device_name = cmd.get('device')
                action = cmd.get('action')
                if action == 'calibrate':
                    client = self.clients.get(device_name)
                    if client and client.is_connected:
                        for attempt in range(3):
                            try:
                                print(f"Sending CALIBRATE to {device_name} (Attempt {attempt+1})")
                                # Use response=False to prevent the PC from waiting for an ACK.
                                # This avoids the "Unreachable" errors caused by the BLE channel being busy.
                                await client.write_gatt_char(self.CHARACTERISTIC_UUID_RX, b"CALIBRATE", response=False)
                                print(f"Successfully sent CALIBRATE to {device_name}")
                                break # Exit retry loop on success
                            except Exception as e:
                                print(f"Failed to send command to {device_name} on attempt {attempt+1}: {e}")
                                await asyncio.sleep(0.5) # Wait before retry
                        
                        # Add a delay between commands to different devices to prevent overwhelming the PC's Bluetooth adapter
                        await asyncio.sleep(0.5)
            await asyncio.sleep(0.1)

    def notification_handler(self, device_name):
        # Closure to capture device_name
        def handler(sender, data: bytearray):
            try:
                line = data.decode('utf-8').strip()
                if line == "CALIB_DONE":
                    self.status_queue.put({'device': device_name, 'status': 'calibrated'})
                    return
                    
                values = line.split(',')
                if len(values) >= 7:
                    sensor_data = {
                        'node_id': device_name,
                        'sys_time': time.time(),
                        'node_time': int(values[0]),
                        'ax': float(values[1]), 'ay': float(values[2]), 'az': float(values[3]),
                        'gx': float(values[4]), 'gy': float(values[5]), 'gz': float(values[6])
                    }
                    if not self.data_queue.full():
                        self.data_queue.put(sensor_data)
            except Exception as e:
                # ignore malformed packets
                pass
        return handler

    async def connect_to_device(self, device_name):
        while self.running:
            client = None
            try:
                async with self.connect_lock:
                    self.status_queue.put({'device': device_name, 'status': 'scanning'})
                    device = await BleakScanner.find_device_by_filter(
                        lambda d, ad: d.name and d.name == device_name,
                        timeout=5.0
                    )
                    if device:
                        self.status_queue.put({'device': device_name, 'status': 'connecting'})
                        client = BleakClient(device)
                        await client.connect()
                        self.clients[device_name] = client
                        await client.start_notify(
                            self.CHARACTERISTIC_UUID_TX,
                            self.notification_handler(device_name)
                        )
                        self.status_queue.put({'device': device_name, 'status': 'connected'})
                    else:
                        self.status_queue.put({'device': device_name, 'status': 'not found'})

                # Stream outside the lock
                if client:
                    while client.is_connected and self.running:
                        await asyncio.sleep(1.0)
            except Exception as e:
                print(f"BLE error {device_name}: {e}")
                self.status_queue.put({'device': device_name, 'status': 'error'})
            finally:
                self.clients.pop(device_name, None)
                if client:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

            if self.running:
                await asyncio.sleep(2.0)

    def stop(self):
        self.running = False
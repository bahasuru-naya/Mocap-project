import threading
import time
from collections import deque

class SyncEngine(threading.Thread):
    def __init__(self, ble_data_queue, cv_data_queue, sync_queue, target_devices, recorder):
        super().__init__()
        self.ble_data_queue = ble_data_queue
        self.cv_data_queue = cv_data_queue
        self.sync_queue = sync_queue
        self.target_devices = target_devices
        self.recorder = recorder
        
        self.running = True
        self.daemon = True
        
        # Buffers to store data within a short sliding window
        self.imu_buffers = {dev: deque(maxlen=200) for dev in self.target_devices}
        self.cv_buffer = deque(maxlen=50)
        
        # Tolerance window (in seconds). 0.05s = 50ms (allows syncing 20-30Hz sources)
        self.sync_tolerance = 0.05 
        
    def run(self):
        while self.running:
            # Empty the queues into buffers
            while not self.ble_data_queue.empty():
                data = self.ble_data_queue.get()
                node_id = data['node_id']
                if node_id in self.imu_buffers:
                    self.imu_buffers[node_id].append(data)
                    
            while not self.cv_data_queue.empty():
                self.cv_buffer.append(self.cv_data_queue.get())
                
            # Attempt synchronization based on the oldest CV frame
            while len(self.cv_buffer) > 0:
                cv_data = self.cv_buffer[0]
                target_time = cv_data['sys_time']
                
                synced_frame = {'sys_time': target_time, 'cv': cv_data['pose'], 'imus': {}}
                
                all_ready = True
                
                for dev in self.target_devices:
                    buffer = self.imu_buffers[dev]
                    
                    closest_sample = None
                    min_diff = float('inf')
                    
                    # Find IMU sample closest to CV frame timestamp
                    for sample in buffer:
                        diff = abs(sample['sys_time'] - target_time)
                        if diff < min_diff:
                            min_diff = diff
                            closest_sample = sample
                            
                    if closest_sample and min_diff <= self.sync_tolerance:
                        synced_frame['imus'][dev] = closest_sample
                    else:
                        # Decide how to handle missing data
                        if len(buffer) > 0 and buffer[-1]['sys_time'] > target_time + self.sync_tolerance:
                            # We missed it completely, accept as missing
                            synced_frame['imus'][dev] = None
                        elif time.time() - target_time > 0.5:
                            # This CV frame is getting too old, give up waiting and mark IMU as missing
                            synced_frame['imus'][dev] = None
                        else:
                            # We might still get data for this timestamp. Wait.
                            all_ready = False
                            break
                            
                if all_ready:
                    # Pop the CV frame as it has been processed
                    self.cv_buffer.popleft()
                    
                    # Put to sync queue for UI
                    if not self.sync_queue.full():
                        self.sync_queue.put(synced_frame)
                        
                    # Send to recorder
                    self.recorder.record_frame(synced_frame)
                else:
                    break # Wait for more data
                    
            # Prevent 100% CPU usage
            time.sleep(0.005)
            
    def stop(self):
        self.running = False

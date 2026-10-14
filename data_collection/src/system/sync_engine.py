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

        self.imu_buffers = {dev: deque(maxlen=300) for dev in self.target_devices}
        self.cv_buffer = deque(maxlen=100)

        # Per-device clock mapping: node millis() -> PC time
        # offset = min(arrival_time - node_time) over a sliding window.
        # The minimum picks the least-delayed packets, which removes BLE batching jitter.
        self.offset_windows = {dev: deque(maxlen=500) for dev in self.target_devices}
        self.offsets = {dev: None for dev in self.target_devices}
        self.last_node_t = {dev: None for dev in self.target_devices}
        self.last_arrival = {dev: 0.0 for dev in self.target_devices}

        self.sync_tolerance = 0.04   # ideal match window (s)
        self.max_hold = 0.05         # accept nearest sample up to this far away (s)
        self.max_wait = 0.25         # max time a CV frame waits for late BLE data (s)
        self.dead_after = 0.3        # device silent this long = treat as absent, never wait for it (s)
        self.time_shift = 0.0        # tune if IMU consistently lags/leads video (s)

    def _ingest_imu(self, data):
        dev = data['node_id']
        if dev not in self.imu_buffers:
            return
        node_t = data['node_time'] / 1000.0
        last = self.last_node_t[dev]
        if last is not None and node_t < last - 1.0:
            # node rebooted, millis() restarted
            self.imu_buffers[dev].clear()
            self.offset_windows[dev].clear()
        self.last_node_t[dev] = node_t

        win = self.offset_windows[dev]
        win.append(data['sys_time'] - node_t)
        self.offsets[dev] = min(win)
        self.last_arrival[dev] = time.time()

        data['node_t'] = node_t
        self.imu_buffers[dev].append(data)

    def _ts(self, dev, sample):
        return sample['node_t'] + self.offsets[dev] + self.time_shift

    def run(self):
        while self.running:
            while not self.ble_data_queue.empty():
                self._ingest_imu(self.ble_data_queue.get())

            while not self.cv_data_queue.empty():
                self.cv_buffer.append(self.cv_data_queue.get())

            now = time.time()
            while len(self.cv_buffer) > 0:
                cv_data = self.cv_buffer[0]
                target_time = cv_data['sys_time']
                age = now - target_time

                synced_frame = {'sys_time': target_time, 'cv': cv_data['pose'], 'imus': {}}
                all_ready = True

                for dev in self.target_devices:
                    buffer = self.imu_buffers[dev]
                    alive = (now - self.last_arrival[dev]) < self.dead_after

                    closest, min_diff, newest_ts = None, float('inf'), None
                    for sample in buffer:
                        ts = self._ts(dev, sample)
                        newest_ts = ts if newest_ts is None or ts > newest_ts else newest_ts
                        diff = abs(ts - target_time)
                        if diff < min_diff:
                            min_diff, closest = diff, sample

                    if closest is not None and min_diff <= self.sync_tolerance:
                        synced_frame['imus'][dev] = closest
                        continue

                    # Data for this timestamp may still be in flight
                    still_coming = alive and (newest_ts is None or newest_ts < target_time + self.sync_tolerance)
                    if still_coming and age < self.max_wait:
                        all_ready = False
                        break

                    # Give up waiting: hold nearest sample if reasonably close, else missing
                    if closest is not None and min_diff <= self.max_hold:
                        synced_frame['imus'][dev] = closest
                    else:
                        synced_frame['imus'][dev] = None

                if all_ready:
                    self.cv_buffer.popleft()
                    if not self.sync_queue.full():
                        self.sync_queue.put(synced_frame)
                    self.recorder.record_frame(synced_frame)
                else:
                    break

            time.sleep(0.005)

    def stop(self):
        self.running = False
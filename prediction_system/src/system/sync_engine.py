import threading
import time
from collections import deque


class SyncEngine(threading.Thread):
    """
    Real-time version of the data-collection SyncEngine.

    The matching logic is the SAME as in data collection, so the model sees
    the same kind of input it was trained on:
      * node millis() is mapped to PC time per device (min-offset trick that
        removes BLE batching jitter),
      * each output slot takes the nearest sample within `sync_tolerance`,
        else holds the nearest one up to `max_hold`, else stays missing (None
        -> NaN -> interpolated by the PredictionEngine),
      * a frame waits (up to `max_wait`) for late BLE packets, but never for a
        device that has been silent for `dead_after` seconds.

    The only difference: there is no camera. In data collection every camera
    frame defined a target time; here a virtual clock produces target times at
    `frame_interval` (0.046 s = median camera frame interval of the training
    recording, ~21.6 Hz).

    Output frames:  {'sys_time': t, 'imu': {device: sample_dict | None}}
    """

    def __init__(self, ble_data_queue, sync_queue, target_devices, frame_interval=0.046):
        super().__init__()
        self.ble_data_queue = ble_data_queue
        self.sync_queue = sync_queue
        self.target_devices = target_devices

        self.running = True
        self.daemon = True

        self.imu_buffers = {dev: deque(maxlen=300) for dev in self.target_devices}

        # Per-device clock mapping: node millis() -> PC time
        # offset = min(arrival_time - node_time) over a sliding window.
        self.offset_windows = {dev: deque(maxlen=500) for dev in self.target_devices}
        self.offsets = {dev: None for dev in self.target_devices}
        self.last_node_t = {dev: None for dev in self.target_devices}
        self.last_arrival = {dev: 0.0 for dev in self.target_devices}

        # --- must equal the values used during data collection ---
        self.sync_tolerance = 0.06   # ideal match window (s)
        self.max_hold = 0.15         # accept nearest sample up to this far away (s)
        self.max_wait = 0.25         # max time a frame waits for late BLE data (s)
        self.dead_after = 0.3        # device silent this long = treat as absent (s)
        self.time_shift = 0.0        # use the same value you used while recording

        # Virtual clock replacing the camera
        self.frame_interval = frame_interval
        self.next_frame_time = None
        self.pending = deque(maxlen=100)     # target times waiting for their IMU data

        # Latest raw sample per device (dashboard live readout)
        self.latest_samples = {dev: None for dev in self.target_devices}

    # ------------------------------------------------------------------ #
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
        self.latest_samples[dev] = data

    def _ts(self, dev, sample):
        return sample['node_t'] + self.offsets[dev] + self.time_shift

    # ------------------------------------------------------------------ #
    def run(self):
        while self.running:
            while not self.ble_data_queue.empty():
                self._ingest_imu(self.ble_data_queue.get())

            now = time.time()

            # ---- virtual clock: create target times (like camera frames) ----
            if self.next_frame_time is None or now - self.next_frame_time > 1.0:
                self.next_frame_time = now           # first start / recover after a stall
            while self.next_frame_time <= now:
                self.pending.append(self.next_frame_time)
                self.next_frame_time += self.frame_interval

            # ---- resolve pending target times in order ----
            while self.pending:
                target_time = self.pending[0]
                age = now - target_time

                frame = {'sys_time': target_time, 'imu': {}}
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
                        frame['imu'][dev] = closest
                        continue

                    # Data for this timestamp may still be in flight
                    still_coming = alive and (newest_ts is None or newest_ts < target_time + self.sync_tolerance)
                    if still_coming and age < self.max_wait:
                        all_ready = False
                        break

                    # Give up waiting: hold nearest sample if reasonably close, else missing
                    if closest is not None and min_diff <= self.max_hold:
                        frame['imu'][dev] = closest
                    else:
                        frame['imu'][dev] = None

                if not all_ready:
                    break

                self.pending.popleft()
                # A frame with no IMU data at all carries no information: skip it
                if any(v is not None for v in frame['imu'].values()) and not self.sync_queue.full():
                    self.sync_queue.put(frame)

            time.sleep(0.005)

    def stop(self):
        self.running = False
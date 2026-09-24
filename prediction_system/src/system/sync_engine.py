import threading
import time
from collections import deque


class SyncEngine(threading.Thread):
    """
    Synchronizes IMU data streams from multiple BLE sensor nodes into
    unified time-aligned frames.

    Unlike the data_collection SyncEngine (which aligns IMU data to camera
    frames), this version runs at a fixed output rate and merges the 5 IMU
    streams purely by timestamp proximity — no camera required.
    """

    def __init__(self, ble_data_queue, sync_queue, target_devices):
        super().__init__()
        self.ble_data_queue = ble_data_queue
        self.sync_queue = sync_queue
        self.target_devices = target_devices

        self.running = True
        self.daemon = True

        # Per-device ring buffers for incoming BLE packets
        self.imu_buffers = {dev: deque(maxlen=200) for dev in target_devices}

        # Tolerance window (seconds). 50 ms matches the data_collection system.
        self.sync_tolerance = 0.05

        # Target output rate — ~30 Hz keeps inference pipeline fed without
        # flooding the queue faster than the model can consume.
        self.sync_interval = 0.033

        # Latest raw sample per device (used by the dashboard for live readout)
        self.latest_samples = {dev: None for dev in target_devices}

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        last_sync_time = time.time()

        while self.running:
            # 1) Drain the BLE queue into per-device buffers
            while not self.ble_data_queue.empty():
                data = self.ble_data_queue.get()
                node_id = data['node_id']
                if node_id in self.imu_buffers:
                    self.imu_buffers[node_id].append(data)
                    self.latest_samples[node_id] = data

            # 2) At fixed intervals, create a synchronized frame
            now = time.time()
            if now - last_sync_time >= self.sync_interval:
                last_sync_time = now
                synced_frame = self._build_synced_frame(now)

                if not self.sync_queue.full():
                    self.sync_queue.put(synced_frame)

                # Prune old samples (keep last 1 second)
                cutoff = now - 1.0
                for dev in self.target_devices:
                    buf = self.imu_buffers[dev]
                    while buf and buf[0]['sys_time'] < cutoff:
                        buf.popleft()

            # Yield CPU to prevent busy-waiting
            time.sleep(0.005)

    # ------------------------------------------------------------------ #
    #  Build a single synchronized frame
    # ------------------------------------------------------------------ #
    def _build_synced_frame(self, reference_time):
        """
        For each device, find the sample closest to *reference_time* within
        the sync tolerance.  If none is available the slot is set to ``None``
        so downstream consumers can interpolate.
        """
        frame = {'sys_time': reference_time, 'imu': {}}

        for dev in self.target_devices:
            buffer = self.imu_buffers[dev]

            closest_sample = None
            min_diff = float('inf')

            for sample in buffer:
                diff = abs(sample['sys_time'] - reference_time)
                if diff < min_diff:
                    min_diff = diff
                    closest_sample = sample

            if closest_sample and min_diff <= self.sync_tolerance:
                frame['imu'][dev] = closest_sample
            else:
                # Mark as missing — the PredictionEngine will interpolate
                frame['imu'][dev] = None

        return frame

    # ------------------------------------------------------------------ #
    def stop(self):
        self.running = False

import queue
import threading
import time


class StreamForwarder(threading.Thread):
    """Forward raw sensor telemetry and samples to the post-recording recorder."""

    def __init__(self, ble_data_queue, cv_data_queue, sync_queue, target_devices, recorder):
        super().__init__()
        self.ble_data_queue = ble_data_queue
        self.cv_data_queue = cv_data_queue
        self.sync_queue = sync_queue
        self.target_devices = target_devices
        self.recorder = recorder

        self.running = True
        self.daemon = True

        self.latest_samples = {device: None for device in target_devices}
        self.last_arrival = {device: 0.0 for device in target_devices}
        self.live_sample_timeout = 0.3

    def _drain_ble_queue(self):
        count = 0
        while True:
            try:
                sample = self.ble_data_queue.get_nowait()
            except queue.Empty:
                break

            self.recorder.record_imu_sample(sample)
            device = sample.get('node_id')
            if device in self.latest_samples:
                self.latest_samples[device] = sample
                self.last_arrival[device] = sample.get(
                    'arrival_sync_time', sample.get('sync_time', time.monotonic())
                )
            count += 1
        return count

    def _drain_cv_queue(self):
        count = 0
        while True:
            try:
                sample = self.cv_data_queue.get_nowait()
            except queue.Empty:
                break
            except OSError:
                break
            self.recorder.record_cv_sample(sample)
            count += 1
        return count

    def _publish_live_readout(self):
        now = time.monotonic()
        frame = {
            'sys_time': time.time(),
            'sync_time': now,
            'imus': {
                device: (
                    self.latest_samples[device]
                    if now - self.last_arrival[device] <= self.live_sample_timeout
                    else None
                )
                for device in self.target_devices
            },
        }
        try:
            self.sync_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.sync_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.sync_queue.put_nowait(frame)
            except queue.Full:
                pass

    def run(self):
        while True:
            imu_count = self._drain_ble_queue()
            cv_count = self._drain_cv_queue()
            if not self.running and not imu_count and not cv_count:
                break
            if imu_count:
                self._publish_live_readout()
            time.sleep(0.005)

    def stop(self):
        self.running = False

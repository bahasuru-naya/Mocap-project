import math
import queue
import threading
import time
from collections import deque

from common.clock_sync import ClockOffsetEstimator, SAMPLE_INTERVAL_SECONDS


class SyncEngine(threading.Thread):
    """Map IMU clocks to PC time and emit synchronized 20 Hz prediction frames."""

    def __init__(
        self, ble_data_queue, sync_queue, target_devices,
        frame_interval=SAMPLE_INTERVAL_SECONDS,
    ):
        super().__init__()
        self.ble_data_queue = ble_data_queue
        self.sync_queue = sync_queue
        self.target_devices = target_devices

        self.running = True
        self.daemon = True
        self.clock_sync = ClockOffsetEstimator(target_devices)
        self.clock_offsets = {device: None for device in target_devices}
        self.clock_sync_locked = False
        self.imu_buffers = {device: deque(maxlen=300) for device in target_devices}
        self.last_arrival = {device: 0.0 for device in target_devices}

        self.max_hold = 0.05
        self.max_interpolation_gap = 0.05
        self.max_wait = 0.25
        self.dead_after = 0.3
        self.time_shift = 0.0

        self.frame_interval = frame_interval
        self.next_frame_time = None
        self.pending = deque(maxlen=100)
        self.latest_samples = {device: None for device in target_devices}
        self.sync_fps = 0
        self._sync_frame_count = 0
        self._last_fps_time = time.monotonic()

    def _ingest_imu(self, data):
        device = data.get('node_id')
        if device not in self.imu_buffers:
            return

        arrival_time = data.get('arrival_sync_time', data.get('sync_time'))
        if arrival_time is None:
            arrival_time = time.monotonic()
        node_elapsed = self.clock_sync.observe_sample(
            device, data.get('node_time'), arrival_time
        )
        if node_elapsed is None:
            return

        sample = dict(data)
        sample['node_t'] = node_elapsed
        sample['arrival_sync_time'] = arrival_time
        self.imu_buffers[device].append(sample)
        self.last_arrival[device] = arrival_time
        self.latest_samples[device] = sample

    def _lock_clock_offsets(self):
        offsets = self.clock_sync.get_offsets()
        if any(offset is None for offset in offsets.values()):
            return False

        self.clock_offsets = self.clock_sync.freeze()
        self.clock_sync_locked = True
        print(
            "[SyncEngine] Clock offsets locked from pre-stream samples: "
            + ", ".join(
                f"{device}={offset:.6f}s"
                for device, offset in self.clock_offsets.items()
            )
        )
        return True

    def _ts(self, device, sample):
        return sample['node_t'] + self.clock_offsets[device] + self.time_shift

    def _interpolate_sample(self, device, target_time):
        before = None
        after = None
        before_ts = float('-inf')
        after_ts = float('inf')

        for sample in self.imu_buffers[device]:
            sample_time = self._ts(device, sample)
            if sample_time <= target_time and sample_time > before_ts:
                before, before_ts = sample, sample_time
            if sample_time >= target_time and sample_time < after_ts:
                after, after_ts = sample, sample_time

        if before is None or after is None:
            return None
        if before is after or before_ts == after_ts:
            return before

        gap = after_ts - before_ts
        if not 0 < gap or not self._within_tolerance(
            gap, self.max_interpolation_gap
        ):
            return None

        ratio = (target_time - before_ts) / gap
        interpolated = dict(before)
        for axis in ('ax', 'ay', 'az', 'gx', 'gy', 'gz'):
            interpolated[axis] = before[axis] + (
                after[axis] - before[axis]
            ) * ratio
        interpolated['sync_time'] = target_time
        interpolated['node_t'] = (
            target_time - self.clock_offsets[device] - self.time_shift
        )
        interpolated['node_time'] = int(interpolated['node_t'] * 1000)
        interpolated['interpolated'] = True
        return interpolated

    @staticmethod
    def _within_tolerance(value, tolerance):
        return value <= tolerance or math.isclose(
            value, tolerance, rel_tol=0.0, abs_tol=1e-9
        )

    def _find_closest(self, device, target_time):
        closest = None
        best_diff = float('inf')
        newest_ts = None

        for sample in self.imu_buffers[device]:
            sample_time = self._ts(device, sample)
            if newest_ts is None or sample_time > newest_ts:
                newest_ts = sample_time
            diff = abs(sample_time - target_time)
            if diff < best_diff:
                best_diff = diff
                closest = sample

        return closest, best_diff, newest_ts

    def _drain_ble_queue(self):
        count = 0
        while True:
            try:
                sample = self.ble_data_queue.get_nowait()
            except queue.Empty:
                break
            self._ingest_imu(sample)
            count += 1
        return count

    def _build_frame(self, target_time, now):
        frame = {
            'sys_time': target_time + (time.time() - now),
            'sync_time': target_time,
            'imu': {},
        }
        age = now - target_time

        for device in self.target_devices:
            alive = now - self.last_arrival[device] < self.dead_after
            closest, min_diff, newest_ts = self._find_closest(device, target_time)
            interpolated = self._interpolate_sample(device, target_time)
            if interpolated is not None:
                frame['imu'][device] = interpolated
                continue

            still_coming = alive and (newest_ts is None or newest_ts < target_time)
            if still_coming and age < self.max_wait:
                return None

            frame['imu'][device] = (
                closest
                if closest is not None and self._within_tolerance(
                    min_diff, self.max_hold
                )
                else None
            )
        return frame

    def run(self):
        while self.running:
            self._drain_ble_queue()
            now = time.monotonic()

            if not self.clock_sync_locked and not self._lock_clock_offsets():
                time.sleep(0.005)
                continue

            if self.next_frame_time is None or now - self.next_frame_time > 1.0:
                self.next_frame_time = now
            while self.next_frame_time <= now:
                self.pending.append(self.next_frame_time)
                self.next_frame_time += self.frame_interval

            while self.pending:
                target_time = self.pending[0]
                frame = self._build_frame(target_time, now)
                if frame is None:
                    break

                self.pending.popleft()
                if any(sample is not None for sample in frame['imu'].values()):
                    try:
                        self.sync_queue.put_nowait(frame)
                        self._sync_frame_count += 1
                    except queue.Full:
                        try:
                            self.sync_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.sync_queue.put_nowait(frame)
                            self._sync_frame_count += 1
                        except queue.Full:
                            pass

            if now - self._last_fps_time >= 1.0:
                self.sync_fps = self._sync_frame_count
                self._sync_frame_count = 0
                self._last_fps_time = now

            time.sleep(0.005)

    def stop(self):
        self.running = False

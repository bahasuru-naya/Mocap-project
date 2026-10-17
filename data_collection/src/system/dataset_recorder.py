import bisect
import csv
import math
import os
import threading
import time
from datetime import datetime

from common.clock_sync import ClockOffsetEstimator, SAMPLE_RATE_HZ


class DatasetRecorder:
    def __init__(self, target_devices):
        self.target_devices = target_devices
        self.recording = False
        self.file = None
        self.writer = None
        self.session_name = ""
        self.save_dir = "recordings"
        self.filepath = None
        self.trim_seconds = 5.0
        self._rows_since_flush = 0
        self._FLUSH_INTERVAL = 100  # flush to disk every N rows for crash safety
        self.sample_rate_hz = SAMPLE_RATE_HZ
        self.max_sync_gap = 0.05
        self.recording_start_sync = None
        self.recording_start_epoch = None
        self.imu_samples = {device: [] for device in target_devices}
        self.cv_samples = []
        self.clock_sync = ClockOffsetEstimator(target_devices)
        self.clock_offsets = {}
        self.lock = threading.Lock()
        os.makedirs(self.save_dir, exist_ok=True)
        
    def start_recording(self, session_name):
        with self.lock:
            sample_counts = self.clock_sync.sample_counts()
            offsets = self.clock_sync.get_offsets()
            unsynced = {
                device: sample_counts[device]
                for device in self.target_devices
                if offsets[device] is None
            }
            if unsynced:
                details = ', '.join(
                    f'{device} ({count}/{self.clock_sync.minimum_samples} samples)'
                    for device, count in unsynced.items()
                )
                raise RuntimeError(
                    f"Cannot start recording: clock sync is not ready for {details}. "
                    "Wait for each IMU to stream at least "
                    f"{self.clock_sync.minimum_samples} samples, then try again."
                )

            self.session_name = session_name
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.save_dir, f"{session_name}_{timestamp}.csv")
            self.clock_offsets = self.clock_sync.freeze()
            self.filepath = filepath
            self.imu_samples = {device: [] for device in self.target_devices}
            self.cv_samples = []
            self.recording_start_sync = time.monotonic()
            self.recording_start_epoch = time.time()
            self.recording = True
        return filepath

    def record_imu_sample(self, sample):
        with self.lock:
            device = sample.get('node_id')
            if device not in self.imu_samples:
                return None

            arrival_time = sample.get('arrival_sync_time', sample.get('sync_time'))
            if arrival_time is None:
                arrival_time = time.monotonic()
            if self.recording:
                node_elapsed = self.clock_sync.unwrap_sample(
                    device, sample.get('node_time')
                )
            else:
                node_elapsed = self.clock_sync.observe_sample(
                    device, sample.get('node_time'), arrival_time
                )
            if node_elapsed is None:
                return None
            sample['node_elapsed'] = node_elapsed

            if not self.recording:
                return None

            recorded_sample = dict(sample)
            recorded_sample['arrival_sync_time'] = sample.get(
                'arrival_sync_time', sample.get('sync_time')
            )
            recorded_sample['sync_time'] = (
                node_elapsed + self.clock_offsets[device]
            )
            self.imu_samples[device].append(recorded_sample)
            return recorded_sample['sync_time']

    def get_clock_offsets(self):
        """Return frozen recording offsets or current pre-record estimates."""
        with self.lock:
            if self.recording:
                return dict(self.clock_offsets)
            return self.clock_sync.get_offsets()

    def record_cv_sample(self, sample):
        with self.lock:
            if self.recording:
                self.cv_samples.append({
                    'sync_time': sample.get('sync_time', time.monotonic()),
                    'pose': [list(point) for point in sample.get('pose', [])],
                })
        
    def stop_recording(self):
        stop_sync = time.monotonic()
        with self.lock:
            self.recording = False
            if not self.filepath:
                return None
            filepath = self.filepath
            start_sync = self.recording_start_sync
            start_epoch = self.recording_start_epoch
            clock_sync = {
                device: {
                    'offset_seconds': self.clock_offsets[device],
                    'pre_record_samples': self.clock_sync.sample_counts()[device],
                }
                for device in self.target_devices
            }
            imu_samples = self.imu_samples
            cv_samples = self.cv_samples
            self.imu_samples = {device: [] for device in self.target_devices}
            self.cv_samples = []
            self.clock_sync.reset_observations()

        header = self._build_header()
        rows = self._build_synchronized_rows(
            stop_sync, start_sync, start_epoch, imu_samples, cv_samples,
        )
        with open(filepath, 'w', newline='') as destination:
            writer = csv.writer(destination)
            writer.writerow(header)
            writer.writerows(rows)

        missing_data = self._summarize_missing_data(header, rows)
        raw_counts = {
            'imu': {device: len(samples) for device, samples in imu_samples.items()},
            'cv': len(cv_samples),
        }
        duration = max(0.0, stop_sync - start_sync)
        trimmed_rows = min(
            int(duration * self.sample_rate_hz),
            int(self.trim_seconds * self.sample_rate_hz),
        )
        with self.lock:
            self.filepath = None
            self.recording_start_sync = None
            self.recording_start_epoch = None
        return {
            'filepath': filepath,
            'removed_rows': trimmed_rows,
            'remaining_rows': len(rows),
            'trim_seconds': self.trim_seconds,
            'missing_data': missing_data,
            'raw_samples': raw_counts,
            'clock_sync': clock_sync,
        }

    def _build_header(self):
        header = ['sys_time', 'local_time']
        for point_index in range(17):
            header.extend([
                f'cv_{point_index}_x',
                f'cv_{point_index}_y',
                f'cv_{point_index}_z',
            ])
        for device in self.target_devices:
            header.extend([
                f'{device}_ax', f'{device}_ay', f'{device}_az',
                f'{device}_gx', f'{device}_gy', f'{device}_gz',
            ])
        return header

    def _build_synchronized_rows(
        self, stop_sync, recording_start_sync, recording_start_epoch,
        imu_samples, cv_samples
    ):
        if recording_start_sync is None or recording_start_epoch is None:
            return []
        end_sync = stop_sync - self.trim_seconds
        if end_sync <= recording_start_sync:
            return []

        imu_streams = {}
        for device, samples in imu_samples.items():
            sorted_samples = sorted(
                (sample for sample in samples if sample.get('sync_time') is not None),
                key=lambda sample: sample['sync_time'],
            )
            imu_streams[device] = (
                [sample['sync_time'] for sample in sorted_samples],
                sorted_samples,
            )
        cv_stream = sorted(
            (sample for sample in cv_samples if sample.get('sync_time') is not None),
            key=lambda sample: sample['sync_time']
        )
        cv_times = [sample['sync_time'] for sample in cv_stream]

        interval = 1.0 / self.sample_rate_hz
        row_count = int((end_sync - recording_start_sync) / interval)
        rows = []
        for frame_index in range(row_count):
            target_sync = recording_start_sync + frame_index * interval
            system_time = recording_start_epoch + (
                target_sync - recording_start_sync
            )
            row = [
                repr(float(system_time)),
                datetime.fromtimestamp(system_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
            ]

            pose = self._nearest_pose(cv_stream, cv_times, target_sync)
            for point_index in range(17):
                if pose is not None and point_index < len(pose) and len(pose[point_index]) >= 3:
                    row.extend(pose[point_index][:3])
                else:
                    row.extend([0, 0, 0])

            for device in self.target_devices:
                sample_times, samples = imu_streams[device]
                values = self._interpolate_imu(sample_times, samples, target_sync)
                row.extend(values if values is not None else ['', '', '', '', '', ''])
            rows.append(row)
        return rows

    def _nearest_pose(self, samples, sample_times, target_sync):
        index = bisect.bisect_left(sample_times, target_sync)
        candidates = samples[max(0, index - 1):index + 1]
        nearest = min(
            candidates,
            key=lambda sample: abs(sample['sync_time'] - target_sync),
            default=None,
        )
        if nearest is None or not self._within_sync_gap(
            abs(nearest['sync_time'] - target_sync)
        ):
            return None
        return nearest['pose']

    def _interpolate_imu(self, sample_times, samples, target_sync):
        index = bisect.bisect_left(sample_times, target_sync)
        before = samples[index - 1] if index > 0 else None
        after = samples[index] if index < len(samples) else None

        if before is not None and after is not None:
            if before is after or before['sync_time'] == after['sync_time']:
                return [before[axis] for axis in ('ax', 'ay', 'az', 'gx', 'gy', 'gz')]
            gap = after['sync_time'] - before['sync_time']
            if 0 < gap and self._within_sync_gap(gap):
                ratio = (target_sync - before['sync_time']) / gap
                return [
                    before[axis] + (after[axis] - before[axis]) * ratio
                    for axis in ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
                ]

        nearest = min(
            (sample for sample in (before, after) if sample is not None),
            key=lambda sample: abs(sample['sync_time'] - target_sync),
            default=None,
        )
        if nearest is not None and self._within_sync_gap(
            abs(nearest['sync_time'] - target_sync)
        ):
            return [nearest[axis] for axis in ('ax', 'ay', 'az', 'gx', 'gy', 'gz')]
        return None

    def _within_sync_gap(self, gap):
        return gap <= self.max_sync_gap or math.isclose(
            gap, self.max_sync_gap, rel_tol=0.0, abs_tol=1e-9
        )

    def _summarize_missing_data(self, header, rows):
        total_rows = len(rows)
        imu_missing = {}
        for dev in self.target_devices:
            columns = [
                header.index(f'{dev}_{feature}')
                for feature in ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
            ]
            missing = sum(
                any(not str(row[column]).strip() for column in columns)
                for row in rows
            )
            imu_missing[dev] = {
                'missing': missing,
                'total': total_rows,
                'percentage': (missing / total_rows * 100) if total_rows else None,
            }

        missing_landmarks = 0
        for row in rows:
            for point_index in range(17):
                coordinates = row[2 + point_index * 3:5 + point_index * 3]
                # record_frame() zero-pads missing CV landmarks.
                if (
                    any(not str(value).strip() for value in coordinates)
                    or all(float(value) == 0 for value in coordinates)
                ):
                    missing_landmarks += 1

        total_landmarks = total_rows * 17
        return {
            'imu': imu_missing,
            'cv': {
                'missing': missing_landmarks,
                'total': total_landmarks,
                'percentage': (
                    missing_landmarks / total_landmarks * 100
                    if total_landmarks else None
                ),
            },
        }

    @staticmethod
    def _resample_rows(header, rows, rate_hz=20):
        if len(rows) < 2:
            return rows

        interval = 1.0 / rate_hz
        start_time = float(rows[0][0])
        output_rows = [rows[0]]
        frame_index = 1

        for left, right in zip(rows, rows[1:]):
            left_time = float(left[0])
            right_time = float(right[0])
            if right_time <= left_time:
                continue

            while start_time + frame_index * interval <= right_time:
                target_time = start_time + frame_index * interval
                ratio = (target_time - left_time) / (right_time - left_time)
                row = [
                    repr(float(target_time)),
                    datetime.fromtimestamp(target_time).strftime('%Y-%m-%d %H:%M:%S.%f'),
                ]

                for column_index in range(2, len(header)):
                    left_value = left[column_index]
                    right_value = right[column_index]
                    if left_value and right_value:
                        value = float(left_value) + (
                            float(right_value) - float(left_value)
                        ) * ratio
                        row.append(repr(value))
                    elif left_value or right_value:
                        row.append(left_value or right_value)
                    else:
                        row.append('')

                output_rows.append(row)
                frame_index += 1

        return output_rows
            
    def record_frame(self, synced_frame):
        if not self.recording or not self.writer:
            return
            
        system_time = synced_frame['sys_time']
        row = [repr(float(system_time)), datetime.fromtimestamp(system_time).strftime('%Y-%m-%d %H:%M:%S.%f')]
        
        # Add CV data
        for pt in synced_frame['cv']:
            row.extend(pt)
            
        # Pad CV data if for some reason it's missing points
        while len(row) < 2 + 17*3:
            row.extend([0, 0, 0])
            
        # Add IMU data
        for dev in self.target_devices:
            imu = synced_frame['imus'].get(dev)
            if imu:
                row.extend([imu['ax'], imu['ay'], imu['az'], imu['gx'], imu['gy'], imu['gz']])
            else:
                row.extend(['', '', '', '', '', '']) # Missing data represented as empty strings
                
        self.writer.writerow(row)

        # Periodic flush for crash safety
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._FLUSH_INTERVAL:
            self._rows_since_flush = 0
            try:
                self.file.flush()
            except Exception:
                pass

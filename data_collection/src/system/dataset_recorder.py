import csv
import os
import time
from datetime import datetime

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
        os.makedirs(self.save_dir, exist_ok=True)
        
    def start_recording(self, session_name):
        self.session_name = session_name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(self.save_dir, f"{session_name}_{timestamp}.csv")
        self.file = open(filepath, 'w', newline='')
        self.filepath = filepath
        self.writer = csv.writer(self.file)
        
        # Build CSV header
        header = ['sys_time', 'local_time']
        
        # 17 CV points * 3 coords (x,y,z) = 51 columns
        for i in range(17):
            header.extend([f'cv_{i}_x', f'cv_{i}_y', f'cv_{i}_z'])
            
        # IMU columns for each target device
        for dev in self.target_devices:
            header.extend([f'{dev}_ax', f'{dev}_ay', f'{dev}_az', f'{dev}_gx', f'{dev}_gy', f'{dev}_gz'])
            
        self.writer.writerow(header)
        self.recording = True
        return filepath
        
    def stop_recording(self):
        self.recording = False
        if not self.file or not self.filepath:
            return None

        filepath = self.filepath
        self.file.close()
        self.file = None
        self.writer = None

        with open(filepath, 'r', newline='') as source:
            rows = list(csv.reader(source))

        header = rows[:1]
        data_rows = self._resample_rows(header[0], rows[1:])
        removed_rows = 0

        if data_rows:
            last_timestamp = float(data_rows[-1][0])
            cutoff = last_timestamp - self.trim_seconds
            kept_rows = []
            for row in data_rows:
                if float(row[0]) < cutoff:
                    kept_rows.append(row)
                else:
                    removed_rows += 1
        else:
            kept_rows = []

        temporary_path = f"{filepath}.tmp"
        with open(temporary_path, 'w', newline='') as destination:
            csv.writer(destination).writerows(header + kept_rows)
        os.replace(temporary_path, filepath)

        self.filepath = None
        return {
            'filepath': filepath,
            'removed_rows': removed_rows,
            'remaining_rows': len(kept_rows),
            'trim_seconds': self.trim_seconds,
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

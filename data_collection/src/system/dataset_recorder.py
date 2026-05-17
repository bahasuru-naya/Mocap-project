import csv
import os
import time

class DatasetRecorder:
    def __init__(self, target_devices):
        self.target_devices = target_devices
        self.recording = False
        self.file = None
        self.writer = None
        self.session_name = ""
        self.save_dir = "recordings"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def start_recording(self, session_name):
        self.session_name = session_name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(self.save_dir, f"{session_name}_{timestamp}.csv")
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Build CSV header
        header = ['sys_time']
        
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
        if self.file:
            self.file.close()
            self.file = None
            
    def record_frame(self, synced_frame):
        if not self.recording or not self.writer:
            return
            
        row = [synced_frame['sys_time']]
        
        # Add CV data
        for pt in synced_frame['cv']:
            row.extend(pt)
            
        # Pad CV data if for some reason it's missing points
        while len(row) < 1 + 17*3:
            row.extend([0, 0, 0])
            
        # Add IMU data
        for dev in self.target_devices:
            imu = synced_frame['imus'].get(dev)
            if imu:
                row.extend([imu['ax'], imu['ay'], imu['az'], imu['gx'], imu['gy'], imu['gz']])
            else:
                row.extend(['', '', '', '', '', '']) # Missing data represented as empty strings
                
        self.writer.writerow(row)

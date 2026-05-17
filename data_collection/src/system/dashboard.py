from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Dashboard(QMainWindow):
    def __init__(self, ble_cmd_queue, frame_queue, sync_queue, ble_status_queue, recorder, target_devices):
        super().__init__()
        self.ble_cmd_queue = ble_cmd_queue
        self.frame_queue = frame_queue
        self.sync_queue = sync_queue
        self.ble_status_queue = ble_status_queue
        self.recorder = recorder
        self.target_devices = target_devices
        self.device_status = {dev: "disconnected" for dev in target_devices}
        
        self.initUI()
        
        # Timer to poll queues and update UI without blocking
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30) # ~33 FPS
        
        self.frames_received = 0
        self.last_fps_time = QDateTime.currentMSecsSinceEpoch()

    def initUI(self):
        self.setWindowTitle("AI MoCap - Data Collection Dashboard")
        self.setGeometry(100, 100, 1200, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # === Left Panel ===
        left_panel = QVBoxLayout()
        
        # Video Feed
        self.video_label = QLabel("Camera Feed (Waiting...)")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; color: #FFFFFF; font-size: 18px; border-radius: 5px;")
        self.video_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.video_label)
        
        # Controls Group
        controls_group = QGroupBox("System Controls")
        controls_layout = QHBoxLayout()
        
        self.calib_btn = QPushButton("Calibrate All Sensors")
        self.calib_btn.setMinimumHeight(40)
        self.calib_btn.setMinimumWidth(140)
        self.calib_btn.clicked.connect(self.calibrate_all)
        controls_layout.addWidget(self.calib_btn)
        
        self.session_input = QLineEdit()
        self.session_input.setPlaceholderText("Enter Session Name...")
        self.session_input.setMinimumHeight(40)
        controls_layout.addWidget(self.session_input)
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.setMinimumWidth(150)
        self.record_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px;")
        self.record_btn.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_btn)
        
        controls_group.setLayout(controls_layout)
        left_panel.addWidget(controls_group)
        
        # Sync Status Group
        sync_group = QGroupBox("Synchronization Monitor")
        sync_layout = QHBoxLayout()
        self.sync_rate_label = QLabel("Sync Rate: 0 FPS")
        self.sync_rate_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        sync_layout.addWidget(self.sync_rate_label)
        sync_group.setLayout(sync_layout)
        left_panel.addWidget(sync_group)
        
        main_layout.addLayout(left_panel, 2)
        
        # === Right Panel ===
        right_panel = QVBoxLayout()
        self.status_labels = {}
        
        # Create a card for each sensor
        for dev in self.target_devices:
            group = QGroupBox(dev)
            group.setStyleSheet("QGroupBox { border: 2px solid #757575; border-radius: 5px; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
            
            glayout = QVBoxLayout()
            
            lbl_status = QLabel("Status: Scanning...")
            lbl_status.setStyleSheet("color: #ffa000;") # Orange
            
            lbl_data = QLabel("Data: N/A")
            lbl_data.setMinimumHeight(40)
            
            btn_calib = QPushButton("Calibrate Node")
            btn_calib.clicked.connect(lambda checked, d=dev: self.calibrate_single(d))
            
            glayout.addWidget(lbl_status)
            glayout.addWidget(lbl_data)
            glayout.addWidget(btn_calib)
            
            group.setLayout(glayout)
            right_panel.addWidget(group)
            
            self.status_labels[dev] = {'status': lbl_status, 'data': lbl_data, 'group': group}
            
        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)

    def calibrate_all(self):
        for dev in self.target_devices:
            self.ble_cmd_queue.put({'device': dev, 'action': 'calibrate'})
            
    def calibrate_single(self, device):
        self.ble_cmd_queue.put({'device': device, 'action': 'calibrate'})
        
    def toggle_recording(self):
        if not self.recorder.recording:
            name = self.session_input.text().strip() or "mocap_session"
            self.recorder.start_recording(name)
            self.record_btn.setText("Stop Recording")
            self.record_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; border-radius: 4px;")
            self.session_input.setEnabled(False)
        else:
            self.recorder.stop_recording()
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px;")
            self.session_input.setEnabled(True)

    def update_ui(self):
        # 1. Update Video Frame
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            # Scale to fit label while preserving aspect ratio
            pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)
            
        # 2. Update BLE Status
        while not self.ble_status_queue.empty():
            stat = self.ble_status_queue.get()
            dev = stat['device']
            status = stat['status']
            self.device_status[dev] = status
            
            if dev in self.status_labels:
                lbl = self.status_labels[dev]['status']
                grp = self.status_labels[dev]['group']
                
                lbl.setText(f"Status: {status.upper()}")
                
                if status == "connected":
                    lbl.setStyleSheet("color: #2e7d32;")
                    grp.setStyleSheet("QGroupBox { border: 2px solid #2e7d32; border-radius: 5px; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
                elif status == "calibrated":
                    lbl.setStyleSheet("color: #1565c0;")
                    grp.setStyleSheet("QGroupBox { border: 2px solid #1565c0; border-radius: 5px; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
                elif status in ["error", "not found", "disconnected"]:
                    lbl.setStyleSheet("color: #c62828;")
                    grp.setStyleSheet("QGroupBox { border: 2px solid #c62828; border-radius: 5px; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
                else:
                    lbl.setStyleSheet("color: #ffa000;")
                    grp.setStyleSheet("QGroupBox { border: 2px solid #ffa000; border-radius: 5px; margin-top: 1ex; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
                    
        # 3. Update Sync Data
        while not self.sync_queue.empty():
            synced_frame = self.sync_queue.get()
            self.frames_received += 1
            
            for dev in self.target_devices:
                imu = synced_frame['imus'].get(dev)
                if imu:
                    text = f"A: {imu['ax']:.2f}, {imu['ay']:.2f}, {imu['az']:.2f}\n" \
                           f"G: {imu['gx']:.2f}, {imu['gy']:.2f}, {imu['gz']:.2f}"
                    self.status_labels[dev]['data'].setText(text)
                else:
                    self.status_labels[dev]['data'].setText("Data: MISSING")
                    
        # 4. Calculate Sync FPS
        current_time = QDateTime.currentMSecsSinceEpoch()
        if current_time - self.last_fps_time >= 1000:
            fps = self.frames_received
            self.sync_rate_label.setText(f"Sync Rate: {fps} FPS")
            self.frames_received = 0
            self.last_fps_time = current_time

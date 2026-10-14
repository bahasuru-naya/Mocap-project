from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os

class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Countdown")
        self.setFixedSize(400, 500)
        self.countdown = 10
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #ffffff; font-family: 'Segoe UI', Inter, sans-serif; }
        """)
        
        layout = QVBoxLayout(self)
        
        instruction = QLabel("Please stand in the Position of Attention\n(Straight, arms at sides)")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet("font-size: 18px; font-weight: bold; color: #bb86fc;")
        layout.addWidget(instruction)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)

        image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "img",
        "PositionofAttention.jpg"
        )
        
        pixmap = QPixmap(image_path)

        # Scale while keeping aspect ratio
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                250, 300,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.img_label.setPixmap(pixmap)
        else:
            self.img_label.setText("Image not found!")

        layout.addWidget(self.img_label)
        
        self.timer_label = QLabel(f"Calibrating in {self.countdown}...")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 36px; color: #cf6679; font-weight: bold;")
        layout.addWidget(self.timer_label)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)
        
    

    def update_timer(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.timer_label.setText(f"Calibrating in {self.countdown}...")
        else:
            self.timer.stop()
            self.accept()

class RecordingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Get Ready to Record")
        self.setFixedSize(400, 125)
        self.countdown = 5
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #ffffff; font-family: 'Segoe UI', Inter, sans-serif; }
        """)
        
        layout = QVBoxLayout(self)
        
        instruction = QLabel("Get into your starting position...")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet("font-size: 18px; font-weight: bold; color: #bb86fc;")
        layout.addWidget(instruction)
        
        self.timer_label = QLabel(f"Recording starts in {self.countdown}...")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 20px; color: #03dac6; font-weight: bold;")
        layout.addWidget(self.timer_label)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)
        
    def update_timer(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.timer_label.setText(f"Recording starts in {self.countdown}...")
        else:
            self.timer.stop()
            self.accept()

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
        
        self.total_frames_recorded = 0
        
        self.initUI()
        
        # Timer to poll queues and update UI without blocking
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30) # ~33 FPS
        
        self.frames_received = 0
        self.last_fps_time = QDateTime.currentMSecsSinceEpoch()

    def initUI(self):
        self.setWindowTitle("AI MoCap - Data Collection Dashboard")
        self.setGeometry(100, 50, 1600, 930)
        
        # --- Modern Premium Dark Theme ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                color: #e0e0e0;
                font-family: 'Segoe UI', Inter, sans-serif;
            }
            QGroupBox {
                border: 2px solid #333333;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                padding-left: 10px;
                padding-right: 10px;
                font-size: 20px;
                font-weight: bold;
                background-color: #1e1e1e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #bb86fc;
                left: 10px;
            }
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 15px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
            QLineEdit {
                background-color: #2c2c2c;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 10px;
                color: #ffffff;
                font-size: 16px;
            }
            QLineEdit:focus {
                border: 1px solid #bb86fc;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # === Left Panel ===
        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)
        
        # Video Feed
        self.video_label = QLabel("Camera Feed (Waiting...)")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                color: #777777;
                font-size: 20px;
                border: 2px solid #333333;
                border-radius: 10px;
            }
        """)
        self.video_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.video_label, stretch=1)

        # Controls Group
        controls_group = QGroupBox("System Controls")
        controls_group_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 15, 0, 15)
        controls_layout.setSpacing(15)

        time_row = QHBoxLayout()
        self.system_time_label = QLabel("System Time: --:--:--")
        self.system_time_label.setAlignment(Qt.AlignCenter)
        self.system_time_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #03dac6;")
        time_row.addWidget(self.system_time_label)
        time_row.setAlignment(Qt.AlignCenter)
        
        self.calib_btn = QPushButton("Calibrate All Sensors")
        self.calib_btn.setMinimumHeight(55)
        self.calib_btn.setStyleSheet("""
            QPushButton {
                background-color: #1f618d; 
                font-size: 16px; 
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.calib_btn.clicked.connect(self.start_calibration_sequence)
        controls_layout.addWidget(self.calib_btn)
        
        self.session_input = QLineEdit()
        self.session_input.setPlaceholderText("Enter Session Name...")
        self.session_input.setMinimumHeight(55)
        controls_layout.addWidget(self.session_input)
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumHeight(55)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; 
                font-size: 18px; 
            }
            QPushButton:hover { background-color: #388e3c; }
        """)
        self.record_btn.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_btn)
        
        controls_group_layout.addLayout(time_row)
        controls_group_layout.addLayout(controls_layout)        
        controls_group.setLayout(controls_group_layout)
        left_panel.addWidget(controls_group)
        
        # Sync Status Group - Enhanced
        sync_group = QGroupBox("System information")
        sync_layout = QHBoxLayout()
        sync_layout.setContentsMargins(20, 20, 20, 20)
        
        self.sync_rate_label = QLabel("Sync Rate: 0 FPS")
        self.sync_rate_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #b3b3b3;")
        sync_layout.addWidget(self.sync_rate_label)
        
        self.active_nodes_label = QLabel("Active Nodes: 0/5")
        self.active_nodes_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #b3b3b3;")
        self.active_nodes_label.setAlignment(Qt.AlignCenter)
        sync_layout.addWidget(self.active_nodes_label)
        
        self.recording_status_label = QLabel("Status: IDLE")
        self.recording_status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #b3b3b3;")
        self.recording_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sync_layout.addWidget(self.recording_status_label)
        
        sync_group.setLayout(sync_layout)
        left_panel.addWidget(sync_group)
        
        main_layout.addLayout(left_panel, 7)
        
        # === Right Panel (Nodes) ===
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; } QScrollBar:vertical { background: #121212; width: 12px; } QScrollBar::handle:vertical { background: #333333; border-radius: 6px; }")
        
        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: transparent;")
        right_scroll.setWidget(right_widget)
        
        right_panel = QVBoxLayout(right_widget)
        right_panel.setSpacing(10)
        right_panel.setContentsMargins(0, 0, 10, 0)
        self.status_labels = {}
        
        # Create a card for each sensor
        for dev in self.target_devices:
            group = QGroupBox(dev)
            
            glayout = QVBoxLayout()
            glayout.setContentsMargins(15, 15, 15, 10)
            glayout.setSpacing(8)
            
            lbl_status = QLabel("Status: SCANNING")
            lbl_status.setStyleSheet("color: #ffa000; font-size: 14px; font-weight: bold; letter-spacing: 1px;") # Orange
            
            lbl_data = QLabel("Data: Waiting for stream...")
            lbl_data.setMinimumHeight(40)
            lbl_data.setStyleSheet("font-size: 13px; font-family: 'Consolas', monospace; color: #a0a0a0;")
            
            btn_calib = QPushButton("Calibrate Node")
            btn_calib.setMinimumHeight(35)
            btn_calib.setStyleSheet("""
                QPushButton { background-color: #2c3e50; font-size: 13px; padding: 5px; }
                QPushButton:hover { background-color: #34495e; }
            """)
            btn_calib.clicked.connect(lambda checked, d=dev: self.calibrate_single(d))
            
            glayout.addWidget(lbl_status)
            glayout.addWidget(lbl_data)
            glayout.addWidget(btn_calib)
            
            group.setLayout(glayout)
            right_panel.addWidget(group)
            
            self.status_labels[dev] = {'status': lbl_status, 'data': lbl_data, 'group': group}
            
        right_panel.addStretch()
        main_layout.addWidget(right_scroll, 3)

    def start_calibration_sequence(self):
        dialog = CalibrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.calibrate_all()

    def calibrate_all(self):
        for dev in self.target_devices:
            self.ble_cmd_queue.put({'device': dev, 'action': 'calibrate'})
            
    def calibrate_single(self, device):
        self.ble_cmd_queue.put({'device': device, 'action': 'calibrate'})
        
    def toggle_recording(self):
        if not self.recorder.recording:
            dialog = RecordingDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                name = self.session_input.text().strip() or "mocap_session"
                self.recorder.start_recording(name)
                self.total_frames_recorded = 0
                self.record_btn.setText("Stop Recording")
                self.record_btn.setStyleSheet("""
                    QPushButton { background-color: #c62828; font-size: 18px; }
                    QPushButton:hover { background-color: #e53935; }
                """)
                self.session_input.setEnabled(False)
                self.recording_status_label.setText("Status: RECORDING (0)")
                self.recording_status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #cf6679;")
        else:
            save_result = self.recorder.stop_recording()
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("""
                QPushButton { background-color: #2e7d32; font-size: 18px; }
                QPushButton:hover { background-color: #388e3c; }
            """)
            self.session_input.setEnabled(True)
            self.recording_status_label.setText("Status: IDLE")
            self.recording_status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #b3b3b3;")

            if save_result:
                message_box = QMessageBox(self)
                message_box.setIcon(QMessageBox.Information)
                message_box.setWindowTitle("Dataset Saved")
                message_box.setText(
                    f"Dataset saved:\n{os.path.basename(save_result['filepath'])}\n\n"
                    f"Removed the last {save_result['trim_seconds']:.0f} seconds "
                    f"({save_result['removed_rows']} records) to exclude unintended movement."
                )
                message_box.setStyleSheet("""
                    QMessageBox {
                        background-color: #1e1e1e;
                    }
                    QMessageBox QLabel {
                        color: #ffffff;
                        font-size: 14px;
                        min-width: 420px;
                    }
                    QMessageBox QPushButton {
                        background-color: #333333;
                        color: #ffffff;
                        border: 1px solid #555555;
                        border-radius: 5px;
                        padding: 7px 20px;
                        min-width: 80px;
                    }
                    QMessageBox QPushButton:hover {
                        background-color: #444444;
                    }
                """)
                message_box.exec_()

    def update_ui(self):
        self.system_time_label.setText(
            f"System Time: {QTime.currentTime().toString('HH:mm:ss')}"
        )

        # 1. Update Video Frame (render only the newest frame)
        frame = None
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
        if frame is not None:
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.video_label.setPixmap(pixmap)

        # 2. Update BLE Status
        active_count = 0
        while not self.ble_status_queue.empty():
            stat = self.ble_status_queue.get()
            dev = stat['device']
            status = stat['status']
            self.device_status[dev] = status
            
            if dev in self.status_labels:
                lbl = self.status_labels[dev]['status']
                grp = self.status_labels[dev]['group']
                
                lbl.setText(f"Status: {status.upper()}")
                
                # Update specific card styling based on state
                if status == "connected":
                    lbl.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold; letter-spacing: 1px;")
                    grp.setStyleSheet("QGroupBox { border: 1px solid #4caf50; border-radius: 8px; margin-top: 15px; padding-top: 15px; background-color: #1e1e1e; } QGroupBox::title { color: #4caf50; subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; }")
                elif status == "calibrated":
                    lbl.setStyleSheet("color: #03dac6; font-size: 14px; font-weight: bold; letter-spacing: 1px;")
                    grp.setStyleSheet("QGroupBox { border: 1px solid #03dac6; border-radius: 8px; margin-top: 15px; padding-top: 15px; background-color: #1e1e1e; } QGroupBox::title { color: #03dac6; subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; }")
                elif status in ["error", "not found", "disconnected"]:
                    lbl.setStyleSheet("color: #cf6679; font-size: 14px; font-weight: bold; letter-spacing: 1px;")
                    grp.setStyleSheet("QGroupBox { border: 1px solid #cf6679; border-radius: 8px; margin-top: 15px; padding-top: 15px; background-color: #1e1e1e; } QGroupBox::title { color: #cf6679; subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; }")
                else:
                    lbl.setStyleSheet("color: #ffa000; font-size: 14px; font-weight: bold; letter-spacing: 1px;")
                    grp.setStyleSheet("QGroupBox { border: 1px solid #ffa000; border-radius: 8px; margin-top: 15px; padding-top: 15px; background-color: #1e1e1e; } QGroupBox::title { color: #ffa000; subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; }")

        # Update active nodes count label
        for st in self.device_status.values():
            if st in ["connected", "calibrated"]:
                active_count += 1
        self.active_nodes_label.setText(f"Active Nodes: {active_count}/5")

        # 3. Update Sync Data (count every frame, render only the newest)
        latest = None
        while not self.sync_queue.empty():
            latest = self.sync_queue.get()
            self.frames_received += 1
            if self.recorder.recording:
                self.total_frames_recorded += 1
        if self.recorder.recording:
            self.recording_status_label.setText(f"Status: RECORDING ({self.total_frames_recorded} frames)")

        if latest is not None:
            if not hasattr(self, '_data_state'):
                self._data_state = {}
            ok_style = "font-size: 13px; font-family: 'Consolas', monospace; color: #03dac6;"
            bad_style = "font-size: 13px; font-family: 'Consolas', monospace; color: #cf6679;"
            for dev in self.target_devices:
                imu = latest['imus'].get(dev)
                lbl = self.status_labels[dev]['data']
                if imu:
                    lbl.setText(f"A: {imu['ax']:>5.2f}, {imu['ay']:>5.2f}, {imu['az']:>5.2f}\n"
                                f"G: {imu['gx']:>5.2f}, {imu['gy']:>5.2f}, {imu['gz']:>5.2f}")
                    state = True
                else:
                    lbl.setText("Data: MISSING")
                    state = False
                if self._data_state.get(dev) != state:
                    lbl.setStyleSheet(ok_style if state else bad_style)
                    self._data_state[dev] = state

        # 4. Calculate Sync FPS
        current_time = QDateTime.currentMSecsSinceEpoch()
        if current_time - self.last_fps_time >= 1000:
            fps = self.frames_received
            self.sync_rate_label.setText(f"Sync Rate: {fps} FPS")
            self.frames_received = 0
            self.last_fps_time = current_time
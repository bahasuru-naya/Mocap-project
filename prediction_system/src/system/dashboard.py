"""
Dashboard — PyQt5 main window with a rotatable 3-D skeleton viewport
and live sensor-status panel for the realtime motion prediction system.
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np
import os

# Matplotlib 3-D canvas embedded in PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)


# ====================================================================== #
#  Skeleton topology — defines bone connections between the 17 keypoints
# ====================================================================== #
SKELETON_CONNECTIONS = [
    # Shoulder line
    (1, 2),
    # Left arm
    (1, 3),
    (3, 5),
    # Right arm
    (2, 4),
    (4, 6),
    # Torso
    (1, 7),
    (2, 8),
    (7, 8),
    # Left leg
    (7, 9),
    (9, 11),
    (11, 13),
    (13, 15),
    # Right leg
    (8, 10),
    (10, 12),
    (12, 14),
    (14, 16),
]

JOINT_LABELS = [
    "Head", "L Shoulder", "R Shoulder",
    "L Elbow", "R Elbow", "L Wrist", "R Wrist",
    "L Hip", "R Hip", "L Knee", "R Knee",
    "L Ankle", "R Ankle", "L Heel", "R Heel",
    "L Foot", "R Foot",
]


# ====================================================================== #
#  Calibration countdown dialog 
# ====================================================================== #
class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Countdown")
        self.setFixedSize(400, 500)
        self.countdown = 5
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #ffffff; font-family: 'Segoe UI', Inter, sans-serif; }
        """)
        
        layout = QVBoxLayout(self)
        
        instruction = QLabel("Please place all 5 IMU sensors on \n a flat surface in the same orientation. \n (Align all sensors in the same direction.)")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet("font-size: 18px; font-weight: bold; color: #bb86fc;")
        layout.addWidget(instruction)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)

        image_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "img",
        "Sensor_position.png"
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


# ====================================================================== #
#  Matplotlib 3-D skeleton canvas
# ====================================================================== #
class Skeleton3DCanvas(FigureCanvas):
    """
    A Matplotlib FigureCanvas that renders the 17-joint skeleton in a
    rotatable / zoomable 3-D viewport.
    """

    def __init__(self, parent=None, width=36, height=32, dpi=200):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor='#121212')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = self.fig.add_subplot(111, projection='3d',
                                        facecolor='#1e1e1e')
        self._style_axes()

        # Default viewpoint
        self.ax.view_init(elev=110, azim=-90)
        

        # Smoothed axis limits (prevent jitter)
        self._axis_limits = None
        self._alpha = 0.3          # EMA smoothing factor

        # Draw an initial "waiting" message
        self.ax.text(0, 0, 0, "Waiting for data...",
                     color='#777777', fontsize=14, ha='center')
        self.draw_idle()

    # ------------------------------------------------------------------ #
    def _style_axes(self):
        ax = self.ax
        ax.set_xlabel("X", color='#888888', fontsize=10)
        ax.set_ylabel("Y", color='#888888', fontsize=10)
        ax.set_zlabel("Z", color='#888888', fontsize=10)

        # Tick & pane colours
        for axis_obj in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis_obj.set_tick_params(colors='#666666', labelsize=8)
            axis_obj.pane.fill = False
            axis_obj.pane.set_edgecolor('#333333')

        ax.grid(True, color='#333333', linewidth=0.5, alpha=0.6)

    # ------------------------------------------------------------------ #
    def update_skeleton(self, coords):
        """
        Redraw the skeleton with new *coords* — a (17, 3) numpy array
        of ``[x, y, z]`` per joint.

        The current camera view (elevation / azimuth) is preserved across
        redraws so the user's rotation is not lost.
        """
        # Preserve current view angle
        elev = self.ax.elev
        azim = self.ax.azim

        self.ax.cla()
        self._style_axes()

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # ---- Draw body joints (indices 1-16) ---- #
        self.ax.scatter(
            x[1:], y[1:], z[1:],
            color='#03dac6', s=50, depthshade=True, zorder=5,
        )

        # ---- Draw head (index 0) in blue ---- #
        self.ax.scatter(
            [x[0]], [y[0]], [z[0]],
            color='#448aff', s=100, depthshade=True, zorder=6,
        )

        # ---- Draw bones ---- #
        for a, b in SKELETON_CONNECTIONS:
            self.ax.plot(
                [x[a], x[b]], [y[a], y[b]], [z[a], z[b]],
                color='#03dac6', linewidth=2.5, zorder=4,
            )

        # ---- Head → shoulder midpoint (neck) ---- #
        shoulder_mid = (coords[1] + coords[2]) / 2.0
        self.ax.plot(
            [x[0], shoulder_mid[0]],
            [y[0], shoulder_mid[1]],
            [z[0], shoulder_mid[2]],
            color='#448aff', linewidth=3, zorder=4,
        )

        # ---- Axis limits (smoothed to reduce jitter) ---- #
        margin = 0.15
        new_limits = np.array([
            [x.min() - margin, x.max() + margin],
            [y.min() - margin, y.max() + margin],
            [z.min() - margin, z.max() + margin],
        ])

        if self._axis_limits is None:
            self._axis_limits = new_limits
        else:
            a = self._alpha
            self._axis_limits = a * new_limits + (1 - a) * self._axis_limits

        self.ax.set_xlim(self._axis_limits[0])
        self.ax.set_ylim(self._axis_limits[1])
        self.ax.set_zlim(self._axis_limits[2])

        # Equal aspect ratio
        try:
            self.ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass  # older Matplotlib

        # Restore camera view
        self.ax.view_init(elev=elev, azim=azim)

        self.draw_idle()


# ====================================================================== #
#  Main Dashboard window
# ====================================================================== #
class Dashboard(QMainWindow):
    """
    PyQt5 main window for the realtime motion prediction system.
    
    """

    def __init__(self, ble_cmd_queue, sync_queue, pose_queue,
                 ble_status_queue, sync_engine, prediction_engine,
                 target_devices):
        super().__init__()
        self.ble_cmd_queue = ble_cmd_queue
        self.sync_queue = sync_queue
        self.pose_queue = pose_queue
        self.ble_status_queue = ble_status_queue
        self.sync_engine = sync_engine
        self.prediction_engine = prediction_engine
        self.target_devices = target_devices
        self.device_status = {dev: "disconnected" for dev in target_devices}

        self._init_ui()

        # Polling timer — ~30 FPS
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(33)

        self._sync_frames = 0
        self._last_fps_time = QDateTime.currentMSecsSinceEpoch()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #
    def _init_ui(self):
        self.setWindowTitle("AI MoCap — Realtime Motion Prediction")
        self.setGeometry(80, 40, 1650, 950)

        # ---- Global dark theme ---- #
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget     { color: #e0e0e0; font-family: 'Segoe UI', Inter, sans-serif; }
            QGroupBox {
                border: 1px solid #333333; 
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
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 10px; color: #bb86fc; left: 10px;
            }
            QPushButton {
                background-color: #333333; color: #ffffff; border: none;
                border-radius: 6px; padding: 10px 15px;
                font-weight: bold; font-size: 14px;
            }
            QPushButton:hover  { background-color: #444444; }
            QPushButton:pressed { background-color: #555555; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ============ Left panel (3-D viewport + controls) ============ #
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        # 3-D canvas
        self.canvas = Skeleton3DCanvas(self, width=9, height=7, dpi=100)
        left_panel.addWidget(self.canvas, stretch=1)

        # Controls bar
        controls_row = QHBoxLayout()
        controls_row.setSpacing(15)

        ctrl_group = QGroupBox("System Controls")
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(15, 15, 15, 15)

        self.calib_btn = QPushButton("Calibrate All Sensors")
        self.calib_btn.setMinimumHeight(55)
        self.calib_btn.setMaximumWidth(350)
        self.calib_btn.setStyleSheet("""
            QPushButton       { background-color: #1f618d; font-size: 16px;  }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.calib_btn.clicked.connect(self._start_calibration)
        ctrl_layout.addWidget(self.calib_btn)

        self.prediction_btn = QPushButton("Start Prediction")
        self.prediction_btn.setMinimumHeight(55)
        self.prediction_btn.setMaximumWidth(350)
        self.prediction_btn.setStyleSheet("""
            QPushButton       { background-color: #2e7d32; font-size: 16px; }
            QPushButton:hover { background-color: #388e3c; }
        """)
        self.prediction_btn.clicked.connect(self._toggle_prediction)
        ctrl_layout.addWidget(self.prediction_btn)
        ctrl_group.setLayout(ctrl_layout)
        controls_row.addWidget(ctrl_group)

        ctrl_group_text = QGroupBox("System information")
        ctrl_layout_text = QHBoxLayout()
        ctrl_layout_text.setContentsMargins(15, 15, 15, 15)
        
        
        self.pred_fps_label = QLabel("Prediction FPS: --")
        self.pred_fps_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #b3b3b3;"
        )
        ctrl_layout_text.addWidget(self.pred_fps_label)

        self.active_nodes_label = QLabel("Active Nodes: 0/5")
        self.active_nodes_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #b3b3b3;"
        )
        self.active_nodes_label.setAlignment(Qt.AlignCenter)
        ctrl_layout_text.addWidget(self.active_nodes_label)

        self.sync_fps_label = QLabel("Sync FPS: 0")
        self.sync_fps_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #b3b3b3;"
        )
        self.sync_fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        ctrl_layout_text.addWidget(self.sync_fps_label)

        ctrl_group_text.setLayout(ctrl_layout_text)
        controls_row.addWidget(ctrl_group_text, stretch=1)

        left_panel.addLayout(controls_row)

        main_layout.addLayout(left_panel, 7)

        # ============ Right panel (sensor cards) ============ #
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet(
            "QScrollArea { border: none; background-color: transparent; }"
            "QScrollBar:vertical { background: #121212; width: 12px; }"
            "QScrollBar::handle:vertical { background: #333333; border-radius: 6px; }"
        )

        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: transparent;")
        right_scroll.setWidget(right_widget)

        right_panel = QVBoxLayout(right_widget)
        right_panel.setSpacing(10)
        right_panel.setContentsMargins(0, 0, 10, 0)
        self.status_labels = {}

        for dev in self.target_devices:
            group = QGroupBox(dev)
            glayout = QVBoxLayout()
            glayout.setContentsMargins(15, 15, 15, 10)
            glayout.setSpacing(8)

            lbl_status = QLabel("Status: SCANNING")
            lbl_status.setStyleSheet(
                "color: #ffa000; font-size: 14px; font-weight: bold; "
                "letter-spacing: 1px;"
            )

            lbl_data = QLabel("Data: Waiting for stream...")
            lbl_data.setMinimumHeight(40)
            lbl_data.setStyleSheet(
                "font-size: 13px; font-family: 'Consolas', monospace; "
                "color: #a0a0a0;"
            )

            btn_calib = QPushButton("Calibrate Node")
            btn_calib.setMinimumHeight(35)
            btn_calib.setStyleSheet("""
                QPushButton       { background-color: #2c3e50; font-size: 13px; padding: 5px; }
                QPushButton:hover { background-color: #34495e; }
            """)
            btn_calib.clicked.connect(
                lambda checked, d=dev: self._calibrate_single(d)
            )

            glayout.addWidget(lbl_status)
            glayout.addWidget(lbl_data)
            glayout.addWidget(btn_calib)
            group.setLayout(glayout)
            right_panel.addWidget(group)

            self.status_labels[dev] = {
                'status': lbl_status,
                'data': lbl_data,
                'group': group,
            }

        right_panel.addStretch()
        main_layout.addWidget(right_scroll, 3)

    # ------------------------------------------------------------------ #
    #  Calibration
    # ------------------------------------------------------------------ #
    def _start_calibration(self):
        dialog = CalibrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            for dev in self.target_devices:
                self.ble_cmd_queue.put({'device': dev, 'action': 'calibrate'})

    def _calibrate_single(self, device):
        self.ble_cmd_queue.put({'device': device, 'action': 'calibrate'})

    def _toggle_prediction(self):
        if self.prediction_engine.is_prediction_enabled():
            self.prediction_engine.stop_prediction()
            self.prediction_btn.setText("Start Prediction")
            self.prediction_btn.setStyleSheet("""
                QPushButton       { background-color: #2e7d32; font-size: 16px; }
                QPushButton:hover { background-color: #388e3c; }
            """)
            return

        if not self.sync_engine.clock_sync_locked:
            sample_counts = self.sync_engine.clock_sync.sample_counts()
            details = ", ".join(
                f"{device}: {count}/{self.sync_engine.clock_sync.minimum_samples}"
                for device, count in sample_counts.items()
            )
            QMessageBox.warning(
                self,
                "Clock Sync Not Ready",
                "Prediction can start after every IMU has enough clock-sync "
                f"samples.\n{details}",
            )
            return

        self.prediction_engine.start_prediction()
        self.prediction_btn.setText("Stop Prediction")
        self.prediction_btn.setStyleSheet("""
            QPushButton       { background-color: #c62828; font-size: 16px; }
            QPushButton:hover { background-color: #e53935; }
        """)

    # ------------------------------------------------------------------ #
    #  Periodic UI update (~30 FPS)
    # ------------------------------------------------------------------ #
    def _poll(self):
        self._update_ble_status()
        self._update_sensor_readouts()
        self._update_skeleton()
        self._update_fps()

    # ---- BLE status badges ---- #
    def _update_ble_status(self):
        while not self.ble_status_queue.empty():
            stat = self.ble_status_queue.get()
            dev, status = stat['device'], stat['status']
            self.device_status[dev] = status

            if dev not in self.status_labels:
                continue
            lbl = self.status_labels[dev]['status']
            grp = self.status_labels[dev]['group']
            lbl.setText(f"Status: {status.upper()}")

            colour_map = {
                'connected':   '#4caf50',
                'calibrated':  '#03dac6',
            }
            err_states = {'error', 'not found', 'disconnected'}
            if status in colour_map:
                c = colour_map[status]
            elif status in err_states:
                c = '#cf6679'
            else:
                c = '#ffa000'

            lbl.setStyleSheet(
                f"color: {c}; font-size: 14px; font-weight: bold; "
                f"letter-spacing: 1px;"
            )
            grp.setStyleSheet(
                f"QGroupBox {{ border: 1px solid {c}; border-radius: 8px; "
                f"margin-top: 15px; padding-top: 15px; "
                f"background-color: #1e1e1e; }}"
                f"QGroupBox::title {{ color: {c}; subcontrol-origin: margin; "
                f"subcontrol-position: top left; padding: 0 10px; left: 10px; }}"
            )

        # Active node count
        active = sum(
            1 for s in self.device_status.values()
            if s in ('connected', 'calibrated')
        )
        self.active_nodes_label.setText(f"Active Nodes: {active}/5")

    # ---- Live IMU readouts from sync engine ---- #
    def _update_sensor_readouts(self):
        for dev in self.target_devices:
            sample = self.sync_engine.latest_samples.get(dev)
            if sample is not None:
                text = (
                    f"A: {sample['ax']:>6.2f}, {sample['ay']:>6.2f}, "
                    f"{sample['az']:>6.2f}\n"
                    f"G: {sample['gx']:>6.2f}, {sample['gy']:>6.2f}, "
                    f"{sample['gz']:>6.2f}"
                )
                self.status_labels[dev]['data'].setText(text)
                self.status_labels[dev]['data'].setStyleSheet(
                    "font-size: 13px; font-family: 'Consolas', monospace; "
                    "color: #03dac6;"
                )

    # ---- 3-D skeleton ---- #
    def _update_skeleton(self):
        latest = None
        while not self.pose_queue.empty():
            latest = self.pose_queue.get()
            self._sync_frames += 1

        if latest is not None:
            self.canvas.update_skeleton(latest['coords'])

    # ---- FPS counters ---- #
    def _update_fps(self):
        now = QDateTime.currentMSecsSinceEpoch()
        if now - self._last_fps_time >= 1000:
            self.sync_fps_label.setText(f"Sync FPS: {self._sync_frames}")
            self._sync_frames = 0
            self._last_fps_time = now

        # Prediction FPS from the engine
        pred_fps = self.prediction_engine.inference_fps
        self.pred_fps_label.setText(f"Prediction FPS: {pred_fps}")

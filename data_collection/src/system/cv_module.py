import cv2
import mediapipe as mp
import time
import queue
import threading
import multiprocessing as mp_proc


class QueuedFrameGrabber(threading.Thread):
    """Captures every frame with an accurate arrival timestamp and queues it in order.
    Inference happens later, so no frames are dropped and timestamps stay evenly
    spaced by the camera, not by inference speed."""
    def __init__(self, camera_index=0):
        super().__init__()
        # Use DirectShow backend on Windows for lower latency
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Unbounded: no frame is ever dropped (this is recording, not real time)
        self.frames = queue.Queue()
        self.running = True
        self.daemon = True

    def run(self):
        while self.running and self.cap.isOpened():
            # grab() returns as soon as the next frame is ready from the camera,
            # so timestamp immediately after it, BEFORE the slower decode step.
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            t_sys = time.time()
            t_mono = time.monotonic()

            success, frame = self.cap.retrieve()
            if not success:
                continue

            self.frames.put((frame, t_sys, t_mono))

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


class CVProcess(mp_proc.Process):
    def __init__(self, frame_queue, cv_data_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.cv_data_queue = cv_data_queue
        self.stop_event = mp_proc.Event()
        self.daemon = True
        
    def run(self):
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=1, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        grabber = QueuedFrameGrabber(0)
        grabber.start()

        allowed_body_indices = list(range(11, 17)) + list(range(23, 33))
        allowed_set = set(allowed_body_indices)
        
        try:
            while not self.stop_event.is_set():
                # Take the next frame IN ORDER (timeout to permit clean shutdown check)
                try:
                    image, frame_time, frame_sync_time = grabber.frames.get(timeout=0.1)
                except queue.Empty:
                    continue

                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                lm_list = []
                if results.pose_world_landmarks:
                    all_world = results.pose_world_landmarks.landmark
                    all_landmarks = results.pose_landmarks.landmark
                    h, w, c = image.shape
                    
                    # Head center
                    left_ear = all_world[7]
                    right_ear = all_world[8]
                    head_x = (left_ear.x + right_ear.x) / 2
                    head_y = (left_ear.y + right_ear.y) / 2
                    head_z = (left_ear.z + right_ear.z) / 2
                    lm_list.append([round(head_x, 4), round(-head_y, 4), round(head_z, 4)])
                    
                    # Body landmarks
                    for idx in allowed_body_indices:
                        lm = all_world[idx]
                        lm_list.append([round(lm.x, 4), round(-lm.y, 4), round(lm.z, 4)])
                    
                    # Draw annotations for UI preview
                    for connection in mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx >= 11 and end_idx >= 11 and start_idx in allowed_set and end_idx in allowed_set:
                            start_lm = all_landmarks[start_idx]
                            end_lm = all_landmarks[end_idx]
                            cv2.line(image_rgb, (int(start_lm.x * w), int(start_lm.y * h)), 
                                     (int(end_lm.x * w), int(end_lm.y * h)), (0, 255, 0), 2)
                                     
                    for idx in allowed_body_indices:
                        lm = all_landmarks[idx]
                        cv2.circle(image_rgb, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                    
                    head_points = all_landmarks[0:11]
                    hx = sum(lm.x for lm in head_points) / len(head_points)
                    hy = sum(lm.y for lm in head_points) / len(head_points)
                    cv2.circle(image_rgb, (int(hx * w), int(hy * h)), 10, (0, 0, 255), -1)
                    
                    l_shoulder = all_landmarks[11]
                    r_shoulder = all_landmarks[12]
                    shoulder_cx = (l_shoulder.x + r_shoulder.x) / 2
                    shoulder_cy = (l_shoulder.y + r_shoulder.y) / 2
                    cv2.line(image_rgb, (int(hx * w), int(hy * h)), (int(shoulder_cx * w), int(shoulder_cy * h)), (0, 255, 0), 2)

                # Send image frame to Dashboard
                try:
                    self.frame_queue.put_nowait(image_rgb)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(image_rgb)
                    except (queue.Empty, queue.Full):
                        pass

                # Send pose with the capture arrival timestamp
                if lm_list:
                    try:
                        self.cv_data_queue.put_nowait({
                            'sys_time': frame_time,
                            'sync_time': frame_sync_time,
                            'pose': lm_list
                        })
                    except queue.Full:
                        pass
        finally:
            grabber.stop()

    def stop(self):
        self.stop_event.set()
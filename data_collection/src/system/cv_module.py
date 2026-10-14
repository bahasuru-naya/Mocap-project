import cv2
import mediapipe as mp
import time
import queue
import multiprocessing as mp_proc

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
        
        cap = cv2.VideoCapture(0)
        
        # Required landmarks (excluding hands/feet intricacies to match original 3dmodel.py logic)
        allowed_body_indices = list(range(11, 17)) + list(range(23, 33))
        allowed_set = set(allowed_body_indices)
        
        while not self.stop_event.is_set() and cap.isOpened():
            success, image = cap.read()
            if not success:
                time.sleep(0.01)
                continue
                
            frame_time = time.time()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            lm_list = []
            
            if results.pose_world_landmarks:
                all_world = results.pose_world_landmarks.landmark
                all_landmarks = results.pose_landmarks.landmark
                h, w, c = image.shape
                
                # Head logic
                left_ear = all_world[7]
                right_ear = all_world[8]
                head_x = (left_ear.x + right_ear.x) / 2
                head_y = (left_ear.y + right_ear.y) / 2
                head_z = (left_ear.z + right_ear.z) / 2
                lm_list.append([round(head_x, 4), round(-head_y, 4), round(head_z, 4)])
                
                # Append required landmarks
                for idx in allowed_body_indices:
                    lm = all_world[idx]
                    lm_list.append([round(lm.x, 4), round(-lm.y, 4), round(lm.z, 4)])
                
                # Draw connections
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx >= 11 and end_idx >= 11 and start_idx in allowed_set and end_idx in allowed_set:
                        start_lm = all_landmarks[start_idx]
                        end_lm = all_landmarks[end_idx]
                        cv2.line(image, (int(start_lm.x * w), int(start_lm.y * h)), 
                                 (int(end_lm.x * w), int(end_lm.y * h)), (0, 255, 0), 2)
                                 
                # Draw points
                for idx in allowed_body_indices:
                    lm = all_landmarks[idx]
                    cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                
                # Draw head
                head_points = all_landmarks[0:11]
                hx = sum(lm.x for lm in head_points) / len(head_points)
                hy = sum(lm.y for lm in head_points) / len(head_points)
                cv2.circle(image, (int(hx * w), int(hy * h)), 10, (0, 0, 255), -1)
                
                # Draw line from head to center of shoulders
                l_shoulder = all_landmarks[11]
                r_shoulder = all_landmarks[12]
                shoulder_cx = (l_shoulder.x + r_shoulder.x) / 2
                shoulder_cy = (l_shoulder.y + r_shoulder.y) / 2
                cv2.line(image, (int(hx * w), int(hy * h)), (int(shoulder_cx * w), int(shoulder_cy * h)), (0, 255, 0), 2)

            
            # Put image frame in queue for Dashboard (convert back to RGB for PyQt)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                self.frame_queue.put_nowait(img_rgb)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # drop oldest to prevent lag
                    self.frame_queue.put_nowait(img_rgb)
                except (queue.Empty, queue.Full):
                    pass

            # Put pose data in queue for Synchronization
            if lm_list:
                try:
                    self.cv_data_queue.put_nowait({
                        'sys_time': frame_time,
                        'pose': lm_list
                    })
                except queue.Full:
                    pass

        cap.release()

    def stop(self):
        self.stop_event.set()
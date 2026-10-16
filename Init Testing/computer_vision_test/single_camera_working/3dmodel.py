import cv2
import mediapipe as mp
import socket
import json
import time

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5052
serverAddressPort = (UDP_IP, UDP_PORT)

# Initialize Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print(f"--- SENDER STARTING ---")
print(f"Target: {UDP_IP}:{UDP_PORT}")
print("Press 'q' on the video window to quit.")
print("\nNew Point Layout (17 points):")
print("0 - head (centroid)")
print("1 - left shoulder")
print("2 - right shoulder")
print("3 - left elbow")
print("4 - right elbow")
print("5 - left wrist")
print("6 - right wrist")
# NOTE: Finger/index/thumb landmarks (original indices 17-22) are intentionally excluded
print("7 - left hip")
print("8 - right hip")
print("9 - left knee")
print("10 - right knee")
print("11 - left ankle")
print("12 - right ankle")
print("13 - left heel")
print("14 - right heel")
print("15 - left foot index")
print("16 - right foot index")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("SENDER ERROR: Failed to grab frame from camera.")
        break

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, c = image.shape
        
        # 1. Extract Landmarks
        all_landmarks = results.pose_landmarks.landmark
        all_world_landmarks = results.pose_world_landmarks.landmark
        
        # Calculate HEAD CENTROID (average of points 0-10)
        # Points 0-10: nose, eyes, ears, mouth
        head_points = all_landmarks[0:11]
        
        head_x = sum(lm.x for lm in head_points) / len(head_points)
        head_y = sum(lm.y for lm in head_points) / len(head_points)
        
        # Convert to pixel coordinates for drawing
        head_pixel_x = int(head_x * w)
        head_pixel_y = int(head_y * h)
        
        # For data sending, use world landmarks
        left_ear = all_world_landmarks[7]
        right_ear = all_world_landmarks[8]

        head_world_x = (left_ear.x + right_ear.x) / 2
        head_world_y = (left_ear.y + right_ear.y) / 2
        head_world_z = (left_ear.z + right_ear.z) / 2
        
        # Build landmark list with HEAD as point 0, then BODY points (11-32)
        lm_list = []
        
        # Add head centroid as first point
        lm_list.append([round(head_world_x, 4), round(-head_world_y, 4), round(head_world_z, 4)])
        
        # Add body points, excluding finger/index/thumb landmarks (original indices 17-22)
        # Keep: 11-16 (shoulders, elbows, wrists) and 23-32 (hips, legs, feet)
        allowed_body_indices = list(range(11, 17)) + list(range(23, 33))
        for idx in allowed_body_indices:
            lm = all_world_landmarks[idx]
            lm_list.append([round(lm.x, 4), round(-lm.y, 4), round(lm.z, 4)])
        
        # 2. Draw on Video
        # Draw body landmarks and connections
        # Draw only connections between allowed body landmarks (skip fingers/thumbs)
        allowed_set = set(allowed_body_indices)
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            # Only draw connections for body landmarks we kept (exclude head-only and excluded fingers)
            if start_idx >= 11 and end_idx >= 11 and start_idx in allowed_set and end_idx in allowed_set:
                start_lm = all_landmarks[start_idx]
                end_lm = all_landmarks[end_idx]

                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))

                cv2.line(image, start_pos, end_pos, (0, 255, 0), 2)
        
        # Draw body landmarks (only allowed indices) as circles
        for idx in allowed_body_indices:
            lm = all_landmarks[idx]
            pos = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, pos, 4, (0, 255, 0), -1)
        
        # Draw HEAD as a larger, more prominent circle
        cv2.circle(image, (head_pixel_x, head_pixel_y), 10, (0, 0, 255), -1)  # Red filled circle
        cv2.circle(image, (head_pixel_x, head_pixel_y), 10, (255, 255, 255), 2)  # White outline        
        
        
        # 2. Package and Send
        data = json.dumps(lm_list)
        sock.sendto(str.encode(data), serverAddressPort)
        
        # 3. DEBUG: Print to Terminal
        # We only print every 1 second so we don't lag the terminal
        if int(time.time() * 10) % 10 == 0:
            print(f"STATUS: Sending {len(lm_list)} points. Head X: {lm_list[0][0]}")
    else:
        # DEBUG: If no person is in frame
        if int(time.time() * 10) % 10 == 0:
            print("STATUS: Camera active, but NO PERSON DETECTED.")

    # Show video for visual confirmation
    cv2.imshow('MediaPipe Sender Debug', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("SENDER CLOSED")
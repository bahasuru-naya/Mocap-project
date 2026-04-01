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

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("SENDER ERROR: Failed to grab frame from camera.")
        break

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_world_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # 1. Extract Landmarks
        lm_list = []
        for lm in results.pose_world_landmarks.landmark:
            # We send [x, y, z]
            # Negating Y here matches Blender's 'Up' axis better
            lm_list.append([round(lm.x, 4), round(-lm.y, 4), round(lm.z, 4)]) 
        
        # 2. Package and Send
        data = json.dumps(lm_list)
        sock.sendto(str.encode(data), serverAddressPort)
        
        # 3. DEBUG: Print to Terminal
        # We only print every 1 second so we don't lag the terminal
        if int(time.time() * 10) % 10 == 0:
            print(f"STATUS: Sending {len(lm_list)} points. Nose X: {lm_list[0][0]}")
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

import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import time

# --- 1. Initialize MediaPipe ---
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture("2.mp4")
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. Initialize MuJoCo ---
model_path = "../mujoco_menagerie#getting-started/unitree_g1/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Find the motor (actuator) ID for the right shoulder pitch (moving arm forward/backward)
# (Note: In the menagerie G1 model, it is usually named 'right_shoulder_pitch')
try:
    shoulder_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_shoulder_pitch')
except:
    print("Warning: Actuator name mismatch. Check g1.xml for exact motor names.")
    shoulder_actuator_id = 0

print("Starting Teleoperation Bridge... Press ESC in the webcam window to quit.")

# --- 3. The Control Loop ---
# launch_passive lets us update the robot while the viewer stays open
with mujoco.viewer.launch_passive(model, data) as viewer:
    while cap.isOpened() and viewer.is_running():
        success, image = cap.read()
        if not success:
            continue
            
        # Process webcam image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Get the human's right wrist Y-coordinate (up/down on screen)
            # MediaPipe Y goes from 0.0 (top of screen) to 1.0 (bottom of screen)
            wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            
            # --- RETARGETING MATH (Very Basic) ---
            # We map the wrist Y position (0.0 to 1.0) to a joint angle in radians
            # Let's say we want the robot arm to sweep between -1.5 rad and +1.5 rad
            target_angle = (wrist_y - 0.5) * -3.0  # Invert and scale
            
            # Send the command to the robot's motor
            data.ctrl[shoulder_actuator_id] = target_angle
            
            # Draw a circle on the wrist in the webcam feed so you see it tracking
            h, w, c = image.shape
            cv2.circle(image, (int(wrist_y * w), int(wrist_y * h)), 10, (0, 255, 0), -1)

        # Step the physics engine forward
        mujoco.mj_step(model, data)
        
        # Sync the viewer to show the new robot position
        viewer.sync()
        
        # Show the webcam
        cv2.imshow('Teleop Camera', image)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break

cap.release()
cv2.destroyAllWindows()
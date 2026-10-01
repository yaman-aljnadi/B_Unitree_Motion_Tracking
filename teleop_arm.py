import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Helper Function for 3D Angle Calculation ---
def calculate_angle(a, b, c):
    """Calculates the 3D angle between three points (a, b, c) with b as the vertex."""
    a = np.array(a) # e.g., Shoulder
    b = np.array(b) # e.g., Elbow
    c = np.array(c) # e.g., Wrist
    
    # Create vectors
    ba = a - b
    bc = c - b
    
    # Calculate cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clip to prevent math domain errors from floating point inaccuracies
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return angle

# --- 1. Video Input ---
VIDEO_PATH = "2.mp4" 
cap = cv2.VideoCapture(VIDEO_PATH)

# --- 2. Initialize MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 3. Initialize MuJoCo ---
model_path = "../mujoco_menagerie#getting-started/unitree_g1/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Get motor IDs for the right arm. 
# We use a helper to safely get IDs in case exact naming varies slightly.
def get_motor_id(name):
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    except:
        print(f"Warning: Actuator {name} not found.")
        return -1

shoulder_pitch_id = get_motor_id('right_shoulder_pitch') # Forward/Back
shoulder_roll_id  = get_motor_id('right_shoulder_roll')  # Up/Down to the side
elbow_pitch_id    = get_motor_id('right_elbow_pitch')    # Bending the elbow

print("Running Full Arm Angle-Based Teleoperation...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while cap.isOpened() and viewer.is_running():
        success, image = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract 3D coordinates for Right Shoulder (12), Right Elbow (14), Right Wrist (16), Right Hip (24)
            # MediaPipe format: [x (width), y (height), z (depth)]
            r_shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
            r_elbow    = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
            r_wrist    = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
            r_hip      = [landmarks[24].x, landmarks[24].y, landmarks[24].z]

            # --- ANGLE CALCULATIONS ---
            
            # 1. Elbow Pitch (Angle between Shoulder, Elbow, Wrist)
            # A straight arm is ~180 deg (PI rad). Bending it decreases the angle.
            elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            # Map MediaPipe's internal angle to the robot's joint limits (approx 0 to 2.0 rad)
            robot_elbow_pitch = np.pi - elbow_angle 

            # 2. Shoulder Roll (Lifting arm out to the side)
            # Angle between Hip, Shoulder, and Elbow
            shoulder_roll_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
            # Subtract baseline resting angle (approx 15 degrees / 0.25 rad)
            robot_shoulder_roll = -1.0 * (shoulder_roll_angle - 0.25)

            # 3. Shoulder Pitch (Lifting arm forward)
            # We look at the Z (depth) and Y (height) difference between elbow and shoulder
            dy = r_elbow[1] - r_shoulder[1]
            dz = r_elbow[2] - r_shoulder[2]
            # Use arctan2 to get the angle of the arm swinging forward/backward
            robot_shoulder_pitch = np.arctan2(dz, dy) * 2.0

            # --- SEND COMMANDS TO MOTORS ---
            if shoulder_pitch_id != -1: data.ctrl[shoulder_pitch_id] = robot_shoulder_pitch
            if shoulder_roll_id != -1:  data.ctrl[shoulder_roll_id] = robot_shoulder_roll
            if elbow_pitch_id != -1:    data.ctrl[elbow_pitch_id] = robot_elbow_pitch

        mujoco.mj_step(model, data)
        viewer.sync()
        
        cv2.imshow('MediaPipe Tracking', image)
        time.sleep(0.01)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
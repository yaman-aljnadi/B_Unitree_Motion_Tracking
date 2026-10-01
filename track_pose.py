import cv2
import mediapipe as mp

# Initialize MediaPipe Pose components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open the default webcam (0 is usually the built-in webcam)
cap = cv2.VideoCapture("3_full_body.mp4")

print("Starting webcam... Press the 'ESC' key on your keyboard to exit.")

# Initialize the pose estimation model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # MediaPipe requires RGB images, but OpenCV captures in BGR. We need to convert it.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find the human pose
        results = pose.process(image_rgb)
        
        # If a human is detected, draw the skeleton and print joint coordinates
        if results.pose_landmarks:
            # Draw the skeleton on the camera feed
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
            # Extract the coordinates of the Right Wrist as a test
            # MediaPipe provides x, y (2D screen position) and z (estimated depth)
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            print(f"Right Wrist [x:{right_wrist.x:.2f} y:{right_wrist.y:.2f} z:{right_wrist.z:.2f}]", end="\r")

        # Show the camera feed with the overlaid skeleton
        cv2.imshow('MediaPipe Human Pose', image)
        
        # Wait for the ESC key (ASCII 27) to close the window
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
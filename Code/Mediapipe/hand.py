import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
import time
import math

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the latest result
latest_result = None
output_image = None

def result_callback(result, image, timestamp_ms: int):
    global latest_result, output_image
    latest_result = result
    output_image = image

def addFrameCounter(annotated_frame, frame_counter, start_time):
    # Add FPS counter
    frame_counter += 1
    if frame_counter % 30 == 0:  # Calculate FPS every 30 frames
        fps = 30 / (time.time() - start_time)
    else:
        fps = 0
    
    if fps > 0:
        cv.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    return annotated_frame

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks and connections on the image."""
    if not detection_result.hand_landmarks:
        return rgb_image
    
    annotated_image = np.copy(rgb_image)
    
    # Hand landmark drawing specifications
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
    
    for hand_landmarks in detection_result.hand_landmarks:
        # Convert normalized coordinates to pixel coordinates
        height, width, _ = annotated_image.shape
        landmark_points = []
        
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmark_points.append((x, y))
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_point = landmark_points[connection[0]]
            end_point = landmark_points[connection[1]]
            cv.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks with labels
        for idx, (x, y) in enumerate(landmark_points):
            cv.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
            cv.putText(annotated_image, str(idx), (x + 5, y - 5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return annotated_image

# Expects rgb format
def addLandmarks(frame, landmarker):
    # Process the frame
    mp_image = mp.Image(mp.ImageFormat.SRGB, frame)

    timestamp = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp)
    
    # Draw landmarks if detection result is available
    if latest_result:
        return draw_landmarks_on_image(frame, latest_result)
    else:
        return frame

# Add this function to parse the landmark data
def parse_hand_landmarks(latest_result):
    """Parse and display meaningful hand landmark information."""
    if not latest_result or not latest_result.hand_landmarks:
        return
    
    landmark_names = [
        "WRIST",           # 0
        "THUMB_CMC",       # 1
        "THUMB_MCP",       # 2  
        "THUMB_IP",        # 3
        "THUMB_TIP",       # 4
        "INDEX_FINGER_MCP", # 5
        "INDEX_FINGER_PIP", # 6
        "INDEX_FINGER_DIP", # 7
        "INDEX_FINGER_TIP", # 8
        "MIDDLE_FINGER_MCP", # 9
        "MIDDLE_FINGER_PIP", # 10
        "MIDDLE_FINGER_DIP", # 11
        "MIDDLE_FINGER_TIP", # 12
        "RING_FINGER_MCP",   # 13
        "RING_FINGER_PIP",   # 14
        "RING_FINGER_DIP",   # 15
        "RING_FINGER_TIP",   # 16
        "PINKY_MCP",         # 17
        "PINKY_PIP",         # 18
        "PINKY_DIP",         # 19
        "PINKY_TIP"          # 20
    ]
    
    for i, hand_landmarks in enumerate(latest_result.hand_landmarks):
        print(f"\n--- Hand {i+1} Key Positions ---")
        
        # Show only important landmarks (fingertips + wrist)
        key_points = [0, 4, 8, 12, 16, 20]  # Wrist + all fingertips
        
        for idx in key_points:
            landmark = hand_landmarks[idx]
            # Convert to pixel coordinates (assuming 640x480)
            x_pixel = int(landmark.x * 640)
            y_pixel = int(landmark.y * 480)
            print(f"{landmark_names[idx]:<18}: ({x_pixel:3d}, {y_pixel:3d}) depth: {landmark.z:.3f}")

def extract_robotic_hand_features(hand_landmarks):
    """Extract features relevant for robotic hand control."""
    features = {}
    
    # 1. FINGER JOINT ANGLES (Most Important)
    finger_joints = {
        'thumb': [2, 3, 4],      # MCP, IP, TIP
        'index': [5, 6, 7, 8],   # MCP, PIP, DIP, TIP  
        'middle': [9, 10, 11, 12], # MCP, PIP, DIP, TIP
        'ring': [13, 14, 15, 16],  # MCP, PIP, DIP, TIP
        'pinky': [17, 18, 19, 20]  # MCP, PIP, DIP, TIP
    }
    
    def calculate_joint_angle(p1, p2, p3):
        """Calculate angle at joint p2 between points p1-p2-p3"""
        # Convert to vectors
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    # Calculate joint angles for each finger
    for finger_name, joints in finger_joints.items():
        finger_angles = []
        
        if finger_name == 'thumb':
            # Thumb: CMC-MCP-IP and MCP-IP-TIP angles
            cmc_mcp_ip = calculate_joint_angle(
                hand_landmarks[1], hand_landmarks[2], hand_landmarks[3])
            mcp_ip_tip = calculate_joint_angle(
                hand_landmarks[2], hand_landmarks[3], hand_landmarks[4])
            finger_angles = [cmc_mcp_ip, mcp_ip_tip]
        else:
            # Other fingers: MCP-PIP-DIP and PIP-DIP-TIP angles
            mcp_pip_dip = calculate_joint_angle(
                hand_landmarks[joints[0]], hand_landmarks[joints[1]], hand_landmarks[joints[2]])
            pip_dip_tip = calculate_joint_angle(
                hand_landmarks[joints[1]], hand_landmarks[joints[2]], hand_landmarks[joints[3]])
            finger_angles = [mcp_pip_dip, pip_dip_tip]
        
        features[f'{finger_name}_angles'] = finger_angles
    
    # 2. FINGER CURL/EXTENSION (Simplified for robotics)
    wrist = hand_landmarks[0]
    features['finger_extensions'] = {}
    
    fingertips = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
    for finger_name, tip_idx in fingertips.items():
        tip = hand_landmarks[tip_idx]
        # Distance from wrist to fingertip (normalized)
        extension = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        features['finger_extensions'][finger_name] = extension
    
    # 3. FINGER SPREAD (Important for robotic hand positioning)
    features['finger_spread'] = {}
    # Angle between adjacent fingers
    index_vec = np.array([hand_landmarks[8].x - wrist.x, hand_landmarks[8].y - wrist.y])
    middle_vec = np.array([hand_landmarks[12].x - wrist.x, hand_landmarks[12].y - wrist.y])
    ring_vec = np.array([hand_landmarks[16].x - wrist.x, hand_landmarks[16].y - wrist.y])
    pinky_vec = np.array([hand_landmarks[20].x - wrist.x, hand_landmarks[20].y - wrist.y])
    
    # Calculate spread angles
    def vector_angle(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    features['finger_spread']['index_middle'] = vector_angle(index_vec, middle_vec)
    features['finger_spread']['middle_ring'] = vector_angle(middle_vec, ring_vec)
    features['finger_spread']['ring_pinky'] = vector_angle(ring_vec, pinky_vec)
    
    # 4. THUMB OPPOSITION (Critical for robotic grasping)
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    thumb_opposition = math.sqrt((thumb_tip.x - index_tip.x)**2 + 
                                (thumb_tip.y - index_tip.y)**2)
    features['thumb_opposition'] = thumb_opposition
    
    return features

def print_robotic_features(latest_result):
    """Print features relevant for robotic hand control."""
    if not latest_result or not latest_result.hand_landmarks:
        print("No hand detected")
        return
    
    for i, hand_landmarks in enumerate(latest_result.hand_landmarks):
        try:
            handedness = latest_result.handedness[i][0].category_name if latest_result.handedness else "Unknown"
        except:
            handedness = "Unknown"
        
        features = extract_robotic_hand_features(hand_landmarks)
        
        print(f"\n=== {handedness} Hand - Robotic Control Features ===")
        
        # Joint angles (most important for servos)
        print("\nJOINT ANGLES (for servo control):")
        for finger, angles in features.items():
            if finger.endswith('_angles'):
                finger_name = finger.replace('_angles', '').upper()
                print(f"  {finger_name:<8}: {[f'{a:.1f}°' for a in angles]}")
        
        # Finger extensions (for grip strength)
        print("\nFINGER EXTENSIONS:")
        for finger, ext in features['finger_extensions'].items():
            print(f"  {finger.capitalize():<8}: {ext:.3f}")
        
        # Finger spread (for object grasping)
        print("\nFINGER SPREAD:")
        for spread, angle in features['finger_spread'].items():
            print(f"  {spread:<15}: {angle:.1f}°")
        
        # Thumb opposition (for precision grip)
        print(f"\nTHUMB OPPOSITION: {features['thumb_opposition']:.3f}")

def main():
    # Create hand landmarker options
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_hands=2,  # Detect up to 2 hands
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera properties for better performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)
    
    print("Starting webcam feed. Press 'q' to quit.")
    
    with HandLandmarker.create_from_options(options) as landmarker:
        frame_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Flip frame horizontally for mirror effect
            frame = cv.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
            landmark_frame = addLandmarks(rgb_frame, landmarker)
        
            # frame_counter += 1
            # annotated_frame = addFrameCounter(landmark_frame, frame_counter, start_time)

            # Display the frame
            bgr_frame = cv.cvtColor(landmark_frame, cv.COLOR_RGB2BGR)            
            cv.imshow('Hand Landmark Detection', bgr_frame)    

            # Break loop on 'q' key press or print landmark on 'p'
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                parse_hand_landmarks(latest_result)
                break
            elif key == ord('r'):  # Press 'r' for robotic features
                print_robotic_features(latest_result)
                break
    
    # Clean up
    cap.release()
    cv.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == "__main__":
    main()
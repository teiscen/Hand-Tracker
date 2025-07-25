import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np
import time

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
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process the frame
            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)
            
            # Draw landmarks if detection result is available
            if latest_result:
                annotated_frame = draw_landmarks_on_image(rgb_frame, latest_result)
                # Convert back to BGR for OpenCV display
                annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR)
            else:
                annotated_frame = frame
            
            # Add FPS counter
            frame_counter += 1
            if frame_counter % 30 == 0:  # Calculate FPS every 30 frames
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
            else:
                fps = 0
            
            if fps > 0:
                cv.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add hand count information
            hand_count = len(latest_result.hand_landmarks) if latest_result else 0
            cv.putText(annotated_frame, f'Hands detected: {hand_count}', (10, 70), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv.imshow('Hand Landmark Detection', annotated_frame)
            
            # Break loop on 'q' key press
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    cv.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == "__main__":
    main()
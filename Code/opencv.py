import mediapipe as mp
import cv2 as cv
from hand_landmarker import HandLandmarkerManagerLive
from parser_model import generate_labelled_data

class Transforms:
    def apply_transforms(bgr_frame, hand_manager):
        rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)

        rgb_frame = Transforms.draw_flipped(rgb_frame)
        rgb_frame = Transforms.draw_landmarks(rgb_frame, hand_manager)

        bgr_frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)       
        return bgr_frame   

    def draw_landmarks(rgb_frame, hand_manager):
        """Draw hand landmarks and connections on the image."""

        hand_manager.detect_async(rgb_frame)
        detection_result = hand_manager.get_latest_result() 

        if not detection_result or not detection_result.hand_landmarks:
            return rgb_frame
            
        for hand_landmarks in detection_result.hand_landmarks:
            # Convert normalized coordinates to pixel coordinates
            height, width, _ = rgb_frame.shape
            landmark_points = []
            
            for landmark in hand_landmarks:
                x = int(landmark.x * width) 
                y = int(landmark.y * height)
                landmark_points.append((x, y))
            
            # Draw connections
            # Hand landmark drawing specifications
            HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
            for connection in HAND_CONNECTIONS:
                start_point = landmark_points[connection[0]]
                end_point = landmark_points[connection[1]]
                cv.line(rgb_frame, start_point, end_point, (0, 255, 0), 2)
            
            # Draw landmarks with labels
            for idx, (x, y) in enumerate(landmark_points):
                cv.circle(rgb_frame, (x, y), 5, (0, 0, 255), -1)
                cv.putText(rgb_frame, str(idx), (x + 5, y - 5), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return rgb_frame

    def draw_flipped(rgb_frame):
        # Flip frame horizontally for mirror effect
        return cv.flip(rgb_frame, 1)

class WebcamManager:
    """Context manager for webcam operations with automatic resource management."""
    
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False
    
    def __enter__(self):
        """Initialize webcam when entering context."""
        print("Initializing webcam...")
        self.cap = cv.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        
        # Set camera properties for better performance
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv.CAP_PROP_FPS, self.fps)
        
        self.is_opened = True
        print(f"Webcam initialized: {self.width}x{self.height} @ {self.fps}fps")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up webcam when exiting context."""
        print("Cleaning up webcam resources...")
        if self.cap and self.is_opened:
            self.cap.release()
        cv.destroyAllWindows()
        self.is_opened = False
        print("Webcam cleanup completed.")
        
        # Handle exceptions gracefully
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions
    
    def read_frame(self):
        """Read a frame from the webcam."""
        if not self.is_opened or not self.cap:
            raise RuntimeError("Webcam not initialized. Use within 'with' statement.")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam.")
        
        return frame
    
    def is_camera_opened(self):
        """Check if camera is properly opened."""
        return self.is_opened and self.cap and self.cap.isOpened()

tracking = -1
frame_list = []
def webcam_loop(webcam_manager, hand_manager):
    """Process a single frame from the webcam."""
    global tracking, frame_list

    try:
        bgr_frame = webcam_manager.read_frame()
    except RuntimeError as e:
        print(f"Error: {e}")
        return False  # Signal to stop the loop

    # Display the frame
    updated_bgr_frame = Transforms.apply_transforms(bgr_frame, hand_manager)
    cv.imshow('Hand Landmark Detection', updated_bgr_frame)    

    # Break loop on 'q' key press
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        return False  # Signal to stop the loop
    if key == ord('r'):
        if tracking == -1:
            tracking = 0

    if tracking > -1 and tracking < 5: 
        frame_list.append(bgr_frame.copy())  # Copy the frame to avoid reference issues
        tracking += 1

        if tracking == 4:
            generate_labelled_data(frame_list[:])  # Pass a copy of the list
            frame_list.clear()
            tracking = -1

    return True  # Continue the loop

def main():
    """Main function using context manager for webcam."""
    try:
        with WebcamManager(camera_id=0, width=640, height=480, fps=30) as webcam:
            with HandLandmarkerManagerLive() as hand_manager:
                print("Starting webcam feed. Press 'q' to quit.")

                while webcam.is_camera_opened():
                    if not webcam_loop(webcam, hand_manager):
                        break
                    
    except RuntimeError as e:
        print(f"Webcam error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("Application stopped.")

if __name__ == "__main__":
    main()
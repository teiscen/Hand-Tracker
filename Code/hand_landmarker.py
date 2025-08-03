import mediapipe as mp
import time

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class HandLandmarkerManagerLive:
    """RAII context manager for MediaPipe hand landmark detection."""
    
    def __init__(self, 
                 model_path='hand_landmarker.task',
                 num_hands=2,
                 min_detection_confidence=0.5,
                 min_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        """Initialize hand landmarker configuration."""
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Will be set in __enter__
        self.landmarker = None
        self.latest_result = None
        self.is_initialized = False
    
    def __enter__(self):
        """Initialize MediaPipe hand landmarker when entering context."""
        print("Initializing MediaPipe hand landmarker...")
        
        try:
            # Create hand landmarker options
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                # running_mode=VisionRunningMode.IMAGE,
                result_callback=self._result_callback,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            # Create the landmarker
            self.landmarker = HandLandmarker.create_from_options(options)
            self.is_initialized = True
            
            print(f"Hand landmarker initialized successfully:")
            print(f"  - Model: {self.model_path}")
            print(f"  - Max hands: {self.num_hands}")
            print(f"  - Detection confidence: {self.min_detection_confidence}")
            
            return self
            
        except Exception as e:
            print(f"Failed to initialize hand landmarker: {e}")
            raise RuntimeError(f"Hand landmarker initialization failed: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up MediaPipe resources when exiting context."""
        print("Cleaning up hand landmarker resources...")
        
        if self.landmarker and self.is_initialized:
            try:
                self.landmarker.close()
                print("Hand landmarker closed successfully.")
            except Exception as e:
                print(f"Error closing hand landmarker: {e}")
        
        self.landmarker = None
        self.latest_result = None
        self.is_initialized = False
        
        # Handle exceptions gracefully
        if exc_type:
            print(f"Exception occurred in hand landmarker: {exc_type.__name__}: {exc_val}")
        
        return False  # Don't suppress exceptions
    
    def _result_callback(self, result, image, timestamp_ms: int):
        """Internal callback for MediaPipe detection results."""
        self.latest_result = result
    
    def detect_async(self, rgb_frame):
        """Detect hands asynchronously in the given RGB frame."""
        if not self.is_initialized or not self.landmarker:
            raise RuntimeError("Hand landmarker not initialized. Use within 'with' statement.")
        
        try:
            # Convert frame to MediaPipe format
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
            timestamp = int(time.time() * 1000)
            
            # Perform async detection
            self.landmarker.detect_async(mp_image, timestamp)
            
        except Exception as e:
            print(f"Error during hand detection: {e}")
            raise
    
    def get_latest_result(self):
        """Get the latest detection result."""
        return self.latest_result
    
    def has_hands(self):
        """Check if hands were detected in the latest result."""
        return (self.latest_result is not None and 
                self.latest_result.hand_landmarks is not None and 
                len(self.latest_result.hand_landmarks) > 0)
    
    def get_hand_count(self):
        """Get the number of detected hands."""
        if not self.has_hands():
            return 0
        return len(self.latest_result.hand_landmarks)
    
    def is_ready(self):
        """Check if the landmarker is ready for detection."""
        return self.is_initialized and self.landmarker is not None

class HandLandmarkerManagerImage:
    """RAII context manager for MediaPipe hand detection in IMAGE mode (batch processing)."""
    
    def __init__(self, 
                 model_path='hand_landmarker.task',
                 num_hands=2,
                 min_detection_confidence=0.5,
                 min_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        """Initialize hand landmarker configuration for IMAGE mode."""
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Will be set in __enter__
        self.landmarker = None
        self.is_initialized = False
    
    def __enter__(self):
        """Initialize MediaPipe hand landmarker for IMAGE mode."""
        print("Initializing MediaPipe hand landmarker (IMAGE mode)...")
        
        try:
            # Create hand landmarker options for IMAGE mode
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,  # IMAGE mode - no callback needed
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            # Create the landmarker
            self.landmarker = HandLandmarker.create_from_options(options)
            self.is_initialized = True
            
            print(f"IMAGE mode hand landmarker initialized successfully:")
            print(f"  - Model: {self.model_path}")
            print(f"  - Max hands: {self.num_hands}")
            print(f"  - Detection confidence: {self.min_detection_confidence}")
            
            return self
            
        except Exception as e:
            print(f"Failed to initialize IMAGE mode hand landmarker: {e}")
            raise RuntimeError(f"Hand landmarker initialization failed: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up MediaPipe resources when exiting context."""
        print("Cleaning up IMAGE mode hand landmarker resources...")
        
        if self.landmarker and self.is_initialized:
            try:
                self.landmarker.close()
                print("IMAGE mode hand landmarker closed successfully.")
            except Exception as e:
                print(f"Error closing IMAGE mode hand landmarker: {e}")
        
        self.landmarker = None
        self.is_initialized = False
        
        # Handle exceptions gracefully
        if exc_type:
            print(f"Exception in IMAGE mode hand landmarker: {exc_type.__name__}: {exc_val}")
        
        return False  # Don't suppress exceptions
    
    def detect(self, rgb_frame):
        """Detect hands synchronously in the given RGB frame."""
        if not self.is_initialized or not self.landmarker:
            raise RuntimeError("IMAGE mode hand landmarker not initialized. Use within 'with' statement.")
        
        try:
            # Convert frame to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Perform synchronous detection
            detection_result = self.landmarker.detect(mp_image)
            return detection_result
            
        except Exception as e:
            print(f"Error during IMAGE mode hand detection: {e}")
            raise
    
    def is_ready(self):
        """Check if the landmarker is ready for detection."""
        return self.is_initialized and self.landmarker is not None
import numpy as np
import math
import cv2 as cv
from scipy.signal import savgol_filter
from hand_landmarker import HandLandmarkerManagerImage
import mediapipe as mp
import sys
from contextlib import redirect_stdout

# Joint Angles
def calculate_joint_angle(p1, p2, p3):
    """Calculate angle at joint p2 between points p1-p2-p3"""
    # Convert to vectors
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_joint_angles(hand_landmarks):
    joint_angles = {}

    # 1. FINGER JOINT ANGLES (Most Important)
    finger_joints = {
        'thumb': [2, 3, 4],         # MCP, IP, TIP
        'index': [5, 6, 7, 8],      # MCP, PIP, DIP, TIP  
        'middle': [9, 10, 11, 12],  # MCP, PIP, DIP, TIP
        'ring': [13, 14, 15, 16],   # MCP, PIP, DIP, TIP
        'pinky': [17, 18, 19, 20]   # MCP, PIP, DIP, TIP
    }
    
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
        
        joint_angles[f'{finger_name}_angles'] = finger_angles

    return joint_angles

# Finger Extension
def calculate_finger_extension(hand_landmarks):
    """Calculate finger extensions and return as dictionary."""
    wrist = hand_landmarks[0]
    finger_extensions = {}  # Create the dictionary
    
    fingertips = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
    for finger_name, tip_idx in fingertips.items():
        tip = hand_landmarks[tip_idx]
        extension = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        finger_extensions[finger_name] = extension
    
    return finger_extensions  # Return the dictionary
    
# Finger Spread
def vector_angle(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def calculate_finger_spread(hand_landmarks):
    wrist = hand_landmarks[0]
    # 3. FINGER SPREAD (Important for robotic hand positioning)
    spread = {}
    # Angle between adjacent fingers
    index_vec = np.array([hand_landmarks[8].x - wrist.x, hand_landmarks[8].y - wrist.y])
    middle_vec = np.array([hand_landmarks[12].x - wrist.x, hand_landmarks[12].y - wrist.y])
    ring_vec = np.array([hand_landmarks[16].x - wrist.x, hand_landmarks[16].y - wrist.y])
    pinky_vec = np.array([hand_landmarks[20].x - wrist.x, hand_landmarks[20].y - wrist.y])
    
    # Calculate spread angles
    spread['index_middle'] = vector_angle(index_vec, middle_vec)
    spread['middle_ring'] = vector_angle(middle_vec, ring_vec)
    spread['ring_pinky'] = vector_angle(ring_vec, pinky_vec)

    return spread
    
# Thumb Opposition
def calculate_thumb_opposition(hand_landmarks):
    # 4. THUMB OPPOSITION (Critical for robotic grasping)
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    return math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    
# Generate Data
def create_example_data(detection_result):
    """Extract features from detection result for right hand only."""
    if not detection_result or not detection_result.hand_landmarks:
        return None
    
    # Find the right hand
    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        try:
            current_handedness = detection_result.handedness[i][0].category_name
            if current_handedness == "Right":
                features = {}
                features['joint_angles']      = calculate_joint_angles(hand_landmarks)
                features['finger_extensions'] = calculate_finger_extension(hand_landmarks)  # Fixed call
                features['finger_spread']     = calculate_finger_spread(hand_landmarks)
                features['thumb_opposition']  = calculate_thumb_opposition(hand_landmarks)
                return features
        except Exception as e:
            print(f"Error processing hand {i}: {e}")
            continue
    
    return None  # No right hand found

# Label Data
# uses: savgol_filter(data, window_length, polyorder)
def label_joint_angles(data_list, window_length, polyorder):
    labelled_joint_angles = {
        'thumb_angles':  [[], []],
        'index_angles':  [[], []],
        'middle_angles': [[], []],
        'ring_angles':   [[], []],
        'pinky_angles':  [[], []]
    }

    for data in data_list:
        print('label_joint_angles')
        labelled_joint_angles['thumb_angles' ][0].append(data['joint_angles']['thumb_angles' ][0])
        labelled_joint_angles['thumb_angles' ][1].append(data['joint_angles']['thumb_angles' ][1])

        labelled_joint_angles['index_angles' ][0].append(data['joint_angles']['index_angles' ][0])
        labelled_joint_angles['index_angles' ][1].append(data['joint_angles']['index_angles' ][1])

        labelled_joint_angles['middle_angles'][0].append(data['joint_angles']['middle_angles'][0])
        labelled_joint_angles['middle_angles'][1].append(data['joint_angles']['middle_angles'][1])

        labelled_joint_angles['ring_angles'  ][0].append(data['joint_angles']['ring_angles'  ][0])
        labelled_joint_angles['ring_angles'  ][1].append(data['joint_angles']['ring_angles'  ][1])

        labelled_joint_angles['pinky_angles' ][0].append(data['joint_angles']['pinky_angles' ][0])
        labelled_joint_angles['pinky_angles' ][1].append(data['joint_angles']['pinky_angles' ][1])

    labelled_joint_angles['thumb_angles' ][0] = savgol_filter(labelled_joint_angles['thumb_angles' ][0], window_length, polyorder)
    labelled_joint_angles['thumb_angles' ][1] = savgol_filter(labelled_joint_angles['thumb_angles' ][1], window_length, polyorder)
    labelled_joint_angles['index_angles' ][0] = savgol_filter(labelled_joint_angles['index_angles' ][0], window_length, polyorder)
    labelled_joint_angles['index_angles' ][1] = savgol_filter(labelled_joint_angles['index_angles' ][1], window_length, polyorder)
    labelled_joint_angles['middle_angles'][0] = savgol_filter(labelled_joint_angles['middle_angles'][0], window_length, polyorder)
    labelled_joint_angles['middle_angles'][1] = savgol_filter(labelled_joint_angles['middle_angles'][1], window_length, polyorder)
    labelled_joint_angles['ring_angles'  ][0] = savgol_filter(labelled_joint_angles['ring_angles'  ][0], window_length, polyorder)
    labelled_joint_angles['ring_angles'  ][1] = savgol_filter(labelled_joint_angles['ring_angles'  ][1], window_length, polyorder)
    labelled_joint_angles['pinky_angles' ][0] = savgol_filter(labelled_joint_angles['pinky_angles' ][0], window_length, polyorder)
    labelled_joint_angles['pinky_angles' ][1] = savgol_filter(labelled_joint_angles['pinky_angles' ][1], window_length, polyorder)

    return labelled_joint_angles

def label_finger_extension(data_list, window_length, polyorder):
    labelled_finger_extensions = {
        'thumb_extensions':  [],
        'index_extensions':  [],
        'middle_extensions': [],
        'ring_extensions':   [],
        'pinky_extensions':  []
    }

    for data in data_list:
        labelled_finger_extensions['thumb_extensions' ].append(data['finger_extensions']['thumb'])
        labelled_finger_extensions['index_extensions' ].append(data['finger_extensions']['index'])
        labelled_finger_extensions['middle_extensions'].append(data['finger_extensions']['middle'])
        labelled_finger_extensions['ring_extensions'  ].append(data['finger_extensions']['ring'])
        labelled_finger_extensions['pinky_extensions' ].append(data['finger_extensions']['pinky'])

    labelled_finger_extensions['thumb_extensions' ] = savgol_filter(labelled_finger_extensions['thumb_extensions' ], window_length, polyorder)
    labelled_finger_extensions['index_extensions' ] = savgol_filter(labelled_finger_extensions['index_extensions' ], window_length, polyorder)
    labelled_finger_extensions['middle_extensions'] = savgol_filter(labelled_finger_extensions['middle_extensions'], window_length, polyorder)
    labelled_finger_extensions['ring_extensions'  ] = savgol_filter(labelled_finger_extensions['ring_extensions'  ], window_length, polyorder)
    labelled_finger_extensions['pinky_extensions' ] = savgol_filter(labelled_finger_extensions['pinky_extensions' ], window_length, polyorder)

    return labelled_finger_extensions

def label_finger_spread(data_list, window_length, polyorder):
    labelled_finger_spread = {
        'index_middle_spread':  [],
        'middle_ring_spread':   [],
        'ring_pinky_spread'    : []
    }

    for data in data_list:
        labelled_finger_spread['index_middle_spread'].append(data['finger_spread']['index_middle'])
        labelled_finger_spread['middle_ring_spread' ].append(data['finger_spread']['middle_ring'])
        labelled_finger_spread['ring_pinky_spread'  ].append(data['finger_spread']['ring_pinky'])

    labelled_finger_spread['index_middle_spread'] = savgol_filter(labelled_finger_spread['index_middle_spread'], window_length, polyorder)
    labelled_finger_spread['middle_ring_spread' ] = savgol_filter(labelled_finger_spread['middle_ring_spread' ], window_length, polyorder)
    labelled_finger_spread['ring_pinky_spread'  ] = savgol_filter(labelled_finger_spread['ring_pinky_spread'  ], window_length, polyorder)

    return labelled_finger_spread

def label_thumb_opposition(data_list, window_length, polyorder):
    labelled_thumb_opposition = []

    for data in data_list:
        labelled_thumb_opposition.append(data['thumb_opposition'])

    labelled_thumb_opposition = savgol_filter(labelled_thumb_opposition, window_length, polyorder)

    return labelled_thumb_opposition

def create_labelled_example(example_features_list):
    # Adjust window_length based on available data
    data_length = len(example_features_list)
    
    # Window length must be odd and <= data length
    if data_length >= 5:
        window_length = 5
    elif data_length >= 3:
        window_length = 3
    else:
        # Not enough data for smoothing, just return the raw data
        print(f"Warning: Only {data_length} frames, skipping smoothing")
        return {
            'joint_angles': example_features_list[0]['joint_angles'] if data_length > 0 else {},
            'finger_extensions': example_features_list[0]['finger_extensions'] if data_length > 0 else {},
            'finger_spread': example_features_list[0]['finger_spread'] if data_length > 0 else {},
            'thumb_opposition': example_features_list[0]['thumb_opposition'] if data_length > 0 else 0
        }
    
    polyorder = min(2, window_length - 1)  # polyorder must be < window_length

    print(f"Using window_length={window_length}, polyorder={polyorder} for {data_length} frames")

    # Create labelled data with proper error handling
    labelled_data = {}
    labelled_data['joint_angles']      = label_joint_angles(example_features_list, window_length, polyorder)
    labelled_data['finger_extensions'] = label_finger_extension(example_features_list, window_length, polyorder)
    labelled_data['finger_spread']     = label_finger_spread(example_features_list, window_length, polyorder)
    labelled_data['thumb_opposition']  = label_thumb_opposition(example_features_list, window_length, polyorder)

    return labelled_data

# print Data
def print_example_data(example_features, file=None):
    """Print example features data - with null check."""
    if example_features is None:
        print("No features to display (example_features is None)", file=file)
        return
    
    # Joint angles (most important for servos)
    print("\nJOINT ANGLES (for servo control):", file=file)
    if 'joint_angles' in example_features:
        joint_angles = example_features['joint_angles']
        for finger, angles in joint_angles.items():
            if finger.endswith('_angles'):
                finger_name = finger.replace('_angles', '').upper()
                # Check if angles is a list of numbersr
                try:
                    formatted_angles = [f'{float(a):.1f}' for a in angles]
                    print(f"  {finger_name:<8}: {formatted_angles}", file=file)
                except (TypeError, ValueError):
                    print(f"  {finger_name:<8}: {angles} (invalid data)", file=file)
    
    # Check if required keys exist
    if 'finger_extensions' not in example_features:
        print("ERROR: 'finger_extensions' key missing", file=file)
        return
    
    # Finger extensions (for grip strength)
    print("\nFINGER EXTENSIONS:", file=file)
    for finger, ext in example_features['finger_extensions'].items():
        try:
            print(f"  {finger.capitalize():<8}: {float(ext):.3f}", file=file)
        except (TypeError, ValueError):
            print(f"  {finger.capitalize():<8}: {ext} (invalid data)", file=file)
    
    # Check if finger_spread exists
    if 'finger_spread' not in example_features:
        print("ERROR: 'finger_spread' key missing", file=file)
        return
    
    # Finger spread (for object grasping)
    print("\nFINGER SPREAD:", file=file)
    for spread, angle in example_features['finger_spread'].items():
        try:
            print(f"  {spread:<15}: {float(angle):.1f}", file=file)
        except (TypeError, ValueError):
            print(f"  {spread:<15}: {angle} (invalid data)", file=file)
    
    # Check thumb_opposition
    if 'thumb_opposition' not in example_features:
        print("ERROR: 'thumb_opposition' key missing", file=file)
        return
    
    # Thumb opposition (for precision grip)
    try:
        thumb_opp = float(example_features['thumb_opposition'])
        print(f"\nTHUMB OPPOSITION: {thumb_opp:.3f}", file=file)
    except (TypeError, ValueError):
        print(f"\nTHUMB OPPOSITION: {example_features['thumb_opposition']} (invalid data)", file=file)

# Fix the print_labelled_data function with proper type checking
def print_labelled_data(labelled_features):
    if labelled_features is None:
        print("No labelled features to display")
        return
    
    # Print Joint Angles
    print("\nLABELLED JOINT ANGLES (smoothed):")
    if 'joint_angles' in labelled_features:
        joint_angles = labelled_features['joint_angles']
        for finger in ['thumb_angles', 'index_angles', 'middle_angles', 'ring_angles', 'pinky_angles']:
            if finger in joint_angles:
                angles = joint_angles[finger]
                finger_name = finger.replace('_angles', '').upper()
                angle_strs = []
                for i, angle_series in enumerate(angles):
                    try:
                        if hasattr(angle_series, '__iter__'):  # Check if it's iterable
                            formatted_series = [f'{float(a):.1f}' for a in angle_series]
                            angle_strs.append(f"{formatted_series}")
                        else:
                            angle_strs.append(f"{float(angle_series):.1f}")
                    except (TypeError, ValueError):
                        angle_strs.append(f"{angle_series} (invalid)")
                print(f"  {finger_name:<8}: {angle_strs}")

    # Print Finger Extensions
    print("\nLABELLED FINGER EXTENSIONS (smoothed):")
    if 'finger_extensions' in labelled_features:
        finger_ext = labelled_features['finger_extensions']
        for finger in ['thumb_extensions', 'index_extensions', 'middle_extensions', 'ring_extensions', 'pinky_extensions']:
            if finger in finger_ext:
                ext_series = finger_ext[finger]
                finger_name = finger.replace('_extensions', '').capitalize()
                try:
                    if hasattr(ext_series, '__iter__'):
                        formatted_ext = [f'{float(e):.3f}' for e in ext_series]
                        print(f"  {finger_name:<8}: {formatted_ext}")
                    else:
                        print(f"  {finger_name:<8}: {float(ext_series):.3f}")
                except (TypeError, ValueError):
                    print(f"  {finger_name:<8}: {ext_series} (invalid)")

    # Print Finger Spread
    print("\nLABELLED FINGER SPREAD (smoothed):")
    if 'finger_spread' in labelled_features:
        finger_spread = labelled_features['finger_spread']
        for spread in ['index_middle_spread', 'middle_ring_spread', 'ring_pinky_spread']:
            if spread in finger_spread:
                spread_series = finger_spread[spread]
                try:
                    if hasattr(spread_series, '__iter__'):
                        formatted_spread = [f'{float(a):.1f}' for a in spread_series]
                        print(f"  {spread:<20}: {formatted_spread}")
                    else:
                        print(f"  {spread:<20}: {float(spread_series):.1f}")
                except (TypeError, ValueError):
                    print(f"  {spread:<20}: {spread_series} (invalid)")

    # Print Thumb Opposition
    print("\nLABELLED THUMB OPPOSITION (smoothed):")
    if 'thumb_opposition' in labelled_features:
        thumb_opp = labelled_features['thumb_opposition']
        try:
            if hasattr(thumb_opp, '__iter__'):
                formatted_thumb = [f'{float(t):.3f}' for t in thumb_opp]
                print(f"  Thumb Opposition: {formatted_thumb}")
            else:
                print(f"  Thumb Opposition: {float(thumb_opp):.3f}")
        except (TypeError, ValueError):
            print(f"  Thumb Opposition: {thumb_opp} (invalid)")

# Fix the generate_labelled_data function with proper error handling
def generate_labelled_data(bgr_frame_list):
    feature_list = []
    example_list = []
    labelled_data = None

    with open('output.txt', 'w') as f:
        with redirect_stdout(f):
            try:
                with HandLandmarkerManagerImage() as image_hand_manager:
                    print(f"Processing {len(bgr_frame_list)} frames...")

                    # Process each frame
                    for i, bgr_frame in enumerate(bgr_frame_list):
                        print(f"Processing frame {i+1}/{len(bgr_frame_list)}")
                        try:
                            rgb_frame = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                            detection_result = image_hand_manager.landmarker.detect(mp_image)
                            feature_list.append(detection_result)
                        except Exception as e:
                            print(f"Error processing frame {i+1}: {e}")
                            feature_list.append(None)

                    # Extract features from detection results
                    for i, detection_result in enumerate(feature_list):
                        print(f"Extracting features from frame {i+1}")
                        if detection_result is not None:
                            example = create_example_data(detection_result)
                            if example is not None:  # Only process valid examples
                                print_example_data(example)
                                example_list.append(example)
                            else:
                                print(f"No right hand detected in frame {i+1}")
                        else:
                            print(f"Frame {i+1} processing failed")

                    # Create labelled data only if we have valid examples
                    if len(example_list) > 0:
                        print(f"\nCreating labelled data from {len(example_list)} valid frames...")
                        labelled_data = create_labelled_example(example_list)
                        if labelled_data:
                            print_labelled_data(labelled_data)
                    else:
                        print("No valid hand data found in any frames")
                        
            except Exception as e:
                print(f"Error in generate_labelled_data: {e}")
                
    print("Labelled data generation complete. Check output.txt for details.")
    return labelled_data



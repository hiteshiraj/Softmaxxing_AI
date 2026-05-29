import numpy as np

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_symmetry(features):
    """
    Calculates a facial symmetry score based on landmark distances.
    """
    if not features:
        return 0, {}

    # Center axis x-coordinate (approximate)
    center_x = features['nose_tip'][0]
    
    # 1. Eye symmetry (distance from nose tip)
    left_eye_dist = calculate_distance(features['left_eye'], features['nose_tip'])
    right_eye_dist = calculate_distance(features['right_eye'], features['nose_tip'])
    eye_ratio = min(left_eye_dist, right_eye_dist) / max(left_eye_dist, right_eye_dist) if max(left_eye_dist, right_eye_dist) > 0 else 0
    
    # 2. Jaw symmetry (distance from nose tip)
    left_jaw_dist = calculate_distance(features['left_jaw'], features['nose_tip'])
    right_jaw_dist = calculate_distance(features['right_jaw'], features['nose_tip'])
    jaw_ratio = min(left_jaw_dist, right_jaw_dist) / max(left_jaw_dist, right_jaw_dist) if max(left_jaw_dist, right_jaw_dist) > 0 else 0

    # 3. Mouth symmetry
    left_mouth_dist = calculate_distance(features['mouth_left'], features['nose_tip'])
    right_mouth_dist = calculate_distance(features['mouth_right'], features['nose_tip'])
    mouth_ratio = min(left_mouth_dist, right_mouth_dist) / max(left_mouth_dist, right_mouth_dist) if max(left_mouth_dist, right_mouth_dist) > 0 else 0

    # Calculate overall symmetry score (0-100)
    symmetry_score = (eye_ratio * 0.4 + jaw_ratio * 0.4 + mouth_ratio * 0.2) * 100
    
    # Cap at 100
    symmetry_score = min(100, max(0, symmetry_score))

    details = {
        'eye_symmetry': eye_ratio * 100,
        'jaw_symmetry': jaw_ratio * 100,
        'mouth_symmetry': mouth_ratio * 100,
        'center_axis_x': center_x
    }
    
    return symmetry_score, details

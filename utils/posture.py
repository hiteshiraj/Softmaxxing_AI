import numpy as np

def analyze_posture(features):
    """
    Estimates head tilt and basic posture alignment.
    """
    if not features:
        return 0, {}

    # 1. Head Tilt (Roll) - Check if eyes are level
    left_eye = features['left_eye_outer']
    right_eye = features['right_eye_outer']
    
    # Calculate angle of the line connecting the eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Perfect alignment is 0 degrees.
    tilt_penalty = min(100, abs(angle) * 5) # 5 points penalty per degree of tilt
    tilt_score = max(0, 100 - tilt_penalty)
    
    # 2. Vertical Alignment (Pitch) - Check ratio of top-to-nose vs nose-to-chin
    top_head = features['top_head']
    nose = features['nose_tip']
    chin = features['chin']
    
    upper_face = abs(nose[1] - top_head[1])
    lower_face = abs(chin[1] - nose[1])
    
    vertical_ratio = upper_face / lower_face if lower_face > 0 else 1
    # Ideal ratio is roughly 1.0 depending on perspective.
    # Significant deviations indicate chin up or chin down.
    pitch_penalty = min(100, abs(vertical_ratio - 1.0) * 100)
    pitch_score = max(0, 100 - pitch_penalty)
    
    posture_score = (tilt_score * 0.6) + (pitch_score * 0.4)
    
    details = {
        'head_tilt_angle': angle,
        'tilt_score': tilt_score,
        'pitch_score': pitch_score
    }
    
    return posture_score, details

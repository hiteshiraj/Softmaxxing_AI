import cv2

def draw_visualizations(image, features, details):
    """
    Draws landmarks, symmetry lines, and posture indicators on the image.
    """
    if not features:
        return image
        
    vis_image = image.copy()
    
    # Define colors (BGR)
    COLOR_DOT = (0, 255, 0)
    COLOR_LINE = (255, 0, 255)
    COLOR_AXIS = (255, 255, 0)
    
    # 1. Draw Key Landmarks
    for name, point in features.items():
        cv2.circle(vis_image, point, 3, COLOR_DOT, -1)
        
    # 2. Draw Symmetry Axis (Nose to Chin)
    nose = features['nose_tip']
    chin = features['chin']
    top_head = features['top_head']
    cv2.line(vis_image, top_head, chin, COLOR_AXIS, 2, cv2.LINE_AA)
    
    # 3. Draw Eye Alignment Line
    left_eye = features['left_eye_outer']
    right_eye = features['right_eye_outer']
    cv2.line(vis_image, left_eye, right_eye, COLOR_LINE, 2, cv2.LINE_AA)
    
    # 4. Draw Posture/Shoulder proxy (using jaw as reference if shoulders not visible)
    left_jaw = features['left_jaw']
    right_jaw = features['right_jaw']
    cv2.line(vis_image, left_jaw, right_jaw, (0, 165, 255), 2, cv2.LINE_AA)
    
    # Draw text overlay of tilt
    posture_details = details.get('posture', {})
    tilt = posture_details.get('head_tilt_angle', 0)
    cv2.putText(vis_image, f"Tilt: {tilt:.1f} deg", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return vis_image

import cv2
import numpy as np

def analyze_hair(image, features):
    """
    Analyzes hair grooming and texture by checking edge density 
    in the region directly above the forehead.
    """
    if not features or 'top_head' not in features:
        return 0, {}
        
    h, w, _ = image.shape
    top_head = features['top_head']
    left_eye = features['left_eye']
    right_eye = features['right_eye']
    
    # Estimate face width to scale the hair bounding box appropriately
    face_width = right_eye[0] - left_eye[0]
    if face_width <= 0:
        face_width = 100
        
    # Define ROI above the top of the head
    # We'll go up by half the face width, and span the face width
    roi_w = int(face_width * 1.5)
    roi_h = int(face_width * 0.8)
    
    center_x = top_head[0]
    start_y = max(0, top_head[1] - roi_h)
    end_y = top_head[1]
    
    start_x = max(0, center_x - roi_w // 2)
    end_x = min(w, center_x + roi_w // 2)
    
    if start_y >= end_y or start_x >= end_x:
        return 50, {'error': 'Hair ROI out of bounds'}
        
    roi = image[start_y:end_y, start_x:end_x]
    
    if roi.size == 0:
        return 50, {'error': 'Empty Hair ROI'}
        
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edge detection to find texture/frizz
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # Calculate edge density (percentage of pixels that are edges)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    
    edge_density = (edge_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Map to score. 
    # Too high density (> 20%) often means frizz or very messy hair.
    # Very low density (< 2%) might mean bald, wearing a hat, or extremely blurred out hair.
    # We will score "grooming" such that moderate density (3-15%) is scored highest.
    if edge_density < 2:
        score = 60 # Neutral/Unknown
    elif edge_density <= 15:
        score = 100 - (abs(edge_density - 8) * 3) # Peaks at 8% density
    else:
        score = max(0, 100 - ((edge_density - 15) * 5))
        
    details = {
        'edge_density': edge_density
    }
    
    return min(100, max(0, score)), details

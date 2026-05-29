import cv2
import numpy as np

def analyze_skin(image, features):
    """
    Analyzes skin quality (smoothness) by measuring texture variance 
    in the cheek and forehead regions.
    """
    if not features or 'left_cheek' not in features:
        return 0, {}
        
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    h, w = gray.shape
    
    # Define a helper to get a safe patch
    def get_patch(center, size=20):
        x, y = center
        x1, y1 = max(0, x - size), max(0, y - size)
        x2, y2 = min(w, x + size), min(h, y + size)
        if x2 > x1 and y2 > y1:
            return gray[y1:y2, x1:x2]
        return None

    # Extract patches
    patches = []
    for point_name in ['left_cheek', 'right_cheek', 'forehead']:
        patch = get_patch(features[point_name], size=15)
        if patch is not None and patch.size > 0:
            patches.append(patch)
            
    if not patches:
        return 50, {'error': 'Could not extract skin patches'}
        
    # Calculate smoothness (lower variance / laplacian = smoother skin)
    # We use standard deviation of pixel intensities as a simple proxy for unevenness (blemishes/shadows).
    total_std = 0
    total_laplacian = 0
    for patch in patches:
        total_std += np.std(patch)
        total_laplacian += cv2.Laplacian(patch, cv2.CV_64F).var()
        
    avg_std = total_std / len(patches)
    avg_laplacian = total_laplacian / len(patches)
    
    # Map to 0-100 score. 
    # High standard deviation or high laplacian variance on the cheeks usually indicates uneven texture, pores, or blemishes.
    # Ideal smooth skin has very low variance.
    # Let's say std of 5 is excellent, std of >30 is poor.
    std_score = 100 - min(100, max(0, (avg_std - 5) * 4)) 
    
    details = {
        'avg_std': avg_std,
        'avg_laplacian': avg_laplacian
    }
    
    return std_score, details

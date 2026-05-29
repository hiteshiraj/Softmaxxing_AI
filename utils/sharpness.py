import cv2

def analyze_sharpness(image):
    """
    Measures image sharpness using the variance of the Laplacian.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Typical threshold for blur is ~100. 
    # High quality images easily exceed 500-1000.
    
    # Map to a 0-100 score
    if laplacian_var < 50:
        score = (laplacian_var / 50) * 40 # 0-40 (Very Blurry)
    elif laplacian_var < 300:
        score = 40 + ((laplacian_var - 50) / 250) * 40 # 40-80 (Acceptable)
    else:
        score = 80 + min(20, ((laplacian_var - 300) / 1000) * 20) # 80-100 (Sharp)
        
    details = {
        'laplacian_variance': laplacian_var,
        'is_blurry': laplacian_var < 100
    }
    
    return min(100, max(0, score)), details

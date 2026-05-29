import numpy as np
import math

def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points (p2 is the vertex)."""
    a = calculate_distance(p2, p1)
    b = calculate_distance(p2, p3)
    c = calculate_distance(p1, p3)
    if a == 0 or b == 0:
        return 0
    # Law of cosines
    val = (a**2 + b**2 - c**2) / (2 * a * b)
    # Clamp to avoid precision domain errors
    val = max(-1.0, min(1.0, val))
    return math.degrees(math.acos(val))

def analyze_facial_geometry(features):
    """
    Calculates heuristic geometrical metrics for the Advanced Facial Structure Analysis.
    """
    if not features or 'glabella' not in features:
        return None
        
    metrics = {}
    
    # 1. Canthal Tilt
    # Left eye: from inner (133) to outer (33)
    # Note: in image coordinates, y increases downwards. A positive tilt (upward) means outer y < inner y.
    l_inner = features['left_eye']
    l_outer = features['left_eye_outer']
    r_inner = features['right_eye']
    r_outer = features['right_eye_outer']
    
    l_dy = l_inner[1] - l_outer[1] # Positive if outer is higher
    l_dx = l_outer[0] - l_inner[0]
    l_tilt = math.degrees(math.atan2(l_dy, l_dx)) if l_dx != 0 else 0
    
    r_dy = r_inner[1] - r_outer[1] # Positive if outer is higher
    r_dx = r_inner[0] - r_outer[0]
    r_tilt = math.degrees(math.atan2(r_dy, r_dx)) if r_dx != 0 else 0
    
    avg_canthal_tilt = (l_tilt + r_tilt) / 2.0
    metrics['canthal_tilt'] = avg_canthal_tilt
    
    # 2. Facial Thirds
    top = features['top_head']
    glabella = features['glabella']
    subnasale = features['subnasale']
    chin = features['chin']
    
    upper_third = max(1, glabella[1] - top[1])
    mid_third = max(1, subnasale[1] - glabella[1])
    lower_third = max(1, chin[1] - subnasale[1])
    total_height = upper_third + mid_third + lower_third
    
    metrics['thirds'] = {
        'upper': (upper_third / total_height) * 100,
        'mid': (mid_third / total_height) * 100,
        'lower': (lower_third / total_height) * 100
    }
    
    # 3. fWHR (Facial Width-to-Height Ratio)
    # Width: distance between left and right zygomatic
    # Height: upper lip (stomion) to brow (glabella)
    bizygomatic_width = calculate_distance(features['left_zygomatic'], features['right_zygomatic'])
    midface_height = abs(features['stomion'][1] - features['glabella'][1])
    
    metrics['fWHR'] = bizygomatic_width / midface_height if midface_height > 0 else 0
    
    # 4. Cheekbone Prominence
    # Ratio of bizygomatic width to bigonial width (jaw width)
    bigonial_width = calculate_distance(features['left_gonion'], features['right_gonion'])
    metrics['cheekbone_prominence'] = bizygomatic_width / bigonial_width if bigonial_width > 0 else 0
    
    # 5. Mandibular Angle Approximation
    # Angle formed by Chin -> Gonion -> Zygomatic
    l_jaw_angle = calculate_angle(chin, features['left_gonion'], features['left_zygomatic'])
    r_jaw_angle = calculate_angle(chin, features['right_gonion'], features['right_zygomatic'])
    
    metrics['mandibular_angle'] = (l_jaw_angle + r_jaw_angle) / 2.0
    
    return metrics

def generate_geometry_feedback(metrics):
    """
    Generates non-toxic, analytical feedback strings based on geometric metrics.
    """
    feedback = []
    if not metrics:
        return feedback
        
    # Canthal Tilt
    tilt = metrics['canthal_tilt']
    if tilt > 3:
        feedback.append(f"Slight upward canthal tilt ({tilt:.1f}°) detected, creating an uplifted eye contour.")
    elif tilt < -3:
        feedback.append(f"Slight downward canthal tilt ({tilt:.1f}°) detected, creating a relaxed eye contour.")
    else:
        feedback.append(f"Neutral canthal tilt ({tilt:.1f}°) detected, showing horizontal eye alignment.")
        
    # Facial Thirds
    thirds = metrics['thirds']
    max_third = max(thirds, key=thirds.get)
    feedback.append(f"Facial thirds distribution: Upper ({thirds['upper']:.1f}%), Mid ({thirds['mid']:.1f}%), Lower ({thirds['lower']:.1f}%). The {max_third} third appears slightly more prominent in this lighting/angle.")
    
    # fWHR
    fwhr = metrics['fWHR']
    if fwhr > 1.9:
        feedback.append("High Facial Width-to-Height Ratio (fWHR) estimated; facial structure appears proportionately wider.")
    elif fwhr < 1.7:
        feedback.append("Lower Facial Width-to-Height Ratio (fWHR) estimated; facial structure appears proportionately taller.")
    else:
        feedback.append("Moderate Facial Width-to-Height Ratio (fWHR) estimated, indicating balanced width and height.")
        
    # Cheekbone Prominence
    prom = metrics['cheekbone_prominence']
    if prom > 1.3:
        feedback.append("Cheekbones measure significantly wider than the jawline, indicating high bizygomatic prominence.")
    else:
        feedback.append("Cheekbone width is closer to jaw width, creating a more uniform lateral profile.")
        
    # Mandibular Angle
    jaw_angle = metrics['mandibular_angle']
    if jaw_angle < 120:
        feedback.append(f"Mandibular angle approximated at {jaw_angle:.1f}°, suggesting a sharper, more squared jawline definition.")
    else:
        feedback.append(f"Mandibular angle approximated at {jaw_angle:.1f}°, suggesting a softer, more sloped jawline definition.")
        
    return feedback

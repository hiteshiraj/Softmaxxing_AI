import cv2
import numpy as np

def draw_advanced_geometry(image, features):
    """
    Draws a cyber-aesthetic overlay highlighting structural facial geometry.
    """
    if not features or 'glabella' not in features:
        return image
        
    vis_image = image.copy()
    h, w, _ = vis_image.shape
    
    # Neon Colors (BGR)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    LIME = (0, 255, 128)
    
    # 1. Jawline / Mandibular Angle Trace (Magenta)
    chin = features['chin']
    l_gonion = features['left_gonion']
    r_gonion = features['right_gonion']
    l_zyg = features['left_zygomatic']
    r_zyg = features['right_zygomatic']
    
    cv2.line(vis_image, l_zyg, l_gonion, MAGENTA, 2, cv2.LINE_AA)
    cv2.line(vis_image, l_gonion, chin, MAGENTA, 2, cv2.LINE_AA)
    cv2.line(vis_image, r_zyg, r_gonion, MAGENTA, 2, cv2.LINE_AA)
    cv2.line(vis_image, r_gonion, chin, MAGENTA, 2, cv2.LINE_AA)
    
    # Highlight gonion vertices
    cv2.circle(vis_image, l_gonion, 4, CYAN, -1)
    cv2.circle(vis_image, r_gonion, 4, CYAN, -1)
    
    # 2. Facial Thirds Dividers (Cyan, dashed/dotted effect)
    top = features['top_head']
    glabella = features['glabella']
    subnasale = features['subnasale']
    
    def draw_horizontal(y, color, width_fraction=0.4):
        cx = w // 2
        lx = int(cx - w * width_fraction)
        rx = int(cx + w * width_fraction)
        cv2.line(vis_image, (lx, y), (rx, y), color, 1, cv2.LINE_AA)
        
    draw_horizontal(top[1], CYAN)
    draw_horizontal(glabella[1], CYAN)
    draw_horizontal(subnasale[1], CYAN)
    draw_horizontal(chin[1], CYAN)
    
    # 3. Canthal Tilt Axes (Lime)
    l_inner = features['left_eye']
    l_outer = features['left_eye_outer']
    r_inner = features['right_eye']
    r_outer = features['right_eye_outer']
    
    # Extend the line slightly
    def extend_line(p1, p2, scale=0.5):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        p3 = (int(p2[0] + dx * scale), int(p2[1] + dy * scale))
        p0 = (int(p1[0] - dx * scale), int(p1[1] - dy * scale))
        return p0, p3
        
    l0, l1 = extend_line(l_inner, l_outer)
    r0, r1 = extend_line(r_inner, r_outer)
    
    cv2.line(vis_image, l0, l1, LIME, 2, cv2.LINE_AA)
    cv2.line(vis_image, r0, r1, LIME, 2, cv2.LINE_AA)
    
    # Draw transparent overlay for a "cyber" HUD feel
    overlay = vis_image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 10, 30), -1)
    cv2.addWeighted(overlay, 0.15, vis_image, 0.85, 0, vis_image)
    
    return vis_image

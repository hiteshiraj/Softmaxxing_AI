def generate_recommendations(scores, details):
    """
    Generates constructive, practical feedback based on the analysis.
    """
    recommendations = []
    
    # 1. Lighting Feedback
    lighting_score = scores.get('lighting', 0)
    lit_details = details.get('lighting', {})
    if lighting_score < 60:
        if lit_details.get('brightness', 128) < 80:
            recommendations.append("The image is quite dark. Try facing a natural light source like a window.")
        elif lit_details.get('contrast', 50) < 30:
            recommendations.append("The lighting lacks contrast, making the image look flat. Use angled lighting to define facial features.")
    elif lighting_score > 85:
        recommendations.append("Excellent lighting. Your facial features are well-illuminated and clear.")
        
    # 2. Sharpness Feedback
    sharpness_score = scores.get('sharpness', 0)
    if sharpness_score < 50:
        recommendations.append("The image appears slightly blurry. Ensure your camera lens is clean and your hands are steady.")
        
    # 3. Posture Feedback
    posture_score = scores.get('posture', 0)
    pos_details = details.get('posture', {})
    if posture_score < 70:
        tilt = pos_details.get('head_tilt_angle', 0)
        pitch_score = pos_details.get('pitch_score', 100)
        
        if abs(tilt) > 5:
            recommendations.append(f"Your head is tilted by about {abs(tilt):.1f} degrees. Keep your head straight for a more professional look.")
        if pitch_score < 80:
            recommendations.append("Camera angle seems off. Keep the camera precisely at eye-level to avoid chin distortion.")
            
    # 4. Symmetry / Composition Feedback
    sym_score = scores.get('symmetry', 0)
    if sym_score < 60:
        recommendations.append("Slight asymmetry detected, which is often caused by off-center camera angles. Try centering the camera perfectly directly in front of you.")
        
    # 5. Skin Feedback
    skin_score = scores.get('skin', 0)
    if skin_score < 60:
        recommendations.append("Skin texture appears slightly uneven. Ensure soft, diffused lighting to reduce harsh shadows on the face.")
        
    # 6. Hair Feedback
    hair_score = scores.get('hair', 0)
    if hair_score < 60:
        recommendations.append("Hair appears slightly messy or highly textured in this lighting. Consider styling or taming flyaways for a sharper look.")
    
    if not recommendations:
        recommendations.append("Great presentation! Your lighting, posture, and image quality are well-optimized.")
        
    return recommendations

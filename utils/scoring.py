import numpy as np
from sklearn.cluster import KMeans
import cv2

class StyleClusterer:
    def __init__(self):
        # Initialize a simple KMeans model for style clustering
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        self._train_mock_clusterer()
        
    def _train_mock_clusterer(self):
        # Features: [brightness, saturation, sharpness]
        # Cluster 0: Aesthetic (high brightness, high saturation, high sharpness)
        # Cluster 1: Casual (medium brightness, medium saturation, medium sharpness)
        # Cluster 2: Low-light (low brightness, high saturation, low/med sharpness)
        # Cluster 3: Professional (high brightness, med saturation, very high sharpness)
        X_train = np.array([
            [180, 150, 800], [170, 160, 750], # Aesthetic
            [130, 100, 300], [120, 110, 250], # Casual
            [50, 130, 100],  [60, 140, 150],  # Low-light
            [200, 90, 1500], [210, 85, 1400], # Professional
        ])
        self.kmeans.fit(X_train)
        # Map cluster indices to labels manually based on centers
        self.labels_map = {
            self.kmeans.predict([[175, 155, 775]])[0]: "Aesthetic",
            self.kmeans.predict([[125, 105, 275]])[0]: "Casual",
            self.kmeans.predict([[55, 135, 125]])[0]: "Low-light",
            self.kmeans.predict([[205, 87, 1450]])[0]: "Professional"
        }
        
    def cluster(self, image, details_dict):
        """
        Assigns an aesthetic style cluster based on image features.
        """
        brightness = details_dict['lighting'].get('brightness', 128)
        sharpness = details_dict['sharpness'].get('laplacian_variance', 300)
        
        # Calculate crude saturation
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:,:,1])
        else:
            saturation = 0
            
        features = np.array([[brightness, saturation, sharpness]])
        cluster_id = self.kmeans.predict(features)[0]
        
        style = self.labels_map.get(cluster_id, "Casual")
        return style

def calculate_overall_score(scores_dict):
    """
    Calculates weighted overall presentation score.
    Symmetry 20%, Lighting 20%, Sharpness 15%, Posture 15%, Skin 15%, Hair 15%
    """
    sym = scores_dict.get('symmetry', 50)
    lit = scores_dict.get('lighting', 50)
    shp = scores_dict.get('sharpness', 50)
    pos = scores_dict.get('posture', 50)
    skin = scores_dict.get('skin', 50)
    hair = scores_dict.get('hair', 50)
    
    final_score = (sym * 0.20) + (lit * 0.20) + (shp * 0.15) + (pos * 0.15) + (skin * 0.15) + (hair * 0.15)
    return min(100, max(0, final_score))

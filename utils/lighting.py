import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class LightingAnalyzer:
    def __init__(self):
        # Initialize a simple mock RandomForestClassifier for MVP
        self.clf = RandomForestClassifier(n_estimators=10, random_state=42)
        self._train_mock_classifier()
        
    def _train_mock_classifier(self):
        # Synthetic data: [brightness, contrast, histogram_spread]
        # Classes: 0 (Poor), 1 (Decent), 2 (Excellent)
        X_train = np.array([
            [50, 20, 30],   # dark, low contrast -> 0
            [30, 30, 40],   # very dark -> 0
            [240, 20, 20],  # overexposed -> 0
            [120, 50, 70],  # decent -> 1
            [140, 60, 80],  # decent -> 1
            [160, 80, 120], # excellent -> 2
            [150, 90, 130], # excellent -> 2
            [170, 85, 125], # excellent -> 2
            [100, 30, 50],  # decent/poor -> 1
            [80, 20, 40],   # poor -> 0
        ])
        y_train = np.array([0, 0, 0, 1, 1, 2, 2, 2, 1, 0])
        self.clf.fit(X_train, y_train)
        
    def analyze(self, image):
        """
        Analyzes image lighting and classifies quality.
        image is expected to be RGB or BGR numpy array.
        """
        # Convert to grayscale for intensity analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Histogram spread (difference between 95th and 5th percentile)
        hist_spread = np.percentile(gray, 95) - np.percentile(gray, 5)
        
        features = np.array([[brightness, contrast, hist_spread]])
        quality_class = self.clf.predict(features)[0]
        
        # Map class to score (0-100)
        score_map = {0: 30, 1: 70, 2: 95}
        # Add some continuous variance based on contrast
        base_score = score_map[quality_class]
        lighting_score = min(100, max(0, base_score + (contrast / 10) - 5))
        
        labels = {0: "Poor", 1: "Decent", 2: "Excellent"}
        
        details = {
            'brightness': brightness,
            'contrast': contrast,
            'histogram_spread': hist_spread,
            'quality_label': labels[quality_class]
        }
        
        return lighting_score, details

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        # We need the task file to be present
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'face_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, image):
        """
        Detects facial landmarks in the given image.
        Returns the original image (RGB) and the raw landmarks if found.
        """
        # Convert the BGR image to RGB before processing.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None, None
            
        landmarks = detection_result.face_landmarks[0]
        return image_rgb, landmarks

    def extract_features(self, image, landmarks):
        """
        Extract key 2D coordinate features from the face mesh.
        Returns a dictionary of key landmark coordinates.
        """
        if not landmarks:
            return None
            
        h, w, _ = image.shape
        coords = {}
        
        # Convert normalized coordinates to pixel coordinates
        for idx, landmark in enumerate(landmarks):
            coords[idx] = (int(landmark.x * w), int(landmark.y * h))
            
        # Select key landmarks (approximate indices for MediaPipe Face Mesh)
        # Left eye: 33, 133, Right eye: 362, 263
        # Nose tip: 1
        # Left mouth corner: 61, Right mouth corner: 291
        # Chin: 152
        # Left jaw: 234, Right jaw: 454
        
        key_features = {
            'left_eye': coords[133], # Inner corner
            'right_eye': coords[362], # Inner corner
            'left_eye_outer': coords[33],
            'right_eye_outer': coords[263],
            'nose_tip': coords[1],
            'mouth_left': coords[61],
            'mouth_right': coords[291],
            'chin': coords[152],
            'left_jaw': coords[234],
            'right_jaw': coords[454],
            'top_head': coords[10],
            'left_cheek': coords[50],
            'right_cheek': coords[280],
            'forehead': coords[151],
            'glabella': coords[9],
            'subnasale': coords[164],
            'stomion': coords[13],
            'left_gonion': coords[132],
            'right_gonion': coords[361],
            'left_zygomatic': coords[127],
            'right_zygomatic': coords[356]
        }
        
        return key_features

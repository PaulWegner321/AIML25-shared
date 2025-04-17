import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import cv2

class KeyPointClassifier(nn.Module):
    def __init__(self):
        super(KeyPointClassifier, self).__init__()
        # Define the model architecture
        self.fc1 = nn.Linear(42, 256)  # 21 landmarks with x,y coordinates = 42 features
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 26)  # 26 classes for A-Z
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class HandDetector:
    def __init__(self, model_path=None):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'keypoint_classifier.pt')
        
        print(f"Initializing hand detector with model: {model_path}")
        
        try:
            # Load the TorchScript model
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print("Successfully loaded TorchScript model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Define the mapping of class indices to letters
        self.class_mapping = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
            9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
            17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z'
        }

    def preprocess_landmarks(self, image):
        """Extract and preprocess hand landmarks from the image."""
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Convert landmarks to relative coordinates
        landmark_list = []
        
        # Use wrist as reference point
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        
        for landmark in hand_landmarks.landmark:
            # Calculate relative coordinates
            rel_x = landmark.x - base_x
            rel_y = landmark.y - base_y
            
            # Normalize coordinates
            landmark_list.extend([rel_x, rel_y])
        
        # Convert to tensor
        landmarks_tensor = torch.tensor(landmark_list, dtype=torch.float32)
        
        # Ensure we have exactly 42 features
        if len(landmarks_tensor) != 42:
            print(f"Warning: Expected 42 features, got {len(landmarks_tensor)}")
            return None
        
        return landmarks_tensor

    def detect_sign(self, image):
        """Detect ASL sign from an image."""
        try:
            # Preprocess image and extract landmarks
            landmarks = self.preprocess_landmarks(image)
            
            if landmarks is None:
                return {
                    'success': False,
                    'error': 'No hand detected in the image'
                }
            
            # Add batch dimension and move to device
            landmarks = landmarks.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                try:
                    outputs = self.model(landmarks)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)
                    confidence = probabilities[0][predicted_class].item()
                    
                    # Get predicted letter
                    predicted_letter = self.class_mapping[predicted_class.item()]
                    
                    return {
                        'success': True,
                        'letter': predicted_letter,
                        'confidence': confidence
                    }
                except Exception as e:
                    print(f"Error during prediction: {str(e)}")
                    print(f"Input shape: {landmarks.shape}")
                    return {
                        'success': False,
                        'error': f'Error during prediction: {str(e)}'
                    }
            
        except Exception as e:
            print(f"Error in detect_sign: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def draw_landmarks(self, image, landmarks):
        """Draw the detected hand landmarks on the image."""
        if landmarks is None:
            return image
            
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create a copy of the image
        annotated_image = image.copy()
        
        # Draw the hand landmarks
        mp_drawing.draw_landmarks(
            annotated_image,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        return annotated_image 
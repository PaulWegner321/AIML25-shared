import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import logging
import mediapipe as mp
import os
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create debug directory if it doesn't exist
debug_dir = Path("debug_images")
debug_dir.mkdir(exist_ok=True)

class ASL_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input 3 channels (RGB), output 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 25)  # 25 classes for ASL alphabet (excluding J and Z which require motion)

    def forward(self, x):
        try:
            logger.info(f"Input tensor shape: {x.shape}")
            
            # Apply convolutional layers with ReLU activation and pooling
            x = self.pool(self.relu(self.conv1(x)))  # 224x224 -> 112x112
            logger.info(f"After conv1: {x.shape}")
            
            x = self.pool(self.relu(self.conv2(x)))  # 112x112 -> 56x56
            logger.info(f"After conv2: {x.shape}")
            
            x = self.pool(self.relu(self.conv3(x)))  # 56x56 -> 28x28
            logger.info(f"After conv3: {x.shape}")
            
            x = self.pool(self.relu(self.conv4(x)))  # 28x28 -> 14x14
            logger.info(f"After conv4: {x.shape}")

            x = self.flatten(x)
            logger.info(f"After flatten: {x.shape}")
            
            x = self.relu(self.fc1(x))
            logger.info(f"After fc1: {x.shape}")
            
            x = self.dropout(x)
            x = self.fc2(x)
            logger.info(f"Final output: {x.shape}")

            return x
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

class NewCNNPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = None
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Standard RGB normalization values
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def process_hand_frame(self, frame, canvas_size=500, offset=60):
        """Process frame to detect hand and create standardized input."""
        try:
            # Generate timestamp for unique filenames
            timestamp = int(time.time() * 1000)  # millisecond timestamp
            
            # Save original input frame
            input_path = debug_dir / f"input_{timestamp}.jpg"
            cv2.imwrite(str(input_path), frame)
            logger.info(f"Saved input frame to {input_path}")
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Create white canvas
            canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
            resized_canvas = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get frame dimensions
                    h, w, _ = frame.shape
                    
                    # Extract landmark coordinates
                    landmark_x = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    landmark_y = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    
                    # Calculate bounding box with offset
                    x_min = max(0, min(landmark_x) - offset)
                    y_min = max(0, min(landmark_y) - offset)
                    x_max = min(w, max(landmark_x) + offset)
                    y_max = min(h, max(landmark_y) + offset)

                    # Extract ROI (hand region)
                    roi = frame.copy()
                    
                    # Draw landmarks on ROI before cropping
                    self.mp_drawing.draw_landmarks(
                        roi,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=3),  # Red dots, thickness 5
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)  # White lines, thickness 3
                    )
                    
                    # Now crop the ROI with landmarks
                    roi = roi[y_min:y_max, x_min:x_max]

                    if roi.size > 0:
                        # Save ROI with landmarks
                        roi_path = debug_dir / f"roi_{timestamp}.jpg"
                        cv2.imwrite(str(roi_path), roi)
                        logger.info(f"Saved ROI with landmarks to {roi_path}")
                        
                        # Calculate offsets to center hand in canvas
                        roi_h, roi_w, _ = roi.shape
                        y_offset = (canvas_size - roi_h) // 2
                        x_offset = (canvas_size - roi_w) // 2

                        # Place ROI on white canvas if it fits
                        if roi_h <= canvas_size and roi_w <= canvas_size:
                            canvas[y_offset:y_offset + roi_h, x_offset:x_offset + roi_w] = roi
                            
                            # Save centered canvas
                            canvas_path = debug_dir / f"canvas_{timestamp}.jpg"
                            cv2.imwrite(str(canvas_path), canvas)
                            logger.info(f"Saved centered canvas to {canvas_path}")
                            
                            # Resize to model input size
                            resized_canvas = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA)
                            
                            # Save final preprocessed image
                            final_path = debug_dir / f"final_{timestamp}.jpg"
                            cv2.imwrite(str(final_path), resized_canvas)
                            logger.info(f"Saved final preprocessed image to {final_path}")
                            
                            return resized_canvas

            logger.warning("No hand landmarks detected in the image")
            resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Save fallback image
            fallback_path = debug_dir / f"fallback_{timestamp}.jpg"
            cv2.imwrite(str(fallback_path), resized_frame)
            logger.info(f"No hand detected. Saved fallback image to {fallback_path}")
            
            return resized_frame
            
        except Exception as e:
            logger.error(f"Error in process_hand_frame: {str(e)}")
            resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Save error case image
            error_path = debug_dir / f"error_{timestamp}.jpg"
            cv2.imwrite(str(error_path), resized_frame)
            logger.error(f"Error occurred. Saved error case image to {error_path}")
            
            return resized_frame
        
    def load_model(self, model_path):
        """Load the model from the specified path"""
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = ASL_CNN().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def preprocess_image(self, image):
        """Preprocess the input image for the model"""
        try:
            timestamp = int(time.time() * 1000)
            logger.info(f"Input image type: {type(image)}, shape: {image.shape if isinstance(image, np.ndarray) else 'N/A'}")
            
            # Process the frame to detect and isolate hand
            processed_frame = self.process_hand_frame(image)
            logger.info(f"Processed frame shape: {processed_frame.shape}")
            
            # Convert to RGB if needed
            if processed_frame.shape[2] == 3:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Save RGB version
                rgb_path = debug_dir / f"rgb_{timestamp}.jpg"
                cv2.imwrite(str(rgb_path), cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved RGB version to {rgb_path}")
            
            # Convert to PIL Image and apply transforms
            image = transforms.ToPILImage()(processed_frame)
            logger.info("Converted to PIL Image")
            
            # Apply transformations
            image = self.transform(image)
            logger.info(f"After transforms - tensor shape: {image.shape}")
            
            # Save normalized tensor as image for visualization
            normalized_img = ((image.cpu().numpy().transpose(1, 2, 0) * 
                             np.array([0.229, 0.224, 0.225]) + 
                             np.array([0.485, 0.456, 0.406])) * 255).astype(np.uint8)
            norm_path = debug_dir / f"normalized_{timestamp}.jpg"
            cv2.imwrite(str(norm_path), cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved normalized version to {norm_path}")
            
            # Add batch dimension
            image = image.unsqueeze(0)
            logger.info(f"Final preprocessed shape: {image.shape}")
            
            return image.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
        
    def predict(self, image):
        """Predict the ASL letter from the input image"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model first.")
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_idx = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
            # Convert index to letter (A=0, B=1, etc.)
            predicted_letter = chr(65 + predicted_idx)  # 65 is ASCII for 'A'
            logger.info(f"Predicted letter: {predicted_letter}, confidence: {confidence}")
            
            return predicted_letter, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise 
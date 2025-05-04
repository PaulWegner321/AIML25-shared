import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import io

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ColorASLCNN(nn.Module):
    def __init__(self):
        super(ColorASLCNN, self).__init__()
        # Define the convolutional layers for grayscale input (1 channel)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 8x8 -> 4x4
        )
        
        # Flatten layer to convert 2D image data into 1D feature vector
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        # After 4 max pooling layers: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # With 128 channels: 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 26)  # Output layer for 26 classes (A-Z)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the feature maps into a 1D vector
        x = self.flatten(x)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer for classification
        
        return x

class SignEvaluator:
    def __init__(self):
        print("Initializing sign evaluator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to 64x64
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
        ])
        # Define class labels (A-Z)
        self.class_mapping = {i: chr(i + 65) for i in range(26)}  # A-Z only
        self.class_mapping[26] = 'space'
        self.class_mapping[27] = 'delete'
        self.class_mapping[28] = 'nothing'

    def _load_model(self):
        # Look for the model weights in the new model folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'app', 'new model', 'asl_cnn_weights_2.pth')
        
        if not os.path.exists(model_path):
            print(f"Model weights not found at: {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
        print(f"Loading model weights from: {model_path}")
        model = ColorASLCNN().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def preprocess_image(self, image_input):
        print("Starting image preprocessing...")
        try:
            # Handle both file paths and PIL Image objects
            if isinstance(image_input, str):
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Input must be either a file path or a PIL Image object")
            
            print(f"Input image size: {image.size}, mode: {image.mode}")
            
            print("Applying transformations...")
            # Apply transformations
            tensor = self.transform(image)
            print(f"Tensor shape: {tensor.shape}")
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            print(f"Final tensor shape: {tensor.shape}")
            
            tensor = tensor.to(self.device)
            print("Image preprocessed successfully")
            return tensor
            
        except Exception as e:
            print(f"Error in preprocess_image: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def evaluate_sign(self, image, expected_sign=None):
        """
        Evaluate an ASL sign from an image.
        
        Args:
            image: PIL Image object
            expected_sign: The expected sign (optional)
            
        Returns:
            dict: Evaluation results
        """
        try:
            print("Starting sign evaluation...")
            
            # Preprocess the image
            print("Preprocessing image...")
            processed_image = self.preprocess_image(image)
            print("Image preprocessed successfully")
            
            # Get prediction from the model
            print("Getting prediction from model...")
            with torch.no_grad():
                output = self.model(processed_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
                predicted_sign = self.class_mapping[predicted_idx.item()]
                confidence = confidence.item()
            print(f"Prediction: {predicted_sign}, Confidence: {confidence}")
            
            # Generate feedback
            print("Generating feedback...")
            if expected_sign:
                is_correct = predicted_sign.lower() == expected_sign.lower()
                if is_correct:
                    feedback = "Good job! Your sign is correct."
                else:
                    feedback = f"Your sign was interpreted as '{predicted_sign}', but the expected sign was '{expected_sign}'. Try again!"
            else:
                is_correct = True
                feedback = f"Your sign was interpreted as '{predicted_sign}' with {confidence:.2%} confidence."
            print(f"Feedback: {feedback}")
            
            return {
                'success': True,
                'predicted_sign': predicted_sign,
                'confidence': confidence,
                'feedback': feedback,
                'is_correct': is_correct
            }
        except Exception as e:
            print(f"Error in evaluate_sign: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'predicted_sign': None,
                'confidence': 0.0,
                'feedback': f"Error evaluating sign: {str(e)}",
                'is_correct': False
            } 
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ASLNet(nn.Module):
    def __init__(self):
        super(ASLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Changed from 3 to 1 for grayscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # Changed from 64 to 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)  # Added conv4 layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1152, 512)  # Changed dimensions to match weights
        self.fc2 = nn.Linear(512, 26)  # Changed from 64 to 512
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Print input shape for debugging
        print(f"Input shape: {x.shape}")
        
        # Apply convolutions and pooling
        x = self.pool(F.relu(self.conv1(x)))
        print(f"After conv1: {x.shape}")
        
        x = self.pool(F.relu(self.conv2(x)))
        print(f"After conv2: {x.shape}")
        
        x = self.pool(F.relu(self.conv3(x)))
        print(f"After conv3: {x.shape}")
        
        x = self.pool(F.relu(self.conv4(x)))  # Added conv4 layer
        print(f"After conv4: {x.shape}")
        
        # Flatten
        x = x.view(-1, 1152)  # Changed dimensions to match weights
        print(f"After flatten: {x.shape}")
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        print(f"After fc1: {x.shape}")
        
        x = self.fc2(x)
        print(f"After fc2: {x.shape}")
        
        return x

class SignEvaluator:
    def __init__(self, model_path=None):
        """
        Initialize the sign evaluator with the trained CNN model.
        
        Args:
            model_path: Path to the model weights. If None, will look in the same directory.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ASLNet().to(self.device)
        
        # If model_path is not provided, try to find it in various locations
        if model_path is None:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # List of possible locations for the model weights file
            possible_paths = [
                os.path.join(current_dir, 'asl_cnn_weights.pth'),
                os.path.join(os.path.dirname(current_dir), 'asl_cnn_weights.pth'),
                os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'app', 'asl_cnn_weights.pth'),
                os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'app', 'models', 'asl_cnn_weights.pth'),
                '/opt/render/project/src/backend/app/models/asl_cnn_weights.pth',
                '/opt/render/project/src/backend/app/asl_cnn_weights.pth'
            ]
            
            # Try each path
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Found model weights at: {path}")
                    break
            
            # If no path was found, use the default
            if model_path is None:
                model_path = os.path.join(current_dir, 'asl_cnn_weights.pth')
                print(f"No model weights found in any location, using default: {model_path}")
        
        # Load model weights
        try:
            print(f"Loading model weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            print(f"Models directory contents: {os.listdir(current_dir) if os.path.exists(current_dir) else 'Directory not found'}")
            raise
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Define class labels (A-Z)
        self.classes = [chr(i) for i in range(65, 91)]  # A to Z

    def preprocess_image(self, image):
        """
        Preprocess an image for the model.
        
        Args:
            image: PIL Image object
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            print("Starting image preprocessing...")
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                print(f"Converting image from {image.mode} to RGB")
                image = image.convert("RGB")
            
            # Resize image
            print("Resizing image...")
            image = image.resize((64, 64))
            
            # Convert to tensor
            print("Converting image to tensor...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image)
            
            # Move to device
            print("Moving tensor to device...")
            image_tensor = image_tensor.to(self.device)
            
            # Add batch dimension
            print("Adding batch dimension...")
            image_tensor = image_tensor.unsqueeze(0)
            
            print("Image preprocessing completed successfully")
            return image_tensor
        except Exception as e:
            print(f"Error in preprocess_image: {str(e)}")
            raise

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
                predicted_sign = self.classes[predicted_idx.item()]
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
            return {
                'success': False,
                'predicted_sign': None,
                'confidence': 0.0,
                'feedback': f"Error evaluating sign: {str(e)}",
                'is_correct': False
            } 
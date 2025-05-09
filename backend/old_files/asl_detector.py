import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

class ASLCNN(nn.Module):
    def __init__(self):
        super(ASLCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1152, 512)  # Changed from 512 * 9 to 1152
        self.fc2 = nn.Linear(512, 26)  # 26 classes for letters A-Z
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 1152)  # Changed from 512 * 9 to 1152
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ASLDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ASLCNN().to(self.device)
        
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
        """Preprocess the input image for the model"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension

    def detect(self, image):
        """Detect ASL sign in the input image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = probabilities[0][predicted_class].item()
            
            # Get predicted letter
            predicted_letter = self.classes[predicted_class.item()]
            
            return {
                'letter': predicted_letter,
                'confidence': confidence,
                'success': True
            }
            
        except Exception as e:
            return {
                'letter': None,
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            } 
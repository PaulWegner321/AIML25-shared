import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLCNNModel(nn.Module):
    def __init__(self):
        super(ASLCNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1152, 512)  # 1152 matches the saved weights
        self.fc2 = nn.Linear(512, 26)  # 26 classes for A-Z
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convert input to grayscale if it's RGB (3 channels)
        if x.size(1) == 3:
            # Convert RGB to grayscale using standard coefficients
            x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        
        # Apply convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the feature maps
        x = x.view(-1, 1152)  # Flatten to match fc1 input size

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNPredictor:
    def __init__(self):
        """Initialize the CNN model predictor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Map indices to ASL letters
        self.idx_to_letter = {i: chr(65 + i) for i in range(26)}  # A-Z mapping
        
    def load_model(self, model_path: str):
        """Load the CNN model from the specified path."""
        try:
            # Initialize the model
            self.model = ASLCNNModel()
            
            # Load the state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess the input image for the model."""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            image = Image.fromarray(image_rgb)
        
        # Apply transformations
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)  # Add batch dimension

    def predict(self, image: np.ndarray):
        """Predict the ASL letter from the input image."""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model first.")
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # Preprocess image
            tensor = self.preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                # Get predicted letter
                letter = self.idx_to_letter[predicted_idx]
                
                return letter, confidence
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise 
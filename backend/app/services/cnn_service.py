import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any
import logging
from ..models.cnn_model import ASLCNNModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNPredictor:
    def __init__(self, model_path: str = None):
        """Initialize the CNN model predictor.
        
        Args:
            model_path (str): Path to the saved model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size according to your model's requirements
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if path is provided
        if model_path:
            self.load_model(model_path)
            
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
        """Preprocess the input image for the model.
        
        Args:
            image (np.ndarray): Input image in OpenCV format (BGR)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transformations
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict the ASL letter from the input image.
        
        Args:
            image (np.ndarray): Input image in OpenCV format (BGR)
            
        Returns:
            dict: Prediction results containing:
                - letter: Predicted ASL letter
                - confidence: Confidence score
                - success: Boolean indicating if prediction was successful
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model first.")
            
            # Preprocess image
            tensor = self.preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get predicted letter and confidence
                letter = self.idx_to_letter[predicted.item()]
                confidence_score = confidence.item()
            
            return {
                "success": True,
                "letter": letter,
                "confidence": confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Initialize the predictor
cnn_predictor = CNNPredictor()  # You'll need to set the model path in your application startup 
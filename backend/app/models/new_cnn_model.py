import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to expected input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Standard RGB normalization values
                std=[0.229, 0.224, 0.225]
            )
        ])
        
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
            logger.info(f"Input image type: {type(image)}, shape: {image.shape if isinstance(image, np.ndarray) else 'N/A'}")
            
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if image is from OpenCV
                if image.shape[2] == 3:  # If it's a color image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    logger.info("Converted BGR to RGB")
                # Convert numpy array to PIL Image
                image = transforms.ToPILImage()(image)
                logger.info("Converted to PIL Image")
            
            # Apply transformations
            image = self.transform(image)
            logger.info(f"After transforms - tensor shape: {image.shape}")
            
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
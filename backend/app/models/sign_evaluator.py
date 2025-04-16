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
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 26 * 26, 64)
        self.fc2 = nn.Linear(64, 26)  # 26 classes for A-Z
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 26 * 26)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
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
        
        # If model_path not provided, look in the app directory
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'asl_cnn_weights.pth')
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define class labels (A-Z)
        self.labels = [chr(i) for i in range(65, 91)]  # A-Z in ASCII

    def preprocess_image(self, image):
        """
        Preprocess the input image for model inference
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def evaluate_sign(self, image, expected_sign):
        """
        Evaluate if the sign in the image matches the expected sign.
        
        Args:
            image: PIL Image or numpy array
            expected_sign: The expected sign (A-Z)
            
        Returns:
            dict: Evaluation results including:
                - correct: Boolean indicating if the sign is correct
                - predicted_sign: The predicted sign
                - confidence: Confidence score for the prediction
                - feedback: Feedback message
        """
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Get model prediction
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted = torch.max(probabilities, 1)
            predicted_sign = self.labels[predicted.item()]
            confidence_score = confidence.item()
            
            # Check if the prediction is correct
            is_correct = predicted_sign == expected_sign.upper()
            
            # Generate feedback
            if is_correct:
                feedback = f"Correct! You signed '{predicted_sign}' correctly."
            else:
                feedback = f"Incorrect. You signed '{predicted_sign}' but the expected sign was '{expected_sign.upper()}'."
            
            return {
                "correct": is_correct,
                "predicted_sign": predicted_sign,
                "confidence": confidence_score,
                "feedback": feedback
            } 
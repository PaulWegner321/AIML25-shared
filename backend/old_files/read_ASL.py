import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Update the model path to point to the new model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_new_Mads', 'keypoint_classifier.pt')

class KeyPointClassifier:
    def __init__(self, model_path='./model_new_Mads/asl_cnn_weights.pth'):
        # Load the PyTorch model
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()  # Set the model to evaluation mode

    def __call__(self, input_tensor):
        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get the predicted class index
        result_index = torch.argmax(output, dim=1).item()
        return result_index

class ASLDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.keypoint_classifier = KeyPointClassifier()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def preprocess_image(self, image):
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to expected input size
            image = image.resize((48, 48), Image.LANCZOS)
            
            # Convert to tensor and normalize
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            return tensor
        except Exception as e:
            print(f"Error in preprocess_image: {str(e)}")
            raise

    def evaluate_sign(self, image, expected_sign=None):
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Get prediction from the model
            with torch.no_grad():
                output = self.keypoint_classifier(processed_image)
                predicted_sign = chr(output + 65)  # Convert index to letter (A-Z)
            
            # Generate feedback
            if expected_sign:
                is_correct = predicted_sign.lower() == expected_sign.lower()
                if is_correct:
                    feedback = "Good job! Your sign is correct."
                else:
                    feedback = f"Your sign was interpreted as '{predicted_sign}', but the expected sign was '{expected_sign}'. Try again!"
            else:
                is_correct = True
                feedback = f"Your sign was interpreted as '{predicted_sign}'."
            
            return {
                'success': True,
                'predicted_sign': predicted_sign,
                'confidence': 1.0,  # The model doesn't provide confidence scores
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
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new model', 'asl_cnn_weights_2.pth')

class ASLCNN(nn.Module):
    def __init__(self):
        super(ASLCNN, self).__init__()
        # Define the convolutional layers for RGB input (3 channels)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 512)  # 128 channels * 4 * 4 = 2048
        self.fc2 = nn.Linear(512, 26)    # A-Z classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ASLDetector:
    def __init__(self, model_path=MODEL_PATH):
        print(f"Initializing ASL detector with model: {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ASLCNN().to(self.device)
        
        # Load model weights
        try:
            print(f"Loading model weights from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            raise
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to 64x64
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB
        ])
        
        # Define class labels (A-Z)
        self.class_mapping = {i: chr(i + 65) for i in range(26)}  # A to Z

    def preprocess_image(self, image):
        """Preprocess the input image for the model"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        
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
            predicted_letter = self.class_mapping[predicted_class.item()]
            
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

def test_detector():
    """Test the ASL detector with a sample image"""
    print("Initializing ASL detector...")
    detector = ASLDetector()
    
    # Create a simple test image (white background with a black letter)
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255  # White background, RGB
    
    # Draw a simple 'A' shape in black
    # Vertical line
    img[15:45, 25:30] = [0, 0, 0]  # Black in RGB
    # Top horizontal line
    img[15:20, 20:35] = [0, 0, 0]
    # Middle horizontal line
    img[30:35, 20:35] = [0, 0, 0]
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)  # RGB mode by default
    
    # Save the test image
    pil_img.save("test_image.png")
    print("Created test image: test_image.png")
    
    # Get prediction
    print("Getting prediction...")
    result = detector.detect(pil_img)
    
    # Print results
    print(f"Evaluation result: {result}")
    
    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Test Image\nPredicted: {result.get('letter', 'N/A')}\nConfidence: {result.get('confidence', 0):.2%}")
    plt.axis('off')
    plt.savefig("test_result.png")
    plt.close()
    print("Saved visualization as: test_result.png")

if __name__ == "__main__":
    test_detector() 
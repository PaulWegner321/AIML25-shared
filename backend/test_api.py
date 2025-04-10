import requests
import os
import sys
import numpy as np
from PIL import Image
import io

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.asl_detector import ASLDetector

def test_api():
    """Test the FastAPI endpoint with a sample image"""
    # Create a simple test image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a simple 'A' shape
    img[50:150, 100:120] = 0  # Vertical line
    img[50:70, 80:140] = 0    # Top horizontal line
    img[100:120, 80:140] = 0  # Middle horizontal line
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    
    # Save the test image
    pil_img.save("test_api_image.png")
    print("Created test image: test_api_image.png")
    
    # Convert to bytes for API request
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Test the API endpoint
    print("Testing API endpoint...")
    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    response = requests.post('http://localhost:8000/process-frame', files=files)
    
    # Print results
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Also test the detector directly for comparison
    print("\nTesting detector directly for comparison...")
    detector = ASLDetector()
    predictions = detector.detect(img)
    print(f"Direct predictions: {predictions}")

if __name__ == "__main__":
    test_api() 
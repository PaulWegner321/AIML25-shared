import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the ASL detector
try:
    from models.asl_detector import ASLDetector
    print("Successfully imported ASLDetector")
except ImportError as e:
    print(f"Failed to import ASLDetector: {e}")
    print("\nTrying alternative import paths...")
    
    # Try different import paths
    try:
        import sys
        sys.path.insert(0, "/opt/render/project/src")
        from models.asl_detector import ASLDetector
        print("Successfully imported ASLDetector from /opt/render/project/src")
    except ImportError as e:
        print(f"Failed to import from /opt/render/project/src: {e}")
        
        try:
            sys.path.insert(0, "/opt/render/project/src/backend")
            from models.asl_detector import ASLDetector
            print("Successfully imported ASLDetector from /opt/render/project/src/backend")
        except ImportError as e:
            print(f"Failed to import from /opt/render/project/src/backend: {e}")
            print("\nAll import attempts failed. Exiting.")
            sys.exit(1)

def test_detector():
    """Test the ASL detector with a sample image"""
    print("Initializing ASL detector...")
    detector = ASLDetector()
    
    # Create a simple test image (white background with a black letter)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a simple 'A' shape
    img[50:150, 100:120] = 0  # Vertical line
    img[50:70, 80:140] = 0    # Top horizontal line
    img[100:120, 80:140] = 0  # Middle horizontal line
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    
    # Save the test image
    pil_img.save("test_image.png")
    print("Created test image: test_image.png")
    
    # Get prediction
    print("Getting prediction...")
    predictions = detector.detect(img)
    
    # Print results
    print(f"Predictions: {predictions}")
    
    # Display the image
    plt.imshow(img)
    plt.title(f"Predicted: {predictions[0][0]} (confidence: {predictions[0][1]:.2f})")
    plt.savefig("test_result.png")
    print("Saved result visualization: test_result.png")

if __name__ == "__main__":
    test_detector() 
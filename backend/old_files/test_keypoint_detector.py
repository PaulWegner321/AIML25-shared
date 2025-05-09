import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from backend.app.models.keypoint_detector import HandDetector

def test_detector():
    # Initialize the hand detector
    detector = HandDetector()
    
    # Use the specific test image path
    image_path = '/Users/henrikjacobsen/Desktop/CBS/Semester 2/Artifical Intelligence and Machine Learning/Final Project/AIML25-shared/backend/app/test.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load test image at {image_path}")
        return
    
    # Detect sign
    result = detector.detect_sign(image)
    
    # Print results
    print("\nDetection Results:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Detected Letter: {result['letter']}")
        print(f"Confidence: {result['confidence']:.2f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Draw landmarks on the image
    annotated_image = detector.draw_landmarks(image, result.get('landmarks'))
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Sign: {result.get('letter', 'None')}")
    plt.axis('off')
    plt.savefig('test_result.png')
    plt.close()
    
    print("\nResults have been saved to 'test_result.png'")

if __name__ == "__main__":
    test_detector() 
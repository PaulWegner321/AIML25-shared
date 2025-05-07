#!/usr/bin/env python3
"""
Test script for the Granite Vision model.
This script performs a basic test of the Granite Vision model to check if it's working correctly.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import json
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.vision_evaluator import VisionEvaluator

def create_test_image(letter='A', size=(300, 300)):
    """Create a simple test image with a letter in it."""
    # Create a blank image
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img.fill(255)  # White background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, letter, (100, 200), font, 5, (0, 0, 0), 10, cv2.LINE_AA)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Test the Granite Vision model.')
    parser.add_argument('--image', type=str, help='Path to an ASL sign image to test (optional)')
    parser.add_argument('--letter', type=str, default='A', help='Letter to generate in test image (if no image path)')
    parser.add_argument('--expected', type=str, help='Expected letter in the image (optional)')
    args = parser.parse_args()
    
    print("Initializing Granite Vision evaluator...")
    evaluator = VisionEvaluator()
    
    if not evaluator.initialization_attempted:
        print("Forcing model initialization...")
        success = evaluator._initialize_model()
        if not success:
            if hasattr(evaluator, 'initialization_error'):
                error = evaluator.initialization_error
            else:
                error = "Unknown error"
            print(f"Failed to initialize model: {error}")
            return 1
    
    if not evaluator.initialized:
        print("Model not initialized. Cannot proceed with test.")
        return 1
    
    # Use provided image or create a test image
    if args.image and os.path.isfile(args.image):
        print(f"Loading image from {args.image}...")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Failed to load image from {args.image}")
            return 1
    else:
        letter = args.letter if args.letter else 'A'
        print(f"Creating test image with letter '{letter}'...")
        image = create_test_image(letter)
    
    expected_sign = args.expected or args.letter
    print(f"Expected sign: {expected_sign}")
    
    print("Evaluating image with Granite Vision...")
    result = evaluator.evaluate(image, expected_sign=expected_sign)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))
    
    if result['success']:
        print("\nTest completed successfully!")
        return 0
    else:
        print("\nTest failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
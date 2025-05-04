import numpy as np
from evaluate_models import evaluate_subset

def test_evaluate_subset():
    # Test case 1: Normal case with flipped images
    predictions = ['A', 'B', 'C', 'D']
    ground_truth = ['A', 'B', 'C', 'D']
    image_paths = ['normal1.jpg', 'flipped1.jpg', 'normal2.jpg', 'flipped2.jpg']
    
    result = evaluate_subset(predictions, ground_truth, image_paths, 'flipped')
    print("\nTest Case 1 - Normal case with flipped images:")
    print(f"Result: {result}")
    
    # Test case 2: No flipped images
    predictions = ['A', 'B']
    ground_truth = ['A', 'B']
    image_paths = ['normal1.jpg', 'normal2.jpg']
    
    result = evaluate_subset(predictions, ground_truth, image_paths, 'flipped')
    print("\nTest Case 2 - No flipped images:")
    print(f"Result: {result}")
    
    # Test case 3: More predictions than paths
    predictions = ['A', 'B', 'C', 'D', 'E']
    ground_truth = ['A', 'B', 'C', 'D', 'E']
    image_paths = ['normal1.jpg', 'flipped1.jpg']
    
    result = evaluate_subset(predictions, ground_truth, image_paths, 'flipped')
    print("\nTest Case 3 - More predictions than paths:")
    print(f"Result: {result}")
    
    # Test case 4: More paths than predictions
    predictions = ['A', 'B']
    ground_truth = ['A', 'B']
    image_paths = ['normal1.jpg', 'flipped1.jpg', 'flipped2.jpg', 'normal2.jpg']
    
    result = evaluate_subset(predictions, ground_truth, image_paths, 'flipped')
    print("\nTest Case 4 - More paths than predictions:")
    print(f"Result: {result}")
    
    # Test case 5: Mixed correct and incorrect predictions
    predictions = ['A', 'B', 'C', 'D']
    ground_truth = ['A', 'X', 'C', 'Y']
    image_paths = ['normal1.jpg', 'flipped1.jpg', 'normal2.jpg', 'flipped2.jpg']
    
    result = evaluate_subset(predictions, ground_truth, image_paths, 'flipped')
    print("\nTest Case 5 - Mixed correct and incorrect predictions:")
    print(f"Result: {result}")

if __name__ == "__main__":
    test_evaluate_subset() 
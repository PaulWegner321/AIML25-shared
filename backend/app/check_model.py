import os
import sys

def check_model():
    """Check if the model weights file exists and is accessible"""
    # Get the path to the models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    # Check if the models directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory not found at {models_dir}")
        print("Checking alternative locations...")
        
        # Check alternative locations
        alt_locations = [
            "/opt/render/project/src/models",
            "/opt/render/project/src/backend/models",
            os.path.join(os.getcwd(), "models"),
            os.path.join(os.getcwd(), "backend", "models")
        ]
        
        for location in alt_locations:
            if os.path.exists(location):
                print(f"Models directory found at {location}")
                models_dir = location
                break
        else:
            print("Models directory not found in any alternative location")
            return False
    
    # Check if the model weights file exists
    weights_file = os.path.join(models_dir, "asl_cnn_weights.pth")
    if not os.path.exists(weights_file):
        print(f"Model weights file not found at {weights_file}")
        print("Checking alternative locations...")
        
        # Check alternative locations
        alt_locations = [
            "/opt/render/project/src/models/asl_cnn_weights.pth",
            "/opt/render/project/src/backend/models/asl_cnn_weights.pth",
            os.path.join(os.getcwd(), "models", "asl_cnn_weights.pth"),
            os.path.join(os.getcwd(), "backend", "models", "asl_cnn_weights.pth")
        ]
        
        for location in alt_locations:
            if os.path.exists(location):
                print(f"Model weights file found at {location}")
                weights_file = location
                break
        else:
            print("Model weights file not found in any alternative location")
            return False
    
    # Check if the model weights file is accessible
    try:
        with open(weights_file, "rb") as f:
            # Just read a small chunk to check if the file is accessible
            f.read(1024)
        print(f"Model weights file is accessible at {weights_file}")
        return True
    except Exception as e:
        print(f"Error accessing model weights file: {e}")
        return False

if __name__ == "__main__":
    check_model() 
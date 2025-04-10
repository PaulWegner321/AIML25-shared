import os
import sys

def check_structure():
    """Check the directory structure and Python path"""
    print("Current working directory:", os.getcwd())
    print("\nPython path:")
    for path in sys.path:
        print(f"  - {path}")
    
    print("\nDirectory structure:")
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
    
    print("\nChecking for models directory:")
    models_paths = [
        "./models",
        "../models",
        "/opt/render/project/src/models",
        "/opt/render/project/src/backend/models"
    ]
    
    for path in models_paths:
        exists = os.path.exists(path)
        print(f"  {path}: {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            print(f"    Contents: {os.listdir(path)}")

if __name__ == "__main__":
    check_structure() 
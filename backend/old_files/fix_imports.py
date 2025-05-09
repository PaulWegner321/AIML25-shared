import os
import sys

def fix_imports():
    """Fix the imports in main.py at runtime"""
    # Get the path to main.py
    main_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    
    # Read the current content
    with open(main_py_path, "r") as f:
        content = f.read()
    
    # Check if the import is already fixed
    if "sys.path.insert(0, BACKEND_DIR)" in content:
        print("Imports already fixed in main.py")
        return
    
    # Replace the import section
    import_section = """# Import model modules
import sys
import os

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the backend directory to the Python path
sys.path.insert(0, BACKEND_DIR)

# Now import the ASL detector
from models.asl_detector import ASLDetector

# Initialize model instances
asl_detector = ASLDetector()"""
    
    # Find the old import section
    old_import_section = """# Import model modules
from models.asl_detector import ASLDetector

# Initialize model instances
asl_detector = ASLDetector()"""
    
    # Replace the old import section with the new one
    new_content = content.replace(old_import_section, import_section)
    
    # Write the new content back to main.py
    with open(main_py_path, "w") as f:
        f.write(new_content)
    
    print("Fixed imports in main.py")

if __name__ == "__main__":
    fix_imports() 
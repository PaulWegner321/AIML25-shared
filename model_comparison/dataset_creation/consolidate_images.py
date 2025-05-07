import os
import shutil
from pathlib import Path

# Define source and destination paths
base_path = Path("/Users/henrikjacobsen/Desktop/CBS/Semester 2/Artifical Intelligence and Machine Learning/Final Project/AIML25-shared/model_comparison")
main_data_path = base_path / "data"
mads_data_path = base_path / "data Mads"
paul_data_path = base_path / "data Paul"

def process_images(source_path, destination_path):
    """Process images from source path to destination path."""
    if not source_path.exists():
        print(f"Source path does not exist: {source_path}")
        return

    # Iterate through all letter directories
    for letter_dir in source_path.glob("*"):
        if not letter_dir.is_dir() or letter_dir.name == ".DS_Store":
            continue

        # Create destination letter directory if it doesn't exist
        dest_letter_dir = destination_path / letter_dir.name
        dest_letter_dir.mkdir(exist_ok=True)

        # Process all jpg files in the letter directory
        for img_file in letter_dir.glob("*.jpg"):
            # Create new filename with correct grayscale naming
            new_name = img_file.name.replace("_greyscaled", "_grayscale")
            dest_file = dest_letter_dir / new_name

            # Copy file if it doesn't already exist
            if not dest_file.exists():
                shutil.copy2(img_file, dest_file)
                print(f"Copied: {img_file.name} -> {dest_file.name}")
            else:
                print(f"File already exists, skipping: {dest_file.name}")

def main():
    # Process Mads' images
    print("\nProcessing Mads' images...")
    process_images(mads_data_path, main_data_path)

    # Process Paul's images
    print("\nProcessing Paul's images...")
    process_images(paul_data_path, main_data_path)

    print("\nImage consolidation complete!")

if __name__ == "__main__":
    main() 
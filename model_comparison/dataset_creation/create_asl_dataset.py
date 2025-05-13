import os
import cv2
import numpy as np
import time
from datetime import datetime
import keyboard
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the dataset structure
BASE_DIR = "data"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def setup_directories():
    """Create the necessary directories for the dataset."""
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        logging.info(f"Created base directory: {BASE_DIR}")
    
    for letter in LETTERS:
        letter_dir = os.path.join(BASE_DIR, letter)
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
            logging.info(f"Created directory for letter: {letter}")

def save_image(image, letter, index):
    """Save the original, flipped, and grayscale versions of the image."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save original image
    original_path = os.path.join(BASE_DIR, letter, f"{letter}_{index}_{timestamp}.jpg")
    cv2.imwrite(original_path, image)
    logging.info(f"Saved original image: {original_path}")
    
    # Save flipped image
    flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
    flipped_path = os.path.join(BASE_DIR, letter, f"{letter}_{index}_{timestamp}_flipped.jpg")
    cv2.imwrite(flipped_path, flipped_image)
    logging.info(f"Saved flipped image: {flipped_path}")
    
    # Save grayscale image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_path = os.path.join(BASE_DIR, letter, f"{letter}_{index}_{timestamp}_grayscale.jpg")
    cv2.imwrite(grayscale_path, grayscale_image)
    logging.info(f"Saved grayscale image: {grayscale_path}")
    
    return index + 1

def capture_dataset():
    """Capture ASL signs from the camera and save them to the dataset."""
    setup_directories()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize counters for each letter
    counters = {letter: 0 for letter in LETTERS}
    
    # Create a window
    cv2.namedWindow("ASL Dataset Creator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Dataset Creator", 1280, 720)
    
    # Instructions
    print("\n" + "="*50)
    print("ASL Dataset Creator")
    print("="*50)
    print("Press a letter key (A-Z) to capture the corresponding ASL sign")
    print("Press 'SPACE' for the space sign")
    print("Press 'DELETE' for the delete sign")
    print("Press 'q' to quit")
    print("="*50 + "\n")
    
    # Flag to track if a key was pressed
    key_pressed = False
    current_letter = None
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break
            
            # Display the frame
            cv2.imshow("ASL Dataset Creator", frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key press
            if key != 255:  # A key was pressed
                if key == ord('a'):
                    break
                elif key == ord(' '):
                    current_letter = "SPACE"
                    key_pressed = True
                elif key == ord('\b'):  # Backspace key
                    current_letter = "DELETE"
                    key_pressed = True
                else:
                    # Convert key to letter
                    letter = chr(key).upper()
                    if letter in LETTERS:
                        current_letter = letter
                        key_pressed = True
            
            # If a key was pressed, save the image
            if key_pressed and current_letter:
                # Save the image
                counters[current_letter] = save_image(frame, current_letter, counters[current_letter])
                
                # Display confirmation
                print(f"Captured {current_letter} sign. Total: {counters[current_letter]}")
                
                # Reset flags
                key_pressed = False
                current_letter = None
                
                # Add a small delay to prevent multiple captures
                time.sleep(0.5)
    
    finally:
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*50)
        print("Dataset Creation Summary")
        print("="*50)
        for letter, count in counters.items():
            if count > 0:
                print(f"{letter}: {count} images (including flipped and grayscale)")
        print("="*50 + "\n")

if __name__ == "__main__":
    capture_dataset() 
import cv2
import numpy as np

def create_test_image():
    # Create a white background
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a simple hand shape (you can modify this to draw different signs)
    # This is a simple example - you should replace this with a real hand image
    cv2.circle(image, (320, 240), 100, (0, 0, 0), 2)  # Outer circle
    cv2.circle(image, (320, 240), 50, (0, 0, 0), 2)   # Inner circle
    
    # Save the image
    cv2.imwrite('test_image.jpg', image)
    print("Test image created: test_image.jpg")

if __name__ == "__main__":
    create_test_image() 
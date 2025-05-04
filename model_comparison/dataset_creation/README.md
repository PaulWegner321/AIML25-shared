# ASL Dataset Creation Tool

This tool helps create a dataset of American Sign Language (ASL) hand gestures using your webcam. It's designed to capture and save images of hand signs for each letter of the alphabet in a structured format.

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages:
  - OpenCV (`cv2`)
  - NumPy
  - Mediapipe
  - Python-dotenv

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd dataset_creation
```

2. Install the required packages:
```bash
pip install opencv-python numpy mediapipe python-dotenv
```

## Usage

1. Run the script:
```bash
python create_asl_dataset.py
```

2. The script will:
   - Create a `data` directory with subdirectories for each letter (A-Z)
   - Open your webcam
   - Display the webcam feed with hand tracking
   - Show instructions on the screen

3. During capture:
   - Press the letter key (A-Z) you want to capture
   - Hold your hand in the ASL position for that letter
   - The script will:
     - Capture the image
     - Save both the original and a flipped version
     - Add a timestamp to the filename
     - Save in grayscale and color formats

4. Additional controls:
   - Press `ESC` to exit the program
   - Press `SPACE` to pause/resume capture
   - The script will automatically detect and track your hand

## Output Structure

The script creates the following directory structure:
```
data/
├── A/
│   ├── A_0_[timestamp]_original.jpg
│   ├── A_0_[timestamp]_flipped.jpg
│   ├── A_0_[timestamp]_grayscale.jpg
├── B/
│   ├── B_0_[timestamp]_original.jpg
│   ├── B_0_[timestamp]_flipped.jpg
│   ├── B_0_[timestamp]_grayscale.jpg
...
```

## Best Practices for Data Collection

1. **Lighting**:
   - Use good, consistent lighting
   - Avoid shadows on your hand
   - Natural light or diffused artificial light works best

2. **Background**:
   - Use a plain, contrasting background
   - Avoid busy or cluttered backgrounds
   - Light or dark solid colors work best

3. **Hand Position**:
   - Keep your hand within the marked area on screen
   - Maintain a consistent distance from the camera
   - Show clear finger positions for each sign

4. **Variations**:
   - Capture signs from slightly different angles
   - Include minor variations in hand position
   - Collect multiple samples for each letter

## Contributing

When contributing to the dataset:
1. Follow the best practices above
2. Verify image quality before submission
3. Ensure correct labeling of letters
4. Include a variety of hand sizes and skin tones
5. Test the signs with native ASL users if possible

## Troubleshooting

1. If the webcam doesn't open:
   - Check if another application is using the camera
   - Try changing the camera index in the script
   - Verify webcam permissions

2. If hand tracking is inconsistent:
   - Improve lighting conditions
   - Use a simpler background
   - Adjust your hand position

3. If images aren't saving:
   - Check write permissions in the data directory
   - Verify the directory structure exists
   - Check available disk space

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
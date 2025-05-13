import os
import json
import logging
import base64
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import io
import time
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from backend/.env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading .env file from: {dotenv_path}")

# Get Google API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found in environment variables. Please check your .env file.")

# Initialize the Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Model configurations
MODEL_CONFIGS = {
    "flash": {
        "name": "gemini-2.0-flash",
        "display_name": "Gemini 2.0 Flash",
        "rate_limit_delay": 5,  # 5 seconds between requests
        "max_retries": 3
    }
}

# Prompt templates
PROMPT_TEMPLATES = {
"zero_shot": """You are an expert in American Sign Language (ASL) recognition. Analyze the provided image and identify the ASL letter being signed (A-Z).

Respond only with a valid JSON object, using this format:
{
  "letter": "A single uppercase letter (A-Z)",
  "confidence": "confidence score (0-1)",
  "feedback": "A short explanation of how the gesture maps to the predicted letter"
}
Be precise and avoid adding anything outside the JSON response.""",

"few_shot": """You are an expert in American Sign Language (ASL) recognition. Analyze the provided image and identify the ASL letter being signed (A-Z).

Here are some known ASL hand signs:
- A: Fist with thumb resting on the side
- B: Flat open hand, fingers extended upward, thumb across the palm
- C: Hand curved into the shape of the letter C
- D: Index finger up, thumb touching middle finger forming an oval
- E: Fingers bent, thumb tucked under

Respond only with a JSON object like this:
{
  "letter": "A single uppercase letter (A-Z)",
  "confidence": "confidence score (0-1)",
  "feedback": "Why this gesture matches the predicted letter"
}
Only return the JSON object. No explanations before or after.""",

"chain_of_thought": """You are an expert in American Sign Language (ASL) recognition. Carefully analyze the provided image step-by-step to identify the ASL letter (A-Z).

1. Describe the hand shape
2. Describe the finger and thumb positions
3. Compare these to known ASL letter signs
4. Identify the most likely letter

Then output your answer as JSON:
{
  "letter": "A single uppercase letter (A-Z)",
  "confidence": "confidence score (0-1),
  "feedback": "Summarize your reasoning in one sentence"
}
Return only the JSON object with no extra text.""",

"visual_grounding": """You are an expert in American Sign Language (ASL) recognition. Carefully analyze the provided image of a hand gesture and determine which ASL letter (A–Z) it represents.

To guide your analysis, consider the following:
- Which fingers are extended or bent?
- Is the thumb visible, and where is it positioned?
- What is the orientation of the palm (facing forward, sideways, etc.)?
- Are there any unique shapes formed (e.g., circles, fists, curves)?

Now, based on this visual inspection, provide your prediction in the following JSON format:

{
  "letter": "predicted letter (A-Z)",
  "confidence": "confidence score (0–1)",
  "feedback": "brief explanation describing the observed hand shape and reasoning"
}

Be precise, use visual clues from the image, and avoid guessing without justification.""",

"contrastive": """You are an expert in American Sign Language (ASL) recognition. Analyze the provided image of a hand gesture and identify the correct ASL letter.

Consider the following candidate letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
(These letters are visually similar and often confused.)

Step-by-step:
1. Observe the hand shape, finger positions, and thumb placement.
2. Compare the observed gesture against the typical signs for each candidate letter.
3. Eliminate unlikely candidates based on visible differences.
4. Choose the most plausible letter and explain your reasoning.

Format your response as JSON:

{
  "letter": "predicted letter from candidates",
  "confidence": "confidence score (0–1)",
  "feedback": "why this letter was selected over the others"
}

Be analytical and compare carefully to avoid misclassification."""

}

def encode_and_convert_image(image_path, target_format="JPEG", quality=95, max_size=(512, 512)):
    """Process image and return raw bytes and MIME type."""
    try:
        with Image.open(image_path) as img:
            # Log original image format
            logging.debug(f"Original image format: {img.format}")
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                logging.debug(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
            
            # Resize if needed
            if max(img.size) > max(max_size):
                logging.debug(f"Resizing image from {img.size} to max {max_size}")
                img.thumbnail(max_size)  # Resize while maintaining aspect ratio
            
            # Save to bytes with specified format
            buffer = io.BytesIO()
            img.save(buffer, format=target_format, quality=quality)
            image_bytes = buffer.getvalue()
            
            # Get the correct MIME type
            mime_type = f"image/{target_format.lower()}"
            logging.debug(f"Using MIME type: {mime_type}")
            logging.debug(f"Image processing successful. Size: {len(image_bytes)} bytes")
            
            return mime_type, image_bytes
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        raise

def get_asl_prediction(image_path: str, model_type: Literal["flash"] = "flash", prompt_strategy: Literal["zero_shot", "few_shot", "chain_of_thought", "visual_grounding", "contrastive"] = "zero_shot") -> dict:
    """Get ASL prediction from specified Gemini model for a given image path, including visibility check."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model type. Must be one of: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    model_name = config["name"]
    display_name = config["display_name"]
    delay_seconds = config["rate_limit_delay"]
    max_retries = config["max_retries"]
    
    # Initialize timing
    start_time = time.time()
    
    try:
        # Process the image and get raw bytes and MIME type
        mime_type, image_bytes = encode_and_convert_image(image_path)
        logging.debug(f"Image processed successfully as {mime_type}. Size: {len(image_bytes)} bytes")

        # Step 1: Visibility Check
        logging.debug(f"Sending visibility check request to {display_name}...")
        visibility_contents = [
            {
                "role": "user",
                "parts": [
                    types.Part.from_bytes(
                        mime_type=mime_type,
                        data=image_bytes
                    ),
                    {
                        "text": "Can you see the image I'm sending? Please respond with ONLY 'yes' or 'no'."
                    }
                ]
            }
        ]
        
        # Count tokens for visibility check
        visibility_tokens = client.models.count_tokens(
            model=model_name,
            contents=visibility_contents
        )
        
        visibility_response = client.models.generate_content(
            model=model_name,
            contents=visibility_contents,
            config=types.GenerateContentConfig(
                temperature=0.05,
                top_p=1.0,
                max_output_tokens=300
            )
        )
        
        visibility_text = visibility_response.text.strip().lower()
        logging.debug(f"Visibility response from {display_name}: {visibility_text}")
        if "yes" not in visibility_text:
            logging.warning(f"{display_name} visibility check failed for {image_path}. Response: {visibility_text}")
            return {
                "error": "Model cannot see the image",
                "metadata": {
                    "response_time_seconds": round(time.time() - start_time, 3),
                    "visibility_check_tokens": visibility_tokens.total_tokens
                }
            }
            
        logging.debug(f"{display_name} confirmed image visibility.")

        # Step 2: Proceed with ASL recognition
        logging.debug(f"Sending ASL recognition request to {display_name}...")
        
        # Get appropriate prompt template
        prompt = PROMPT_TEMPLATES[prompt_strategy]
        
        # Prepare contents for ASL recognition
        asl_contents = [
            {
                "role": "user",
                "parts": [
                    types.Part.from_bytes(
                        mime_type=mime_type,
                        data=image_bytes
                    ),
                    {
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Count tokens for ASL recognition
        asl_tokens = client.models.count_tokens(
            model=model_name,
            contents=asl_contents
        )
        
        # Make prediction
        response = client.models.generate_content(
            model=model_name,
            contents=asl_contents,
            config=types.GenerateContentConfig(
                temperature=0.05,
                top_p=1.0,
                max_output_tokens=300
            )
        )
        
        # Calculate total response time
        response_time = time.time() - start_time
        
        # Extract the response text
        response_text = response.text.strip()
        logging.debug(f"Raw ASL response from {display_name}: {response_text}")
        
        # Try to parse the JSON response
        try:
            # Find the JSON object in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Add timing and token information to the result
                result["metadata"] = {
                    "response_time_seconds": round(response_time, 3),
                    "visibility_check_tokens": visibility_tokens.total_tokens,
                    "asl_recognition_tokens": asl_tokens.total_tokens,
                    "total_tokens": visibility_tokens.total_tokens + asl_tokens.total_tokens
                }
                return result
            else:
                logging.warning(f"No JSON object found in {display_name} ASL response for {image_path}")
                return {
                    "error": "No JSON found in response",
                    "metadata": {
                        "response_time_seconds": round(response_time, 3),
                        "visibility_check_tokens": visibility_tokens.total_tokens,
                        "asl_recognition_tokens": asl_tokens.total_tokens,
                        "total_tokens": visibility_tokens.total_tokens + asl_tokens.total_tokens
                    }
                }
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON from {display_name} ASL response for {image_path}")
            return {
                "error": "Invalid JSON response",
                "metadata": {
                    "response_time_seconds": round(response_time, 3),
                    "visibility_check_tokens": visibility_tokens.total_tokens,
                    "asl_recognition_tokens": asl_tokens.total_tokens,
                    "total_tokens": visibility_tokens.total_tokens + asl_tokens.total_tokens
                }
            }
            
    except Exception as e:
        response_time = time.time() - start_time
        logging.error(f"Error getting prediction from {display_name} for {image_path}: {e}")
        # Check for specific API errors (like google.api_core.exceptions.InvalidArgument)
        error_response = {
            "error": f"API Error: {e.message}" if hasattr(e, 'message') else str(e),
            "metadata": {
                "response_time_seconds": round(response_time, 3)
            }
        }
        return error_response
    finally:
        # Add a delay to respect rate limits
        logging.debug(f"Waiting {delay_seconds}s before next {display_name} request...")
        time.sleep(delay_seconds)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Gemini models for ASL recognition')
    parser.add_argument('--image', type=str, default="data/V/V_0_20250428_114109_flipped.jpg",
                        help='Path to the image file to test')
    parser.add_argument('--model', type=str, default="flash",
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model to use (flash)')
    parser.add_argument('--prompt-strategy', type=str, 
                        choices=['zero_shot', 'few_shot', 'chain_of_thought', 'visual_grounding', 'contrastive'], 
                        default='zero_shot',
                        help='Prompting strategy to use')
    args = parser.parse_args()
    
    image_path = args.image
    model_type = args.model
    
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        exit(1)
    
    # Test the model
    config = MODEL_CONFIGS[model_type]
    logging.info(f"Testing {config['display_name']} model with image: {image_path}")
    result = get_asl_prediction(image_path, model_type, args.prompt_strategy)
    
    # Print the result
    print(f"\n{config['display_name']} result:")
    print(json.dumps(result, indent=2))
    
    # Save the result to a file
    timestamp = int(time.time())
    results_file = f"{model_type}_{args.prompt_strategy}_result_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {results_file}")

if __name__ == "__main__":
    main()
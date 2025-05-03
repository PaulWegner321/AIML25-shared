import os
import json
import logging
import base64
import time
import requests
import argparse
from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the correct path
dotenv_path = os.path.join('../../backend', '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading .env file from: {dotenv_path}")

# Get WatsonX credentials
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
GRANITE_MODEL_ID = "ibm/granite-vision-3-2-2b"

# Prompt templates
PROMPT_TEMPLATES = {
    "zero_shot": """You are an ASL (American Sign Language) recognition expert. 
Analyze the image and identify the ASL letter being signed.

IMPORTANT INSTRUCTIONS:
1. Focus ONLY on the person's hand in the image - ignore everything else
2. Look specifically at:
   - Hand shape and finger positions
   - Orientation of the hand
   - Any specific gestures or movements shown
3. Ignore any text, labels, or other elements in the image
4. The image shows a single hand performing an ASL sign for one letter
5. Pay special attention to:
   - Which fingers are extended and which are folded
   - The angle between extended fingers
   - The overall hand orientation (palm facing in/out/up/down)
   - Any specific finger configurations (e.g., crossed fingers, spread fingers)

Please respond in the following JSON format:
{
    "letter": "",  // The letter being signed (A-Z)
    "confidence": 0.95,  // Your confidence in the prediction (0-1)
    "explanation": "Brief explanation focusing on the hand position and shape that indicates this letter"
}

Valid letters are: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z""",

    "few_shot": """You are an ASL (American Sign Language) recognition expert. 
Analyze the image and identify the ASL letter being signed.

Here are some examples of ASL letters and their characteristics:
- A: Fist with thumb alongside fingers
- B: Flat hand, fingers together, thumb alongside
- C: Curved hand, thumb and fingers forming a C shape
- D: Index finger pointing up, other fingers closed
- E: Fingers slightly curved, thumb alongside
- V: Index and middle fingers extended and spread apart, other fingers folded
- Z: Index and middle fingers extended and crossed
- W: Three fingers extended (index, middle, ring)
- L: Index finger extended, thumb extended perpendicular to it

IMPORTANT INSTRUCTIONS:
1. Focus ONLY on the person's hand in the image - ignore everything else
2. Look specifically at:
   - Hand shape and finger positions
   - Orientation of the hand
   - Any specific gestures or movements shown
3. Ignore any text, labels, or other elements in the image
4. The image shows a single hand performing an ASL sign for one letter
5. Pay special attention to:
   - Which fingers are extended and which are folded
   - The angle between extended fingers
   - The overall hand orientation (palm facing in/out/up/down)
   - Any specific finger configurations (e.g., crossed fingers, spread fingers)

Please respond in the following JSON format:
{
    "letter": "",  // The letter being signed (A-Z)
    "confidence": 0.95,  // Your confidence in the prediction (0-1)
    "explanation": "Brief explanation focusing on the hand position and shape that indicates this letter"
}

Valid letters are: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z""",

    "chain_of_thought": """You are an ASL (American Sign Language) recognition expert. 
Analyze the image and identify the ASL letter being signed.

Let's analyze this step by step:
1. First, observe the overall hand shape and orientation
2. Then, examine the position of each finger and the thumb
3. Compare these observations with known ASL letter formations
4. Consider any unique characteristics or distinguishing features
5. Finally, make your prediction based on the complete analysis

IMPORTANT INSTRUCTIONS:
1. Focus ONLY on the person's hand in the image - ignore everything else
2. Look specifically at:
   - Hand shape and finger positions
   - Orientation of the hand
   - Any specific gestures or movements shown
3. Ignore any text, labels, or other elements in the image
4. The image shows a single hand performing an ASL sign for one letter
5. Pay special attention to:
   - Which fingers are extended and which are folded
   - The angle between extended fingers
   - The overall hand orientation (palm facing in/out/up/down)
   - Any specific finger configurations (e.g., crossed fingers, spread fingers)

Please respond in the following JSON format:
{
    "letter": "",  // The letter being signed (A-Z)
    "confidence": 0.95,  // Your confidence in the prediction (0-1)
    "explanation": "Brief explanation focusing on the hand position and shape that indicates this letter"
}

Valid letters are: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z"""
}

def encode_image_base64(image_path, resize_dim=(512, 512)):
    """Process image and return base64 string with data URI prefix."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if needed
            if max(img.size) > max(resize_dim):
                img.thumbnail(resize_dim)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()
            
            # Create base64 string with data URI prefix
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{base64_str}"
            
            logging.debug(f"Image processed successfully. Size: {len(base64_str)} bytes")
            return data_uri
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        raise

def get_watsonx_token(api_key):
    """Get or refresh the IBM Cloud IAM token."""
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    
    try:
        response = requests.post(auth_url, headers=headers, data=data, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    except Exception as e:
        logging.error(f"Error getting WatsonX token: {e}")
        return None

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text using a simple approximation."""
    # Rough estimation: 1 token ≈ 4 characters for English text
    # This is a conservative estimate that tends to be higher than actual token count
    return max(1, len(text) // 4)

def get_asl_prediction(image_path: str, prompt_strategy: Literal["zero_shot", "few_shot", "chain_of_thought"] = "zero_shot") -> dict:
    """Get ASL prediction from Granite Vision model."""
    start_time = time.time()
    
    try:
        token = get_watsonx_token(WATSONX_API_KEY)
        if not token:
            return {"error": "Failed to get authentication token", "metadata": {"response_time_seconds": round(time.time() - start_time, 3)}}

        # Process image and get data URI
        image_data_uri = encode_image_base64(image_path)

        # First, test if the model can see the image
        visibility_test_prompt = "Can you see the image I provided? Please respond with 'Yes, I can see the image' or 'No, I cannot see the image'."
        visibility_check_tokens = estimate_tokens(visibility_test_prompt)

        visibility_test_payload = {
            "model_id": GRANITE_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": visibility_test_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri
                            }
                        }
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Make the visibility test API request
        visibility_response = requests.post(
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
            headers=headers,
            json=visibility_test_payload,
            timeout=30
        )

        if visibility_response.status_code != 200:
            logging.error(f"Visibility test failed: {visibility_response.status_code} - {visibility_response.text}")
            return {
                "error": f"Visibility test failed: {visibility_response.status_code}",
                "metadata": {
                    "response_time_seconds": round(time.time() - start_time, 3),
                    "visibility_check_tokens": visibility_check_tokens
                }
            }

        visibility_result = visibility_response.json()
        visibility_text = visibility_result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if "cannot see" in visibility_text.lower():
            logging.error("Model cannot see the image. Please check the image format and encoding.")
            return {
                "error": "Model cannot see the image",
                "metadata": {
                    "response_time_seconds": round(time.time() - start_time, 3),
                    "visibility_check_tokens": visibility_check_tokens
                }
            }

        # If visibility test passed, proceed with the ASL sign recognition
        logging.info("Proceeding with ASL sign recognition...")

        # Get appropriate prompt template
        prompt = PROMPT_TEMPLATES[prompt_strategy]
        asl_recognition_tokens = estimate_tokens(prompt)

        # Prepare the API request for ASL recognition
        asl_payload = {
            "model_id": GRANITE_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri
                            }
                        }
                    ]
                }
            ]
        }

        # Make the ASL recognition API request
        response = requests.post(
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
            headers=headers,
            json=asl_payload,
            timeout=30
        )

        if response.status_code != 200:
            logging.error(f"API request failed: {response.status_code} - {response.text}")
            return {
                "error": f"API request failed: {response.status_code}",
                "metadata": {
                    "response_time_seconds": round(time.time() - start_time, 3),
                    "visibility_check_tokens": visibility_check_tokens,
                    "asl_recognition_tokens": asl_recognition_tokens,
                    "total_tokens": visibility_check_tokens + asl_recognition_tokens
                }
            }

        response_data = response.json()
        generated_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Calculate total response time
        response_time = time.time() - start_time

        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = generated_text[json_start:json_end]
            result = json.loads(json_str)
            
            # Add timing and token information to the result
            result["metadata"] = {
                "response_time_seconds": round(response_time, 3),
                "visibility_check_tokens": visibility_check_tokens,
                "asl_recognition_tokens": asl_recognition_tokens,
                "total_tokens": visibility_check_tokens + asl_recognition_tokens
            }
            return result
        else:
            return {
                "error": "No JSON found in response",
                "raw_response": generated_text,
                "metadata": {
                    "response_time_seconds": round(response_time, 3),
                    "visibility_check_tokens": visibility_check_tokens,
                    "asl_recognition_tokens": asl_recognition_tokens,
                    "total_tokens": visibility_check_tokens + asl_recognition_tokens
                }
            }

    except Exception as e:
        response_time = time.time() - start_time
        logging.error(f"Error getting prediction: {e}")
        return {
            "error": str(e),
            "metadata": {"response_time_seconds": round(response_time, 3)}
        }
    finally:
        # Add a delay to respect rate limits
        time.sleep(2)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Granite Vision model for ASL recognition')
    parser.add_argument('--image', type=str, default="data/V/V_0_20250428_114109_flipped.jpg",
                        help='Path to the image file to test')
    parser.add_argument('--prompt-strategy', type=str, 
                        choices=['zero_shot', 'few_shot', 'chain_of_thought'], 
                        default='zero_shot',
                        help='Prompting strategy to use')
    args = parser.parse_args()
    
    image_path = args.image
    
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        exit(1)
    
    # Test the model
    logging.info(f"Testing Granite Vision model with image: {image_path}")
    result = get_asl_prediction(image_path, args.prompt_strategy)
    
    # Print the result
    print("\nGranite Vision result:")
    print(json.dumps(result, indent=2))
    
    # Save the result to a file
    timestamp = int(time.time())
    results_file = f"granite_vision_{args.prompt_strategy}_result_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {results_file}")

if __name__ == "__main__":
    main() 
import os
import json
import logging
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io
import time
import re
import argparse
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from backend/.env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading .env file from: {dotenv_path}")

# Get WatsonX credentials
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not all([WATSONX_API_KEY, WATSONX_PROJECT_ID]):
    raise ValueError("WatsonX credentials not found in environment variables. Please check your .env file.")

# Model ID
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"

# Prompting strategies
ZERO_SHOT_PROMPT = """You are a professional ASL interpreter. Analyze the hand gesture shown in the image.

It is one of these 29 American Sign Language symbols:
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, space, delete, nothing.

Respond **only** with a JSON object in the following format:
{
  "letter": "your_single_choice_from_above",
  "confidence": number_from_0_to_100,
  "feedback": "brief explanation of visible hand shape"
}

Do not infer based on context or imagination. Only use the visible hand shape."""

FEW_SHOT_PROMPT = """You are a professional ASL interpreter. Here are some examples of ASL hand gestures and their interpretations:

Example 1:
Hand shape: Index and middle fingers forming a V shape, palm facing forward
Letter: V
Confidence: 95
Feedback: Clear V-shape formed by extended index and middle fingers

Example 2:
Hand shape: Closed fist with thumb across palm
Letter: A
Confidence: 90
Feedback: Fist position with thumb laid flat against side

Now, analyze the hand gesture shown in the image.
It is one of these 29 American Sign Language symbols:
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, space, delete, nothing.

Respond **only** with a JSON object in the following format:
{
  "letter": "your_single_choice_from_above",
  "confidence": number_from_0_to_100,
  "feedback": "brief explanation of visible hand shape"
}

Do not infer based on context or imagination. Only use the visible hand shape."""

CHAIN_OF_THOUGHT_PROMPT = """You are a professional ASL interpreter. Let's analyze the hand gesture shown in the image step by step:

1. First, observe the overall hand position and orientation
2. Look at the finger positions and relationships
3. Note any distinctive features (bent fingers, touching points, etc.)
4. Compare to known ASL letter formations
5. Make your determination based on the closest match

The sign must be one of these 29 American Sign Language symbols:
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, space, delete, nothing.

After your analysis, respond **only** with a JSON object in the following format:
{
  "letter": "your_single_choice_from_above",
  "confidence": number_from_0_to_100,
  "feedback": "brief explanation including your step-by-step analysis"
}

Do not infer based on context or imagination. Only use the visible hand shape."""

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string (1 token ≈ 4 characters)."""
    return len(text) // 4

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_watsonx_token(api_key: str) -> str:
    """Get a token for WatsonX API authentication with retry logic."""
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    
    try:
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        logging.error(f"Error getting WatsonX token: {e}")
        raise

def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image if needed
            max_size = (800, 800)  # Reduced from 1024x1024 to 800x800
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes with higher quality
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=100)  # Increased quality from 95 to 100
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_api_request(token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make API request with retry logic."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.post(
        f"{WATSONX_URL}/ml/v1/text/chat?version=2023-05-29",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()

def get_asl_prediction(image_path: str, strategy: str = "zero-shot") -> Dict[str, Any]:
    """Get ASL prediction from Llama Scout 17B model with specified prompting strategy."""
    start_time = time.time()
    
    try:
        # Get authentication token
        token = get_watsonx_token(WATSONX_API_KEY)
        
        # Encode image
        image_base64 = encode_image_base64(image_path)
        logging.info(f"Image encoded successfully. Base64 size: {len(image_base64)} characters")
        
        # First, test if the model can see the image
        visibility_test_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Look at the image I provided. Can you see a hand gesture? If yes, describe what you see. If no, say 'No, I cannot see the image'."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        visibility_test_payload = {
            "model_id": MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": visibility_test_messages
        }
        
        # Make the visibility test API request
        logging.info("Testing image visibility...")
        visibility_result = make_api_request(token, visibility_test_payload)
        
        # Extract the visibility response
        visibility_text = visibility_result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logging.info(f"Visibility test response: {visibility_text}")
        
        if "no, i cannot see the image" in visibility_text.lower():
            raise ValueError("Model cannot see the image")
        
        # If visibility test passed, proceed with the ASL sign recognition
        logging.info("Proceeding with ASL sign recognition...")
        
        # Select prompt based on strategy
        prompt_map = {
            "zero-shot": ZERO_SHOT_PROMPT,
            "few-shot": FEW_SHOT_PROMPT,
            "chain-of-thought": CHAIN_OF_THOUGHT_PROMPT
        }
        prompt = prompt_map.get(strategy, ZERO_SHOT_PROMPT)
        
        # Create the message with image and prompt
        messages = [
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
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        payload = {
            "model_id": MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": messages
        }
        
        # Make the API request
        result = make_api_request(token, payload)
        
        # Extract the generated text
        generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logging.info(f"Generated text: {generated_text}")
        
        # Calculate response time and token estimates
        response_time = time.time() - start_time
        prompt_tokens = estimate_tokens(prompt)
        response_tokens = estimate_tokens(generated_text)
        total_tokens = prompt_tokens + response_tokens
        
        # Try to parse the JSON response
        try:
            # Find the JSON object in the response
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                prediction = json.loads(json_str)
                
                # Add metadata to the response
                final_result = {
                    "prediction": prediction,
                    "metadata": {
                        "model": "llama-scout-17b",
                        "strategy": strategy,
                        "response_time": round(response_time, 3),
                        "tokens": {
                            "prompt": prompt_tokens,
                            "response": response_tokens,
                            "total": total_tokens
                        }
                    }
                }
                return final_result
            else:
                raise ValueError("No JSON object found in response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}")
            
    except Exception as e:
        logging.error(f"Error in ASL prediction: {e}")
        return {
            "error": str(e),
            "metadata": {
                "model": "llama-scout-17b",
                "strategy": strategy,
                "response_time": round(time.time() - start_time, 3)
            }
        }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Llama Scout 17B model with different prompting strategies')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--prompt-strategy', type=str, choices=['zero-shot', 'few-shot', 'chain-of-thought'],
                      default='zero-shot', help='Prompting strategy to use')
    args = parser.parse_args()
    
    # Use provided image path or default
    if args.image:
        image_path = args.image
    else:
        # Use a default image path
        image_path = "/Users/henrikjacobsen/Desktop/CBS/Semester 2/Artifical Intelligence and Machine Learning/Final Project/AIML25-shared/data/V/V_17_20250428_114126.jpg"
        print(f"Using default image path: {image_path}")
    
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        exit(1)
    
    # Test the specified strategy
    logging.info(f"\nTesting Llama Scout 17B model with {args.prompt_strategy} strategy...")
    result = get_asl_prediction(image_path, args.prompt_strategy)
    print(f"\nLlama Scout 17B {args.prompt_strategy} result:")
    print(json.dumps(result, indent=2))
    
    # Save the result to a file
    timestamp = int(time.time())
    results_file = f"llama_scout_17b_{args.prompt_strategy}_result_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {results_file}")

if __name__ == "__main__":
    main() 
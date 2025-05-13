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
from typing import Dict, Any, List, Literal
from tenacity import retry, stop_after_attempt, wait_exponential
from io import BytesIO

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

def encode_image_base64(image_path):
    """Encode image to base64 string with proper format for Llama Scout."""
    try:
        # Open and convert to RGB
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get original dimensions
            logging.info(f"Original image dimensions: {img.size}")
            
            # Resize to smaller dimensions
            new_size = (512, 512)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logging.info(f"Resized image dimensions: {img.size}")
            
            # Save to bytes with JPEG format
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            
            # Get buffer size
            buffer_size = len(buffer.getvalue())
            logging.info(f"Buffer size before base64: {buffer_size} bytes")
            
            # Encode to base64
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            logging.info(f"Base64 string length: {len(img_str)}")
            
            return img_str
    except Exception as e:
        logging.error(f"Error encoding image: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_api_request(token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make API request with retry logic."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.post(
            f"{WATSONX_URL}/ml/v1/text/chat?version=2023-05-29",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            logging.error(f"API Error Response: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {str(e)}")
        if hasattr(e.response, 'text'):
            logging.error(f"Error Response: {e.response.text}")
        raise
    except Exception as e:
        logging.error(f"API Request Error: {str(e)}")
        raise

def get_asl_prediction(image_path: str, strategy: Literal["zero_shot", "few_shot", "chain_of_thought", "visual_grounding", "contrastive"] = "zero_shot") -> dict:
    """Get ASL prediction from Llama Scout model."""
    start_time = time.time()
    
    try:
        token = get_watsonx_token(WATSONX_API_KEY)
        if not token:
            return {
                "error": "Failed to get authentication token",
                "metadata": {
                    "response_time": round(time.time() - start_time, 3),
                    "model": "llama_scout_17b",
                    "strategy": strategy
                }
            }

        # Process image and get data URI
        image_data_uri = encode_image_base64(image_path)

        # Get the appropriate prompt template
        prompt_template = PROMPT_TEMPLATES.get(strategy, PROMPT_TEMPLATES["zero_shot"])

        # Create the payload with the actual ASL prediction request
        payload = {
            "model_id": MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in American Sign Language (ASL) recognition. Analyze the provided image and identify the ASL letter being signed (A-Z)."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_template
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data_uri}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.05,
            "top_p": 1.0,
            "max_tokens": 300
        }
        
        result = make_api_request(token, payload)
        
        # Extract the generated text
        generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Clean the response text - remove any markdown code blocks
        generated_text = generated_text.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        try:
            prediction = json.loads(generated_text)
            if not isinstance(prediction, dict):
                raise ValueError("Response is not a valid JSON object")
            
            if "letter" not in prediction:
                raise ValueError("Response missing 'letter' field")

            # Calculate response time
            response_time = time.time() - start_time
            
            # Return the prediction in the format expected by evaluate_models.py
            return {
                "letter": prediction["letter"],
                "confidence": prediction.get("confidence", 0.0),
                "feedback": prediction.get("feedback", ""),
                "metadata": {
                    "response_time": round(response_time, 3),
                    "model": "llama_scout_17b",
                    "strategy": strategy
                }
            }
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            logging.error(f"Raw response: {generated_text}")
            return {
                "error": "Invalid JSON response from model",
                "raw_response": generated_text,
                "metadata": {
                    "response_time": round(time.time() - start_time, 3),
                    "model": "llama_scout_17b",
                    "strategy": strategy
                }
            }
        except ValueError as e:
            logging.error(f"Invalid response format: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "response_time": round(time.time() - start_time, 3),
                    "model": "llama_scout_17b",
                    "strategy": strategy
                }
            }

    except Exception as e:
        response_time = time.time() - start_time
        logging.error(f"Error in ASL prediction: {e}")
        return {
            "error": str(e),
            "metadata": {
                "response_time": round(response_time, 3),
                "model": "llama_scout_17b",
                "strategy": strategy
            }
        }
    finally:
        # Add a delay to respect rate limits
        time.sleep(2)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Llama Scout 17B model with different prompting strategies')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--prompt-strategy', type=str, choices=['zero_shot', 'few_shot', 'chain_of_thought', 'visual_grounding', 'contrastive'],
                      default='zero_shot', help='Prompting strategy to use')
    args = parser.parse_args()
    
    # Use provided image path or default
    if args.image:
        image_path = args.image
    else:
        # Use a default image path
        image_path = "/Users/henrikjacobsen/Desktop/CBS/Semester 2/Artifical Intelligence and Machine Learning/Final Project/AIML25-shared/model_comparison/data/V/V_17_20250428_114126.jpg"
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
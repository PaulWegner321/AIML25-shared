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
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Any, Literal

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
MODEL_ID = "mistralai/pixtral-12b"

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
        response = requests.post(auth_url, headers=headers, data=data, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    except Exception as e:
        logging.error(f"Error getting token: {e}")
        raise

def encode_image_base64(image_path: str) -> str:
    """Reads and returns the base64 encoded string of an image."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
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
    
    try:
        response = requests.post(
            f"{WATSONX_URL}/ml/v1/text/chat?version=2023-05-29",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error response: {e.response.status_code} - {e.response.text}")
        raise

def get_asl_prediction(image_path: str, strategy: Literal["zero_shot", "few_shot", "chain_of_thought"] = "zero_shot") -> Dict[str, Any]:
    """Get ASL prediction from Mistral model with specified prompting strategy."""
    start_time = time.time()
    
    try:
        # Get authentication token
        token = get_watsonx_token(WATSONX_API_KEY)
        
        # Encode image
        image_base64 = encode_image_base64(image_path)
        logging.info(f"Image encoded successfully. Base64 size: {len(image_base64)} characters")
        
        # First, test if the model can see the image
        visibility_test_payload = {
            "model_id": MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": [
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
            ],
            "temperature": 0.05,
            "top_p": 1.0,
            "max_tokens": 300
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
        prompt = PROMPT_TEMPLATES.get(strategy, PROMPT_TEMPLATES["zero_shot"])
        
        # Create the message with image and prompt
        asl_payload = {
            "model_id": MODEL_ID,
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
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.05,
            "top_p": 1.0,
            "max_tokens": 300
        }
        
        # Make the API request
        result = make_api_request(token, asl_payload)
        
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
                        "model": "mistral-7b-instruct-v2",
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
                "model": "mistral-7b-instruct-v2",
                "strategy": strategy,
                "response_time": round(time.time() - start_time, 3)
            }
        }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test Mistral model for ASL recognition')
    parser.add_argument('--image', type=str, default="data/V/V_0_20250428_114109_flipped.jpg",
                        help='Path to the image file to test')
    parser.add_argument('--prompt-strategy', type=str, 
                        choices=['zero_shot', 'few_shot', 'chain_of_thought', 'visual_grounding', 'contrastive'], 
                        default='zero_shot',
                        help='Prompting strategy to use')
    args = parser.parse_args()
    
    image_path = args.image
    
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        exit(1)
    
    # Test the model
    logging.info(f"Testing Mistral model with image: {image_path}")
    result = get_asl_prediction(image_path, args.prompt_strategy)
    
    # Print the result
    print("\nMistral result:")
    print(json.dumps(result, indent=2))
    
    # Save the result to a file
    timestamp = int(time.time())
    results_file = f"mistral_{args.prompt_strategy}_result_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {results_file}")

if __name__ == "__main__":
    main() 
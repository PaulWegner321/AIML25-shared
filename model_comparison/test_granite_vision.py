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
GRANITE_MODEL_ID = "ibm/granite-vision-3-2-2b"

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

def encode_image_base64(image_path):
    """Encode image to base64 string with proper format for Granite Vision."""
    try:
        # Open and convert to RGB
        img = Image.open(image_path).convert('RGB')
        
        # Calculate new dimensions while maintaining aspect ratio
        max_size = 1024
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        
        # Resize with high-quality resampling
        img = img.resize(new_size, Image.Resampling.LANCZOS)
            
        # Save to bytes with JPEG format and maximum quality
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=100, optimize=True)
        buffer.seek(0)
            
        # Encode to base64
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Clean the base64 string
        img_str = img_str.replace('\n', '')
        
        # Create the full data URI with proper MIME type
        data_uri = f"data:image/jpeg;base64,{img_str}"
            
        # Log the data URI format (truncated for readability)
        logging.info(f"Image data URI format: {data_uri[:100]}...")
        
        return data_uri
    except Exception as e:
        logging.error(f"Error encoding image: {str(e)}")
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

def get_asl_prediction(image_path: str, prompt_strategy: Literal["zero_shot", "few_shot", "chain_of_thought", "visual_grounding", "contrastive"] = "zero_shot") -> dict:
    """Get ASL prediction from Granite Vision model."""
    start_time = time.time()
    
    try:
        token = get_watsonx_token(WATSONX_API_KEY)
        if not token:
            return {
                "error": "Failed to get authentication token",
                "metadata": {
                    "response_time": round(time.time() - start_time, 3),
                    "model": "granite_vision",
                    "strategy": prompt_strategy
                }
            }

        # Process image and get data URI
        image_data_uri = encode_image_base64(image_path)

        # Get the appropriate prompt template
        prompt_template = PROMPT_TEMPLATES.get(prompt_strategy, PROMPT_TEMPLATES["zero_shot"])

        # Create the payload with the actual ASL prediction request
        payload = {
            "model_id": GRANITE_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "messages": [
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
                                "url": image_data_uri
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
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
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
                    "model": "granite_vision",
                    "strategy": prompt_strategy
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
                    "model": "granite_vision",
                    "strategy": prompt_strategy
                }
            }
        except ValueError as e:
            logging.error(f"Invalid response format: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "response_time": round(time.time() - start_time, 3),
                    "model": "granite_vision",
                    "strategy": prompt_strategy
                }
            }

    except Exception as e:
        response_time = time.time() - start_time
        logging.error(f"Error in ASL prediction: {e}")
        return {
            "error": str(e),
            "metadata": {
                "response_time": round(response_time, 3),
                "model": "granite_vision",
                "strategy": prompt_strategy
            }
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
                        choices=['zero_shot', 'few_shot', 'chain_of_thought', 'visual_grounding', 'contrastive'], 
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
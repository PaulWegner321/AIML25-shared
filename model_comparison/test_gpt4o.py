import os
import json
import logging
import base64
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io
import time
import openai
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the correct path
dotenv_path = os.path.join('../../backend', '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Loading .env file from: {dotenv_path}")

# Get OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables. Please check your .env file.")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Prompt templates
PROMPT_TEMPLATES = {
    "zero_shot": """You are a professional ASL interpreter. Analyze the hand gesture shown in the image.

It is one of these 29 American Sign Language symbols:
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, space, delete, nothing.

Respond **only** with a JSON object in the following format:
{
  "letter": "your_single_choice_from_above",
  "confidence": number_from_0_to_100,
  "feedback": "brief explanation of visible hand shape"
}

Do not infer based on context or imagination. Only use the visible hand shape.""",

    "few_shot": """You are a professional ASL interpreter. Analyze the hand gesture shown in the image.

Here are some examples of ASL letters and their characteristics:
- A: Fist with thumb alongside fingers
- B: Flat hand, fingers together, thumb alongside
- C: Curved hand, thumb and fingers forming a C shape
- D: Index finger pointing up, other fingers closed
- E: Fingers slightly curved, thumb alongside

It is one of these 29 American Sign Language symbols:
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, space, delete, nothing.

Respond **only** with a JSON object in the following format:
{
  "letter": "your_single_choice_from_above",
  "confidence": number_from_0_to_100,
  "feedback": "brief explanation of visible hand shape"
}

Do not infer based on context or imagination. Only use the visible hand shape.""",

    "chain_of_thought": """You are a professional ASL interpreter. Analyze the hand gesture shown in the image.

Let's analyze this step by step:
1. First, observe the overall hand shape and orientation
2. Then, examine the position of each finger and the thumb
3. Compare these observations with known ASL letter formations
4. Consider any unique characteristics or distinguishing features
5. Finally, make your prediction based on the complete analysis

It is one of these 29 American Sign Language symbols:
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, space, delete, nothing.

Respond **only** with a JSON object in the following format:
{
  "letter": "your_single_choice_from_above",
  "confidence": number_from_0_to_100,
  "feedback": "brief explanation of visible hand shape"
}

Do not infer based on context or imagination. Only use the visible hand shape."""
}

def encode_image_base64(image_path):
    """Encode image to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image if needed
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        raise

def get_asl_prediction(image_path: str, prompt_strategy: Literal["zero_shot", "few_shot", "chain_of_thought"] = "zero_shot") -> dict:
    """Get ASL prediction from GPT-4o model."""
    start_time = time.time()
    
    try:
        # Encode image
        image_base64 = encode_image_base64(image_path)
        logging.info(f"Image encoded successfully. Base64 size: {len(image_base64)} characters")
        
        # Get appropriate prompt template
        prompt = PROMPT_TEMPLATES[prompt_strategy]
        
        # Create the message with image and prompt
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional ASL interpreter. Your task is to analyze hand gestures and identify the correct ASL letter."
                },
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
            max_tokens=300,
            temperature=0.0
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Extract the generated text and token usage
        generated_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        logging.info(f"Generated text: {generated_text}")
        
        # Try to parse the JSON response
        try:
            # Find the JSON object in the response
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Add timing and token information to the result
                result["metadata"] = {
                    "response_time_seconds": round(response_time, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                return result
            else:
                logging.warning("No JSON object found in response")
                return {
                    "error": "No JSON found in response",
                    "metadata": {
                        "response_time_seconds": round(response_time, 3),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON from response")
            return {
                "error": "Invalid JSON response",
                "metadata": {
                    "response_time_seconds": round(response_time, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
    except Exception as e:
        response_time = time.time() - start_time
        logging.error(f"Error testing GPT-4o: {e}")
        return {
            "error": str(e),
            "metadata": {
                "response_time_seconds": round(response_time, 3)
            }
        }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test GPT-4o model for ASL recognition')
    parser.add_argument('--image', type=str, default="data/V/V_17_20250428_114126.jpg",
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
    logging.info(f"Testing GPT-4o model with image: {image_path}")
    result = get_asl_prediction(image_path, args.prompt_strategy)
    
    # Print the result
    print("\nGPT-4o result:")
    print(json.dumps(result, indent=2))
    
    # Save the result to a file
    timestamp = int(time.time())
    results_file = f"gpt4o_{args.prompt_strategy}_result_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {results_file}")

if __name__ == "__main__":
    main() 
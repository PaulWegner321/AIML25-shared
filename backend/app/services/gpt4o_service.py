import os
import json
import logging
import base64
import openai
from PIL import Image
import io
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Visual grounding prompt template
VISUAL_GROUNDING_PROMPT = """You are an expert in American Sign Language (ASL) recognition. Carefully analyze the provided image of a hand gesture and determine which ASL letter (A–Z) it represents.

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

Be precise, use visual clues from the image, and avoid guessing without justification."""

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

def get_asl_prediction(image_path: str) -> dict:
    """Get ASL prediction from GPT-4o model using visual grounding prompt."""
    start_time = time.time()
    
    try:
        # Encode image
        image_base64 = encode_image_base64(image_path)
        
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
                            "text": VISUAL_GROUNDING_PROMPT
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
        
        # Extract the generated text
        generated_text = response.choices[0].message.content
        
        # Try to parse the JSON response
        try:
            # Find the JSON object in the response
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                return {
                    "error": "No JSON found in response",
                    "letter": None,
                    "confidence": 0,
                    "feedback": "Failed to parse response"
                }
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON response",
                "letter": None,
                "confidence": 0,
                "feedback": "Failed to parse response"
            }
            
    except Exception as e:
        logging.error(f"Error in GPT-4o prediction: {e}")
        return {
            "error": str(e),
            "letter": None,
            "confidence": 0,
            "feedback": "An error occurred during prediction"
        } 
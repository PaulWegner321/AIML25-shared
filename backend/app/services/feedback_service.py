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

# Feedback prompt template
FEEDBACK_PROMPT = """
You are an expert ASL instructor. Analyze the provided image and give concise, actionable feedback for the user's sign language performance.

- If the sign is correct: Give brief praise and one tip for further improvement (if any).
- If the sign is incorrect: Briefly state what is wrong and give 2-3 practical tips to improve. Do not repeat the correct sign or expected letter, as this is already shown to the user. Focus only on what the user can do to improve their handshape, position, or movement. Keep your response short and practical.

Do not include headings or repeat information already shown on the website. Only provide new, helpful feedback.

Expected Sign: {expected_sign}
Detected Sign: {detected_sign}
Is Correct: {is_correct}
"""

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

def get_sign_feedback(image_path: str, expected_sign: str, detected_sign: str, is_correct: bool) -> dict:
    """Get feedback from GPT-4o model using feedback prompt."""
    start_time = time.time()
    
    try:
        # Encode image
        image_base64 = encode_image_base64(image_path)
        
        # Format the prompt with the provided information
        formatted_prompt = FEEDBACK_PROMPT.format(
            expected_sign=expected_sign,
            detected_sign=detected_sign,
            is_correct=str(is_correct).lower()
        )
        
        # Create the message with image and prompt
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional ASL instructor providing constructive feedback on sign language performance."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_prompt
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
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract the generated text
        feedback = response.choices[0].message.content
        
        return {
            "success": True,
            "feedback": feedback
        }
            
    except Exception as e:
        logging.error(f"Error in GPT-4o feedback: {e}")
        return {
            "success": False,
            "error": str(e),
            "feedback": "An error occurred while generating feedback"
        } 
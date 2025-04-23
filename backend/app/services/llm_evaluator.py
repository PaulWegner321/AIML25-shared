import os
import cv2
import numpy as np
from dotenv import load_dotenv
import base64
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# Load environment variables
load_dotenv()

class LLMEvaluator:
    def __init__(self):
        """Initialize the LLM evaluator with WatsonX credentials."""
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.url = os.getenv("WATSONX_URL")
        
        if not self.api_key or not self.project_id:
            print("Warning: WatsonX API credentials not found in .env file. LLM evaluation will not work.")
            self.client = None
            self.model = None
            return

        # Initialize WatsonX client
        try:
            credentials = Credentials(
                url=self.url,
                api_key=self.api_key
            )

            self.client = APIClient(
                credentials=credentials,
                project_id=self.project_id
            )

            # Set up model parameters
            self.params = TextGenParameters(
                temperature=0.2,            # Slightly higher temperature for more natural responses
                max_new_tokens=500,         # Allow longer responses
                min_new_tokens=50,          # Ensure we get at least some meaningful feedback
                repetition_penalty=1.2,     # Prevent repetitive text
            )

            self.model = ModelInference(
                api_client=self.client,
                model_id="ibm/granite-13b-instruct-v2",
                params=self.params
            )

            print("WatsonX client initialized successfully")
        except Exception as e:
            print(f"Error initializing WatsonX client: {str(e)}")
            self.client = None
            self.model = None

    def evaluate(self, image, detected_sign=None, expected_sign=None, mode='full'):
        """
        Evaluate an image using the Watson LLM.
        
        Args:
            image: OpenCV image
            detected_sign: The sign detected by the CNN model (optional)
            expected_sign: The expected sign (optional)
            mode: 'full' for complete evaluation, 'feedback' for improvement suggestions
            
        Returns:
            dict: Evaluation result with success, feedback, and confidence
        """
        if not self.model:
            return {
                'success': True,
                'feedback': f"This is a dummy feedback for the sign '{detected_sign}'. WatsonX API is not available.",
                'confidence': 0.85
            }
        
        try:
            # Convert OpenCV image to base64 (we might use this later for image analysis)
            _, buffer = cv2.imencode('.jpg', image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            if mode == 'feedback':
                # Create prompt for improvement feedback
                prompt = f"""You are an American Sign Language (ASL) expert. The CNN model detected the sign as '{detected_sign}' when the expected sign was '{expected_sign}'.

Please provide specific feedback on:
1. Why the sign might have been misinterpreted
2. Key differences between '{detected_sign}' and '{expected_sign}'
3. Specific tips to improve the signing of '{expected_sign}'

Keep your response focused on helping the user improve their signing.

Feedback:"""
            else:
                # Create prompt for full evaluation
                prompt = f"""You are an American Sign Language (ASL) expert evaluating a sign image. 

You must respond with a JSON object in exactly this format:
{{
    "letter": "A",  // The letter you believe is being signed (A-Z or 0-9)
    "confidence": 0.85,  // Your confidence score between 0 and 1
    "feedback": "Provide specific, actionable feedback. If the sign is correct but could be improved, mention those small adjustments. If it's perfect, simply say 'Perfect! Your hand position and form are exactly right.' For incorrect signs, explain exactly what needs to be fixed."
}}

When providing feedback:
1. Focus on actionable improvements, not just descriptions
2. For correct signs:
   - If perfect: Keep it simple and congratulatory
   - If minor issues: Suggest small refinements (e.g., "Try keeping your fingers slightly closer together")
3. For incorrect signs:
   - Explain specifically what needs to change
   - Provide clear steps for improvement
   - Compare with the correct form

Keep feedback concise and action-oriented. Avoid just describing what you see.
Ensure your response is ONLY the JSON object with these exact fields.

Response:"""
            
            # Generate response from WatsonX
            response = self.model.generate(prompt)
            
            if mode == 'feedback':
                feedback = response["results"][0]["generated_text"].strip()
                return {
                    'success': True,
                    'feedback': feedback,
                    'confidence': detected_sign == expected_sign
                }
            else:
                # Parse the LLM's JSON response
                try:
                    import json
                    raw_text = response["results"][0]["generated_text"].strip()
                    print(f"Raw LLM response: {raw_text}")  # Debug log
                    
                    # Try to find and extract just the JSON object
                    import re
                    json_match = re.search(r'\{[^}]+\}', raw_text)
                    if json_match:
                        json_text = json_match.group(0)
                        # Fix potential capitalization issues in the feedback field
                        json_text = json_text.replace('"Feedback":', '"feedback":')
                        result = json.loads(json_text)
                        
                        # Validate required fields
                        if not all(key in result for key in ['letter', 'confidence', 'feedback']):
                            raise ValueError("Missing required fields in LLM response")
                            
                        return {
                            'success': True,
                            'letter': result['letter'],
                            'confidence': float(result['confidence']),
                            'feedback': result['feedback']
                        }
                    else:
                        raise ValueError("No JSON object found in LLM response")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing LLM response: {str(e)}")
                    print(f"Raw response was: {raw_text}")
                    
                    # Extract any useful information from the raw text if possible
                    letter_match = re.search(r'"letter":\s*"([^"]+)"', raw_text)
                    confidence_match = re.search(r'"confidence":\s*([\d.]+)', raw_text)
                    feedback_match = re.search(r'"[Ff]eedback":\s*"([^"]+)"', raw_text)
                    
                    return {
                        'success': True,
                        'letter': letter_match.group(1) if letter_match else detected_sign or 'Unknown',
                        'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
                        'feedback': feedback_match.group(1) if feedback_match else "I apologize, but I couldn't properly analyze your sign. Please try again, making sure your hand is clearly visible and well-lit."
                    }
                except Exception as e:
                    print(f"Unexpected error processing LLM response: {str(e)}")
                    return {
                        'success': False,
                        'error': "An error occurred while analyzing your sign. Please try again."
                    }
            
        except Exception as e:
            print(f"Error evaluating with LLM: {str(e)}")
            return {
                'success': False,
                'error': f"Error evaluating with LLM: {str(e)}"
            } 
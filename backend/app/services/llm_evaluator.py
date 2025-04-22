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

    def evaluate(self, image, detected_sign):
        """
        Evaluate an image using the Watson LLM.
        
        Args:
            image: OpenCV image
            detected_sign: The sign detected by the CNN model
            
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
            
            # Create the prompt for the LLM
            prompt = f"""You are an American Sign Language (ASL) expert. I will show you a sign that was detected as the letter '{detected_sign}'. 
            
Please provide a brief but detailed feedback focusing on:
1. Accuracy of the detected sign
2. Common confusions with other letters
3. Key hand positions for this sign
4. Quick tips for improvement

Keep your response concise but informative. Focus on practical advice.

Feedback:"""
            
            # Generate response from WatsonX
            response = self.model.generate(prompt)
            feedback = response["results"][0]["generated_text"].strip()
            
            # Calculate a confidence score based on the length and specificity of the feedback
            # This is a simple heuristic - you might want to adjust this
            confidence = min(0.95, 0.5 + (len(feedback.split()) / 100))
            
            return {
                'success': True,
                'feedback': feedback,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Error evaluating with LLM: {str(e)}")
            return {
                'success': False,
                'error': f"Error evaluating with LLM: {str(e)}"
            } 
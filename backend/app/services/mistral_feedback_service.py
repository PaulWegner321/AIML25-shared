from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get IBM Watson credentials
WX_API_KEY = os.getenv("WATSONX_API_KEY")
WX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WX_URL = os.getenv("WATSONX_URL")

if not WX_API_KEY:
    raise ValueError("WATSONX_API_KEY not found in environment variables")
if not WX_PROJECT_ID:
    raise ValueError("WATSONX_PROJECT_ID not found in environment variables")
if not WX_URL:
    raise ValueError("WATSONX_URL not found in environment variables")

class MistralFeedbackService:
    def __init__(self):
        # Setup Watson credentials
        credentials = Credentials(
            url=WX_URL,
            api_key=WX_API_KEY
        )
        
        self.client = APIClient(
            credentials=credentials,
            project_id=WX_PROJECT_ID
        )
        
        # Setup model parameters
        self.params = TextGenParameters(
            temperature=0.05,
            max_new_tokens=300
        )
        
        # Initialize model
        self.model = ModelInference(
            api_client=self.client,
            params=self.params,
            model_id="mistralai/mistral-large"
        )

    def generate_feedback(self, image_description: str, expected_sign: str, detected_sign: str) -> Dict[str, List[str]]:
        """Generate structured feedback based on GPT-4V's image description."""
        is_correct = expected_sign.upper() == detected_sign.upper()
        
        prompt = f"""You are an ASL instructor. Based on the detailed description of a hand gesture below, provide structured feedback for the user.

Image Description: {image_description}

Expected Sign: {expected_sign}
Detected Sign: {detected_sign}
Is Correct: {is_correct}

Analyze the description and provide:
1. A clear description of what the user did
2. Specific steps to improve their sign (at least 3 numbered steps)
3. Helpful tips for better execution (at least 2 bullet points)

Format your response exactly like this:
DESCRIPTION: [One clear sentence about what you see]

STEPS:
1. [First improvement step]
2. [Second improvement step]
3. [Third improvement step]

TIPS:
- [First practical tip]
- [Second practical tip]
"""

        try:
            response = self.model.generate(prompt=prompt)
            generated_text = response['results'][0]['generated_text']
            
            # Process the response
            # Split by section headers
            sections = []
            if "DESCRIPTION:" in generated_text:
                sections = generated_text.split("DESCRIPTION:")[1].split("STEPS:")
                description_section = sections[0].strip()
                
                if len(sections) > 1 and "TIPS:" in sections[1]:
                    steps_and_tips = sections[1].split("TIPS:")
                    steps_section = steps_and_tips[0].strip()
                    tips_section = steps_and_tips[1].strip() if len(steps_and_tips) > 1 else ""
                else:
                    steps_section = sections[1].strip() if len(sections) > 1 else ""
                    tips_section = ""
            else:
                # Fallback if format wasn't followed
                description_section = generated_text.strip()
                steps_section = ""
                tips_section = ""
            
            # Process steps - looking for numbered items
            steps = []
            for line in steps_section.split('\n'):
                line = line.strip()
                if line and line[0].isdigit() and '.' in line:
                    step_text = line.split('.', 1)[1].strip()
                    steps.append(step_text)
            
            # Process tips - looking for bullet points
            tips = []
            for line in tips_section.split('\n'):
                line = line.strip()
                if line and line.startswith('-'):
                    tip_text = line[1:].strip()
                    tips.append(tip_text)
            
            # Default responses if nothing was generated
            if not description_section:
                if is_correct:
                    description_section = f"You correctly signed the letter '{expected_sign}'."
                else:
                    description_section = f"The model detected that you signed '{detected_sign}', but the expected sign was '{expected_sign}'."
            
            # Default steps if none were generated
            if not steps:
                if is_correct:
                    steps = ["Continue practicing to maintain consistency", 
                            "Try signing at different speeds to build fluency", 
                            "Practice transitioning between this sign and others"]
                else:
                    steps = [f"Study the correct hand position for '{expected_sign}'", 
                            "Practice the correct position slowly", 
                            "Compare your sign with reference images"]
            
            # Default tips if none were generated
            if not tips:
                if is_correct:
                    tips = ["Keep your hand in good lighting for better visibility",
                           "Maintain proper hand positioning for clarity"]
                else:
                    tips = ["Ensure your hand is fully visible to the camera",
                           "Pay attention to finger positioning and hand orientation"]
            
            return {
                "description": description_section,
                "steps": steps,
                "tips": tips
            }
            
        except Exception as e:
            print(f"Error generating feedback with Mistral: {str(e)}")
            return {
                "description": f"The model detected that you signed '{detected_sign}', but the expected sign was '{expected_sign}'.",
                "steps": ["Study the correct hand position", "Practice the correct position slowly", "Compare your sign with reference images"],
                "tips": ["Ensure your hand is fully visible to the camera", "Pay attention to finger positioning and hand orientation"]
            }

# Create a singleton instance
mistral_feedback_service = MistralFeedbackService() 
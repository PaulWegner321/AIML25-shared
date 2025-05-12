from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters
from typing import Dict, List
import os
from dotenv import load_dotenv
import re

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

class LLaMaFeedbackService:
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
            model_id="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        
        # ASL Sign Knowledge Base
        self.Sign_knowledge = {
            "A": "Thumb: Curled alongside the side of the index finger, resting against it. Index: Bent downward into the palm, creating a firm curve. Middle: Bent downward in line with the index. Ring: Bent downward. Pinky: Bent downward. Palm Orientation: Facing forward (away from your body). Wrist/Forearm: Neutral position; elbow bent naturally. Movement: None. Note: Represents the shape of a capital 'A'.",
            "B": "Thumb: Folded tightly across the center of the palm, held flat. Index: Extended straight up and held close to the middle finger. Middle: Extended straight up next to the index finger. Ring: Extended straight up next to the middle finger. Pinky: Extended straight up, close to the ring finger. Palm Orientation: Facing forward. Wrist/Forearm: Upright, fingers vertical. Movement: None. Note: Resembles the vertical line of the letter 'B'.",
            "C": "Thumb: Curved naturally to oppose the fingers and help form a half-circle. Index: Curved downward and to the side to help form the top of the 'C'. Middle: Curved to follow the shape created by the index. Ring: Curved in alignment with the rest to form the side of the 'C'. Pinky: Curved slightly to close the 'C' shape. Palm Orientation: Slightly angled outward (to mimic letter curvature). Wrist/Forearm: Slight bend at wrist to angle the 'C'. Movement: None. Note: Entire hand forms a visible capital letter 'C'.",
            "D": "Thumb: Pads rest against the tips of the middle, ring, and pinky fingers. Index: Fully extended upward and isolated from other fingers. Middle: Curved downward to meet the thumb. Ring: Curved downward to meet the thumb. Pinky: Curved downward to meet the thumb. Palm Orientation: Facing forward. Wrist/Forearm: Neutral vertical. Movement: None. Note: Mimics the shape of a capital 'D' with the index as the upright line.",
            "E": "Thumb: Pressed against the palm and touching curled fingers from below. Index: Curled downward toward the palm to meet the thumb. Middle: Curled downward toward the palm. Ring: Curled downward toward the palm. Pinky: Curled downward toward the palm. Palm Orientation: Facing forward. Wrist: Neutral or slightly rotated outward. Movement: None. Note: Shape resembles the loop and middle bar of the letter 'E'.",
            "F": "Thumb: Touches tip of the index finger to form a closed circle. Index: Touches the thumb to complete the circle. Middle: Extended straight up and relaxed, slightly separated. Ring: Extended straight up and relaxed, slightly separated. Pinky: Extended straight up and relaxed, slightly separated. Palm Orientation: Facing forward. Wrist: Neutral to slightly outward. Movement: None. Note: The circle represents the opening in the letter 'F'.",
            "G": "Thumb: Extended sideways, parallel to index. Index: Extended sideways, forming a flat, straight line with thumb. Middle: Folded inward against the palm. Ring: Folded inward against the palm. Pinky: Folded inward against the palm. Palm Orientation: Inward (side of hand faces viewer). Wrist: Horizontal; hand like a gun shape. Movement: None. Note: Emulates the lower stroke of a 'G'.",
            "H": "Thumb: Tucked over curled ring and pinky. Index: Extended to the side. Middle: Extended to the side, beside index. Ring: Curled tightly in palm. Pinky: Curled tightly in palm. Palm Orientation: Facing down or slightly out. Wrist: Flat or slightly turned. Movement: None. Note: Represents two parallel lines, like a sideways 'H'.",
            "I": "Thumb: Folded across or tucked alongside curled fingers. Index: Curled into the palm. Middle: Curled into the palm. Ring: Curled into the palm. Pinky: Extended straight up. Palm Orientation: Facing forward. Wrist: Neutral vertical. Movement: None. Note: Pinky alone resembles a lowercase 'i'.",
            "J": "Thumb: Folded against curled fingers. Index: Curled into the palm. Middle: Curled into the palm. Ring: Curled into the palm. Pinky: Extended and used to trace a 'J' in the air. Palm Orientation: Starts forward, rotates slightly. Movement: Trace 'J' downward, left, then up. Note: Motion is essential to identify this as 'J'.",
            "K": "Thumb: Between index and middle fingers, touching base of middle. Index: Extended diagonally upward. Middle: Extended diagonally upward, apart from index. Ring: Curled into palm. Pinky: Curled into palm. Palm Orientation: Facing out or slightly angled. Wrist: Upright or angled. Movement: None. Note: Mimics the open shape of the letter 'K'.",
            "L": "Thumb: Extended horizontally. Index: Extended vertically. Middle: Curled into palm. Ring: Curled into palm. Pinky: Curled into palm. Palm Orientation: Facing forward. Wrist: Upright. Movement: None. Note: Clearly forms a visual 'L'.",
            "M": "Thumb: Tucked under index, middle, and ring fingers. Index: Folded over the thumb. Middle: Folded over the thumb. Ring: Folded over the thumb. Pinky: Curled beside ring or relaxed. Palm Orientation: Facing out. Wrist: Neutral. Movement: None. Note: Three fingers over thumb = 3 strokes = 'M'.",
            "N": "Thumb: Tucked under index and middle fingers. Index: Folded over thumb. Middle: Folded over thumb. Ring: Folded into palm. Pinky: Folded into palm. Palm Orientation: Facing out. Movement: None. Note: Two fingers over thumb = 2 strokes = 'N'.",
            "O": "Thumb: Curved inward to meet fingertips. Index: Curved downward to meet thumb. Middle: Curved downward to meet thumb. Ring: Curved downward to meet thumb. Pinky: Curved downward to meet thumb. Palm Orientation: Facing forward. Wrist: Upright or slightly turned. Movement: None. Note: Clear circular 'O' shape.",
            "P": "Thumb: Between and touching middle finger. Index: Extended downward and slightly angled. Middle: Extended and separated from index. Ring: Folded into the palm. Pinky: Folded into the palm. Palm Orientation: Tilted downward. Wrist: Bent downward. Movement: None. Note: Downward angle distinguishes from K.",
            "Q": "Thumb: Parallel to index. Index: Points downward. Middle: Curled into palm. Ring: Curled into palm. Pinky: Curled into palm. Palm Orientation: Downward. Wrist: Bent downward. Movement: None. Note: Like G but rotated to point down.",
            "R": "Thumb: Resting against curled fingers. Index: Crossed over middle finger tightly. Middle: Crossed under index. Ring: Folded into palm. Pinky: Folded into palm. Palm Orientation: Facing forward. Movement: None. Note: Finger crossing symbolizes 'R'.",
            "S": "Thumb: Crossed tightly over the front of curled fingers. Index: Curled into a fist. Middle: Curled into a fist. Ring: Curled into a fist. Pinky: Curled into a fist. Palm Orientation: Facing forward. Wrist: Upright. Movement: None. Note: Fist shape resembles bold 'S'.",
            "T": "Thumb: Inserted between index and middle fingers. Index: Curled downward over the thumb. Middle: Curled downward over the thumb. Ring: Curled into the palm. Pinky: Curled into the palm. Palm Orientation: Facing forward. Movement: None. Note: Thumb poking between fingers resembles old-style 'T'.",
            "U": "Thumb: Folded against palm. Index: Extended straight upward. Middle: Extended straight upward, held together with index. Ring: Folded into the palm. Pinky: Folded into the palm. Palm Orientation: Facing forward. Movement: None. Note: Two fingers = 2 strokes of 'U'.",
            "V": "Thumb: Folded in or at side. Index: Extended upward. Middle: Extended upward, spread apart from index. Ring: Folded into palm. Pinky: Folded into palm. Palm Orientation: Facing forward. Movement: None. Note: Clear 'V' shape.",
            "W": "Thumb: Tucked or relaxed. Index: Extended upward. Middle: Extended upward. Ring: Extended upward, spread slightly. Pinky: Folded into the palm. Palm Orientation: Facing forward. Movement: None. Note: Three fingers = 'W'.",
            "X": "Thumb: Resting at side or across curled fingers. Index: Bent to form a hook. Middle: Folded into palm. Ring: Folded into palm. Pinky: Folded into palm. Palm Orientation: Facing forward. Movement: None. Note: Hooked finger mimics 'X'.",
            "Y": "Thumb: Extended sideways. Index: Folded into palm. Middle: Folded into palm. Ring: Folded into palm. Pinky: Extended in opposite direction from thumb. Palm Orientation: Facing forward. Movement: None. Note: Thumb and pinky spread = 'Y' shape.",
            "Z": "Thumb: Folded against curled fingers or at the side. Index: Extended and used to draw a 'Z' in the air. Middle: Curled into palm. Ring: Curled into palm. Pinky: Curled into palm. Palm Orientation: Faces slightly forward, rotating with the movement. Movement: Trace 'Z' in air from top-left to bottom-right."
        }

    def generate_feedback(self, image_description: str, expected_sign: str, detected_sign: str) -> Dict[str, List[str]]:
        """Generate structured feedback based on GPT-4V's image description."""
        is_correct = expected_sign.upper() == detected_sign.upper()
        
        # Get the correct sign details
        correct_sign_details = self.Sign_knowledge.get(expected_sign.upper(), "Sign details not available.")
        
        prompt = f"""You are an expert ASL instructor providing feedback directly to a student. Based on the detailed description of their hand gesture below, provide clear, concise, and actionable feedback.

Image Description: {image_description}

Expected Sign: {expected_sign}
Detected Sign: {detected_sign}
Is Correct: {is_correct}

Correct Sign Reference:
{correct_sign_details}

Your feedback guidelines:
- Speak directly to the student in a friendly, encouraging tone
- Do not refer to "the user" - use "you" instead
- The correct hand position can be reviewed in the "Lookup" part of the web application, where the user can access text-based definitions
- Be concise but specific
- If the sign is correct: Give brief praise and 1-2 tips for further improvement
- If the sign is incorrect: Briefly explain what needs adjustment, then provide specific tips to improve

Format your response exactly like this:
DESCRIPTION: [One clear, conversational sentence about what you see]

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
            
            # Clean placeholder text patterns from the description
            placeholder_patterns = [
                r'\[Optional:.*?\]',
                r'\[Optional.*?\]',
                r'\[Write.*?\]',
                r'\[First.*?\]',
                r'\[Second.*?\]',
                r'\[Third.*?\]',
                r'\[In the case of.*?\]',
                r'\[.*?provide.*?\]',
                r'\[.*?improvement.*?\]',
            ]
            
            for pattern in placeholder_patterns:
                description_section = re.sub(pattern, '', description_section)
                steps_section = re.sub(pattern, '', steps_section)
                tips_section = re.sub(pattern, '', tips_section)
            
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
                    description_section = f"Great job! You correctly signed the letter '{expected_sign}' with excellent form and precision. Your hand position shows great attention to detail."
                else:
                    description_section = f"I noticed that you signed what looks like '{detected_sign}', but we were practicing the letter '{expected_sign}'."
            
            # Default steps if none were generated
            if not steps:
                if is_correct:
                    steps = ["Continue practicing to maintain your excellent form", 
                            "Try signing at different speeds to build your fluency", 
                            "Practice transitioning between this sign and others to build muscle memory"]
                else:
                    steps = [f"Review the correct hand position for '{expected_sign}' shown in the guide", 
                            "Practice the correct position slowly in front of a mirror", 
                            "Compare your sign with the reference images to spot differences"]
            
            # Default tips if none were generated
            if not tips:
                if is_correct:
                    tips = ["Keep your hand in good lighting for better visibility when practicing with the app",
                           "Maintain consistent hand positioning for clarity in real conversations"]
                else:
                    tips = ["Ensure your entire hand is visible to the camera when practicing",
                           "Pay close attention to specific finger positions described in the reference"]
            
            return {
                "description": description_section,
                "steps": steps,
                "tips": tips
            }
            
        except Exception as e:
            print(f"Error generating feedback with LLaMa Scout: {str(e)}")
            return {
                "description": f"I noticed you signed what looks like '{detected_sign}', but we were practicing the letter '{expected_sign}'. Let me help you improve!",
                "steps": ["Review the correct hand position on the lookup page", "Practice forming the sign slowly and deliberately", "Compare your sign with the explanations to spot differences"],
                "tips": ["Make sure your entire hand is clearly visible to the camera", "Pay special attention to the finger positions described in the guide"]
            }

# Create a singleton instance
llama_feedback_service = LLaMaFeedbackService() 
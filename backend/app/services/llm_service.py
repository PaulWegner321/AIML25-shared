from fastapi import HTTPException
from pydantic import BaseModel
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import uuid
import time

# Load environment variables
load_dotenv()

# Get IBM Watson credentials
WX_API_KEY = os.getenv("WATSONX_API_KEY")
WX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# Flag to track if Watson is available
watson_available = bool(WX_API_KEY and WX_PROJECT_ID and WX_URL)

if not watson_available:
    print("WARNING: IBM WatsonX credentials not found. Using fallback mode.")
else:
    print("IBM WatsonX credentials found. Using WatsonX for sign descriptions.")

# ASL Sign Knowledge Base
Sign_knowledge = {
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

class SignRequest(BaseModel):
    sign_name: str

class SignResponse(BaseModel):
    word: str
    description: str
    steps: List[str]
    tips: List[str]

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    sign_name: str
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

def create_prompt(sign_name: str) -> str:
    sign_details = Sign_knowledge.get(sign_name.upper(), "Sign not found.")

    return (
        f"You are an American Sign Language (ASL) teacher. Your task is to explain how to perform ASL signs.\n\n"
        f"Here is an example of how to format your response for the word 'all':\n\n"
        f"The sign for 'all' is made by moving your dominant hand in a circular motion over your non-dominant hand, ending with both palms facing each other.\n\n"
        f"1. Hold your non-dominant hand in front of you, palm facing in.\n"
        f"2. Place your dominant hand behind your non-dominant hand, palm facing out.\n"
        f"3. Move your dominant hand in a circular motion over the back of your non-dominant hand.\n"
        f"4. End with both hands open, palms facing each other.\n\n"
        f"- Keep your movements smooth and controlled\n"
        f"- Make sure both hands are clearly visible\n\n"
        f"Now, please explain how to perform the ASL sign for the letter '{sign_name}' in the same format:\n\n"
        f"First write a brief description of the overall sign.\n\n"
        f"Then list the steps, with each step on a new line starting with a number:\n"
        f"1. First step\n"
        f"2. Second step\n"
        f"etc.\n\n"
        f"Finally, provide 1-2 specific tips for performing this sign correctly, each on a new line starting with a bullet point (-).\n\n"
        f"Do not use any markdown formatting (no **, *, or other special characters).\n\n"
        f"Here is the reference information for the letter '{sign_name}':\n"
        f"{sign_details}\n\n"
        f"Keep your response clear and beginner-friendly. If you cannot generate a description, output: 'Sorry, I cannot help with this sign'\n\n"
    )

def process_llm_response(text: str) -> Dict:
    """Process the LLM response into structured format."""
    # Split the response into sections based on empty lines
    sections = [s.strip() for s in text.split('\n\n') if s.strip()]
    
    # Initialize with defaults
    description = ""
    steps = []
    tips = []
    
    # Process each section
    for section in sections:
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        
        # Skip section headers (Description, Steps, Tips)
        if len(lines) == 1 and (lines[0].lower() in ['description', 'steps', 'tips']):
            continue
            
        # If any line starts with a number, this is the steps section
        if any(line[0].isdigit() for line in lines):
            for line in lines:
                if line[0].isdigit():
                    # Remove any duplicate numbering (e.g., "1. 1." or "1.1.")
                    # First, remove the initial number and any dots/spaces
                    cleaned_step = line.lstrip('0123456789. ')
                    # Then remove any remaining numbers at the start
                    cleaned_step = cleaned_step.lstrip('0123456789. ')
                    steps.append(f"{len(steps) + 1}. {cleaned_step}")
        # If any line starts with a dash or bullet point, these are tips
        elif any(line.startswith('-') or line.startswith('•') for line in lines):
            tips.extend(line.lstrip('-•').strip() for line in lines if line.startswith('-') or line.startswith('•'))
        # Otherwise, this is part of the description
        else:
            # Remove any markdown formatting
            clean_text = ' '.join(lines).replace('**', '').replace('*', '')
            if description:
                description += "\n" + clean_text
            else:
                description = clean_text
    
    # If no tips were found, add a general tip
    if not tips:
        tips = ["Keep your hand steady and well-lit for clear signing"]
    
    return {
        "description": description,
        "steps": steps,
        "tips": tips
    }

class LLMService:
    def __init__(self):
        # Flag to track if Watson is available
        self.watson_available = watson_available
        # Store chat sessions
        self.chat_sessions = {}
        # Session expiry time (2 hours in seconds)
        self.session_expiry = 7200
        # Clean expired sessions periodically
        self.last_cleanup = time.time()
        
        if self.watson_available:
            try:
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
                print("Successfully initialized WatsonX connection")
            except Exception as e:
                print(f"Error initializing WatsonX: {e}")
                self.watson_available = False
        else:
            print("Using static sign descriptions (fallback mode)")

    def _clean_expired_sessions(self):
        """Clean expired chat sessions to prevent memory leaks"""
        current_time = time.time()
        # Only clean every 10 minutes
        if current_time - self.last_cleanup < 600:
            return
            
        expired_sessions = []
        for session_id, session_data in self.chat_sessions.items():
            if current_time - session_data["last_access"] > self.session_expiry:
                expired_sessions.append(session_id)
                
        for session_id in expired_sessions:
            del self.chat_sessions[session_id]
            
        self.last_cleanup = current_time
        print(f"Cleaned {len(expired_sessions)} expired chat sessions. Active sessions: {len(self.chat_sessions)}")

    async def lookup_sign(self, request: SignRequest) -> SignResponse:
        sign_name = request.sign_name.upper()
        
        try:
            # Check if we have a static definition for this sign
            sign_details = Sign_knowledge.get(sign_name)
            
            if not sign_details:
                return SignResponse(
                    word=sign_name,
                    description=f"Description for sign '{sign_name}' not found.",
                    steps=["1. Check that you've entered a valid ASL letter sign (A-Z)."],
                    tips=["Try looking up a different letter."]
                )
            
            if self.watson_available:
                try:
                    # Generate response using WatsonX
                    prompt = create_prompt(sign_name)
                    response = self.model.generate(prompt=prompt)
                    generated_text = response['results'][0]['generated_text']
                    
                    # Process the response into structured format
                    processed_response = process_llm_response(generated_text)
                    
                    return SignResponse(
                        word=sign_name,
                        description=processed_response["description"],
                        steps=processed_response["steps"],
                        tips=processed_response["tips"]
                    )
                except Exception as e:
                    print(f"WatsonX error: {e}. Falling back to static description.")
                    # Fall back to static description if WatsonX fails
            
            # Fallback to static descriptions
            return self._create_static_response(sign_name, sign_details)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    def _create_static_response(self, sign_name: str, sign_details: str) -> SignResponse:
        """Create a structured response from static sign description."""
        # Parse the static knowledge into usable format
        parts = sign_details.split('. ')
        
        # Extract basic components for a simple structured response
        description = f"The sign for letter '{sign_name}' in American Sign Language."
        
        # Create simple step-by-step instructions from the knowledge base
        steps = [
            f"1. Position your hand as follows: {parts[0]}",
            f"2. Make sure your {parts[1] if len(parts) > 1 else 'hand is in the correct position'}",
            f"3. Keep your {parts[2] if len(parts) > 2 else 'fingers properly aligned'}"
        ]
        
        # Standard tips
        tips = [
            f"Practice in front of a mirror to check your form",
            f"The sign for '{sign_name}' should be clear and deliberate"
        ]
        
        return SignResponse(
            word=sign_name,
            description=description,
            steps=steps,
            tips=tips
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle chat interactions about a specific ASL sign"""
        self._clean_expired_sessions()
        
        sign_name = request.sign_name.upper()
        user_message = request.message
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create chat history
        if session_id not in self.chat_sessions:
            # Get sign details
            sign_details = Sign_knowledge.get(sign_name, "Unknown sign")
            
            # Format initial context with sign details 
            system_context = (
                f"You are an ASL tutor helping with the sign for the letter '{sign_name}'. "
                f"Reference information: {sign_details} "
                f"Be friendly, helpful and concise in your responses. Focus on teaching the correct ASL finger positions."
            )
            
            # Initialize session with system message and timestamp
            self.chat_sessions[session_id] = {
                "messages": [{"role": "system", "content": system_context}],
                "sign_name": sign_name, 
                "last_access": time.time()
            }
            
            # If this is a new session, provide initial greeting
            if not request.message or request.message.strip() == "":
                greeting = (
                    f"Hi! I'm your ASL tutor for the letter '{sign_name}'. "
                    f"You can ask me questions about how to form this sign, common mistakes to avoid, "
                    f"or request more detailed explanations for any aspect of the sign."
                )
                self.chat_sessions[session_id]["messages"].append({"role": "assistant", "content": greeting})
                return ChatResponse(response=greeting, session_id=session_id)
        
        # Update session access time
        self.chat_sessions[session_id]["last_access"] = time.time()
        
        # Add user message to history
        self.chat_sessions[session_id]["messages"].append({"role": "user", "content": user_message})
        
        # Get response using WatsonX or fallback
        if self.watson_available:
            try:
                # Format chat history for prompt
                chat_history = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in self.chat_sessions[session_id]["messages"]
                ])
                
                prompt = f"""
                You are an interactive ASL tutor specialized in teaching American Sign Language.
                
                Current sign being discussed: {sign_name}
                
                Previous conversation:
                {chat_history}
                
                Respond to the user's latest question with helpful, clear information about ASL.
                Keep your response concise (15-40 words) and informative, focusing specifically on what was asked.
                Do not use phrases like "I would say" or "Your response should be" - just provide the direct answer.
                Do not start with "assistant:" or similar prefixes.
                """
                
                response = self.model.generate(prompt=prompt)
                assistant_response = response['results'][0]['generated_text'].strip()
                
                # Clean up response if needed
                if assistant_response.startswith("assistant:"):
                    assistant_response = assistant_response[10:].strip()
                
                # Remove any instructions or meta text that might be included
                common_prefixes = [
                    "your response should be",
                    "the answer is",
                    "i would say",
                    "in response to your question",
                    "to answer your question",
                ]
                
                for prefix in common_prefixes:
                    if assistant_response.lower().startswith(prefix):
                        assistant_response = assistant_response[len(prefix):].strip()
                
                # Remove period at end if there's a second sentence starting point
                parts = assistant_response.split('. ', 1)
                if len(parts) > 1 and len(parts[0]) < 50:
                    assistant_response = parts[0].strip()

            except Exception as e:
                print(f"WatsonX chat error: {e}")
                assistant_response = self._generate_simple_response(user_message, sign_name)
        else:
            # Simple fallback for non-Watson mode
            assistant_response = self._generate_simple_response(user_message, sign_name)
        
        # Store assistant's response
        self.chat_sessions[session_id]["messages"].append({"role": "assistant", "content": assistant_response})
        
        return ChatResponse(
            response=assistant_response,
            session_id=session_id
        )
    
    def _generate_simple_response(self, message, sign_name):
        """Generate a simple rule-based response for fallback mode"""
        message_lower = message.lower()
        sign_details = Sign_knowledge.get(sign_name, "")
        parts = sign_details.split('. ', 2)  # Split into up to 3 parts
        
        # Check for common question patterns
        if any(word in message_lower for word in ["how", "form", "make", "do"]):
            return f"{parts[0]}."
            
        elif any(word in message_lower for word in ["difficult", "hard", "challenge"]):
            return f"The sign for '{sign_name}' is straightforward with practice. Focus on proper finger positioning and hand orientation."
            
        elif any(phrase in message_lower for phrase in ["common mistake", "error", "wrong", "incorrect"]):
            return f"Common mistakes include incorrect finger positioning or hand orientation. Make sure your {parts[0].lower()}."
            
        elif any(word in message_lower for word in ["compare", "difference", "similar", "versus", "vs"]):
            return f"The sign '{sign_name}' is distinct in its finger positioning. The key feature is {parts[0].lower()}."
            
        elif any(word in message_lower for word in ["practice", "exercise", "improve", "better"]):
            return f"Practice in front of a mirror. Focus on {parts[0].lower()}."
            
        elif any(word in message_lower for word in ["thank", "thanks", "appreciate"]):
            return f"You're welcome! Happy to help with the '{sign_name}' sign."
            
        else:
            return f"For the '{sign_name}' sign: {parts[0]}. What else would you like to know?"

# Create a singleton instance
llm_service = LLMService() 
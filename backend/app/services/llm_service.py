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
import logging
import re

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
        f"You are an American Sign Language (ASL) teacher.\n\n"
        f"Please clearly explain how to perform the ASL sign on a beginner level for the letter '{sign_name}'. "
        f"Use simple language and full sentences. Do not assume any prior knowledge about ASL.\n\n"
        f"Here is relevant information for the letter '{sign_name}':\n"
        f"{sign_details}\n\n"
        f"Refer to the following examples for how to structure your response:\n"
        f"1. sign: 'Hello' - explanation: Begin with the side of your index finger against your forehead and then move your hand up and away from your head.\n"
        f"2. sign: 'Customer' - explanation: Begin with your hands on each side of the top of your chest with your palms oriented toward each other and your thumbs touching your chest. Move your hands off your chest and bring them down and press them against your midsection.\n"
        f"3. sign: 'Become' - explanation: Begin with both palms oriented towards each other with your hands perpendicular to each other. Then, rotate your wrists until your hands are perpendicular to each other in the opposite direction.\n"
        f"4. sign: 'Certain' - explanation: Begin with your index finger touching your mouth and pointing up. Then, bring it forward and down until your index finger is facing forwards.\n"
        f"5. sign: 'All' - explanation: Begin with both hands in front of you. Your non-dominant hand should be closer to you and be oriented towards yourself. Your dominant hand should be oriented away from yourself. Rotate your dominant hand so that its palm is oriented toward yourself and then rest the back of your dominant hand against the palm of your non-dominant hand.\n\n"
        f"Your response must follow this exact format:\n\n"
        f"[Write a brief 1-2 sentence description of the overall sign, mentioning what it looks like or what distinguishing features it has]\n\n"
        f"Steps\n"
        f"1. [First step]\n"
        f"2. [Second step]\n"
        f"3. [Third step]\n"
        f"... and so on\n\n"
        f"Tips\n"
        f"- [First tip for performing this sign correctly]\n"
        f"- [Second tip if applicable]\n\n"
        f"Do not use any markdown formatting (no **, *, or other special characters).\n\n"
        f"If you cant generate a description based on the relevant information, output: 'Sorry, I cant help with this sign'\n\n"
        f"Only output the formatted response once. Do not include any other text. If appropriate, use fewer tokens than available.\n"
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
                
                # Setup parameters for sign explanation model (Llama)
                self.sign_params = TextGenParameters(
                    temperature=0.05,
                    max_new_tokens=300
                )
                
                # Setup parameters for chat model (Mistral)
                self.chat_params = TextGenParameters(
                    temperature=0.3,
                    max_new_tokens=500
                )
                
                # Initialize Llama model for sign explanations
                self.sign_model = ModelInference(
                    api_client=self.client,
                    params=self.sign_params,
                    model_id="meta-llama/llama-4-scout-17b-16e-instruct"
                )
                
                # Initialize Mistral model for chat interactions
                self.chat_model = ModelInference(
                    api_client=self.client,
                    params=self.chat_params,
                    model_id="mistralai/mistral-large"
                )
                
                print("Successfully initialized WatsonX connection with Llama for signs and Mistral for chat")
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
                    print(f"LOOKUP - Sending prompt to WatsonX for sign '{sign_name}'")
                    print(f"LOOKUP - Using parameters: temperature={self.sign_params.temperature}, max_new_tokens={self.sign_params.max_new_tokens}")
                    
                    response = self.sign_model.generate(prompt=prompt)
                    generated_text = response['results'][0]['generated_text']
                    
                    print(f"LOOKUP - Raw generated text length: {len(generated_text)}")
                    print(f"LOOKUP - First 100 chars: {generated_text[:100]}...")
                    
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
        
        print(f"Chat request received - sign: {sign_name}, session: {session_id}, message: '{user_message}'")
        
        # Get or create chat history
        if session_id not in self.chat_sessions:
            # Get sign details
            sign_details = Sign_knowledge.get(sign_name, "Unknown sign")
            
            # Format initial context with sign details 
            system_context = (
                f"You are an ASL tutor helping with the sign for the letter '{sign_name}'. "
                f"Reference information: {sign_details} "
                f"Be friendly, helpful and concise in your responses. Focus on teaching the correct ASL finger positions. "
                f"If asked about topics unrelated to ASL or the letter '{sign_name}', politely explain that you're an ASL tutor and can only help with ASL-related questions."
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
                print(f"New session greeting: '{greeting}'")
                return ChatResponse(response=greeting, session_id=session_id)
        
        # Update session access time
        self.chat_sessions[session_id]["last_access"] = time.time()
        
        # Add user message to history
        self.chat_sessions[session_id]["messages"].append({"role": "user", "content": user_message})
        
        # Get response using WatsonX or fallback
        assistant_response = ""  # Initialize with empty string
        
        if self.watson_available:
            try:
                # Format chat history for prompt
                chat_history = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in self.chat_sessions[session_id]["messages"]
                ])
                
                # Create a simpler prompt format for Mistral
                prompt = (
                    f"You are an ASL tutor specialized in teaching American Sign Language for the letter '{sign_name}'.\n\n"
                    f"Here's what you know about signing letter '{sign_name}':\n{Sign_knowledge.get(sign_name, 'Unknown sign')}\n\n"
                    f"Previous conversation:\n{chat_history}\n\n"
                    f"Please respond to the user's latest question with helpful information about ASL sign '{sign_name}'. "
                    f"Keep responses concise (20-40 words) and direct. If the user asks about topics unrelated to ASL, politely "
                    f"redirect them to ASL topics for letter '{sign_name}'."
                )
                
                print(f"Sending prompt to WatsonX")
                print(f"FULL PROMPT SENT TO WATSONX:\n{'-'*50}\n{prompt}\n{'-'*50}")
                print(f"Starting WatsonX API call with params: temperature={self.chat_params.temperature}, max_new_tokens={self.chat_params.max_new_tokens}, model_id={self.chat_model.model_id}")
                
                # Set detailed logging for the request/response
                logging.getLogger('ibm_watsonx_ai').setLevel(logging.DEBUG)
                logging.getLogger('httpx').setLevel(logging.DEBUG)
                
                response = self.chat_model.generate(prompt=prompt)
                print(f"Raw WatsonX API response:\n{'-'*50}\n{response}\n{'-'*50}")
                
                assistant_response = response['results'][0]['generated_text'].strip()
                print(f"Raw WatsonX response text: '{assistant_response}'")
                print(f"Response text length: {len(assistant_response)}")
                
                # If response is empty, try with different parameters
                if not assistant_response or assistant_response.strip() == "":
                    print("Empty response received, trying with higher temperature...")
                    # Create a one-time use model with higher temperature
                    retry_params = TextGenParameters(
                        temperature=0.7,
                        max_new_tokens=500
                    )
                    retry_model = ModelInference(
                        api_client=self.client,
                        params=retry_params,
                        model_id=self.chat_model.model_id
                    )
                    print("Retrying with temperature=0.7, max_new_tokens=500")
                    try:
                        retry_response = retry_model.generate(prompt=prompt)
                        print(f"Retry response:\n{'-'*50}\n{retry_response}\n{'-'*50}")
                        assistant_response = retry_response['results'][0]['generated_text'].strip()
                        print(f"Retry response text: '{assistant_response}'")
                    except Exception as retry_error:
                        print(f"Error during retry: {str(retry_error)}")
                
                # Clean the response using our helper method
                assistant_response = self._clean_response(assistant_response, user_message)
                
                # Don't truncate responses about off-topic questions
                if "I'm your ASL tutor for the letter" in assistant_response and "I can't help with that" in assistant_response:
                    # Keep the full response for off-topic warnings
                    pass
                else:
                    # For regular responses, consider shortening very long ones
                    parts = assistant_response.split('. ', 1)
                    if len(parts) > 1 and len(parts[0]) < 50 and len(assistant_response) > 200:
                        assistant_response = parts[0].strip() + "."

            except Exception as e:
                print(f"WatsonX chat error: {str(e)}")
                assistant_response = self._generate_simple_response(user_message, sign_name)
                # Also clean fallback responses
                assistant_response = self._clean_response(assistant_response, user_message)
        else:
            # Simple fallback for non-Watson mode
            print(f"WatsonX not available, using fallback response generator")
            assistant_response = self._generate_simple_response(user_message, sign_name)
            # Also clean fallback responses
            assistant_response = self._clean_response(assistant_response, user_message)
        
        # Ensure we have a non-empty response
        if not assistant_response or assistant_response.strip() == "":
            print(f"WARNING: Empty response generated. Using fallback response.")
            assistant_response = self._generate_simple_response(user_message, sign_name)
            # If still empty, use a generic response
            if not assistant_response or assistant_response.strip() == "":
                assistant_response = f"I'm sorry, I couldn't generate a proper response about the '{sign_name}' sign. Could you please try asking your question differently?"
        
        print(f"Final response: '{assistant_response}'")
        
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
        
        print(f"Generating simple response for message: '{message_lower}', sign: '{sign_name}'")
        
        # List of ASL-related terms to check if the message is on-topic
        asl_terms = ["asl", "sign", "hand", "finger", "position", "thumb", "index", "middle", "ring", "pinky", 
                    "palm", "wrist", "movement", "alphabet", "learn", "practice", "form", "signing", "language"]
                
        # Check if message appears to be off-topic (doesn't contain ASL-related terms)
        is_off_topic = True
        for term in asl_terms:
            if term in message_lower:
                is_off_topic = False
                break
                
        # Handle potentially off-topic questions with a complete response
        if is_off_topic and len(message_lower.split()) > 2:  # Only trigger for longer queries
            return f"I'm your ASL tutor for the letter '{sign_name}'. I can't help with that, but I'd be happy to answer questions about forming the '{sign_name}' sign, common mistakes to avoid, or tips for practice."
        
        # Simple one-word queries - treat as greetings
        if len(message_lower.split()) <= 2:
            if any(word in message_lower for word in ["hi", "hello", "hey", "greetings"]):
                return f"Hi there! I'm here to help you learn the ASL sign for the letter '{sign_name}'. What would you like to know about it?"
        
        # Check for common question patterns
        if any(word in message_lower for word in ["how", "form", "make", "do", "sign"]):
            return f"To form the letter '{sign_name}' in ASL: {parts[0]}. Make sure your {parts[1].lower() if len(parts) > 1 else 'hand is in the correct position'}."
            
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
            
        # For off-topic queries
        elif any(word in message_lower for word in ["cookie", "recipe", "bake", "cook", "food", "eat"]):
            return f"I'm here to help with ASL! Let's focus on the sign for '{sign_name}'. Remember, the thumb curls alongside the index finger, and all fingers bend downward with the palm facing forward."
            
        else:
            # Default response for any other message
            return f"For the '{sign_name}' sign: {parts[0]}. What else would you like to know about forming this sign correctly?"
            
    def _clean_response(self, response_text, user_message=""):
        """Clean up the LLM response text by removing formatting artifacts."""
        print(f"Original response before cleaning: '{response_text}'")
        assistant_response = response_text.strip()
        
        # Handle responses with visual separators like "---"
        lines = assistant_response.split('\n')
        filtered_lines = []
        
        # Special case: Check for the pattern in the most recent screenshot
        # Where "---" is followed by "assistant: I'm here to help with ASL!"
        separator_index = -1
        for i, line in enumerate(lines):
            if line.strip() and all(c == '-' for c in line.strip()):
                separator_index = i
                break
                
        if separator_index >= 0 and separator_index + 1 < len(lines):
            # Check if the line after the separator contains an assistant message
            after_separator = '\n'.join(lines[separator_index+1:])
            if "assistant:" in after_separator.lower():
                # Extract just that part
                assistant_response = after_separator
                filtered_lines = []  # Skip regular filtering
            
        # Regular filtering for other cases
        if filtered_lines == []:  # Only if not already handled by special case
            for line in lines:
                # Skip separator lines with just dashes
                if line.strip() and not all(c == '-' for c in line.strip()):
                    filtered_lines.append(line)
                    
        if filtered_lines:
            assistant_response = '\n'.join(filtered_lines)
        
        # Find any assistant message in a conversation format
        # This handles formats like "user: X\n\nassistant: I'm here to help with ASL!"
        assistant_regex = r'(?:^|\n)(?:assistant|a):\s*(.*?)(?=\n\w+:|$)'
        assistant_matches = re.findall(assistant_regex, assistant_response, re.DOTALL | re.IGNORECASE)
        if assistant_matches:
            # If we find an assistant part, use the most relevant one (typically the last)
            assistant_response = assistant_matches[-1].strip()
        else:
            # Continue with other cleaning methods if no assistant part found
            
            # Basic cleanup
            if assistant_response.startswith("assistant:"):
                assistant_response = assistant_response[10:].strip()
                
            # Remove common meta-instruction prefixes
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
            
            # Check for the specific pattern: "user: Question\n\nA: Answer"
            if assistant_response.lower().startswith("user:") and "\n\n" in assistant_response:
                parts = assistant_response.split("\n\n", 1)
                if len(parts) == 2 and (parts[1].lower().startswith("a:") or parts[1].lower().startswith("assistant:")):
                    # Extract just the answer part
                    answer_part = parts[1]
                    for prefix in ["a:", "assistant:"]:
                        if answer_part.lower().startswith(prefix):
                            answer_part = answer_part[len(prefix):].strip()
                            break
                    assistant_response = answer_part
            
            # Check for 'user:' pattern elsewhere in the response
            elif "user:" in assistant_response.lower():
                parts = assistant_response.split("\n\n")
                filtered_parts = []
                
                for part in parts:
                    if not part.strip().lower().startswith("user:"):
                        filtered_parts.append(part)
                
                if filtered_parts:
                    assistant_response = "\n\n".join(filtered_parts).strip()
            
            # Look for and remove "assistant:" role labels anywhere in text
            lines = assistant_response.split('\n')
            for i, line in enumerate(lines):
                line_lower = line.strip().lower()
                # Check for standalone role indicators
                if line_lower == "assistant:" or line_lower == "a:":
                    lines[i] = ""
                # Check for role indicators at start of lines
                elif line_lower.startswith("assistant:") or line_lower.startswith("a:"):
                    for prefix in ["assistant:", "a:"]:
                        if line_lower.startswith(prefix):
                            lines[i] = line[len(prefix):].strip()
                            break
            
            # Rejoin lines, removing any empty ones
            assistant_response = '\n'.join(line for line in lines if line.strip())
        
        # Further cleanup for all paths
        
        # Check for "Your response:" prefix and remove it
        if assistant_response.startswith("Your response:"):
            assistant_response = assistant_response[14:].strip()
        elif assistant_response.lower().startswith("your response:"):
            assistant_response = assistant_response[14:].strip()
        elif assistant_response.startswith("Your response -"):
            assistant_response = assistant_response[14:].strip()
        elif assistant_response.lower().startswith("your response -"):
            assistant_response = assistant_response[14:].strip()
        elif assistant_response.lower().startswith("your response "):
            assistant_response = assistant_response[14:].strip()
        
        # Remove the user's message if it somehow got into the response
        if user_message and len(user_message) > 3 and user_message.lower() in assistant_response.lower():
            parts = assistant_response.lower().split(user_message.lower(), 1)
            if len(parts) > 1 and parts[1].strip():
                assistant_response = parts[1].strip()
        
        # Final cleanup to remove any remaining role prefixes that may be embedded in the response
        assistant_response = re.sub(r'(?:^|\s)(?:assistant|a):\s*', ' ', assistant_response, flags=re.IGNORECASE)
        assistant_response = assistant_response.strip()
        
        print(f"Cleaned response: '{assistant_response}'")
        return assistant_response

# Create a singleton instance
llm_service = LLMService() 
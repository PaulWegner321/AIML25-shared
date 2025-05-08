import json
import logging
import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models.schema import TextGenParameters
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ASL letter definitions (ground truth)
asl_definitions = {
    "A": "Make a fist with your thumb resting on the side of your index finger.",
    "B": "Hold your hand up, fingers together and straight, thumb folded across your palm.",
    "C": "Curve your fingers and thumb to form the shape of the letter C.",
    "D": "Hold up your index finger, touch the tip of your thumb to the tip of your middle finger, other fingers curled down.",
    "E": "Curl your fingers down to touch your thumb, keeping the thumb underneath the fingers.",
    "F": "Touch the tip of your index finger to the tip of your thumb, other fingers extended upward.",
    "G": "Hold your hand sideways, index finger and thumb extended parallel, other fingers closed.",
    "H": "Extend your index and middle fingers together, palm facing sideways, other fingers closed.",
    "I": "Make a fist and extend your pinky finger upward.",
    "J": "Make the sign for I, then draw the letter J in the air with your pinky.",
    "K": "Extend your index and middle fingers upward and apart, thumb between them.",
    "L": "Extend your thumb and index finger to form an L shape, other fingers closed.",
    "M": "Place your thumb under your first three fingers, fingers together.",
    "N": "Place your thumb under your first two fingers, fingers together.",
    "O": "Curve all your fingers and thumb to touch, forming an O shape.",
    "P": "Make the K sign, then point your hand downward.",
    "Q": "Make the G sign, then point your hand downward.",
    "R": "Cross your index and middle fingers, other fingers closed.",
    "S": "Make a fist with your thumb across the front of your fingers.",
    "T": "Make a fist and place your thumb between your index and middle fingers.",
    "U": "Extend your index and middle fingers together, other fingers closed.",
    "V": "Extend your index and middle fingers apart to form a V, other fingers closed.",
    "W": "Extend your index, middle, and ring fingers upward, other fingers closed.",
    "X": "Make a fist and bend your index finger to form a hook.",
    "Y": "Extend your thumb and pinky finger, other fingers closed.",
    "Z": "Extend your index finger and draw a Z shape in the air."
}

def get_asl_sign_context(letter):
    """Get the ground truth context for a single letter."""
    if letter.upper() not in asl_definitions:
        return None
    return asl_definitions[letter.upper()]

def create_tutor_prompt(letter):
    """Create a detailed prompt for the ASL tutor."""
    sign_definition = get_asl_sign_context(letter)
    if not sign_definition:
        return None
    
    prompt = f"""You are an American Sign Language (ASL) teacher.

GROUND TRUTH DEFINITION TO USE:
{sign_definition}

Your task is to explain how to perform the ASL sign '{letter}' in a clear and simple way. Do not assume any prior knowledge about ASL.

Think step-by-step how to teach the sign for "{letter}". Think about handshape, location of both hands, orientation, movement, and a helpful everyday analogy.
After thinking, clearly explain how to perform the ASL sign on a beginner level for '{letter}'.

Only output the explanation for the requested sign once. Try to only use three sentences in the explanation. If appropriate, use less tokens than available. Do not include any other text. Ensure that you exclusively explain the sign '{letter}'.

Stay strictly faithful to the ground truth definition provided above."""
    
    return prompt

def ask_tutor_for_explanation(letter, client, model):
    """Get a detailed tutorial for a single ASL letter sign."""
    try:
        # Create the tutoring prompt
        prompt = create_tutor_prompt(letter)
        if not prompt:
            return f"Error: Letter '{letter}' is not in the ASL alphabet."

        # Generate response using the model
        response = model.generate(prompt)
        return response["results"][0]["generated_text"].strip()

    except Exception as e:
        logging.error(f"Error in ASL tutorial: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Load .env file from backend directory
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', '.env')
    load_dotenv(dotenv_path=env_path)

    # Get credentials from environment
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    
    if not api_key or not project_id:
        raise ValueError("API key or project ID not found in .env file.")

    # Initialize WatsonX client
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=api_key
    )

    client = APIClient(
        credentials=credentials,
        project_id=project_id
    )

    # Set up model parameters
    params = TextGenParameters(
        temperature=0.5,  # Balance between creativity and accuracy
        max_new_tokens=500,  # Allow for detailed explanations
        min_new_tokens=100,  # Ensure comprehensive responses
        stop_sequences=["\n\n\n"],  # Stop at major breaks
        top_k=50,  # Diverse but relevant token selection
        top_p=0.9  # High-quality output while maintaining some creativity
    )

    model = ModelInference(
        api_client=client,
        model_id="ibm/granite-13b-instruct-v2",
        params=params
    )

    # Test with a single letter
    letter = "A"  # Change this to test other letters
    print(f"\nGetting detailed tutorial for signing the letter '{letter}' in ASL...")
    
    try:
        result = ask_tutor_for_explanation(letter, client, model)
        print("\nASL Tutorial:")
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}") 
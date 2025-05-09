import os
import sys
import json
from typing import Dict, List, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RAGDescription:
    def __init__(self):
        """
        Initialize the RAG description generator.
        This is a placeholder implementation that will be replaced with a real RAG system.
        """
        # Load mock ASL descriptions
        self.asl_descriptions = {
            # Letters
            "A": "Make a fist with your thumb on the side of your index finger.",
            "B": "Hold your hand open with fingers together and palm facing forward.",
            "C": "Curve your fingers to form a 'C' shape.",
            "D": "Point your index finger up with your thumb touching the middle finger.",
            "E": "Bend your fingers down to touch your thumb.",
            "F": "Touch your thumb and index finger together, other fingers extended.",
            "G": "Point your index finger to the side.",
            "H": "Point your index and middle fingers to the side.",
            "I": "Point your pinky finger up.",
            "J": "Point your pinky finger up and make a 'J' motion.",
            "K": "Point your index and middle fingers up in a 'V' shape.",
            "L": "Point your thumb and index finger to form an 'L'.",
            "M": "Tuck your thumb between your index and middle fingers.",
            "N": "Tuck your thumb between your index and middle fingers, with middle finger slightly bent.",
            "O": "Touch your thumb and fingers together to form an 'O'.",
            "P": "Point your index finger down with your thumb touching the middle finger.",
            "Q": "Point your index finger down.",
            "R": "Cross your index and middle fingers.",
            "S": "Make a fist with your thumb over your fingers.",
            "T": "Tuck your thumb between your index and middle fingers, with index finger extended.",
            "U": "Point your index and middle fingers up together.",
            "V": "Point your index and middle fingers up in a 'V' shape.",
            "W": "Point your thumb, index, and middle fingers up.",
            "X": "Bend your index finger.",
            "Y": "Point your thumb and pinky finger out.",
            "Z": "Point your index finger and make a 'Z' motion.",
            
            # Numbers
            "1": "Point your index finger up.",
            "2": "Point your index and middle fingers up in a 'V' shape.",
            "3": "Point your thumb, index, and middle fingers up.",
            "4": "Point all fingers up except your thumb.",
            "5": "Point all fingers up.",
            "6": "Touch your thumb and pinky finger together, other fingers extended.",
            "7": "Point your index finger down with your thumb touching the middle finger.",
            "8": "Point your index and middle fingers down.",
            "9": "Bend your index finger down.",
            "10": "Point your index finger up and shake it.",
            
            # Common words
            "HELLO": "Wave your hand from side to side.",
            "THANK YOU": "Touch your chin with your fingers and move your hand forward and down.",
            "PLEASE": "Rub your chest in a circular motion with your open hand.",
            "SORRY": "Make a fist and rub it in a circular motion on your chest.",
            "YES": "Make a fist and move it up and down like nodding.",
            "NO": "Touch your index and middle fingers to your thumb and move them apart.",
            "HELP": "Place your right hand on your left palm and lift both hands up.",
            "WATER": "Tap your index finger on your chin.",
            "FOOD": "Touch your fingers to your mouth.",
            "BATHROOM": "Make a 'T' shape with your index fingers and shake it.",
            "GOOD": "Place your right hand on your chin and move it forward.",
            "BAD": "Place your right hand on your chin and move it down.",
            "HAPPY": "Rub your chest in upward circular motions with both hands.",
            "SAD": "Move your index fingers down your cheeks.",
            "ANGRY": "Claw your fingers and move them down your face.",
            "SICK": "Place your middle fingers on your forehead and move them down.",
            "TIRED": "Place your fingers on your chest and move them down.",
            "SLEEP": "Place your hand on your cheek and close your eyes.",
            "WAKE UP": "Open your eyes and move your hand away from your face.",
            "GOOD MORNING": "Place your right hand on your chin and move it forward, then move your hand up.",
            "GOOD NIGHT": "Place your right hand on your chin and move it forward, then place your hand on your cheek and close your eyes.",
            "I LOVE YOU": "Point your thumb, index finger, and pinky finger up.",
            "FRIEND": "Hook your index fingers together and move them in a circular motion.",
            "FAMILY": "Make 'F' hands and move them in a circle.",
            "NAME": "Cross your index and middle fingers and tap them on your other index finger.",
            "MEET": "Hold your hands up with palms facing each other and move them toward each other.",
            "NICE": "Place your right hand on your left palm and move it forward.",
            "MEET YOU": "Hold your hands up with palms facing each other and move them toward each other, then point to the person.",
        }

    def get_sign_description(self, word: str) -> Dict[str, Any]:
        """
        Get a description of how to sign a word in ASL.
        
        Args:
            word: The word to get a description for
            
        Returns:
            dict: Description results including:
                - word: The input word
                - description: The ASL description
                - steps: List of steps to perform the sign
                - tips: List of tips for performing the sign correctly
        """
        # Convert to uppercase for consistency
        word = word.upper()
        
        # Check if we have a description for this word
        if word in self.asl_descriptions:
            description = self.asl_descriptions[word]
            
            # Generate steps and tips based on the description
            steps = self._generate_steps(description)
            tips = self._generate_tips(word, description)
            
            return {
                "word": word,
                "description": description,
                "steps": steps,
                "tips": tips
            }
        else:
            # For words we don't have a description for, generate a generic response
            return {
                "word": word,
                "description": f"To sign '{word}', spell it out using the ASL alphabet.",
                "steps": [
                    "Spell out each letter of the word using the ASL alphabet.",
                    "Make sure to form each letter clearly and distinctly.",
                    "Keep your hand movements smooth and fluid."
                ],
                "tips": [
                    "Practice each letter individually before trying to spell the whole word.",
                    "Make sure your hand is positioned correctly for each letter.",
                    "Keep your movements clear and deliberate."
                ]
            }
    
    def _generate_steps(self, description: str) -> List[str]:
        """
        Generate steps for performing a sign based on its description.
        
        Args:
            description: The description of the sign
            
        Returns:
            list: Steps for performing the sign
        """
        # This is a placeholder implementation
        # In a real RAG system, this would be more sophisticated
        steps = []
        
        # Split the description into sentences
        sentences = description.split(". ")
        
        # Add each sentence as a step
        for sentence in sentences:
            if sentence:
                steps.append(sentence + ".")
        
        # If we don't have enough steps, add some generic ones
        while len(steps) < 3:
            steps.append("Practice the sign repeatedly to improve your form.")
        
        return steps
    
    def _generate_tips(self, word: str, description: str) -> List[str]:
        """
        Generate tips for performing a sign correctly.
        
        Args:
            word: The word being signed
            description: The description of the sign
            
        Returns:
            list: Tips for performing the sign correctly
        """
        # This is a placeholder implementation
        # In a real RAG system, this would be more sophisticated
        tips = [
            "Make sure your hand is positioned correctly.",
            "Keep your movements clear and deliberate.",
            "Practice the sign repeatedly to improve your form."
        ]
        
        # Add a word-specific tip
        if len(word) == 1:
            tips.append(f"Focus on forming the letter '{word}' correctly.")
        else:
            tips.append(f"Make sure to sign each part of '{word}' clearly.")
        
        return tips 
from typing import List, Dict
from config.watsonx_client import watsonx_client

class TranslationJudge:
    def __init__(self):
        """Initialize the translation judge with Watsonx client."""
        self.client = watsonx_client
    
    def judge_translation(self, tokens: List[str], translated_text: str) -> Dict:
        """
        Evaluate the quality of an ASL translation.
        
        Args:
            tokens: Original ASL tokens
            translated_text: The translated English text
            
        Returns:
            Dictionary containing score and suggestions
        """
        try:
            judgment = self.client.judge_translation(tokens, translated_text)
            return judgment
        except Exception as e:
            raise Exception(f"Translation judgment failed: {str(e)}")
    
    def judge_with_criteria(self, tokens: List[str], translated_text: str, 
                          criteria: Dict[str, float]) -> Dict:
        """
        Evaluate translation with specific criteria weights.
        
        Args:
            tokens: Original ASL tokens
            translated_text: The translated English text
            criteria: Dictionary of criteria and their weights (e.g., {"accuracy": 0.6, "fluency": 0.4})
            
        Returns:
            Dictionary containing detailed scores and suggestions
        """
        try:
            # Add criteria to the judgment request
            criteria_str = ", ".join(f"{k}: {v}" for k, v in criteria.items())
            judgment = self.client.judge_translation(
                tokens + [f"Criteria: {criteria_str}"], 
                translated_text
            )
            return judgment
        except Exception as e:
            raise Exception(f"Criteria-based judgment failed: {str(e)}")
    
    def batch_judge(self, translations: List[Dict[str, str]]) -> List[Dict]:
        """
        Evaluate multiple translations.
        
        Args:
            translations: List of dictionaries containing tokens and translated_text
            
        Returns:
            List of judgment results
        """
        try:
            judgments = []
            for translation in translations:
                judgment = self.judge_translation(
                    translation["tokens"],
                    translation["translated_text"]
                )
                judgments.append(judgment)
            return judgments
        except Exception as e:
            raise Exception(f"Batch judgment failed: {str(e)}")

# Create a singleton instance
judge = TranslationJudge() 
from typing import List, Dict
from config.watsonx_client import watsonx_client

class ASLTranslator:
    def __init__(self):
        """Initialize the ASL translator with Watsonx client."""
        self.client = watsonx_client
    
    def translate_tokens(self, tokens: List[str]) -> str:
        """
        Translate ASL tokens to English using Watsonx.
        
        Args:
            tokens: List of ASL tokens to translate
            
        Returns:
            Translated English text
        """
        try:
            translated_text = self.client.translate(tokens)
            return translated_text
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")
    
    def translate_with_context(self, tokens: List[str], context: Dict[str, str]) -> str:
        """
        Translate ASL tokens with additional context using Watsonx.
        
        Args:
            tokens: List of ASL tokens to translate
            context: Dictionary of additional context (e.g., domain, style)
            
        Returns:
            Translated English text with context consideration
        """
        try:
            # Add context to the translation request
            context_str = ", ".join(f"{k}: {v}" for k, v in context.items())
            tokens_with_context = tokens + [f"Context: {context_str}"]
            translated_text = self.client.translate(tokens_with_context)
            return translated_text
        except Exception as e:
            raise Exception(f"Contextual translation failed: {str(e)}")
    
    def batch_translate(self, token_sequences: List[List[str]]) -> List[str]:
        """
        Translate multiple sequences of ASL tokens.
        
        Args:
            token_sequences: List of token lists to translate
            
        Returns:
            List of translated English texts
        """
        try:
            translations = []
            for tokens in token_sequences:
                translated_text = self.translate_tokens(tokens)
                translations.append(translated_text)
            return translations
        except Exception as e:
            raise Exception(f"Batch translation failed: {str(e)}")

# Create a singleton instance
translator = ASLTranslator() 
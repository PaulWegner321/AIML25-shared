import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as MetaNames

# Load environment variables
load_dotenv()

class WatsonxClient:
    def __init__(self):
        """Initialize the Watsonx client with credentials."""
        self.api_key = os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.model_id = os.getenv("WATSONX_MODEL_ID")
        self.url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")  # Default to US South if not specified
        
        if not all([self.api_key, self.project_id, self.model_id]):
            raise ValueError("Missing required Watsonx credentials in environment variables")
        
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Watsonx model with credentials."""
        try:
            model = Model(
                model_id=self.model_id,
                credentials={
                    "apikey": self.api_key,
                    "url": self.url
                },
                project_id=self.project_id
            )
            return model
        except Exception as e:
            raise Exception(f"Failed to initialize Watsonx model: {str(e)}")
    
    def translate(self, tokens: list) -> str:
        """Translate ASL tokens to English using Watsonx."""
        prompt = f"Convert the following ASL tokens into fluent English: {', '.join(tokens)}"
        
        parameters = {
            MetaNames.DECODING_METHOD: "greedy",
            MetaNames.MAX_NEW_TOKENS: 100,
            MetaNames.MIN_NEW_TOKENS: 1,
            MetaNames.STOP_SEQUENCES: ["\n\n"],
            MetaNames.REPETITION_PENALTY: 1
        }
        
        try:
            response = self.model.generate(prompt, parameters)
            return response.generated_text
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")
    
    def judge_translation(self, tokens: list, translated_text: str) -> dict:
        """Judge the quality of the translation using Watsonx."""
        prompt = f"""Evaluate the following ASL to English translation:
        Original tokens: {', '.join(tokens)}
        Translated text: {translated_text}
        
        Provide a score from 1-10 and specific suggestions for improvement."""
        
        parameters = {
            MetaNames.DECODING_METHOD: "greedy",
            MetaNames.MAX_NEW_TOKENS: 150,
            MetaNames.MIN_NEW_TOKENS: 1,
            MetaNames.STOP_SEQUENCES: ["\n\n"],
            MetaNames.REPETITION_PENALTY: 1
        }
        
        try:
            response = self.model.generate(prompt, parameters)
            # TODO: Parse the response to extract score and suggestions
            return {
                "score": 8.5,  # Placeholder
                "suggestions": response.generated_text
            }
        except Exception as e:
            raise Exception(f"Judgment failed: {str(e)}")

# Create a singleton instance
watsonx_client = WatsonxClient() 
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import io
import sys
import datetime

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ASL Learning Platform API",
    description="API for ASL learning platform with sign evaluation and description capabilities.",
    version="1.0.0"
)

# Get environment variables
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://asl-edu-platform.vercel.app")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://asl-edu-platform.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class TranslationRequest(BaseModel):
    tokens: List[str]

class JudgmentRequest(BaseModel):
    translation: str
    tokens: List[str]

# Define response models
class TranslationResponse(BaseModel):
    translation: str

class JudgmentResponse(BaseModel):
    feedback: str
    score: float

class VideoFrameResponse(BaseModel):
    letter: str
    confidence: float

class SignEvaluationResponse(BaseModel):
    predicted_sign: str
    confidence: float
    feedback: str
    is_correct: bool

class SignDescriptionResponse(BaseModel):
    word: str
    description: str
    steps: List[str]
    tips: List[str]

class ErrorResponse(BaseModel):
    error: str

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the backend directory to the Python path
sys.path.insert(0, BACKEND_DIR)

# Now import the ASL detector
from app.models.asl_detector import ASLDetector

# Import our models
from app.models.sign_evaluator import SignEvaluator
from app.models.rag_description import RAGDescription

# Initialize model instances
asl_detector = ASLDetector()
sign_evaluator = SignEvaluator()
rag_description = RAGDescription()

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "ASL Learning Platform API",
        "version": "1.0.0",
        "description": "API for ASL learning platform with sign evaluation and description capabilities."
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        # For now, return a dummy translation
        # In the future, use the translator model
        translation = f"Dummy translation for tokens: {', '.join(request.tokens)}"
        return TranslationResponse(translation=translation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/judge", response_model=JudgmentResponse)
async def judge(request: JudgmentRequest):
    try:
        # For now, return dummy feedback
        # In the future, use the judge model
        feedback = "This is a dummy feedback for the translation."
        score = 0.75
        return JudgmentResponse(feedback=feedback, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-frame", response_model=VideoFrameResponse)
async def process_frame(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array for processing
        image_array = np.array(image)
        
        # Get prediction from ASL detector
        predictions = asl_detector.detect(image_array)
        
        if not predictions:
            raise HTTPException(status_code=400, detail="No ASL letter detected in the frame")
            
        # Return the first prediction (we're only processing one frame at a time)
        letter, confidence = predictions[0]
        return VideoFrameResponse(
            letter=letter,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process frame: {str(e)}")

@app.post("/evaluate-sign", response_model=SignEvaluationResponse, responses={500: {"model": ErrorResponse}})
async def evaluate_sign(
    file: UploadFile = File(...),
    expected_sign: Optional[str] = None
):
    """
    Evaluate an ASL sign from an uploaded image.
    
    Args:
        file: The image file containing the ASL sign
        expected_sign: The expected sign (optional, for providing feedback)
        
    Returns:
        SignEvaluationResponse: Evaluation results including predicted sign and confidence
    """
    try:
        # Read and validate the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get prediction from the model
        result = sign_evaluator.evaluate_sign(image, expected_sign)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['feedback'])
        
        return SignEvaluationResponse(
            predicted_sign=result['predicted_sign'],
            confidence=result['confidence'],
            feedback=result['feedback'],
            is_correct=result['is_correct']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sign-description/{word}", response_model=SignDescriptionResponse, responses={500: {"model": ErrorResponse}})
async def get_sign_description(word: str):
    """
    Get a description of how to sign a word in ASL.
    
    Args:
        word: The word to get a description for
        
    Returns:
        SignDescriptionResponse: Description results including steps and tips
    """
    try:
        # Get description from the RAG model
        description = rag_description.get_sign_description(word)
        
        return SignDescriptionResponse(
            word=description["word"],
            description=description["description"],
            steps=description["steps"],
            tips=description["tips"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    try:
        # Check if model files are accessible
        model_check = check_model()
        return {
            "status": "healthy" if model_check else "degraded",
            "model_files": "accessible" if model_check else "not accessible",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 
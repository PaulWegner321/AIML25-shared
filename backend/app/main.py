from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import numpy as np
import cv2

# Import the HandDetector
from .models.keypoint_detector import HandDetector

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ASL Translation API",
    description="API for ASL detection, translation, and evaluation",
    version="1.0.0"
)

# Get environment variables
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://asltranslate-p4sndxrkd-henriks-projects-f6f15939.vercel.app")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:3000",
        "https://asltranslate-p4sndxrkd-henriks-projects-f6f15939.vercel.app",
        "https://asltranslate-c8qu1q97f-henriks-projects-f6f15939.vercel.app",
        "https://asl-edu-platform.vercel.app"

    ] if ENVIRONMENT == "production" else ["*"],
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

class SignEvaluationResponse(BaseModel):
    success: bool
    letter: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

# Define response models
class TranslationResponse(BaseModel):
    translation: str

class JudgmentResponse(BaseModel):
    feedback: str
    score: float

# Initialize model instances
hand_detector = HandDetector()

@app.get("/")
async def root():
    return {
        "message": "ASL Translation API",
        "status": "healthy",
        "environment": ENVIRONMENT
    }

@app.post("/evaluate-sign", response_model=SignEvaluationResponse)
async def evaluate_sign(file: UploadFile = File(...), expected_sign: str = None):
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect sign
        result = hand_detector.detect_sign(image)
        
        if not result['success']:
            return SignEvaluationResponse(
                success=False,
                error=result.get('error', 'Failed to detect hand in image')
            )
        
        # If an expected sign was provided, check if the prediction matches
        if expected_sign:
            is_correct = result['letter'].lower() == expected_sign.lower()
            if is_correct:
                feedback = "Good job! Your sign is correct."
            else:
                feedback = f"Your sign was interpreted as '{result['letter']}', but the expected sign was '{expected_sign}'. Try again!"
            
            return SignEvaluationResponse(
                success=True,
                letter=result['letter'],
                confidence=result['confidence'],
                error=None if is_correct else feedback
            )
        
        return SignEvaluationResponse(
            success=True,
            letter=result['letter'],
            confidence=result['confidence']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        # For now, return a dummy translation
        # In the future, use the translator model
        # translation = translator.translate(request.tokens)
        translation = f"Dummy translation for tokens: {', '.join(request.tokens)}"
        return TranslationResponse(translation=translation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/judge", response_model=JudgmentResponse)
async def judge(request: JudgmentRequest):
    try:
        # For now, return dummy feedback
        # In the future, use the judge model
        # feedback, score = judge.evaluate(request.translation, request.tokens)
        feedback = "This is a dummy feedback for the translation."
        score = 0.75
        return JudgmentResponse(feedback=feedback, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 
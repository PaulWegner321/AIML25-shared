from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

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
        "https://asltranslate-c8qu1q97f-henriks-projects-f6f15939.vercel.app"
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

# Define response models
class TranslationResponse(BaseModel):
    translation: str

class JudgmentResponse(BaseModel):
    feedback: str
    score: float

# Import model modules (to be implemented)
# from models.asl_detector import ASLDetector
# from models.translator import Translator
# from models.judge import Judge
# from models.rag_pipeline import RAGPipeline

# Initialize model instances (to be implemented)
# asl_detector = ASLDetector()
# translator = Translator()
# judge = Judge()
# rag_pipeline = RAGPipeline()

@app.get("/")
async def root():
    return {
        "message": "ASL Translation API",
        "status": "healthy",
        "environment": ENVIRONMENT
    }

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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="ASL Translation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    tokens: List[str]

class TranslationResponse(BaseModel):
    translated_text: str

class JudgeRequest(BaseModel):
    tokens: List[str]
    translated_text: str

class JudgeResponse(BaseModel):
    score: float
    suggestions: str

@app.get("/")
async def root():
    return {"message": "ASL Translation API is running"}

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        # TODO: Integrate with actual Watsonx translation
        # For now, return mock translation
        translated_text = f"Mock translation of tokens: {', '.join(request.tokens)}"
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/judge", response_model=JudgeResponse)
async def judge_translation(request: JudgeRequest):
    try:
        # TODO: Integrate with actual Watsonx judgment
        # For now, return mock judgment
        return JudgeResponse(
            score=8.5,
            suggestions="Good translation, but could be more natural in some parts."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
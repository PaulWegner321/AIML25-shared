from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import numpy as np
import cv2
import datetime
from pathlib import Path
import time
import glob

# Import the HandDetector
from .models.keypoint_detector import HandDetector
from .services.vision_evaluator import VisionEvaluator
from .services.gpt4o_service import get_asl_prediction as gpt4o_predict
from .services.cnn_service import cnn_predictor

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ASL Translation API",
    description="API for ASL detection, translation, and evaluation",
    version="1.0.0"
)

# Get environment variables
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://asl-edu-platform.vercel.app")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://asltranslate-p4sndxrkd-henriks-projects-f6f15939.vercel.app",
        "https://asltranslate-c8qu1q97f-henriks-projects-f6f15939.vercel.app",
        "https://asl-edu-platform.vercel.app",
        "http://localhost:8000",  # Local backend
        "https://asl-api.onrender.com"  # Deployed backend
    ] if ENVIRONMENT == "production" else ["*"],  # Allow all origins in development
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
    feedback: Optional[str] = None

# Define response models
class TranslationResponse(BaseModel):
    translation: str

class JudgmentResponse(BaseModel):
    feedback: str
    score: float

# Initialize model instances
hand_detector = HandDetector()
vision_evaluator = VisionEvaluator()

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    try:
        # Load CNN model
        model_path = os.path.join(os.path.dirname(__file__), "models", "weights", "cnn_model.pth")
        cnn_predictor.load_model(model_path)
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        # Don't raise the error, let the application start without the CNN model
        # The endpoints will handle the error gracefully

@app.get("/")
async def root():
    return {
        "message": "ASL Translation API",
        "status": "healthy",
        "environment": ENVIRONMENT
    }

@app.post("/evaluate-sign", response_model=SignEvaluationResponse)
async def evaluate_sign(
    file: UploadFile = File(...), 
    expected_sign: str = Form(None),
    model_id: str = Form("model1"),
    model_type: str = Form("image_processing")
):
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Use appropriate model based on model_id
        if model_id == "model1":
            # Use the new CNN model
            result = cnn_predictor.predict(image)
        else:
            # Use the existing hand detector for other models
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
                error=None if is_correct else feedback,
                feedback=feedback
            )
        
        return SignEvaluationResponse(
            success=True,
            letter=result['letter'],
            confidence=result['confidence']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-vision", response_model=SignEvaluationResponse)
async def evaluate_vision_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("full"),
    expected_sign: str = Form(None),
    detected_sign: str = Form(None),
    user_intention: str = Form(None),
    model_id: str = Form("granite-vision"),
    model_type: str = Form("llm")
):
    """Endpoint for objective sign analysis using vision models"""
    try:
        print(f"Received request for /evaluate-vision with model_id={model_id}")
        print(f"File info: name={file.filename}, content_type={file.content_type}")
        print(f"Expected sign: '{expected_sign}', Detected sign: '{detected_sign}'")
        if user_intention:
            print(f"User intention: {user_intention}")
        
        # Read and decode image
        contents = await file.read()
        print(f"Read {len(contents)} bytes from uploaded file")
        
        try:
            # Try to decode as an image
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Failed to decode image, treating as test file")
                # This might be a test file, not an image
                if file.content_type == 'text/plain' or file.filename.endswith('.txt'):
                    # For test files, return a success response
                    return SignEvaluationResponse(
                        success=True,
                        letter="T",
                        confidence=0.99,
                        feedback="This is a test response for a text file."
                    )
                else:
                    raise HTTPException(status_code=400, detail="Invalid image file format")
            
            # Log image details
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            
            # Save a debug copy of the received image with timestamp in filename to prevent caching
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            timestamp = int(time.time())
            debug_image_path = debug_dir / f"raw_image_{timestamp}.jpg"
            
            try:
                cv2.imwrite(str(debug_image_path), image)
                print(f"Debug image saved to {debug_image_path}")
                
                # Also save a different view for comparison
                # Convert to grayscale to better see hand features
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_path = debug_dir / f"gray_image_{timestamp}.jpg"
                cv2.imwrite(str(gray_path), gray_image)
                print(f"Grayscale debug image saved to {gray_path}")
            except Exception as e:
                print(f"Could not save debug image: {str(e)}")
            
            # Create a metadata file with all request parameters for debugging
            try:
                metadata_path = debug_dir / f"request_metadata_{timestamp}.json"
                with open(metadata_path, "w") as f:
                    import json
                    json.dump({
                        "timestamp": timestamp,
                        "file_info": {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size_bytes": len(contents)
                        },
                        "parameters": {
                            "mode": mode,
                            "expected_sign": expected_sign,
                            "detected_sign": detected_sign,
                            "user_intention": user_intention,
                            "model_id": model_id,
                            "model_type": model_type
                        }
                    }, f, indent=2)
                print(f"Request metadata saved to {metadata_path}")
            except Exception as e:
                print(f"Could not save request metadata: {str(e)}")
            
            # Call vision_evaluator.evaluate_vision directly with NO EXPECTATION HINTS
            # Include current timestamp to force a fresh API call
            print("Performing objective sign detection with no expected sign hints...")
            vision_result = vision_evaluator.evaluate_vision(
                image=image,
                timestamp=timestamp  # Add timestamp to prevent using cached results
                # Do NOT pass expected_sign or detected_sign here
            )
            
            if "error" in vision_result:
                return SignEvaluationResponse(
                    success=False,
                    error=vision_result.get("error", "Failed to evaluate with Vision API")
                )
            
            detected_letter = vision_result.get("letter", None)
            confidence = vision_result.get("confidence", 0.0)
            feedback = vision_result.get("feedback", "No feedback available")
            
            print(f"DETECTION RESULT: Model detected '{detected_letter}', expected was '{expected_sign}', with confidence: {confidence}")
            
            # Post-detection comparison with expected sign (if provided)
            # Only do this AFTER the model has made its determination
            if expected_sign and detected_letter:
                is_match = detected_letter.upper() == expected_sign.upper()
                if not is_match:
                    # If the detection doesn't match expected, add comparison to feedback
                    feedback = f"{feedback}\n\nYou signed '{detected_letter}', but the expected sign was '{expected_sign}'. "
                    if mode == "feedback":
                        # Get improvement feedback for mismatched signs
                        try:
                            # Get sign-specific improvement feedback 
                            print(f"Getting improvement feedback for expected sign '{expected_sign}'")
                            feedback += f"To correctly sign '{expected_sign}', " + vision_evaluator.get_sign_guidance(expected_sign)
                        except Exception as fb_error:
                            print(f"Error getting sign guidance: {str(fb_error)}")
                            feedback += f"Please check proper hand position for '{expected_sign}'."
            
            return SignEvaluationResponse(
                success=True,
                letter=detected_letter,
                confidence=confidence,
                feedback=feedback
            )
            
        except Exception as img_error:
            print(f"Error processing image: {str(img_error)}")
            print(f"Error type: {type(img_error)}")
            # If there's an error processing as an image, try to interpret as a test
            if file.content_type == 'text/plain' or (file.filename and file.filename.endswith('.txt')):
                print("Treating as a test file")
                return SignEvaluationResponse(
                    success=True,
                    letter="Test",
                    confidence=0.5,
                    feedback="This is a test response. The file was recognized as a text file, not an image."
                )
            else:
                raise
            
    except Exception as e:
        print(f"Detailed error in evaluate_vision: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return SignEvaluationResponse(
            success=False,
            error=f"Server error: {str(e)}"
        )

@app.post("/evaluate-llm", response_model=SignEvaluationResponse)
async def evaluate_vision(
    file: UploadFile = File(...),
    mode: str = Form("full"),
    expected_sign: str = Form(None),
    detected_sign: str = Form(None),
    model_id: str = Form("granite-vision"),
    model_type: str = Form("llm")
):
    """Legacy endpoint kept for compatibility"""
    # Just forward to the new endpoint
    return await evaluate_vision_endpoint(
        file=file,
        mode=mode,
        expected_sign=expected_sign,
        detected_sign=detected_sign,
        model_id=model_id,
        model_type=model_type
    )

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

@app.get("/ping")
async def ping():
    """Simple endpoint to test connectivity"""
    print("Ping received!")
    return {
        "message": "pong",
        "time": str(datetime.datetime.now()),
        "status": "healthy"
    }

@app.options("/evaluate-vision")
async def options_evaluate_vision():
    """Handle preflight CORS requests for the vision endpoint"""
    print("OPTIONS request received for /evaluate-vision")
    return {}

@app.get("/debug-images")
async def get_debug_images():
    """Endpoint to list available debug images"""
    try:
        debug_dir = Path("debug_images")
        if not debug_dir.exists():
            return {"images": []}
            
        debug_images = []
        # Get all image files and sort by modification time (most recent first)
        image_files = list(debug_dir.glob("*.jpg"))
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Return the most recent 20 images
        for image_path in image_files[:20]:
            filename = image_path.name
            debug_images.append({
                "filename": filename,
                "timestamp": int(image_path.stat().st_mtime),
                "url": f"/debug-image/{filename}"
            })
        return {"images": debug_images}
    except Exception as e:
        print(f"Error listing debug images: {str(e)}")
        return {"error": str(e), "images": []}

@app.get("/debug-image/{filename}")
async def get_debug_image(filename: str):
    """Endpoint to retrieve a specific debug image by filename"""
    try:
        # Validate filename to prevent directory traversal attacks
        if ".." in filename or "/" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
            
        image_path = Path("debug_images") / filename
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
            
        return FileResponse(image_path)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error serving debug image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

@app.get("/diagnostic")
async def get_diagnostic():
    """Endpoint to provide details about the last processed image"""
    try:
        debug_dir = Path("debug_images")
        if not debug_dir.exists():
            return {"status": "No debug directory found"}
            
        # Find the most recent raw image and its paired gray image
        raw_images = list(debug_dir.glob("raw_image_*.jpg"))
        if not raw_images:
            return {"status": "No raw images found"}
            
        # Sort by modification time (most recent first)
        raw_images.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        most_recent = raw_images[0]
        
        # Get the timestamp from the filename
        timestamp = most_recent.name.replace("raw_image_", "").replace(".jpg", "")
        
        # Look for the corresponding grayscale image and response file
        gray_image = debug_dir / f"gray_image_{timestamp}.jpg"
        api_image = debug_dir / f"vision_api_input_{timestamp}.jpg"
        response_file = debug_dir / f"vision_api_response_{timestamp}.json"
        
        # Read response data if available
        response_data = None
        if response_file.exists():
            import json
            with open(response_file, "r") as f:
                response_data = json.load(f)
        
        return {
            "status": "success",
            "timestamp": timestamp,
            "images": {
                "raw": f"/debug-image/{most_recent.name}",
                "grayscale": f"/debug-image/{gray_image.name}" if gray_image.exists() else None,
                "api_input": f"/debug-image/{api_image.name}" if api_image.exists() else None
            },
            "api_response": response_data
        }
    except Exception as e:
        print(f"Error generating diagnostic: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/evaluate-gpt4o", response_model=SignEvaluationResponse)
async def evaluate_gpt4o(
    file: UploadFile = File(...),
    expected_sign: str = Form(None)
):
    """Endpoint for GPT-4o vision model with visual grounding prompt"""
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save image temporarily
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        temp_image_path = temp_dir / f"temp_image_{timestamp}.jpg"
        cv2.imwrite(str(temp_image_path), image)
        
        try:
            # Get prediction from GPT-4o
            result = gpt4o_predict(str(temp_image_path))
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
            if "error" in result:
                return SignEvaluationResponse(
                    success=False,
                    error=result["error"]
                )
            
            # Extract prediction details
            predicted_letter = result.get("letter")
            confidence = result.get("confidence", 0.0)
            feedback = result.get("feedback", "")
            
            # If an expected sign was provided, check if the prediction matches
            if expected_sign:
                is_correct = predicted_letter.upper() == expected_sign.upper()
                if not is_correct:
                    feedback = f"{feedback}\n\nYou signed '{predicted_letter}', but the expected sign was '{expected_sign}'."
            
            return SignEvaluationResponse(
                success=True,
                letter=predicted_letter,
                confidence=confidence,
                feedback=feedback
            )
            
        finally:
            # Ensure temp file is cleaned up even if there's an error
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
    except Exception as e:
        return SignEvaluationResponse(
            success=False,
            error=f"Server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 
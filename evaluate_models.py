import os
import json
import logging
import argparse
import traceback
import time
import random
import requests
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import model predictors from model_comparison directory
import sys
sys.path.append('./model_comparison')
try:
    from test_gemini_2_flash import get_asl_prediction as get_gemini_flash_prediction
    from test_gemini_1_5_pro import get_asl_prediction as get_gemini_pro_prediction
    from test_llama_90b_vision import get_asl_prediction as get_llama_90b_prediction
    from test_llama_maverick_17b import get_asl_prediction as get_llama_maverick_prediction
    from test_llama_scout_17b import get_asl_prediction as get_llama_scout_prediction
    from test_pixtral_12b import get_asl_prediction as get_mistral_prediction
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import predictor from test script: {e}")
    MODELS_AVAILABLE = False

# Add retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
)
def get_prediction_with_retry(predictor_func, image_path):
    """Wrapper function to add retry logic to model predictions"""
    return predictor_func(image_path)

def handle_result(model_name, prediction_result, true_letter, image_path):
    """Handles the prediction result dictionary for any model."""
    results[model_name]["total"] += 1
    
    if prediction_result is None:
        logging.error(f"Prediction result is None for {model_name} on {image_path}")
        results[model_name]["errors"] += 1
        return

    if "error" in prediction_result:
        error_msg = prediction_result["error"]
        logging.warning(f"{model_name} prediction error for {Path(image_path).name}: {error_msg}")
        
        # Enhanced error categorization
        if "Model cannot see the image" in error_msg:
            results[model_name]["visibility_failed"] += 1
            # Add delay after visibility failure
            time.sleep(random.uniform(2, 5))
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            results[model_name]["rate_limit_errors"] += 1
            results[model_name]["errors"] += 1
            # Add longer delay after rate limit
            time.sleep(random.uniform(10, 20))
        else:
            results[model_name]["errors"] += 1
            # Add small delay after other errors
            time.sleep(random.uniform(1, 3))
        return

    # Add delay between successful predictions to avoid rate limits
    time.sleep(random.uniform(1, 2))

    predicted_letter = prediction_result.get("letter")
    confidence = prediction_result.get("confidence", "N/A")
    
    if not predicted_letter or not isinstance(predicted_letter, str) or len(predicted_letter) != 1 or predicted_letter not in VALID_CLASSES:
        logging.warning(f"Invalid predicted letter '{predicted_letter}' from {model_name} for {Path(image_path).name}. Result: {prediction_result}")
        results[model_name]["errors"] += 1
        return

    predicted_letter = predicted_letter.upper()

    if predicted_letter == true_letter:
        results[model_name]["correct"] += 1
        logging.info(f"{model_name} CORRECT: True={true_letter}, Pred={predicted_letter} (Conf: {confidence}%) | File: {Path(image_path).name}")
    else:
        logging.info(f"{model_name} INCORRECT: True={true_letter}, Pred={predicted_letter} (Conf: {confidence}%) | File: {Path(image_path).name}")
        if true_letter not in misclassified[model_name]:
            misclassified[model_name][true_letter] = []
        misclassified[model_name][true_letter].append({
            "image_path": image_path,
            "predicted_letter": predicted_letter,
            "confidence": confidence,
            "feedback": prediction_result.get("feedback", "N/A")
        })

def main():
    # ... existing code ...
    
    # --- Run Evaluation --- #
    start_time = time.time()
    for image_path, true_letter in tqdm(dataset_sample, desc="Evaluating Images"):
        logging.info(f"\n--- Evaluating Image: {Path(image_path).name} (True Label: {true_letter}) ---")
        
        # Shuffle model order for each image to distribute rate limits
        model_items = list(model_predictors.items())
        random.shuffle(model_items)
        
        for model_name, predictor_func in model_items:
            logging.debug(f"Running prediction for {model_name}...")
            try:
                # Use retry wrapper for predictions
                prediction_result = get_prediction_with_retry(predictor_func, image_path)
                handle_result(model_name, prediction_result, true_letter, image_path)
            except Exception as e:
                logging.error(f"Unhandled exception during {model_name} prediction for {image_path}: {e}")
                traceback.print_exc()
                if model_name in results:
                    results[model_name]["errors"] += 1
                    results[model_name]["total"] += 1
                else:
                    logging.error(f"Model {model_name} not found in results dict during exception handling.")
            
            # Add delay between different models
            time.sleep(random.uniform(2, 4))

    # ... rest of existing code ... 
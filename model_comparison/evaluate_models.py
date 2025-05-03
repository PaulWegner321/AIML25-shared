# evaluate_models.py
import os
import argparse
import base64
import io
import json
import time
from pathlib import Path
import requests
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
import logging
import traceback
import random
import torch.nn.functional as F

# --- Import Model Predictors --- #
try:
    # Assuming evaluate_models.py is at the root or similar
    from backend.app.models.sign_evaluator import ColorASLCNN
except ImportError as e:
    logging.error(f"Failed to import ColorASLCNN: {e}")
    logging.error("Please ensure evaluate_models.py is run from a location where 'backend' is a reachable package, or adjust the import path.")
    ColorASLCNN = None

# Keep these imports active
try:
    from test_gemini_2_flash import get_asl_prediction as get_gemini_flash_prediction
    from test_gemini_2_flash import get_asl_prediction as get_gemini_flash_lite_prediction
    from test_gemini_2_5_pro import get_asl_prediction as get_gemini_pro_prediction
    from test_llama_90b_vision import get_asl_prediction as get_llama_90b_prediction
    from test_llama_maverick_17b import get_asl_prediction as get_llama_maverick_prediction
    from test_llama_scout_17b import get_asl_prediction as get_llama_scout_prediction
    from test_pixtral_12b import get_asl_prediction as get_mistral_prediction
    from test_gpt4_turbo import get_asl_prediction as get_gpt4_turbo_prediction
    from test_gpt4o import get_asl_prediction as get_gpt4o_prediction
    from test_granite_vision import get_asl_prediction as get_granite_vision_prediction
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import predictor from test script: {e}")
    MODELS_AVAILABLE = False
    # Handle missing predictor functions if necessary

# Load environment variables from backend/.env
dotenv_path = os.path.join('../../backend', '.env')
load_dotenv(dotenv_path=dotenv_path)
logging.info(f"Attempting to load .env file from: {dotenv_path}")

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VALID_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_MAP = {label: i for i, label in enumerate(VALID_CLASSES)}
INDEX_TO_CLASS = {i: label for label, i in CLASS_MAP.items()}

# Define models to evaluate based on availability
MODELS_TO_EVALUATE = []
if MODELS_AVAILABLE:
    MODELS_TO_EVALUATE.extend([
        "gpt4_turbo",
        "gpt4o",
        "gemini_flash",
        "gemini_flash_lite",
        "gemini_pro",
        "llama_90b",
        "llama_maverick",
        "llama_scout",
        "mistral",
        "granite_vision"
    ])

# Global variables for tracking results
results = {model: {
    "correct": 0, 
    "total": 0, 
    "errors": 0, 
    "visibility_failed": 0, 
    "rate_limit_errors": 0,
    "response_times": [],
    "token_usage": []
} for model in MODELS_TO_EVALUATE}
misclassified = {model: {} for model in MODELS_TO_EVALUATE}

# Initialize misclassified dictionary for each model and letter
for model in misclassified:
    for letter in VALID_CLASSES:
        misclassified[model][letter] = []

# Define the transform for preprocessing images (used by CNN)
cnn_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Helper Functions ---

def get_watsonx_token(api_key):
    """Gets or refreshes the IBM Cloud IAM token."""
    global watsonx_token, watsonx_token_expiry
    now = time.time()

    if watsonx_token and watsonx_token_expiry and now < watsonx_token_expiry:
        # --- Log Token Snippet --- #
        token_snippet = f"{watsonx_token[:5]}...{watsonx_token[-5:]}" if watsonx_token else "None"
        logging.debug(f"Using existing token: {token_snippet}")
        # --- End Log Token Snippet --- #
        return watsonx_token

    logging.info("Refreshing WatsonX authentication token...")
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    try:
        response = requests.post(auth_url, headers=headers, data=data, timeout=15)
        response.raise_for_status() # Raise error for bad status codes
        token_data = response.json()
        watsonx_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        watsonx_token_expiry = now + expires_in - 300 # 5 min buffer
        # --- Log Token Snippet --- #
        token_snippet = f"{watsonx_token[:5]}...{watsonx_token[-5:]}" if watsonx_token else "None"
        logging.info(f"WatsonX token refreshed. New token: {token_snippet}")
        # --- End Log Token Snippet --- #
        return watsonx_token
    except requests.exceptions.RequestException as e:
        logging.error(f"Authentication failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Error refreshing token: {e}")
        return None


def encode_image_base64(image_path, resize_dim=(256, 256)):
    """Reads, resizes, and returns the base64 encoded string of an image."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if resize_dim:
                logging.debug(f"Resizing image {Path(image_path).name} to {resize_dim}")
                img = img.resize(resize_dim)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_str = ''.join(base64_str.split())
            
            logging.debug(f"Base64 image string (first 20 chars): {base64_str[:20]}...")
            return base64_str
    except Exception as e:
        logging.error(f"Error encoding/resizing image {image_path}: {e}")
        traceback.print_exc()
        return None

def load_dataset_sample(dataset_path_str, sample_size=2):
    """Loads dataset from folders, taking a random sample per class."""
    dataset_path = Path(dataset_path_str)
    if not dataset_path.is_dir():
        logging.error(f"Dataset path not found or not a directory: {dataset_path}")
        return []

    images_by_label = {}
    logging.info(f"Loading dataset from: {dataset_path}")

    for label_dir in dataset_path.iterdir():
        if label_dir.is_dir():
            label = label_dir.name.upper()
            if label in VALID_CLASSES:
                image_files = []
                for image_file in label_dir.glob('*'):
                    if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                         image_files.append(str(image_file))
                
                if image_files:
                    images_by_label[label] = image_files
                else:
                    logging.warning(f"No valid image files found for label: {label}")
            else:
                 logging.debug(f"Skipping directory with non-alphabet label: {label_dir.name}")

    sampled_image_paths = []
    total_loaded = 0
    for label, image_files in images_by_label.items():
        num_available = len(image_files)
        actual_sample_size = min(sample_size, num_available)
        
        if actual_sample_size > 0:
            sampled_files = random.sample(image_files, actual_sample_size)
            logging.debug(f"Sampled {actual_sample_size} images for label '{label}' from {num_available} available.")
            for file_path in sampled_files:
                sampled_image_paths.append((file_path, label))
                total_loaded += 1
        else:
            logging.warning(f"Could not sample {sample_size} images for label '{label}', only {num_available} available.")

    logging.info(f"Loaded a sample of {total_loaded} images across {len(images_by_label)} classes (up to {sample_size} per class).")
    if not sampled_image_paths:
        logging.warning("No images loaded after sampling. Check dataset path and structure.")
    return sampled_image_paths

def get_prediction(model_name, image_path, prompt_strategy="zero_shot"):
    """Get prediction from the specified model."""
    start_time = time.time()
    try:
        if model_name == "gpt4_turbo":
            result = get_gpt4_turbo_prediction(image_path)
        elif model_name == "gpt4o":
            result = get_gpt4o_prediction(image_path)
        elif model_name == "gemini_flash":
            result = get_gemini_flash_prediction(image_path, model_type="flash", prompt_strategy=prompt_strategy)
        elif model_name == "gemini_flash_lite":
            result = get_gemini_flash_prediction(image_path, model_type="flash-lite", prompt_strategy=prompt_strategy)
        elif model_name == "gemini_pro":
            result = get_gemini_pro_prediction(image_path)
        elif model_name == "llama_90b":
            result = get_llama_90b_prediction(image_path)
        elif model_name == "llama_maverick":
            result = get_llama_maverick_prediction(image_path)
        elif model_name == "llama_scout":
            result = get_llama_scout_prediction(image_path)
        elif model_name == "mistral":
            result = get_mistral_prediction(image_path)
        elif model_name == "granite_vision":
            result = get_granite_vision_prediction(image_path)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        response_time = time.time() - start_time
        return result, response_time, None  # No token usage tracking for these models
    except Exception as e:
        logging.error(f"Error getting prediction from {model_name} for {image_path}: {e}")
        return {"error": str(e)}, time.time() - start_time, None

def handle_result(model_name, prediction_result, true_letter, image_path, response_time=None, token_usage=None):
    """Handles the prediction result dictionary for any model."""
    results[model_name]["total"] += 1
    
    if response_time is not None:
        results[model_name]["response_times"].append(response_time)
    
    if token_usage is not None:
        results[model_name]["token_usage"].append(token_usage)

    if "error" in prediction_result:
        results[model_name]["errors"] += 1
        error_msg = prediction_result["error"]
        if "rate limit" in error_msg.lower():
             results[model_name]["rate_limit_errors"] += 1
        logging.error(f"{model_name} error for {image_path}: {error_msg}")
        return
    
    if "visibility_failed" in prediction_result:
        results[model_name]["visibility_failed"] += 1
        logging.warning(f"{model_name} visibility check failed for {image_path}")
        return

    predicted_letter = prediction_result.get("letter", "").upper()
    if predicted_letter == true_letter:
        results[model_name]["correct"] += 1
    else:
        if true_letter not in misclassified[model_name]:
             misclassified[model_name][true_letter] = []
        misclassified[model_name][true_letter].append({
            "image": image_path,
            "predicted": predicted_letter,
            "confidence": prediction_result.get("confidence", 0),
            "feedback": prediction_result.get("feedback", "")
        })

def evaluate_models(dataset_path, sample_size=2, output_dir="evaluation_results"):
    """Main evaluation function."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset sample
    image_paths = load_dataset_sample(dataset_path, sample_size)
    if not image_paths:
        logging.error("No images to evaluate. Exiting.")
        return

    # Define prompting strategies to test
    prompt_strategies = ["zero_shot", "few_shot", "chain_of_thought"]
    
    # Evaluate each model
    for model_name in MODELS_TO_EVALUATE:
        logging.info(f"\nEvaluating {model_name}...")
        
        # For Gemini models, test each prompting strategy
        if model_name in ["gemini_flash", "gemini_flash_lite"]:
            for strategy in prompt_strategies:
                logging.info(f"\nTesting {model_name} with {strategy} prompting...")
                for image_path, true_letter in tqdm(image_paths, desc=f"Processing {model_name} ({strategy})"):
                    prediction_result, response_time, token_usage = get_prediction(model_name, image_path, strategy)
                    handle_result(model_name, prediction_result, true_letter, image_path, response_time, token_usage)
                    
                    # Add delay between requests to respect rate limits
                    time.sleep(2)
        else:
            # For other models, use default prompting
            for image_path, true_letter in tqdm(image_paths, desc=f"Processing {model_name}"):
                prediction_result, response_time, token_usage = get_prediction(model_name, image_path)
                handle_result(model_name, prediction_result, true_letter, image_path, response_time, token_usage)
                
                # Add delay between requests to respect rate limits
                time.sleep(2)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    misclassified_file = os.path.join(output_dir, f"misclassified_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(misclassified_file, 'w') as f:
        json.dump(misclassified, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    for model_name in MODELS_TO_EVALUATE:
        model_results = results[model_name]
        accuracy = (model_results["correct"] / model_results["total"] * 100) if model_results["total"] > 0 else 0
        avg_time = sum(model_results["response_times"]) / len(model_results["response_times"]) if model_results["response_times"] else 0
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Total images: {model_results['total']}")
        print(f"  Correct: {model_results['correct']}")
        print(f"  Errors: {model_results['errors']}")
        print(f"  Visibility failed: {model_results['visibility_failed']}")
        print(f"  Rate limit errors: {model_results['rate_limit_errors']}")
        print(f"  Average response time: {avg_time:.2f}s")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Misclassified examples saved to: {misclassified_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate ASL recognition models')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--sample-size', type=int, default=2,
                        help='Number of images to sample per class (default: 2)')
    parser.add_argument('--output-dir', type=str, default="evaluation_results",
                        help='Directory to save evaluation results (default: evaluation_results)')
    
    args = parser.parse_args()
    
    evaluate_models(args.dataset, args.sample_size, args.output_dir)

if __name__ == "__main__":
    main() 
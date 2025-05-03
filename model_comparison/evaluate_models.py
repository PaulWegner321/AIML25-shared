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
import numpy as np
import sys
import re

# --- Import Model Predictors --- #
try:
    # Add the project root to Python path for backend imports
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_root)
    
    # Now we can import from backend
    from backend.app.models.sign_evaluator import ColorASLCNN
except ImportError as e:
    logging.error(f"Failed to import ColorASLCNN: {e}")
    logging.error("Please ensure evaluate_models.py is run from a location where 'backend' is a reachable package, or adjust the import path.")
    ColorASLCNN = None

# Keep these imports active
try:
    from test_gemini_2_flash import get_asl_prediction as get_gemini_flash_prediction
    from test_gemini_2_flash_lite import get_asl_prediction as get_gemini_flash_lite_prediction
    from test_gemini_1_5_pro import get_asl_prediction as get_gemini_pro_prediction
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
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend', '.env')
load_dotenv(dotenv_path=dotenv_path)
logging.info(f"Attempting to load .env file from: {dotenv_path}")

# Debug: Print all environment variables (excluding sensitive ones)
logging.info("Available environment variables:")
for key in os.environ:
    if 'key' not in key.lower() and 'secret' not in key.lower() and 'token' not in key.lower():
        logging.info(f"  {key} = {os.environ[key]}")
    else:
        logging.info(f"  {key} = [REDACTED]")

# --- Configuration & Setup ---
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate multiple ASL recognition models')
    parser.add_argument('--dataset_path', type=str, default='/Users/henrikjacobsen/Desktop/CBS/Semester 2/Artifical Intelligence and Machine Learning/Final Project/AIML25-shared/model_comparison/data',
                       help='Path to the dataset directory (default: project data directory)')
    parser.add_argument('--sample_size', type=int, default=1,
                       help='Number of images to sample from each letter')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    dataset_path = args.dataset_path
    logging.info(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        logging.error(f"Dataset path not found or not a directory: {dataset_path}")
        exit(1)

    # Get all image files
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        logging.error("No images found in dataset directory")
        exit(1)

    # Group images by letter
    images_by_letter = {}
    for image_path in image_files:
        letter = os.path.basename(os.path.dirname(image_path))
        if letter not in images_by_letter:
            images_by_letter[letter] = []
        images_by_letter[letter].append(image_path)

    # Sample images
    sampled_images = []
    for letter, images in images_by_letter.items():
        if len(images) > args.sample_size:
            sampled_images.extend(random.sample(images, args.sample_size))
        else:
            sampled_images.extend(images)

    if not sampled_images:
        logging.error("No images loaded for evaluation")
        exit(1)

    logging.info(f"Loaded {len(sampled_images)} images for evaluation")

    # For testing, only use one letter
    VALID_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # All ASL letters
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

    # Define available prompt strategies
    PROMPT_STRATEGIES = [
        "zero_shot"  # Only test with zero-shot for now
    ]

    # Global variables for tracking results
    results = {model: {
        "correct": 0, 
        "total": 0, 
        "errors": 0, 
        "visibility_failed": 0, 
        "rate_limit_errors": 0,
        "response_times": [],
        "token_usage": [],
        "prompt_strategy_results": {strategy: {
            "correct": 0,
            "total": 0,
            "response_times": [],
            "token_usage": []
        } for strategy in PROMPT_STRATEGIES}
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

    def load_dataset_sample(dataset_path_str, sample_size=1):
        """Load a single random image from the dataset."""
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        
        all_images = []
        for letter_dir in dataset_path.iterdir():
            if letter_dir.is_dir():
                letter = letter_dir.name
                images = list(letter_dir.glob("*.jpg"))
                if images:
                    all_images.extend([(str(img_path), letter) for img_path in images])
        
        if not all_images:
            raise ValueError("No images found in the dataset")
        
        # Return a single random image
        return [random.choice(all_images)]

    def get_prediction(model_name, image_path, prompt_strategy="zero_shot"):
        """Get prediction from the specified model."""
        start_time = time.time()
        try:
            if model_name == "gpt4_turbo":
                result = get_gpt4_turbo_prediction(image_path, prompt_strategy=prompt_strategy)
            elif model_name == "gpt4o":
                result = get_gpt4o_prediction(image_path, prompt_strategy=prompt_strategy)
            elif model_name == "gemini_flash":
                result = get_gemini_flash_prediction(image_path, model_type="flash", prompt_strategy=prompt_strategy)
            elif model_name == "gemini_flash_lite":
                result = get_gemini_flash_lite_prediction(image_path, prompt_strategy=prompt_strategy)
            elif model_name == "gemini_pro":
                result = get_gemini_pro_prediction(image_path, prompt_strategy=prompt_strategy)
            elif model_name == "llama_90b":
                result = get_llama_90b_prediction(image_path, strategy=prompt_strategy)
            elif model_name == "llama_maverick":
                result = get_llama_maverick_prediction(image_path, strategy=prompt_strategy)
            elif model_name == "llama_scout":
                result = get_llama_scout_prediction(image_path, strategy=prompt_strategy)
            elif model_name == "mistral":
                result = get_mistral_prediction(image_path, strategy=prompt_strategy)
            elif model_name == "granite_vision":
                result = get_granite_vision_prediction(image_path, prompt_strategy=prompt_strategy)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            response_time = time.time() - start_time
            
            # Extract token usage if available
            token_usage = 0
            if isinstance(result, dict):
                if "metadata" in result:
                    if "total_tokens" in result["metadata"]:
                        token_usage = result["metadata"]["total_tokens"]
                    elif "tokens" in result["metadata"] and "total" in result["metadata"]["tokens"]:
                        token_usage = result["metadata"]["tokens"]["total"]
                
                # If the result has a "prediction" key, use that as the actual result
                if "prediction" in result:
                    result = result["prediction"]
            
            return result, response_time, token_usage
        except Exception as e:
            logging.error(f"Error getting prediction from {model_name} for {image_path}: {e}")
            return {"error": str(e)}, time.time() - start_time, 0

    def handle_result(model_name, prediction_result, true_letter, image_path, response_time=None, token_usage=None, prompt_strategy="zero_shot"):
        """Handle and process model prediction results."""
        try:
            # Handle string responses
            if isinstance(prediction_result, str):
                # Clean the response string
                raw = prediction_result.replace("```json", "").replace("```", "").strip()
                try:
                    prediction_result = json.loads(raw)
                except json.JSONDecodeError:
                    # Try to extract JSON-like content
                    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                    if json_match:
                        prediction_result = json.loads(json_match.group())
                    else:
                        return {
                            "model": model_name,
                            "error": "Invalid JSON response",
                            "raw_response": prediction_result,
                            "metadata": {
                                "response_time_seconds": response_time,
                                "token_usage": token_usage,
                                "prompt_strategy": prompt_strategy
                            }
                        }

            # Handle error responses
            if isinstance(prediction_result, dict) and "error" in prediction_result:
                return {
                    "model": model_name,
                    "error": prediction_result["error"],
                    "metadata": {
                        "response_time_seconds": response_time,
                        "token_usage": token_usage,
                        "prompt_strategy": prompt_strategy
                    }
                }

            # Extract prediction and confidence
            prediction = prediction_result.get("letter", "").upper()
            confidence = prediction_result.get("confidence", 0)
            
            # Convert confidence to float if it's a string
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0

            # Validate prediction
            if not prediction or prediction not in VALID_CLASSES:
                return {
                    "model": model_name,
                    "error": f"Invalid prediction: {prediction}",
                    "metadata": {
                        "response_time_seconds": response_time,
                        "token_usage": token_usage,
                        "prompt_strategy": prompt_strategy
                    }
                }

            # Calculate accuracy
            is_correct = prediction == true_letter.upper()
            
            return {
                "model": model_name,
                "prediction": prediction,
                "true_letter": true_letter.upper(),
                "is_correct": is_correct,
                "confidence": confidence,
                "feedback": prediction_result.get("feedback", ""),
                "metadata": {
                    "response_time_seconds": response_time,
                    "token_usage": token_usage,
                    "prompt_strategy": prompt_strategy,
                    "image_path": str(image_path)
                }
            }

        except Exception as e:
            return {
                "model": model_name,
                "error": f"Error processing result: {str(e)}",
                "raw_response": str(prediction_result),
                "metadata": {
                    "response_time_seconds": response_time,
                    "token_usage": token_usage,
                    "prompt_strategy": prompt_strategy
                }
            }

    def calculate_statistics(results, misclassified):
        """Calculate statistics for each model."""
        stats = {}
        for model_name, model_results in results.items():
            total = model_results["total"]
            correct = model_results["correct"]
            errors = model_results["errors"]
            visibility_failed = model_results["visibility_failed"]
            rate_limit_errors = model_results["rate_limit_errors"]
            
            # Calculate rates (handle division by zero)
            accuracy = correct / total if total > 0 else 0
            error_rate = errors / total if total > 0 else 0
            visibility_failure_rate = visibility_failed / total if total > 0 else 0
            rate_limit_error_rate = rate_limit_errors / total if total > 0 else 0
            
            # Calculate response time statistics
            response_times = model_results["response_times"]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            std_response_time = (sum((x - avg_response_time) ** 2 for x in response_times) / len(response_times)) ** 0.5 if response_times else 0
            
            # Calculate token usage statistics
            token_usage = model_results["token_usage"]
            avg_token_usage = sum(token_usage) / len(token_usage) if token_usage else 0
            std_token_usage = (sum((x - avg_token_usage) ** 2 for x in token_usage) / len(token_usage)) ** 0.5 if token_usage else 0
            
            # Calculate prompt strategy results
            prompt_strategy_results = {}
            for strategy, strategy_results in model_results["prompt_strategy_results"].items():
                strategy_total = strategy_results["total"]
                strategy_correct = strategy_results["correct"]
                strategy_response_times = strategy_results["response_times"]
                strategy_token_usage = strategy_results["token_usage"]
                
                strategy_accuracy = strategy_correct / strategy_total if strategy_total > 0 else 0
                strategy_avg_response_time = sum(strategy_response_times) / len(strategy_response_times) if strategy_response_times else 0
                strategy_avg_token_usage = sum(strategy_token_usage) / len(strategy_token_usage) if strategy_token_usage else 0
                
                prompt_strategy_results[strategy] = {
                    "accuracy": strategy_accuracy,
                    "avg_response_time": strategy_avg_response_time,
                    "avg_token_usage": strategy_avg_token_usage
                }
            
            stats[model_name] = {
                "accuracy": accuracy,
                "error_rate": error_rate,
                "visibility_failure_rate": visibility_failure_rate,
                "rate_limit_error_rate": rate_limit_error_rate,
                "avg_response_time": avg_response_time,
                "std_response_time": std_response_time,
                "avg_token_usage": avg_token_usage,
                "std_token_usage": std_token_usage,
                "prompt_strategy_results": prompt_strategy_results
            }
        
        return stats

    def evaluate_models(dataset_path, sample_size=1, output_dir="evaluation_results"):
        """Evaluates all models on the dataset with all prompt strategies."""
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset sample - get one random image
        image_paths = load_dataset_sample(dataset_path, sample_size)
        if not image_paths:
            logging.error("No images loaded for evaluation")
            return
        
        # Get the single image and its true letter
        image_path, true_letter = image_paths[0]
        logging.info(f"\nEvaluating all models on image: {image_path} (true letter: {true_letter})")
        
        # Initialize results tracking
        results = {model: {
            "correct": 0,
            "total": 0,
            "errors": 0,
            "visibility_failed": 0,
            "rate_limit_errors": 0,
            "response_times": [],
            "token_usage": [],
            "prompt_strategy_results": {strategy: {
                "correct": 0,
                "total": 0,
                "response_times": [],
                "token_usage": []
            } for strategy in PROMPT_STRATEGIES}
        } for model in MODELS_TO_EVALUATE}
        
        # Initialize misclassified dictionary for each model and letter
        misclassified = {model: {letter: [] for letter in VALID_CLASSES} for model in MODELS_TO_EVALUATE}
        
        # Evaluate each model with each prompt strategy
        for model_name in MODELS_TO_EVALUATE:
            logging.info(f"\nEvaluating {model_name}...")
            
            for prompt_strategy in PROMPT_STRATEGIES:
                logging.info(f"\nTesting {model_name} with {prompt_strategy} prompting...")
                
                prediction_result, response_time, token_usage = get_prediction(
                    model_name,
                    image_path,
                    prompt_strategy=prompt_strategy
                )
                
                result = handle_result(
                    model_name,
                    prediction_result,
                    true_letter,
                    image_path,
                    response_time,
                    token_usage,
                    prompt_strategy=prompt_strategy
                )
                
                # Add delay between requests to respect rate limits
                time.sleep(2)
                
                # Update results
                results[model_name]["total"] += 1
                results[model_name]["response_times"].append(response_time)
                results[model_name]["token_usage"].append(token_usage)
                
                results[model_name]["prompt_strategy_results"][prompt_strategy]["total"] += 1
                results[model_name]["prompt_strategy_results"][prompt_strategy]["response_times"].append(response_time)
                results[model_name]["prompt_strategy_results"][prompt_strategy]["token_usage"].append(token_usage)
                
                if "error" in result:
                    logging.error(f"Error for {model_name}: {result['error']}")
                    results[model_name]["errors"] += 1
                    if "cannot see the image" in result["error"].lower():
                        results[model_name]["visibility_failed"] += 1
                    elif "rate limit" in result["error"].lower():
                        results[model_name]["rate_limit_errors"] += 1
                else:
                    if result["is_correct"]:
                        results[model_name]["correct"] += 1
                        results[model_name]["prompt_strategy_results"][prompt_strategy]["correct"] += 1
                        logging.info(f"Correct prediction for {model_name}: {result['prediction']}")
                    else:
                        misclassified[model_name][true_letter].append((image_path, result["prediction"]))
                        logging.info(f"Misclassification for {model_name}: predicted {result['prediction']}, actual {true_letter}")
        
        # Calculate statistics
        stats = calculate_statistics(results, misclassified)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "dataset_path": str(dataset_path),
                "sample_size": sample_size,
                "results": results,
                "misclassified": misclassified,
                "statistics": stats
            }, f, indent=2)
        
        logging.info(f"\nResults saved to {results_file}")
        logging.info("\nModel Performance Summary:")
        for model_name, model_stats in stats.items():
            logging.info(f"\n{model_name}:")
            logging.info(f"  Overall Accuracy: {model_stats['accuracy']:.2%}")
            logging.info(f"  Error Rate: {model_stats['error_rate']:.2%}")
            logging.info(f"  Visibility Failure Rate: {model_stats['visibility_failure_rate']:.2%}")
            logging.info(f"  Rate Limit Error Rate: {model_stats['rate_limit_error_rate']:.2%}")
            logging.info(f"  Average Response Time: {model_stats['avg_response_time']:.2f}s")
            logging.info(f"  Response Time Std Dev: {model_stats['std_response_time']:.2f}s")
            logging.info(f"  Average Token Usage: {model_stats['avg_token_usage']:.0f}")
            logging.info(f"  Token Usage Std Dev: {model_stats['std_token_usage']:.0f}")
            
            logging.info("\n  Prompt Strategy Results:")
            for strategy, strategy_stats in model_stats["prompt_strategy_results"].items():
                logging.info(f"    {strategy}:")
                logging.info(f"      Accuracy: {strategy_stats['accuracy']:.2%}")
                logging.info(f"      Average Response Time: {strategy_stats['avg_response_time']:.2f}s")
                logging.info(f"      Average Token Usage: {strategy_stats['avg_token_usage']:.0f}")

    evaluate_models(
        dataset_path,
        args.sample_size,
        args.output_dir
    )

if __name__ == "__main__":
    main() 
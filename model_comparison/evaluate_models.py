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
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
AVAILABLE_MODELS = {}

try:
    from test_gpt4_turbo import get_asl_prediction as get_gpt4_turbo_prediction
    AVAILABLE_MODELS["gpt4_turbo"] = get_gpt4_turbo_prediction
except ImportError as e:
    logging.warning(f"GPT-4 Turbo model not available: {e}")

try:
    from test_gpt4o import get_asl_prediction as get_gpt4o_prediction
    AVAILABLE_MODELS["gpt4o"] = get_gpt4o_prediction
except ImportError as e:
    logging.warning(f"GPT-4 Vision model not available: {e}")

try:
    from test_gemini_2_flash import get_asl_prediction as get_gemini_flash_prediction
    AVAILABLE_MODELS["gemini_flash"] = get_gemini_flash_prediction
except ImportError as e:
    logging.warning(f"Gemini Flash model not available: {e}")

try:
    from test_gemini_2_flash_lite import get_asl_prediction as get_gemini_flash_lite_prediction
    AVAILABLE_MODELS["gemini_flash_lite"] = get_gemini_flash_lite_prediction
except ImportError as e:
    logging.warning(f"Gemini Flash Lite model not available: {e}")

try:
    from test_llama_90b_vision import get_asl_prediction as get_llama_90b_prediction
    AVAILABLE_MODELS["llama_90b"] = get_llama_90b_prediction
except ImportError as e:
    logging.warning(f"Llama 90B Vision model not available: {e}")

try:
    from test_llama_maverick_17b import get_asl_prediction as get_llama_maverick_prediction
    AVAILABLE_MODELS["llama_maverick"] = get_llama_maverick_prediction
except ImportError as e:
    logging.warning(f"Llama Maverick model not available: {e}")

try:
    from test_llama_scout_17b import get_asl_prediction as get_llama_scout_prediction
    AVAILABLE_MODELS["llama_scout"] = get_llama_scout_prediction
except ImportError as e:
    logging.warning(f"Llama Scout model not available: {e}")

try:
    from test_pixtral_12b import get_asl_prediction as get_mistral_prediction
    AVAILABLE_MODELS["mistral"] = get_mistral_prediction
except ImportError as e:
    logging.warning(f"Mistral (Pixtral) model not available: {e}")

try:
    from test_granite_vision import get_asl_prediction as get_granite_vision_prediction
    AVAILABLE_MODELS["granite_vision"] = get_granite_vision_prediction
except ImportError as e:
    logging.warning(f"Granite Vision model not available: {e}")

# Define models to evaluate based on availability
MODELS_TO_EVALUATE = list(AVAILABLE_MODELS.keys())
logging.info(f"Available models: {MODELS_TO_EVALUATE}")

if not MODELS_TO_EVALUATE:
    logging.warning("No models available for evaluation. Please check your environment variables and dependencies.")

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

# Define valid classes and class mapping
VALID_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # All ASL letters
CLASS_MAP = {label: i for i, label in enumerate(VALID_CLASSES)}
INDEX_TO_CLASS = {i: label for label, i in CLASS_MAP.items()}

# Define available prompt strategies
PROMPT_STRATEGIES = [
    "zero_shot"  # Only test with zero-shot for now
]

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
        return watsonx_token

    logging.info("Refreshing WatsonX authentication token...")
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    try:
        response = requests.post(auth_url, headers=headers, data=data, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        watsonx_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        watsonx_token_expiry = now + expires_in - 300  # 5 min buffer
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
                img = img.resize(resize_dim)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_str = ''.join(base64_str.split())
            return base64_str
    except Exception as e:
        logging.error(f"Error encoding/resizing image {image_path}: {e}")
        traceback.print_exc()
        return None

def load_dataset_sample(dataset_path_str, sample_size=1):
    """Load a single specific image for testing."""
    test_image = "/Users/henrikjacobsen/Desktop/CBS/Semester 2/Artifical Intelligence and Machine Learning/Final Project/AIML25-shared/model_comparison/data/V/V_0_20250428_114109_flipped.jpg"
    true_letter = "V"  # The letter V is the true label for this image
    
    logging.info(f"Using single test image: {test_image} (true letter: {true_letter})")
    return [(test_image, true_letter)]

def get_prediction(model_name, image_path, prompt_strategy="zero_shot"):
    """Get prediction from the specified model."""
    start_time = time.time()
    try:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} is not available")
            
        # Get the prediction function for this model
        predict_func = AVAILABLE_MODELS[model_name]
        
        # Handle different parameter naming conventions
        if model_name in ["gemini_flash", "gemini_flash_lite", "granite_vision"]:
            result = predict_func(image_path, prompt_strategy=prompt_strategy)
        elif model_name in ["llama_90b", "llama_maverick", "llama_scout", "mistral"]:
            result = predict_func(image_path, strategy=prompt_strategy)
        else:  # GPT-4 models
            result = predict_func(image_path, prompt_strategy=prompt_strategy)
        
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
        logging.error(f"Error getting prediction from {model_name}: {str(e)}")
        response_time = time.time() - start_time
        return {"error": str(e)}, response_time, 0

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

def calculate_statistics(results: Dict[str, Any], misclassified: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Any]:
    """Calculate statistics for each model."""
    stats = {}
    for model_name, model_results in results.items():
        if isinstance(model_results, dict):
            total = model_results.get("total", 0)
            correct = model_results.get("correct", 0)
            errors = model_results.get("errors", 0)
            visibility_failed = model_results.get("visibility_failed", 0)
            rate_limit_errors = model_results.get("rate_limit_errors", 0)
            
            # Calculate rates (handle division by zero)
            accuracy = correct / total if total > 0 else 0
            error_rate = errors / total if total > 0 else 0
            visibility_failure_rate = visibility_failed / total if total > 0 else 0
            rate_limit_error_rate = rate_limit_errors / total if total > 0 else 0
            
            # Calculate response time statistics
            response_times = model_results.get("response_times", [])
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            std_response_time = (sum((x - avg_response_time) ** 2 for x in response_times) / len(response_times)) ** 0.5 if response_times else 0
            
            # Calculate token usage statistics
            token_usage = model_results.get("token_usage", [])
            avg_token_usage = sum(token_usage) / len(token_usage) if token_usage else 0
            std_token_usage = (sum((x - avg_token_usage) ** 2 for x in token_usage) / len(token_usage)) ** 0.5 if token_usage else 0
            
            # Calculate prompt strategy results
            prompt_strategy_results = {}
            for strategy, strategy_results in model_results.get("prompt_strategy_results", {}).items():
                strategy_total = strategy_results.get("total", 0)
                strategy_correct = strategy_results.get("correct", 0)
                strategy_response_times = strategy_results.get("response_times", [])
                strategy_token_usage = strategy_results.get("token_usage", [])
                
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

def calculate_metrics(predictions: List[str], ground_truth: List[str], prediction_probs: np.ndarray = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    # Convert to numpy arrays for easier calculations
    preds = np.array(predictions)
    truth = np.array(ground_truth)
    
    # 1. Basic Accuracy
    metrics["accuracy"] = np.mean(preds == truth)
    
    # 2. Top-k Accuracy (if prediction probabilities are available)
    if prediction_probs is not None:
        for k in [2, 3, 5]:
            metrics[f"top_{k}_accuracy"] = top_k_accuracy_score(truth, prediction_probs, k=k)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(truth, preds, labels=VALID_CLASSES)
    metrics["confusion_matrix"] = cm.tolist()
    
    # 4. Classification Report
    report = classification_report(truth, preds, labels=VALID_CLASSES, output_dict=True)
    metrics["classification_report"] = report
    
    # 5. Macro and Micro Averages
    metrics["macro_avg"] = {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1_score": report["macro avg"]["f1-score"]
    }
    
    metrics["micro_avg"] = {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }
    
    return metrics

def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: str) -> str:
    """Plot and save confusion matrix as a heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=VALID_CLASSES,
                yticklabels=VALID_CLASSES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def evaluate_subset(predictions: List[str], ground_truth: List[str], image_paths: List[str], subset_type: str) -> Dict[str, Any]:
    """Evaluate model performance on a specific subset of images."""
    subset_indices = []
    for i, path in enumerate(image_paths):
        if subset_type == 'grayscale' and 'grayscale' in path.lower():
            subset_indices.append(i)
        elif subset_type == 'flipped' and 'flipped' in path.lower():
            subset_indices.append(i)
    
    if not subset_indices:
        return {
            "count": 0,
            "accuracy": None,
            "error_rate": None
        }
    
    # Only include indices that have corresponding predictions
    valid_indices = [i for i in subset_indices if i < len(predictions)]
    
    if not valid_indices:
        return {
            "count": 0,
            "accuracy": None,
            "error_rate": None
        }
    
    subset_preds = [predictions[i] for i in valid_indices]
    subset_truth = [ground_truth[i] for i in valid_indices]
    
    accuracy = np.mean(np.array(subset_preds) == np.array(subset_truth))
    error_rate = 1 - accuracy
    
    return {
        "count": len(valid_indices),
        "accuracy": accuracy,
        "error_rate": error_rate
    }

def evaluate_models(dataset_path: str, sample_size: int = 1, output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """Evaluate all models on the dataset with comprehensive metrics."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    
    # Load and sample images
    sampled_images = load_dataset_sample(dataset_path, sample_size)
    logging.info(f"Evaluating {len(sampled_images)} images across all models")
    
    # Initialize results tracking
    results = {model: {
        "correct": 0,
        "total": 0,
        "errors": 0,
        "visibility_failed": 0,
        "rate_limit_errors": 0,
        "response_times": [],
        "token_usage": [],
        "predictions": [],
        "ground_truth": [],
        "image_paths": [],
        "prompt_strategy_results": {strategy: {
            "correct": 0,
            "total": 0,
            "response_times": [],
            "token_usage": [],
            "predictions": [],
            "ground_truth": []
        } for strategy in PROMPT_STRATEGIES}
    } for model in MODELS_TO_EVALUATE}
    
    misclassified = {model: {letter: [] for letter in VALID_CLASSES} for model in MODELS_TO_EVALUATE}
    
    # Evaluate each image with each model
    for image_path, true_letter in sampled_images:
        logging.info(f"\nEvaluating image: {image_path} (true letter: {true_letter})")
        
        for model_name in MODELS_TO_EVALUATE:
            logging.info(f"\nTesting model: {model_name}")
            for prompt_strategy in PROMPT_STRATEGIES:
                try:
                    prediction_result, response_time, token_usage = get_prediction(model_name, image_path, prompt_strategy)
                    result = handle_result(
                        model_name,
                        prediction_result,
                        true_letter,
                        image_path,
                        response_time=response_time,
                        token_usage=token_usage,
                        prompt_strategy=prompt_strategy
                    )
                    
                    # Update results
                    results[model_name]["total"] += 1
                    results[model_name]["response_times"].append(response_time)
                    results[model_name]["token_usage"].append(token_usage)
                    results[model_name]["image_paths"].append(image_path)
                    
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
                        results[model_name]["predictions"].append(result["prediction"])
                        results[model_name]["ground_truth"].append(result["true_letter"])
                        results[model_name]["prompt_strategy_results"][prompt_strategy]["predictions"].append(result["prediction"])
                        results[model_name]["prompt_strategy_results"][prompt_strategy]["ground_truth"].append(result["true_letter"])
                        
                        if result["is_correct"]:
                            results[model_name]["correct"] += 1
                            results[model_name]["prompt_strategy_results"][prompt_strategy]["correct"] += 1
                            logging.info(f"Correct prediction for {model_name}: {result['prediction']}")
                        else:
                            misclassified[model_name][true_letter].append((image_path, result["prediction"]))
                            logging.info(f"Misclassification for {model_name}: predicted {result['prediction']}, actual {true_letter}")
                except Exception as e:
                    logging.error(f"Error evaluating {model_name} on {image_path}: {str(e)}")
                    results[model_name]["errors"] += 1
                    results[model_name]["total"] += 1
    
    # Calculate comprehensive metrics for each model
    for model_name in MODELS_TO_EVALUATE:
        if results[model_name]["predictions"]:
            # Calculate basic statistics
            stats = calculate_statistics(results[model_name], misclassified[model_name])
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(
                results[model_name]["predictions"],
                results[model_name]["ground_truth"]
            )
            
            # Calculate subset accuracies
            subset_metrics = {
                "grayscale": evaluate_subset(
                    results[model_name]["predictions"],
                    results[model_name]["ground_truth"],
                    results[model_name]["image_paths"],
                    "grayscale"
                ),
                "flipped": evaluate_subset(
                    results[model_name]["predictions"],
                    results[model_name]["ground_truth"],
                    results[model_name]["image_paths"],
                    "flipped"
                )
            }
            
            # Calculate tokens per second
            avg_response_time = np.mean(results[model_name]["response_times"])
            avg_token_usage = np.mean(results[model_name]["token_usage"])
            tokens_per_second = avg_token_usage / avg_response_time if avg_response_time > 0 else 0
            
            # Update results with new metrics
            results[model_name]["metrics"] = {
                **metrics,
                "tokens_per_second": tokens_per_second,
                "subset_metrics": subset_metrics
            }
            
            # Plot confusion matrix
            cm_path = plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                model_name,
                output_dir
            )
            results[model_name]["metrics"]["confusion_matrix_plot"] = cm_path
    
    # Save results
    evaluation_results = {
        "timestamp": timestamp,
        "dataset_path": dataset_path,
        "sample_size": sample_size,
        "results": results,
        "misclassified": misclassified
    }
    
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logging.info(f"\nEvaluation results saved to: {output_file}")
    return evaluation_results

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

    # Run evaluation
    evaluate_models(args.dataset_path, args.sample_size, args.output_dir)

if __name__ == "__main__":
    main() 
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
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import string

# Set up logging
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_dir = "evaluation_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logging.info(f"Logging to file: {log_file}")

# Initialize global results dictionary
results = {
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "dataset_path": "",
    "sample_size": 0,
    "results": {}
}

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
AVAILABLE_MODELS = {}

try:
    from test_gpt4_turbo import get_asl_prediction as get_gpt4_turbo_prediction
    AVAILABLE_MODELS["gpt4_turbo"] = get_gpt4_turbo_prediction
except ImportError as e:
    logging.warning(f"GPT-4 Turbo model not available: {e}")

try:
    from test_gpt4o import get_asl_prediction as get_gpt4o_prediction
    AVAILABLE_MODELS["gpt4o"] = get_gpt4o_prediction
except ImportError as e:
    logging.warning(f"GPT-4 Vision model not available: {e}")

try:
    from test_gemini_2_flash import get_asl_prediction as get_gemini_flash_prediction
    AVAILABLE_MODELS["gemini_flash"] = get_gemini_flash_prediction
except ImportError as e:
    logging.warning(f"Gemini Flash model not available: {e}")

try:
    from test_gemini_2_flash_lite import get_asl_prediction as get_gemini_flash_lite_prediction
    AVAILABLE_MODELS["gemini_flash_lite"] = get_gemini_flash_lite_prediction
except ImportError as e:
    logging.warning(f"Gemini Flash Lite model not available: {e}")

try:
    from test_llama_90b_vision import get_asl_prediction as get_llama_90b_prediction
    AVAILABLE_MODELS["llama_90b"] = get_llama_90b_prediction
except ImportError as e:
    logging.warning(f"Llama 90B Vision model not available: {e}")

try:
    from test_llama_maverick_17b import get_asl_prediction as get_llama_maverick_prediction
    AVAILABLE_MODELS["llama_maverick"] = get_llama_maverick_prediction
except ImportError as e:
    logging.warning(f"Llama Maverick model not available: {e}")

try:
    from test_llama_scout_17b import get_asl_prediction as get_llama_scout_prediction
    AVAILABLE_MODELS["llama_scout"] = get_llama_scout_prediction
except ImportError as e:
    logging.warning(f"Llama Scout model not available: {e}")

try:
    from test_pixtral_12b import get_asl_prediction as get_mistral_prediction
    AVAILABLE_MODELS["mistral"] = get_mistral_prediction
except ImportError as e:
    logging.warning(f"Mistral (Pixtral) model not available: {e}")

try:
    from test_granite_vision import get_asl_prediction as get_granite_vision_prediction
    AVAILABLE_MODELS["granite_vision"] = get_granite_vision_prediction
except ImportError as e:
    logging.warning(f"Granite Vision model not available: {e}")

# Define models to evaluate based on availability
MODELS_TO_EVALUATE = list(AVAILABLE_MODELS.keys())
logging.info(f"Available models: {MODELS_TO_EVALUATE}")

if not MODELS_TO_EVALUATE:
    logging.warning("No models available for evaluation. Please check your environment variables and dependencies.")

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

# Define valid classes and class mapping
VALID_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # All ASL letters
CLASS_MAP = {label: i for i, label in enumerate(VALID_CLASSES)}
INDEX_TO_CLASS = {i: label for label, i in CLASS_MAP.items()}

# Define available prompt strategies
PROMPT_STRATEGIES = [
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "visual_grounding",
    "contrastive"
]

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
        return watsonx_token

    logging.info("Refreshing WatsonX authentication token...")
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    try:
        response = requests.post(auth_url, headers=headers, data=data, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        watsonx_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)
        watsonx_token_expiry = now + expires_in - 300  # 5 min buffer
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
                img = img.resize(resize_dim)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_str = ''.join(base64_str.split())
            return base64_str
    except Exception as e:
        logging.error(f"Error encoding/resizing image {image_path}: {e}")
        traceback.print_exc()
        return None

def load_dataset_sample(dataset_path_str, sample_size=30):
    """Load random samples from the dataset for each letter."""
    dataset_path = Path(dataset_path_str)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    
    samples = []
    for letter in VALID_CLASSES:
        letter_dir = dataset_path / letter
        if not letter_dir.exists():
            logging.warning(f"Directory for letter {letter} not found: {letter_dir}")
            continue
            
        # Get all image files in the directory
        image_files = list(letter_dir.glob("*.jpg"))
        if not image_files:
            logging.warning(f"No images found for letter {letter} in {letter_dir}")
            continue
            
        # Randomly sample images
        if len(image_files) < sample_size:
            logging.warning(f"Not enough images for letter {letter}. Found {len(image_files)}, requested {sample_size}")
            selected_images = image_files
        else:
            selected_images = random.sample(image_files, sample_size)
            
        # Add selected images to samples list
        for img_path in selected_images:
            samples.append((str(img_path), letter))
    
    logging.info(f"Loaded {len(samples)} images across all letters")
    return samples

def get_prediction(model_name, image_path, prompt_strategy="zero_shot"):
    """Get prediction from the specified model."""
    start_time = time.time()
    try:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} is not available")
            
        # Get the prediction function for this model
        predict_func = AVAILABLE_MODELS[model_name]
        
        # Handle different parameter naming conventions
        if model_name in ["gemini_flash", "gemini_flash_lite", "granite_vision"]:
            result = predict_func(image_path, prompt_strategy=prompt_strategy)
        elif model_name in ["llama_90b", "llama_maverick", "llama_scout", "mistral"]:
            result = predict_func(image_path, strategy=prompt_strategy)
        else:  # GPT-4 models
            result = predict_func(image_path, prompt_strategy=prompt_strategy)
        
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
        logging.error(f"Error getting prediction from {model_name}: {str(e)}")
        response_time = time.time() - start_time
        return {"error": str(e)}, response_time, 0

def handle_result(model_name, prediction_result, true_letter, image_path, response_time=None, token_usage=None, prompt_strategy="zero_shot"):
    """Handle prediction result and update global results."""
    if model_name not in results["results"]:
        results["results"][model_name] = {
            "predictions": [],
            "ground_truth": [],
            "response_times": [],
            "token_usage": [],
            "prompt_strategy_results": {}
        }
    
    # Add to overall results
    results["results"][model_name]["predictions"].append(prediction_result)
    results["results"][model_name]["ground_truth"].append(true_letter)
    results["results"][model_name]["response_times"].append(response_time if response_time else 0)
    results["results"][model_name]["token_usage"].append(token_usage if token_usage else 0)
    
    # Add to strategy-specific results
    if prompt_strategy not in results["results"][model_name]["prompt_strategy_results"]:
        results["results"][model_name]["prompt_strategy_results"][prompt_strategy] = {
            "predictions": [],
            "ground_truth": [],
            "response_times": [],
            "token_usage": []
        }
    
    strategy_data = results["results"][model_name]["prompt_strategy_results"][prompt_strategy]
    strategy_data["predictions"].append(prediction_result)
    strategy_data["ground_truth"].append(true_letter)
    strategy_data["response_times"].append(response_time if response_time else 0)
    strategy_data["token_usage"].append(token_usage if token_usage else 0)

def calculate_statistics(results: Dict[str, Any], misclassified: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Any]:
    """Calculate statistics for each model and strategy."""
    for model_name, model_data in results["results"].items():
        # Calculate overall metrics
        predictions = model_data["predictions"]
        ground_truth = model_data["ground_truth"]
        response_times = model_data["response_times"]
        token_usage = model_data["token_usage"]
        
        # Calculate overall statistics
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        total = len(predictions)
        errors = total - correct
        
        model_data.update({
            "correct": correct,
            "total": total,
            "errors": errors,
            "metrics": calculate_metrics(predictions, ground_truth)
        })
        
        # Calculate strategy-specific statistics
        for strategy, strategy_data in model_data["prompt_strategy_results"].items():
            strategy_predictions = strategy_data["predictions"]
            strategy_ground_truth = strategy_data["ground_truth"]
            strategy_response_times = strategy_data["response_times"]
            strategy_token_usage = strategy_data["token_usage"]
            
            strategy_correct = sum(1 for p, g in zip(strategy_predictions, strategy_ground_truth) if p == g)
            strategy_total = len(strategy_predictions)
            
            strategy_data.update({
                "correct": strategy_correct,
                "total": strategy_total,
                "response_times": strategy_response_times,
                "token_usage": strategy_token_usage,
                "metrics": calculate_metrics(strategy_predictions, strategy_ground_truth)
            })
    
    return results

def calculate_metrics(predictions: List[str], ground_truth: List[str], prediction_probs: np.ndarray = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    # Convert to numpy arrays for easier calculations
    preds = np.array(predictions)
    truth = np.array(ground_truth)
    
    # 1. Basic Accuracy
    metrics["accuracy"] = np.mean(preds == truth)
    
    # 2. Top-k Accuracy (if prediction probabilities are available)
    if prediction_probs is not None:
        for k in [2, 3, 5]:
            metrics[f"top_{k}_accuracy"] = top_k_accuracy_score(truth, prediction_probs, k=k)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(truth, preds, labels=VALID_CLASSES)
    metrics["confusion_matrix"] = cm.tolist()
    
    # 4. Classification Report
    report = classification_report(truth, preds, labels=VALID_CLASSES, output_dict=True)
    metrics["classification_report"] = report
    
    # 5. Macro and Micro Averages
    metrics["macro_avg"] = {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1_score": report["macro avg"]["f1-score"]
    }
    
    metrics["micro_avg"] = {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }
    
    return metrics

def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: str) -> str:
    """Plot and save confusion matrix as a heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=VALID_CLASSES,
                yticklabels=VALID_CLASSES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def generate_confusion_matrices(data, output_dir):
    """Generate confusion matrices for each model and strategy."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all possible labels (A-Z)
    labels = list(string.ascii_uppercase)
    
    for model_name, model_data in data['results'].items():
        # Overall model confusion matrix
        cm = confusion_matrix(
            model_data['ground_truth'],
            model_data['predictions'],
            labels=labels
        )
        plot_confusion_matrix(
            cm, 
            labels,
            f'Confusion Matrix - {model_name}',
            os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png')
        )
        
        # Strategy-specific confusion matrices
        if 'prompt_strategy_results' in model_data:
            for strategy, strategy_data in model_data['prompt_strategy_results'].items():
                cm = confusion_matrix(
                    strategy_data['ground_truth'],
                    strategy_data['predictions'],
                    labels=labels
                )
                plot_confusion_matrix(
                    cm,
                    labels,
                    f'Confusion Matrix - {model_name} - {strategy}',
                    os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}_{strategy.lower()}.png')
                )

def evaluate_subset(predictions: List[str], ground_truth: List[str], image_paths: List[str], subset_type: str) -> Dict[str, Any]:
    """Evaluate model performance on a specific subset of images."""
    subset_indices = []
    for i, path in enumerate(image_paths):
        if subset_type == 'grayscale' and 'grayscale' in path.lower():
            subset_indices.append(i)
        elif subset_type == 'flipped' and 'flipped' in path.lower():
            subset_indices.append(i)
    
    if not subset_indices:
        return {
            "count": 0,
            "accuracy": None,
            "error_rate": None
        }
    
    # Only include indices that have corresponding predictions
    valid_indices = [i for i in subset_indices if i < len(predictions)]
    
    if not valid_indices:
        return {
            "count": 0,
            "accuracy": None,
            "error_rate": None
        }
    
    subset_preds = [predictions[i] for i in valid_indices]
    subset_truth = [ground_truth[i] for i in valid_indices]
    
    accuracy = np.mean(np.array(subset_preds) == np.array(subset_truth))
    error_rate = 1 - accuracy
    
    return {
        "count": len(valid_indices),
        "accuracy": accuracy,
        "error_rate": error_rate
    }

def save_intermediate_results(output_dir: str, model_name: str) -> None:
    """Save intermediate results to a temporary file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_file = os.path.join(output_dir, f"temp_results_{model_name}_{timestamp}.json")
    with open(temp_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved intermediate results to {temp_file}")

def evaluate_models(dataset_path: str, sample_size: int = 30, output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """Evaluate all available models on the dataset."""
    # Update global results with dataset info
    results["dataset_path"] = dataset_path
    results["sample_size"] = sample_size
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset samples
    samples = load_dataset_sample(dataset_path, sample_size)
    
    # Evaluate each model
    for model_name in MODELS_TO_EVALUATE:
        logging.info(f"Evaluating model: {model_name}")
        
        # Evaluate each prompt strategy
        for strategy in PROMPT_STRATEGIES:
            logging.info(f"Using prompt strategy: {strategy}")
            
            # Evaluate each sample
            for image_path, true_letter in tqdm(samples, desc=f"{model_name} - {strategy}"):
                try:
                    # Get prediction with timing
                    start_time = time.time()
                    prediction_result = get_prediction(model_name, image_path, strategy)
                    response_time = time.time() - start_time
                    
                    # Get token usage (if available)
                    token_usage = None
                    if hasattr(prediction_result, 'token_usage'):
                        token_usage = prediction_result.token_usage
                    
                    # Handle the result
                    handle_result(
                        model_name,
                        prediction_result,
                        true_letter,
                        image_path,
                        response_time,
                        token_usage,
                        strategy
                    )
                    
                except Exception as e:
                    logging.error(f"Error evaluating {model_name} on {image_path}: {e}")
                    continue
    
    # Calculate final statistics
    final_results = calculate_statistics(results, {})
    
    # Generate confusion matrices
    generate_confusion_matrices(final_results, output_dir)
    
    # Save results
    output_file = os.path.join(output_dir, f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate multiple ASL recognition models')
    parser.add_argument('--dataset_path', type=str, default='/Users/henrikjacobsen/Desktop/CBS/Semester 2/Artifical Intelligence and Machine Learning/Final Project/AIML25-shared/model_comparison/data',
                       help='Path to the dataset directory (default: project data directory)')
    parser.add_argument('--sample_size', type=int, default=30,
                       help='Number of images to sample from each letter')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run a quick test with one random image and zero-shot prompting only')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick_test:
        # For quick test, override the sample loading to get just one random image
        def quick_test_sample(dataset_path_str, sample_size=None):  # Added sample_size parameter
            dataset_path = Path(dataset_path_str)
            all_images = []
            for letter in VALID_CLASSES:
                letter_dir = dataset_path / letter
                if letter_dir.exists():
                    image_files = list(letter_dir.glob("*.jpg"))
                    if image_files:
                        all_images.append((str(random.choice(image_files)), letter))
            if all_images:
                return [random.choice(all_images)]
            return []
        
        # Override the load_dataset_sample function temporarily
        global load_dataset_sample
        original_load_dataset = load_dataset_sample
        load_dataset_sample = quick_test_sample
        
        # Override the prompt strategies temporarily
        global PROMPT_STRATEGIES
        original_strategies = PROMPT_STRATEGIES
        PROMPT_STRATEGIES = ["zero_shot"]
        
        try:
            # Run evaluation with quick test settings
            results = evaluate_models(args.dataset_path, args.sample_size, args.output_dir)
        finally:
            # Restore original functions
            load_dataset_sample = original_load_dataset
            PROMPT_STRATEGIES = original_strategies
    else:
        # Run normal evaluation
        evaluate_models(args.dataset_path, args.sample_size, args.output_dir)

if __name__ == "__main__":
    main() 
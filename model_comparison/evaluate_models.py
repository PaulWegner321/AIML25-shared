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
import string
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

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

# Initialize global variables for token management
watsonx_token = None
watsonx_token_expiry = None

# --- Import Model Predictors --- #

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

# Define valid classes and class mapping
VALID_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # All ASL letters
CLASS_MAP = {label: i for i, label in enumerate(VALID_CLASSES)}
INDEX_TO_CLASS = {i: label for label, i in CLASS_MAP.items()}

# Define available prompt strategies (use all 5)
PROMPT_STRATEGIES = [
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "visual_grounding",
    "contrastive"
]

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
                "metrics": calculate_metrics(strategy_predictions, strategy_ground_truth)
            })
    
    return results

def calculate_metrics(predictions: List[str], ground_truth: List[str], prediction_probs: np.ndarray = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    if not predictions or not ground_truth:
        return {"error": "No predictions or ground truth data available"}
    
    metrics = {}
    
    # Convert to numpy arrays for easier calculations
    preds = np.array(predictions)
    truth = np.array(ground_truth)
    
    # 1. Basic Accuracy
    metrics["accuracy"] = np.mean(preds == truth)
    
    # 2. Confusion Matrix (if we have enough data)
    try:
        cm = confusion_matrix(truth, preds, labels=VALID_CLASSES)
        metrics["confusion_matrix"] = cm.tolist()
    except Exception as e:
        logging.error(f"Error calculating confusion matrix: {e}")
        metrics["confusion_matrix"] = None
    
    # 3. Classification Report (if we have enough data)
    try:
        report = classification_report(truth, preds, labels=VALID_CLASSES, output_dict=True)
        metrics["classification_report"] = report
        
        # Extract macro and weighted averages
        metrics["macro_avg"] = {
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"]
        }
        
        metrics["weighted_avg"] = {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"]
        }
    except Exception as e:
        logging.error(f"Error calculating classification report: {e}")
        metrics["classification_report"] = None
        metrics["macro_avg"] = None
        metrics["weighted_avg"] = None
    
    return metrics

def plot_confusion_matrix(cm, labels, title, output_path):
    """Plot and save a confusion matrix as a heatmap."""
    plt.figure(figsize=(12, 10))
    
    # Normalize the confusion matrix for display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
    
    # Create the heatmap
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def generate_confusion_matrices(data, output_dir):
    """Generate confusion matrices for each model and strategy."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all possible labels (A-Z)
    labels = list(string.ascii_uppercase)
    
    for model_name, model_data in data['results'].items():
        # Check if we have prediction data
        if 'predictions' not in model_data or not model_data['predictions']:
            logging.warning(f"No prediction data for model {model_name}")
            continue
            
        # Overall model confusion matrix
        try:
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
        except Exception as e:
            logging.error(f"Error generating confusion matrix for {model_name}: {e}")
        
        # Strategy-specific confusion matrices
        if 'prompt_strategy_results' in model_data:
            for strategy, strategy_data in model_data['prompt_strategy_results'].items():
                if not strategy_data.get('predictions'):
                    continue
                    
                try:
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
                except Exception as e:
                    logging.error(f"Error generating confusion matrix for {model_name} - {strategy}: {e}")

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
                    prediction_result, response_time, token_usage = get_prediction(model_name, image_path, strategy)
                    
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
            
            # Save intermediate results after each strategy is complete
            save_intermediate_results(output_dir, f"{model_name}_{strategy}")
    
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
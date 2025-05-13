# ASL Recognition Model Comparison

This project evaluates multiple vision-language models on American Sign Language (ASL) recognition tasks. It compares the performance of various large multimodal models including GPT-4o, Gemini, Llama, and others on their ability to correctly identify ASL hand signs.

## Project Structure

```
model_comparison/
├── data/                    # ASL image dataset (A-Z folders)
├── Notebooks/               # Jupyter notebooks for hand in of the paper and understanding how the notebooks run and how the output looks like
├── dataset_creation/        # Tools for creating the ASL dataset
├── evaluation_results/      # Results from model evaluations
├── evaluation_logs/         # Logs from evaluation runs
├── test_*.py                # Individual model test scripts
└── evaluate_models.py       # Main evaluation framework
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up API keys:
   - Create a `.env` file in the `backend/` directory
   - Add your API keys for the models you want to test:
   ```
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   IBM_CLOUD_API_KEY=your_ibm_key
   ```

## Main Components

### Dataset Creation

The `dataset_creation/` directory contains tools for creating an ASL dataset using a webcam:

- `create_asl_dataset.py`: Captures ASL hand signs from webcam
- `consolidate_images.py`: Merges multiple datasets into one

### Model Test Scripts

Each `test_*.py` file implements a specific model's ASL prediction capability:

- `test_gpt4o.py` / `test_gpt4_turbo.py`: OpenAI's GPT-4 Vision models
- `test_gemini_2_flash.py` / `test_gemini_2_flash_lite.py`: Google's Gemini models
- `test_llama_*.py`: Llama family models (90B, Maverick, Scout)
- `test_pixtral_12b.py`: Mistral's Pixtral model
- `test_granite_vision.py`: IBM's Granite Vision model

Each test script follows a common structure:
1. Image processing and encoding
2. Multiple prompting strategies (zero-shot, few-shot, chain-of-thought, etc.)
3. API request handling and retries
4. Result parsing and prediction extraction

### Evaluation Framework

The `evaluate_models.py` script is the main evaluation framework that:

1. Loads available models
2. Tests each model against the dataset
3. Calculates performance metrics
4. Generates confusion matrices
5. Outputs comprehensive evaluation results

## Usage

### Testing a Single Model

To test a specific model with a single image:

```bash
python test_gpt4o.py --image path/to/image.jpg --prompt-strategy zero_shot
```

Available prompt strategies:
- `zero_shot`: Basic prompt with no examples
- `few_shot`: Include examples of ASL signs
- `chain_of_thought`: Step-by-step reasoning approach
- `visual_grounding`: Focus on visual details
- `contrastive`: Compare against multiple candidates

### Running Full Evaluation

To evaluate all available models on the dataset:

```bash
python evaluate_models.py --dataset_path ./data --sample_size 30 --output_dir evaluation_results
```

Options:
- `--dataset_path`: Path to the dataset directory (default: ./data)
- `--sample_size`: Number of images to sample from each letter (default: 30)
- `--output_dir`: Directory to save evaluation results (default: evaluation_results)
- `--quick_test`: Run a quick test with one random image and zero-shot prompting only

## Results

Evaluation results include:
- Accuracy metrics for each model
- Confusion matrices showing misclassifications
- Response time and token usage statistics
- Performance by prompting strategy

Results are saved as JSON files and visualizations in the `evaluation_results/` directory.

## License

This project is for educational and research purposes.

## Acknowledgments

This project was developed as part of the Artificial Intelligence and Machine Learning course at Copenhagen Business School. 
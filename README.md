# Multimodal Sentiment Analysis System

A sophisticated sentiment analysis system that combines Natural Language Processing (NLP) and Computer Vision (CV) techniques to classify sentiment based on both textual and image data.

## Project Overview

This project implements a multimodal sentiment analysis system capable of detecting positive, neutral, and negative sentiment from text and images. By combining both modalities using an attention-based fusion mechanism, the model achieves higher accuracy than either modality alone.

### Features

- Text sentiment analysis using BERT
- Image sentiment analysis using ResNet50
- Multimodal fusion with attention mechanism
- Support for text-only inference when images are unavailable
- Interactive command-line interface for sentiment prediction

## Model Architecture

The system consists of three main components:

1. **Text Model (`text_model.py`)**

   - Uses BERT (bert-base-uncased) for feature extraction
   - Fine-tuned with a classification head for sentiment prediction
   - 768-dimensional BERT embeddings → 256-dimensional features
   - Includes dropout for regularization

2. **Image Model (`image_model.py`)**

   - Uses ResNet50 pre-trained on ImageNet
   - Adapts the CNN for sentiment analysis with custom classification layers
   - 2048-dimensional image features → 256-dimensional features
   - Includes dropout for regularization

3. **Fusion Model (`fusion_model.py`)**

   - Combines text and image features using one of two methods:
     - **Concatenation**: Simply concatenates the feature vectors
     - **Attention**: Uses attention mechanism to dynamically weight features
   - Final classification through fully connected layers

4. **Combined Model (`model.py`)**
   - Integrates all three components for end-to-end sentiment analysis
   - Can work with both text and images or text-only input

### Fusion Methods

Two fusion approaches are implemented:

1. **Concatenation Fusion**:

   - Simple concatenation of text and image feature vectors
   - Combined dimensions: 512 (256 + 256)

2. **Attention Fusion**:
   - Dynamic attention mechanism to focus on more informative features
   - Attention weights calculated based on both modalities
   - Output dimension: 256

## Performance

The model achieves the following performance metrics on the test set:

- **Overall Accuracy**: 77.67%
- **Positive Sentiment**: Precision 0.87, Recall 0.84, F1 0.85
- **Neutral Sentiment**: Precision 0.69, Recall 0.76, F1 0.73
- **Negative Sentiment**: Precision 0.78, Recall 0.72, F1 0.75
- **Macro-average F1**: 0.78

The fusion model consistently outperforms single-modality models, demonstrating the benefit of the multimodal approach.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers (Hugging Face)
- torchvision
- pandas
- matplotlib
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model with default parameters:

```bash
python train.py --data_path dataset/LabeledText.xlsx --images_dir dataset --epochs 20 --batch_size 16
```

For improved training with better hyperparameters:

```bash
python train_improved_model.py --data_path dataset/LabeledText.xlsx --images_dir dataset --epochs 50 --batch_size 16 --learning_rate 2e-5 --weight_decay 0.01
```

Training options:

- `--freeze_bert`: Freeze BERT parameters for faster training
- `--freeze_resnet`: Freeze ResNet parameters for faster training
- `--fusion_method`: Choose fusion method ('concat' or 'attention')
- `--save_dir`: Directory to save model checkpoints
- `--force_cpu`: Force CPU usage even if GPU is available
- `--text_only_mode`: Use only text data (if images are unavailable)
- `--use_class_weights`: Apply class weighting to handle imbalanced data

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --data_path dataset/LabeledText.xlsx --images_dir dataset --checkpoint checkpoints/best_model.pth
```

### Testing with Sample Sentences

To test the model with predefined positive, neutral, and negative sentences:

```bash
python test_probabilities.py --model_path checkpoints/best_model.pth
```

### Interactive Sentiment Analysis

To analyze sentiment of specific text and optional image:

```bash
python run_sentiment_analysis.py --text "Your text here" --visualize
```

With an image:

```bash
python run_sentiment_analysis.py --text "Your text here" --image dataset/Images/Images/positive/4128.jpg --visualize
```

To run in fully interactive mode:

```bash
python run_sentiment_analysis.py
```

## Project Structure

```
.
├── data_utils.py            # Data handling and preprocessing
├── text_model.py            # BERT-based text sentiment model
├── image_model.py           # ResNet-based image sentiment model
├── fusion_model.py          # Feature fusion implementation
├── model.py                 # Combined multimodal model
├── train.py                 # Standard training script
├── train_improved_model.py  # Enhanced training with better optimization
├── evaluate.py              # Model evaluation script
├── test_probabilities.py    # Test with example sentences
├── run_sentiment_analysis.py # Interactive sentiment analysis tool
├── requirements.txt         # Dependencies
└── dataset/                 # Dataset directory
    ├── LabeledText.xlsx     # Text data with sentiment labels
    └── Images/              # Image directory
```

## Dataset Format

The system expects data in the following format:

1. An Excel file (LabeledText.xlsx) with columns:

   - `Text`: The text content
   - `Sentiment`: The sentiment label (positive, neutral, negative)
   - `ImageID`: ID linking to the corresponding image

2. Images organized in sentiment-specific folders:
   ```
   dataset/Images/Images/positive/
   dataset/Images/Images/Neutral/
   dataset/Images/Images/Negative/
   ```

## Common Issues and Solutions

- **Missing Images**: If images are not found, the system will use dummy tensors and rely on text-only predictions.
- **Memory Issues**: Use `--freeze_bert` and `--freeze_resnet` to reduce memory usage.
- **GPU vs CPU**: Use `--force_cpu` if you encounter GPU-related errors.

## Future Improvements

- Implement data augmentation for better generalization
- Explore different fusion mechanisms (bilinear pooling, etc.)
- Add support for more sentiment classes or emotion detection
- Create a web interface for easier interaction

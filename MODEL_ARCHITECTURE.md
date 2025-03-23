# Multimodal Sentiment Analysis: Model Architecture

This document provides a detailed explanation of the model architecture used in our multimodal sentiment analysis system.

## Overview

The model architecture follows a late fusion approach, where:

1. Text and images are processed independently through specialized neural networks
2. Features from both modalities are extracted
3. These features are combined using a fusion mechanism
4. The combined representation is used for final sentiment classification

```
                                ┌─────────────────┐
                                │                 │
Text Input  ─────────────────► │   BERT Model    │ ─┐
                                │                 │  │    ┌─────────────────┐    ┌─────────────────┐
                                └─────────────────┘  │    │                 │    │                 │
                                                     ├─► │  Fusion Layer   │ ─► │  Classifier     │ ─► Sentiment
                                ┌─────────────────┐  │    │                 │    │                 │
                                │                 │  │    └─────────────────┘    └─────────────────┘
Image Input ─────────────────► │  ResNet Model   │ ─┘
                                │                 │
                                └─────────────────┘
```

## Component Details

### 1. Text Sentiment Model (`text_model.py`)

```python
class TextSentimentModel(nn.Module):
    def __init__(self, num_classes=3, dropout=0.1, freeze_bert=True):
        # Initialization...
```

- **Base Model**: BERT (bert-base-uncased)
- **Feature Extraction**:
  - BERT embedding output: 768 dimensions
  - Feature extraction layer: 768 → 256 dimensions
- **Classification Head**:
  - Input: 256 dimensions
  - Output: 3 classes (positive, neutral, negative)
- **Training Options**:
  - Option to freeze BERT parameters for faster training
  - Dropout for regularization (default: 0.1)

### 2. Image Sentiment Model (`image_model.py`)

```python
class ImageSentimentModel(nn.Module):
    def __init__(self, num_classes=3, dropout=0.1, freeze_backbone=True):
        # Initialization...
```

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Feature Extraction**:
  - ResNet feature output: 2048 dimensions
  - Feature extraction layer: 2048 → 256 dimensions
- **Classification Head**:
  - Input: 256 dimensions
  - Output: 3 classes (positive, neutral, negative)
- **Training Options**:
  - Option to freeze ResNet parameters for faster training
  - Dropout for regularization (default: 0.1)

### 3. Feature Fusion Model (`fusion_model.py`)

```python
class FeatureFusionModel(nn.Module):
    def __init__(self, text_feature_dim=256, image_feature_dim=256, hidden_dim=512,
                 num_classes=3, dropout=0.1, fusion_method='concat'):
        # Initialization...
```

The fusion model supports two different fusion methods:

#### Concatenation Fusion

```
Text Features (256) ──┐
                      │─► Concatenate ──► FC Layer ──► ReLU ──► Dropout ──► FC Layer ──► Sentiment
Image Features (256) ─┘       (512)        (512→512)               (0.1)     (512→3)     Output
```

- Simple concatenation of feature vectors
- Input dimension: 256 (text) + 256 (image) = 512
- Hidden layer: 512 dimensions with ReLU activation
- Output: 3 classes (positive, neutral, negative)

#### Attention Fusion

```
                         ┌─► Linear ──► Tanh ──► Linear ──► Softmax ──┐
                         │    (512→256)          (256→1)              │
Text Features (256) ───┬─┘                                    Weighted Sum ──► FC Layer ──► ReLU ──► Dropout ──► FC Layer ──► Sentiment
                       │                                          │            (256→512)               (0.1)     (512→3)     Output
Image Features (256) ──┴─────────────────────────────────────────┘
```

- Dynamic weighting of features through attention mechanism
- Combined features: text_features + image_features (512 dimensions)
- Attention weights calculated using a 2-layer neural network
- Output dimension: 256
- Hidden layer: 512 dimensions with ReLU activation
- Output: 3 classes (positive, neutral, negative)

### 4. Multimodal Sentiment Model (`model.py`)

```python
class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes=3, text_dropout=0.1, image_dropout=0.1, fusion_dropout=0.1,
                 freeze_bert=True, freeze_resnet=True, fusion_method='concat'):
        # Initialization...
```

- **Integration of Components**:
  - TextSentimentModel
  - ImageSentimentModel
  - FeatureFusionModel
- **Forward Pass**:
  - Process text and image in parallel
  - Extract features from both modalities
  - Fuse features
  - Generate predictions from each component:
    - Text-only prediction
    - Image-only prediction
    - Fusion (combined) prediction
- **Return Values**:
  - Logits from each component
  - Feature vectors from each component

## Model Training Process

The training process involves:

1. **Data Preparation**:

   - Text preprocessing (tokenization, etc.)
   - Image preprocessing (resizing, normalization)
   - Label encoding

2. **Training Loop**:

   - Forward pass through the multimodal model
   - Calculate losses for each component
     - Text loss: CrossEntropy(text_logits, labels)
     - Image loss: CrossEntropy(image_logits, labels)
     - Fusion loss: CrossEntropy(fusion_logits, labels)
   - Combined loss: fusion_loss + 0.3 _ text_loss + 0.3 _ image_loss
   - Backpropagation and parameter updates

3. **Validation**:

   - Regular evaluation on validation set
   - Model selection based on validation accuracy
   - Early stopping to prevent overfitting

4. **Optimization Techniques**:
   - Learning rate scheduling
   - Weight decay for regularization
   - Gradient clipping
   - Class weighting for imbalanced data

## Model Dimensions Summary

| Component           | Input Dimension   | Output Dimension | Parameters |
| ------------------- | ----------------- | ---------------- | ---------- |
| BERT                | Text Tokens       | 768              | 110M       |
| Text Feature Layer  | 768               | 256              | 197K       |
| ResNet50            | Image (3x224x224) | 2048             | 23M        |
| Image Feature Layer | 2048              | 256              | 524K       |
| Concat Fusion       | 512               | 3                | 263K       |
| Attention Fusion    | 256               | 3                | 133K       |

Total parameters: ~134M (with BERT and ResNet)

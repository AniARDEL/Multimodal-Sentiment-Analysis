# Multimodal Sentiment Analysis: Quick Start Guide

This guide will help you quickly set up and run the multimodal sentiment analysis system.

## Installation

1. **Clone the repository** (if you haven't already):

   ```
   git clone <repository-url>
   cd Multimodal-Sentiment-Analysis
   ```

2. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```
   python -c "import torch; import transformers; import torchvision; print('All dependencies installed!')"
   ```

## Dataset Preparation

The system expects your data in a specific format:

1. **Excel file** (e.g., `dataset/LabeledText.xlsx`) with columns:

   - `Text`: The text content
   - `Sentiment`: The sentiment label (positive, neutral, negative)
   - `ImageID`: ID linking to the corresponding image

2. **Images** organized in folders:
   ```
   dataset/Images/Images/positive/
   dataset/Images/Images/Neutral/
   dataset/Images/Images/Negative/
   ```

## Quick Test (Without Training)

To test the system with pre-trained models:

1. **Test with predefined examples**:

   ```
   python test_probabilities.py --model_path checkpoints/best_model.pth
   ```

2. **Analyze your own text**:

   ```
   python run_sentiment_analysis.py --text "Your text here" --visualize
   ```

3. **Analyze text and image**:
   ```
   python run_sentiment_analysis.py --text "Your text here" --image path/to/your/image.jpg --visualize
   ```

## Training

To train the model on your dataset:

1. **Basic training**:

   ```
   python train.py --data_path dataset/LabeledText.xlsx --images_dir dataset --epochs 20 --batch_size 16 --freeze_bert --freeze_resnet
   ```

2. **Advanced training** (better results):
   ```
   python train_improved_model.py --data_path dataset/LabeledText.xlsx --images_dir dataset --epochs 50 --batch_size 16 --learning_rate 2e-5 --weight_decay 0.01 --use_class_weights
   ```

## Evaluation

To evaluate your trained model:

```
python evaluate.py --data_path dataset/LabeledText.xlsx --images_dir dataset --checkpoint checkpoints/best_model.pth
```

## Troubleshooting

1. **Memory issues**:

   - Try reducing batch size: `--batch_size 8`
   - Use model freezing: `--freeze_bert --freeze_resnet`

2. **GPU errors**:

   - Force CPU usage: `--force_cpu`

3. **Missing images**:

   - Try text-only mode: `--text_only_mode`

4. **Performance issues**:
   - For better neutral predictions, train with class weights: `--use_class_weights`
   - Try attention fusion: `--fusion_method attention`

## Example Workflow

Here's a typical workflow:

1. **Prepare your data** in the required format
2. **Train the model** with appropriate parameters
3. **Evaluate the model** to measure performance
4. **Test with specific examples** to verify results
5. **Use in interactive mode** for on-the-fly analysis

## Next Steps

- Check README.md for more detailed documentation
- Explore MODEL_ARCHITECTURE.md for technical details
- Try different hyperparameters for better results

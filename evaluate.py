import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_utils import get_data_loaders
from model import MultimodalSentimentModel


def evaluate_model(args):
    """
    Evaluate the multimodal sentiment analysis model
    
    Args:
        args: Command line arguments
    """
    # Set device
    if args.force_cpu:
        device = torch.device('cpu')
        print("Forcing CPU usage as requested")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders (only need test loader)
    _, _, test_loader = get_data_loaders(
        text_data_path=args.data_path,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        val_split=0,  # No need for validation split
        test_split=1.0,  # Use all data for testing
        max_length=args.max_length,
        text_only_mode=args.text_only_mode
    )
    
    print(f"Test samples: {len(test_loader.sampler)}")
    
    # Initialize model
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=0.1,
        image_dropout=0.1,
        fusion_dropout=0.1,
        freeze_bert=True,
        freeze_resnet=True,
        fusion_method=args.fusion_method
    ).to(device)
    
    # Load model checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file {args.checkpoint} not found.")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluation metrics
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Lists to store predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    # Component evaluation metrics
    text_correct = 0
    image_correct = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, images)
            
            # Calculate loss
            loss = criterion(outputs['fusion_logits'], labels)
            
            # Update statistics for fusion model
            test_loss += loss.item()
            _, predicted = torch.max(outputs['fusion_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update statistics for component models
            _, text_predicted = torch.max(outputs['text_logits'], 1)
            text_correct += (text_predicted == labels).sum().item()
            
            _, image_predicted = torch.max(outputs['image_logits'], 1)
            image_correct += (image_predicted == labels).sum().item()
            
            # Store predictions and true labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': test_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
    
    # Calculate average test loss and accuracy
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    text_acc = 100 * text_correct / total
    image_acc = 100 * image_correct / total
    
    # Print test statistics
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Fusion Model Accuracy: {test_acc:.2f}%")
    print(f"Text-only Model Accuracy: {text_acc:.2f}%")
    print(f"Image-only Model Accuracy: {image_acc:.2f}%")
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    class_names = ['Positive', 'Neutral', 'Negative']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(all_true_labels, all_predictions, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Fusion Model Accuracy: {test_acc:.2f}%\n")
        f.write(f"Text-only Model Accuracy: {text_acc:.2f}%\n")
        f.write(f"Image-only Model Accuracy: {image_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate multimodal sentiment analysis model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the labeled text Excel file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing the image folders')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for BERT tokenizer')
    
    # Model parameters
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention'],
                        help='Method for fusing features')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for saving evaluation results')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--text_only_mode', action='store_true', default=True,
                        help='Use only text data with dummy images')
    
    args = parser.parse_args()
    
    # Evaluate the model
    evaluate_model(args) 
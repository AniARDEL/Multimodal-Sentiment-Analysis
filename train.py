import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import get_data_loaders
from model import MultimodalSentimentModel


def train_model(args):
    """
    Train the multimodal sentiment analysis model
    
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
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        text_data_path=args.data_path,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        max_length=args.max_length,
        text_only_mode=args.text_only_mode
    )
    
    print(f"Training samples: {len(train_loader.sampler)}")
    print(f"Validation samples: {len(val_loader.sampler)}")
    print(f"Testing samples: {len(test_loader.sampler)}")
    
    # Initialize model
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=args.text_dropout,
        image_dropout=args.image_dropout,
        fusion_dropout=args.fusion_dropout,
        freeze_bert=args.freeze_bert,
        freeze_resnet=args.freeze_resnet,
        fusion_method=args.fusion_method
    ).to(device)
    
    # Define loss function and optimizer
    # Use weighted loss to handle class imbalance
    if args.use_class_weights:
        # Initialize with default weights
        class_weights = torch.FloatTensor([1.0, 1.0, 1.0]).to(device)
        
        # Count samples per class in the training data
        class_counts = torch.zeros(3)
        for batch in train_loader:
            labels = batch['label']
            for i in range(3):
                class_counts[i] += (labels == i).sum().item()
        
        # If we have samples for all classes, compute weights
        if torch.all(class_counts > 0):
            # Inverse frequency weighting
            class_weights = torch.max(class_counts) / class_counts
            class_weights = class_weights / class_weights.sum() * 3  # Normalize
            class_weights = class_weights.to(device)
            print(f"Using class weights: {class_weights}")
        else:
            print("Not all classes represented in training data, using uniform weights")
    else:
        class_weights = None
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use a slightly warmer learning rate with a decay schedule
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training metrics
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training phase
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask, images)
            
            # Calculate loss
            loss = criterion(outputs['fusion_logits'], labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs['fusion_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # Calculate average training loss and accuracy
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
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
                
                # Update statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs['fusion_logits'], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': val_loss / (progress_bar.n + 1),
                    'acc': 100 * correct / total
                })
        
        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save model if it has the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with val accuracy: {val_acc:.2f}%")
        
        # Save model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_loss_model.pth'))
            print(f"Saved best loss model with val loss: {val_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
    plt.close()
    
    print("Training complete!")
    
    # Test the model with the best validation accuracy
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test phase
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Confusion matrix
    confusion_matrix = torch.zeros(3, 3)
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
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
            
            # Update statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs['fusion_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': test_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
    
    # Calculate average test loss and accuracy
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    
    # Print test statistics
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    print("\nRow: Actual, Column: Predicted")
    
    # Calculate precision, recall, and F1 score for each class
    precision = torch.zeros(3)
    recall = torch.zeros(3)
    f1 = torch.zeros(3)
    
    for i in range(3):
        precision[i] = confusion_matrix[i, i] / confusion_matrix[:, i].sum() if confusion_matrix[:, i].sum() > 0 else 0
        recall[i] = confusion_matrix[i, i] / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0
    
    # Print precision, recall, and F1 score for each class
    class_names = ['Positive', 'Neutral', 'Negative']
    for i in range(3):
        print(f"{class_names[i]} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    
    # Print macro and weighted average F1 score
    macro_f1 = f1.mean().item()
    weighted_f1 = (f1 * confusion_matrix.sum(1) / confusion_matrix.sum()).sum().item()
    
    print(f"Macro-average F1 Score: {macro_f1:.4f}")
    print(f"Weighted-average F1 Score: {weighted_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multimodal sentiment analysis model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the labeled text Excel file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing the image folders')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for BERT tokenizer')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Proportion of data for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Proportion of data for testing')
    
    # Model parameters
    parser.add_argument('--freeze_bert', action='store_true',
                        help='Freeze BERT parameters')
    parser.add_argument('--freeze_resnet', action='store_true',
                        help='Freeze ResNet parameters')
    parser.add_argument('--text_dropout', type=float, default=0.3,
                        help='Dropout probability for text model')
    parser.add_argument('--image_dropout', type=float, default=0.3,
                        help='Dropout probability for image model')
    parser.add_argument('--fusion_dropout', type=float, default=0.3,
                        help='Dropout probability for fusion model')
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention'],
                        help='Method for fusing features')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='Learning rate for optimizer')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory for saving models and results')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--text_only_mode', action='store_true', default=False,
                        help='Use only text data with dummy images')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights to handle imbalanced data')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(args) 
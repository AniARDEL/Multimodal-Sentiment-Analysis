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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def train_improved_model(args):
    """
    Train the multimodal sentiment analysis model with improved parameters
    
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
    
    # Initialize model with lower dropout for early training
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=args.text_dropout,
        image_dropout=args.image_dropout,
        fusion_dropout=args.fusion_dropout,
        freeze_bert=args.freeze_bert,
        freeze_resnet=args.freeze_resnet,
        fusion_method=args.fusion_method
    ).to(device)
    
    # Use class weights to handle imbalance
    class_weights = None
    if args.use_class_weights:
        # Count samples per class in the training data
        class_counts = torch.zeros(3)
        for batch in train_loader:
            labels = batch['label']
            for i in range(3):
                class_counts[i] += (labels == i).sum().item()
        
        # If we have samples for all classes, compute weights
        if torch.all(class_counts > 0):
            # Inverse frequency weighting with stronger emphasis
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.min() * 1.0  # Scale
            class_weights = class_weights.to(device)
            print(f"Using class weights: {class_weights}")
        else:
            print("Not all classes represented in training data, using uniform weights")
    
    # Loss function with stronger class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True
    )
    
    # Learning rate scheduler with warmup
    warmup_steps = len(train_loader) * 2  # 2 epochs
    total_steps = len(train_loader) * args.epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Training metrics
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    patience = args.patience
    patience_counter = 0
    
    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop with early stopping
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
            
            # Calculate loss with label smoothing
            loss = criterion(outputs['fusion_logits'], labels)
            
            # Add L1 regularization (feature selection)
            if args.l1_reg > 0:
                l1_reg = 0
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += args.l1_reg * l1_reg
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs['fusion_logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total,
                'lr': scheduler.get_last_lr()[0]
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
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
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
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
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
            
            # Get predictions and probabilities
            _, predicted = torch.max(outputs['fusion_logits'], 1)
            probs = torch.nn.functional.softmax(outputs['fusion_logits'], dim=1)
            
            # Update statistics
            test_loss += loss.item()
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
    
    # Save probability distribution example for each class
    print("\nSample probability distributions:")
    
    # Find examples of each class in test data
    class_examples = {}
    
    # Enable model gradients for attention visualization later
    model.train()
    
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        texts = batch['text']
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, images)
            probs = torch.nn.functional.softmax(outputs['fusion_logits'], dim=1)
        
        # Get predictions
        _, predicted = torch.max(probs, 1)
        
        # Store correctly classified examples
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            
            # Only store correctly classified examples
            if true_label == pred_label and true_label not in class_examples:
                class_examples[true_label] = {
                    'text': texts[i],
                    'probs': probs[i].cpu().numpy()
                }
            
            # Break if we found one example for each class
            if len(class_examples) == 3:
                break
        
        if len(class_examples) == 3:
            break
    
    # Print example probability distributions
    for label in sorted(class_examples.keys()):
        example = class_examples[label]
        print(f"\n{class_names[label]} example: '{example['text']}'")
        print("Probability distribution:")
        for i, name in enumerate(class_names):
            prob = example['probs'][i] * 100
            bar = '█' * int(prob / 5) + '░' * (20 - int(prob / 5))
            print(f"  {name}: {bar} {prob:.2f}%")
    
    # Save model metadata
    metadata = {
        'accuracy': test_acc,
        'f1_score': macro_f1,
        'class_names': class_names,
        'confusion_matrix': confusion_matrix.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
    }
    
    # Save metadata to a file
    import json
    with open(os.path.join(args.save_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model, test_acc, macro_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an improved multimodal sentiment analysis model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='dataset/LabeledText.xlsx',
                        help='Path to the labeled text Excel file')
    parser.add_argument('--images_dir', type=str, default='dataset',
                        help='Directory containing the image folders')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for BERT tokenizer')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Proportion of data for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Proportion of data for testing')
    
    # Model parameters
    parser.add_argument('--freeze_bert', action='store_true', default=False,
                        help='Freeze BERT parameters')
    parser.add_argument('--freeze_resnet', action='store_true', default=True,
                        help='Freeze ResNet parameters')
    parser.add_argument('--text_dropout', type=float, default=0.3,
                        help='Dropout probability for text model')
    parser.add_argument('--image_dropout', type=float, default=0.3,
                        help='Dropout probability for image model')
    parser.add_argument('--fusion_dropout', type=float, default=0.3,
                        help='Dropout probability for fusion model')
    parser.add_argument('--fusion_method', type=str, default='attention', choices=['concat', 'attention'],
                        help='Method for fusing features')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for regularization')
    parser.add_argument('--l1_reg', type=float, default=1e-5,
                        help='L1 regularization coefficient')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--save_dir', type=str, default='improved_checkpoints',
                        help='Directory for saving models and results')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--text_only_mode', action='store_true', default=False,
                        help='Use only text data with dummy images')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights to handle imbalanced data')
    
    args = parser.parse_args()
    
    # Train the model
    train_improved_model(args) 
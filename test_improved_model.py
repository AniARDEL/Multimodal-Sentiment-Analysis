import os
import time
import subprocess
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer
import torchvision.transforms as transforms

from model import MultimodalSentimentModel

# Sample test sentences with varying sentiments
TEST_SENTENCES = [
    "This is absolutely amazing! I couldn't be happier with the results.",  # Strong positive
    "The product was great, but the delivery was a bit slow.",             # Mixed (slightly positive)
    "It's an average product, nothing special about it.",                  # Neutral
    "I'm not very satisfied with my purchase. It was disappointing.",      # Negative
    "This is terrible! I hate it and I want a refund immediately."         # Strong negative
]


def train_quick_model(args):
    """Train a quick model for testing"""
    print("\n" + "="*60)
    print("TRAINING A QUICK MODEL FOR TESTING")
    print("="*60)
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(args.checkpoint) and not args.retrain:
        print(f"Using existing model at {args.checkpoint}")
        print("To retrain the model, use --retrain")
        return True
    
    # Build training command
    cmd = [
        "python", "train.py",
        "--data_path", args.data_path,
        "--images_dir", args.images_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", "0.0003",
        "--force_cpu" if args.force_cpu else "",
        "--freeze_bert", "--freeze_resnet",
        "--use_class_weights",
        "--save_dir", os.path.dirname(args.checkpoint)
    ]
    
    # Remove any empty strings
    cmd = [x for x in cmd if x]
    
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd)
    
    return process.returncode == 0


def test_model_with_sentences(args):
    """Test the model with sample sentences"""
    print("\n" + "="*60)
    print("TESTING MODEL WITH SAMPLE SENTENCES")
    print("="*60)
    
    # Set device
    device = torch.device('cpu') if args.force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file {args.checkpoint} not found.")
        return False
    
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=0.3,
        image_dropout=0.3,
        fusion_dropout=0.3,
        freeze_bert=True,
        freeze_resnet=True,
        fusion_method='concat'
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Set up tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Find a test image if needed
    has_image = False
    image_path = None
    image_tensor = None
    
    if not args.text_only:
        for sentiment in ['positive', 'Neutral', 'Negative']:
            img_dir = os.path.join(args.images_dir, 'Images', sentiment)
            if os.path.exists(img_dir):
                img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
                if img_files:
                    image_path = os.path.join(img_dir, img_files[0])
                    has_image = True
                    
                    # Load and transform the image
                    image = Image.open(image_path).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    print(f"Using image: {image_path}")
                    break
        
        if not has_image:
            print("No images found. Using text-only mode.")
            image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    else:
        print("Using text-only mode.")
        image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    
    # Create visualization
    class_names = ['Positive', 'Neutral', 'Negative']
    results = []
    
    for text in TEST_SENTENCES:
        # Tokenize text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            
            fusion_probs = torch.nn.functional.softmax(outputs['fusion_logits'], dim=1)
            text_probs = torch.nn.functional.softmax(outputs['text_logits'], dim=1)
            
            # Get predicted class
            fusion_class = torch.argmax(fusion_probs, dim=1).item()
            text_class = torch.argmax(text_probs, dim=1).item()
        
        # Print results with colored bars
        print(f"\nText: \"{text}\"")
        print(f"Predicted sentiment: {class_names[fusion_class]}")
        
        print("\nText model probabilities:")
        for i, name in enumerate(class_names):
            percentage = text_probs[0, i].item() * 100
            bar = get_progress_bar(percentage)
            print(f"  {name}: {bar} {percentage:.2f}%")
        
        print("\nFusion model probabilities:")
        for i, name in enumerate(class_names):
            percentage = fusion_probs[0, i].item() * 100
            bar = get_progress_bar(percentage)
            print(f"  {name}: {bar} {percentage:.2f}%")
        
        results.append({
            'text': text, 
            'sentiment': class_names[fusion_class],
            'probabilities': fusion_probs[0].tolist()
        })
    
    # Visualize comparison of all results
    if args.visualize:
        visualize_results(results)
    
    return True


def get_progress_bar(percentage, width=20):
    """Create a text-based progress bar for visualization"""
    filled_width = int(width * percentage / 100)
    bar = '█' * filled_width + '░' * (width - filled_width)
    return bar


def visualize_results(results):
    """Create a visualization of all test sentences and their sentiment probabilities"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    texts = [r['text'][:50] + '...' if len(r['text']) > 50 else r['text'] for r in results]
    pos_probs = [r['probabilities'][0] * 100 for r in results]
    neu_probs = [r['probabilities'][1] * 100 for r in results]
    neg_probs = [r['probabilities'][2] * 100 for r in results]
    
    # Set up bar positions
    x = np.arange(len(texts))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, pos_probs, width, label='Positive', color='green')
    bars2 = ax.bar(x, neu_probs, width, label='Neutral', color='blue')
    bars3 = ax.bar(x + width, neg_probs, width, label='Negative', color='red')
    
    # Highlight the predicted sentiment
    for i, result in enumerate(results):
        if result['sentiment'] == 'Positive':
            bars1[i].set_edgecolor('black')
            bars1[i].set_linewidth(2)
        elif result['sentiment'] == 'Neutral':
            bars2[i].set_edgecolor('black')
            bars2[i].set_linewidth(2)
        elif result['sentiment'] == 'Negative':
            bars3[i].set_edgecolor('black')
            bars3[i].set_linewidth(2)
    
    # Add text and labels
    ax.set_ylabel('Probability (%)')
    ax.set_title('Sentiment Analysis Across Test Sentences')
    ax.set_xticks(x)
    ax.set_xticklabels(texts, rotation=30, ha='right')
    ax.legend()
    
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test the improved sentiment analysis model")
    parser.add_argument('--data_path', type=str, default='dataset/LabeledText.xlsx', help='Path to the dataset file')
    parser.add_argument('--images_dir', type=str, default='dataset', help='Path to the images directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to the model checkpoint')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--text_only', action='store_true', help='Use text-only mode')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for quick training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model even if it exists')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    
    args = parser.parse_args()
    
    # Train a quick model if needed
    if train_quick_model(args):
        # Test the model with sample sentences
        test_model_with_sentences(args)
    else:
        print("Failed to train or load the model. Please check the error messages above.")


if __name__ == "__main__":
    main() 
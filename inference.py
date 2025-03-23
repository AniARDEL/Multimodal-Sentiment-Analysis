import argparse
import os
import torch
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import MultimodalSentimentModel


def predict_sentiment(args):
    """
    Predict sentiment for a single text and image
    
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
    
    # Initialize model
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=0.3,
        image_dropout=0.3,
        fusion_dropout=0.3,
        freeze_bert=True,
        freeze_resnet=True,
        fusion_method=args.fusion_method
    ).to(device)
    
    # Load model checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file {args.checkpoint} not found.")
        
        if args.try_train:
            print("Attempting to train a quick model first...")
            import subprocess
            train_cmd = f"python train.py --data_path {args.data_path} --images_dir {args.images_dir} --epochs 5 --batch_size 8 --freeze_bert --freeze_resnet"
            result = subprocess.run(train_cmd, shell=True)
            if result.returncode != 0:
                print("Training failed. Please train the model manually first.")
                return
        else:
            return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer.encode_plus(
        args.text,
        add_special_tokens=True,
        max_length=args.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Process image or create dummy image
    has_image = args.image and os.path.exists(args.image) and not args.text_only_mode
    
    if has_image:
        try:
            print(f"Loading image from: {args.image}")
            image = Image.open(args.image).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image {args.image}: {e}")
            has_image = False
            image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    else:
        if args.text_only_mode:
            print("Using text-only mode with dummy image tensor")
        elif args.image:
            print(f"Image file {args.image} not found. Using a dummy image instead.")
        else:
            print("No image provided. Using a dummy image tensor.")
        
        image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image_tensor)
        
        # Get predictions
        fusion_probs = torch.nn.functional.softmax(outputs['fusion_logits'], dim=1)
        text_probs = torch.nn.functional.softmax(outputs['text_logits'], dim=1)
        image_probs = torch.nn.functional.softmax(outputs['image_logits'], dim=1)
        
        # Get predicted class
        fusion_class = torch.argmax(fusion_probs, dim=1).item()
        text_class = torch.argmax(text_probs, dim=1).item()
        image_class = torch.argmax(image_probs, dim=1).item()
    
    # Map class indices to labels
    class_names = ['Positive', 'Neutral', 'Negative']
    
    # Print prediction results
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*50)
    print(f"Text: '{args.text}'")
    
    if has_image:
        print(f"Image: {args.image}")
    else:
        print("Image: [No valid image provided]")
    
    print("\nPredictions:")
    print(f"✓ Fusion Model: {class_names[fusion_class]} ({fusion_probs[0, fusion_class].item() * 100:.2f}%)")
    print(f"✓ Text-only Model: {class_names[text_class]} ({text_probs[0, text_class].item() * 100:.2f}%)")
    
    if has_image:
        print(f"✓ Image-only Model: {class_names[image_class]} ({image_probs[0, image_class].item() * 100:.2f}%)")
    else:
        print("✓ Image-only Model: N/A (no valid image)")
    
    # Print probability scores
    print("\nDetailed Probability Breakdown:")
    
    print("\nFusion Model (Text + Image):")
    for i, name in enumerate(class_names):
        confidence = fusion_probs[0, i].item() * 100
        bar = get_progress_bar(confidence)
        print(f"  {name}: {bar} {confidence:.2f}%")
    
    print("\nText-only Model:")
    for i, name in enumerate(class_names):
        confidence = text_probs[0, i].item() * 100
        bar = get_progress_bar(confidence)
        print(f"  {name}: {bar} {confidence:.2f}%")
    
    if has_image:
        print("\nImage-only Model:")
        for i, name in enumerate(class_names):
            confidence = image_probs[0, i].item() * 100
            bar = get_progress_bar(confidence)
            print(f"  {name}: {bar} {confidence:.2f}%")
    
    print("="*50)
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(args, has_image, class_names, fusion_probs, text_probs, image_probs, image if has_image else None)


def get_progress_bar(percentage, width=20):
    """
    Create a progress bar for visualization in terminal
    
    Args:
        percentage (float): Value between 0 and 100
        width (int): Width of the progress bar
        
    Returns:
        str: ASCII progress bar
    """
    filled_width = int(width * percentage / 100)
    bar = '█' * filled_width + '░' * (width - filled_width)
    return bar


def visualize_results(args, has_image, class_names, fusion_probs, text_probs, image_probs, image=None):
    """
    Visualize the sentiment analysis results
    
    Args:
        args: Command line arguments
        has_image (bool): Whether a valid image was provided
        class_names (list): List of class names
        fusion_probs (torch.Tensor): Fusion model probabilities
        text_probs (torch.Tensor): Text model probabilities
        image_probs (torch.Tensor): Image model probabilities
        image (PIL.Image): Original image if available
    """
    # Setup figure
    if has_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 2]})
        
        # Display image
        ax1.imshow(image)
        ax1.set_title("Input Image")
        ax1.axis('off')
        
        # Use ax2 for the bars
        ax = ax2
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    bar_width = 0.25
    x = range(len(class_names))
    
    # Plot probability bars
    bars1 = ax.bar([p - bar_width for p in x], [fusion_probs[0, i].item() * 100 for i in range(3)], 
                   bar_width, label='Fusion Model', color='blue')
    bars2 = ax.bar(x, [text_probs[0, i].item() * 100 for i in range(3)], 
                   bar_width, label='Text Model', color='green')
    
    if has_image:
        bars3 = ax.bar([p + bar_width for p in x], [image_probs[0, i].item() * 100 for i in range(3)], 
                       bar_width, label='Image Model', color='red')
    
    # Add values at the top of the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', rotation=0, fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    if has_image:
        add_labels(bars3)
    
    # Add labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('Sentiment Analysis Results', fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    
    # Add text at the bottom
    plt.figtext(0.5, 0.01, f"Text: {args.text[:80] + '...' if len(args.text) > 80 else args.text}", 
                ha='center', fontsize=10, wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save or display the visualization
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
        plt.savefig(args.output_file)
        print(f"Visualization saved to {args.output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with multimodal sentiment analysis model')
    
    # Input parameters
    parser.add_argument('--text', type=str, required=True,
                        help='Input text for sentiment analysis')
    parser.add_argument('--image', type=str, default='',
                        help='Path to input image for sentiment analysis')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for BERT tokenizer')
    
    # Model parameters
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'attention'],
                        help='Method for fusing features')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='dataset/LabeledText.xlsx',
                        help='Path to data for quick training if needed')
    parser.add_argument('--images_dir', type=str, default='dataset',
                        help='Path to images for quick training if needed')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize prediction results')
    parser.add_argument('--output_file', type=str, default='',
                        help='Path to save visualization (if not provided, display instead)')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--text_only_mode', action='store_true', default=False,
                        help='Use only text data with dummy images')
    parser.add_argument('--try_train', action='store_true',
                        help='Try to train a quick model if checkpoint not found')
    
    args = parser.parse_args()
    
    # Run inference
    predict_sentiment(args) 
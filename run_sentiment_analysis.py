import argparse
import os
import torch
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import glob
import numpy as np

from model import MultimodalSentimentModel


def analyze_sentiment(text, image_path=None, checkpoint='checkpoints/best_model.pth', force_cpu=True, text_only_mode=False):
    """
    Analyze sentiment of text and optional image input
    
    Args:
        text (str): Input text for sentiment analysis
        image_path (str): Path to input image for sentiment analysis (optional)
        checkpoint (str): Path to model checkpoint
        force_cpu (bool): Whether to force CPU usage
        text_only_mode (bool): If True, only use text data
        
    Returns:
        dict: Sentiment analysis results
    """
    # Set device
    device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directories if they don't exist
    os.makedirs(os.path.dirname(checkpoint) if os.path.dirname(checkpoint) else '.', exist_ok=True)
    
    # If no checkpoint exists yet, inform the user
    if not os.path.exists(checkpoint):
        print(f"No checkpoint found at {checkpoint}. Please train the model first using:")
        print("python train.py --data_path dataset/LabeledText.xlsx --images_dir dataset --epochs 5 --batch_size 8 --freeze_bert --freeze_resnet")
        return None
    
    # Initialize model
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=0.3,
        image_dropout=0.3,
        fusion_dropout=0.3,
        freeze_bert=True,
        freeze_resnet=True,
        fusion_method='concat'
    ).to(device)
    
    # Load model checkpoint
    try:
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"Loaded model from {checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize text
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return None
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Process image or create dummy image
    has_image = image_path and os.path.exists(image_path) and not text_only_mode
    original_image = None
    
    if has_image:
        try:
            print(f"Loading image from: {image_path}")
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()  # Store original for visualization
            
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
            print(f"Error loading image {image_path}: {e}")
            has_image = False
            image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    else:
        if text_only_mode:
            print("Using text-only mode with dummy image tensor")
        elif image_path:
            print(f"Image file {image_path} not found. Using a dummy image tensor.")
        else:
            print("No image provided. Using a dummy image tensor.")
            
        image_tensor = torch.zeros(1, 3, 224, 224).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image_tensor)
        
        # Get probabilities
        fusion_probs = torch.nn.functional.softmax(outputs['fusion_logits'], dim=1)
        text_probs = torch.nn.functional.softmax(outputs['text_logits'], dim=1)
        image_probs = torch.nn.functional.softmax(outputs['image_logits'], dim=1)
        
        # Get predicted classes - standard approach, no threshold
        fusion_pred = torch.argmax(fusion_probs, dim=1).item()
        text_pred = torch.argmax(text_probs, dim=1).item()
        image_pred = torch.argmax(image_probs, dim=1).item()
    
    # Map predictions to sentiment labels
    sentiments = ['Positive', 'Neutral', 'Negative']
    result = {
        'text': text,
        'image_path': image_path,
        'fusion': {
            'prediction': fusion_pred,
            'sentiment': sentiments[fusion_pred],
            'probabilities': fusion_probs[0].cpu().numpy(),
            'confidence': float(fusion_probs[0, fusion_pred].item())
        },
        'text_only': {
            'prediction': text_pred,
            'sentiment': sentiments[text_pred],
            'probabilities': text_probs[0].cpu().numpy()
        },
        'image_only': {
            'prediction': image_pred,
            'sentiment': sentiments[image_pred],
            'probabilities': image_probs[0].cpu().numpy()
        }
    }
    
    return result


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


def plot_sentiment_results(results, output_file=None):
    """
    Plot sentiment analysis results
    
    Args:
        results (dict): Results from sentiment analysis
        output_file (str, optional): Path to save the plot
    """
    # Define sentiment classes
    sentiments = ['Positive', 'Neutral', 'Negative']
    
    # Create figure with 3 subplots (one per model)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot fusion model results
    fusion_probs = results['fusion']['probabilities'] * 100
    axes[0].bar(sentiments, fusion_probs, color=['green', 'gray', 'red'])
    axes[0].set_title('Fusion Model (Text + Image)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(fusion_probs):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Plot text-only model results
    text_probs = results['text_only']['probabilities'] * 100
    axes[1].bar(sentiments, text_probs, color=['green', 'gray', 'red'])
    axes[1].set_title('Text-Only Model')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(text_probs):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Plot image-only model results
    image_probs = results['image_only']['probabilities'] * 100
    axes[2].bar(sentiments, image_probs, color=['green', 'gray', 'red'])
    axes[2].set_title('Image-Only Model')
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(image_probs):
        axes[2].text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Add overall title and adjust layout
    plt.suptitle('Sentiment Analysis Results', fontsize=16)
    plt.tight_layout()
    
    # Save plot if output file provided
    if output_file:
        plt.savefig(output_file)
    
    # Show plot
    plt.show()


def find_example_images():
    """Find some example images from the dataset for demo purposes"""
    image_dirs = ['dataset/Images/Images/positive', 'dataset/Images/Images/Neutral', 'dataset/Images/Images/Negative']
    example_images = []
    
    for image_dir in image_dirs:
        if os.path.exists(image_dir):
            images = glob.glob(os.path.join(image_dir, "*.jpg"))
            if images:
                example_images.append(images[0])  # Add the first image from each sentiment directory
    
    return example_images


def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Tool')
    
    parser.add_argument('--text', type=str, default=None,
                        help='Input text for sentiment analysis')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image for sentiment analysis')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize prediction results')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save visualization')
    parser.add_argument('--force_cpu', action='store_true', default=True,
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--text_only', action='store_true', default=False,
                        help='Use text-only mode, ignoring images')
    
    args = parser.parse_args()
    
    # Find example images for interactive mode
    example_images = find_example_images()
    has_examples = len(example_images) > 0
    
    # Interactive mode if no text is provided
    if args.text is None:
        print("="*50)
        print("Welcome to the Multimodal Sentiment Analysis Tool!")
        print("="*50)
        print("This tool analyzes the sentiment of text and/or images as positive, neutral, or negative.")
        print("Enter 'q' or 'quit' to exit.")
        
        if has_examples:
            print("\nExample images available:")
            for i, img in enumerate(example_images):
                print(f"  {i+1}. {img}")
        
        print("="*50)
        
        while True:
            text = input("\nEnter text for sentiment analysis: ")
            if text.lower() in ['q', 'quit', 'exit']:
                break
                
            if not text.strip():
                continue
            
            # Ask for image if examples are available
            image_path = None
            if has_examples and not args.text_only:
                use_image = input("Use an image for multimodal analysis? (y/n): ").lower()
                if use_image == 'y':
                    if len(example_images) == 1:
                        image_path = example_images[0]
                        print(f"Using example image: {image_path}")
                    else:
                        try:
                            choice = input(f"Choose an image (1-{len(example_images)}) or enter a custom path: ")
                            if choice.isdigit() and 1 <= int(choice) <= len(example_images):
                                image_path = example_images[int(choice)-1]
                                print(f"Using example image: {image_path}")
                            elif os.path.exists(choice):
                                image_path = choice
                                print(f"Using custom image: {image_path}")
                            else:
                                print("Invalid choice or path. Using text-only analysis.")
                        except:
                            print("Invalid input. Using text-only analysis.")
            
            results = analyze_sentiment(
                text, 
                image_path=image_path, 
                checkpoint=args.checkpoint, 
                force_cpu=args.force_cpu,
                text_only_mode=args.text_only
            )
            
            if results and (args.visualize or input("\nVisualize results? (y/n): ").lower() == 'y'):
                plot_sentiment_results(results, args.output_file)
    else:
        # One-time analysis
        results = analyze_sentiment(
            args.text, 
            image_path=args.image, 
            checkpoint=args.checkpoint, 
            force_cpu=args.force_cpu,
            text_only_mode=args.text_only
        )
        if results:
            # Display the results
            display_prediction(results)
            
            if args.visualize:
                plot_sentiment_results(results, args.output_file)


def display_prediction(result):
    """
    Display prediction results
    
    Args:
        result (dict): Prediction results from analyze_sentiment
    """
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nInput text: \"{result['text']}\"")
    if result['image_path']:
        print(f"Input image: {result['image_path']}")
    
    # Format for displaying probabilities
    def format_probs(probs):
        sentiments = ['Positive', 'Neutral', 'Negative']
        lines = []
        for i, sentiment in enumerate(sentiments):
            percentage = probs[i] * 100
            bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
            lines.append(f"  {sentiment}: {bar} {percentage:.2f}%")
        return '\n'.join(lines)
    
    # Display results for each model
    print("\nText-only model:")
    print(f"  Predicted sentiment: {result['text_only']['sentiment']}")
    print("\n" + format_probs(result['text_only']['probabilities']))
    
    if result['image_path']:
        print("\nImage-only model:")
        print(f"  Predicted sentiment: {result['image_only']['sentiment']}")
        print("\n" + format_probs(result['image_only']['probabilities']))
    
    print("\nFusion model (combined text and image):")
    print(f"  Predicted sentiment: {result['fusion']['sentiment']}")
    print("\n" + format_probs(result['fusion']['probabilities']))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main() 
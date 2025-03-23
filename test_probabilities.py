import os
import torch
from transformers import BertTokenizer
import argparse
import matplotlib.pyplot as plt
import numpy as np

from model import MultimodalSentimentModel

# Define test sentences with clear sentiments
POSITIVE_SENTENCES = [
    "This is absolutely amazing! I love it so much!",
    "The experience was fantastic and exceeded all my expectations.",
    "I'm extremely happy with the results. Best purchase ever!"
]

NEUTRAL_SENTENCES = [
    "This product is average, neither good nor bad.",
    "The weather today is partly cloudy with mild temperatures.",
    "It works as expected, nothing special to report."
]

NEGATIVE_SENTENCES = [
    "This is terrible! I hate it and want a refund immediately.",
    "The worst experience of my life. Everything was broken.",
    "I'm extremely disappointed with the terrible quality."
]

def test_sentiment_model(model_path, force_cpu=True):
    """
    Test the sentiment model with predefined sentences and display probability distributions
    
    Args:
        model_path (str): Path to the model checkpoint
        force_cpu (bool): Whether to force CPU usage
    """
    # Set device
    device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}")
        return
    
    # Load model
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=0.3,
        image_dropout=0.3,
        fusion_dropout=0.3,
        freeze_bert=True,
        freeze_resnet=True,
        fusion_method='attention'
    ).to(device)
    
    # Load model weights with compatibility handling
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter out incompatible keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    # Update model dict
    model_dict.update(pretrained_dict)
    
    # Load the filtered weights
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded compatible weights from {model_path}")
    
    model.eval()
    
    # Set up tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create a dummy image tensor (we're focusing on text only)
    dummy_image = torch.zeros(1, 3, 224, 224).to(device)
    
    # Process all sentences
    all_results = []
    all_sentences = []
    
    # Process positive sentences
    print("\nTesting POSITIVE sentences:")
    for sentence in POSITIVE_SENTENCES:
        result = process_sentence(sentence, model, tokenizer, dummy_image, device)
        all_results.append(result)
        all_sentences.append(sentence)
        print_result(sentence, result)
    
    # Process neutral sentences
    print("\nTesting NEUTRAL sentences:")
    for sentence in NEUTRAL_SENTENCES:
        result = process_sentence(sentence, model, tokenizer, dummy_image, device)
        all_results.append(result)
        all_sentences.append(sentence)
        print_result(sentence, result)
    
    # Process negative sentences
    print("\nTesting NEGATIVE sentences:")
    for sentence in NEGATIVE_SENTENCES:
        result = process_sentence(sentence, model, tokenizer, dummy_image, device)
        all_results.append(result)
        all_sentences.append(sentence)
        print_result(sentence, result)
    
    # Visualize all results
    visualize_results(all_sentences, all_results)


def process_sentence(sentence, model, tokenizer, image, device):
    """
    Process a single sentence and return prediction results
    
    Args:
        sentence (str): The input sentence
        model: The sentiment model
        tokenizer: The BERT tokenizer
        image: The image tensor (dummy for text-only)
        device: The device to run on
        
    Returns:
        dict: Prediction results
    """
    # Tokenize text
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image)
        
        # Calculate probabilities
        fusion_probs = torch.nn.functional.softmax(outputs['fusion_logits'], dim=1)
        text_probs = torch.nn.functional.softmax(outputs['text_logits'], dim=1)
        
        # Get predicted class - standard approach, no threshold
        fusion_class = torch.argmax(fusion_probs, dim=1).item()
        text_class = torch.argmax(text_probs, dim=1).item()
    
    # Return results
    return {
        'fusion_class': fusion_class,
        'text_class': text_class,
        'fusion_probs': fusion_probs[0].cpu().numpy(),
        'text_probs': text_probs[0].cpu().numpy()
    }


def print_result(sentence, result):
    """
    Print prediction results for a sentence
    
    Args:
        sentence (str): The input sentence
        result (dict): Prediction results
    """
    class_names = ['Positive', 'Neutral', 'Negative']
    print(f"\nSentence: \"{sentence}\"")
    print(f"Predicted sentiment: {class_names[result['fusion_class']]}")
    
    print("\nText model probabilities:")
    for i, name in enumerate(class_names):
        percentage = result['text_probs'][i] * 100
        bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
        print(f"  {name}: {bar} {percentage:.2f}%")
    
    print("\nFusion model probabilities:")
    for i, name in enumerate(class_names):
        percentage = result['fusion_probs'][i] * 100
        bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
        print(f"  {name}: {bar} {percentage:.2f}%")


def visualize_results(sentences, results):
    """
    Visualize prediction results for all sentences
    
    Args:
        sentences (list): List of input sentences
        results (list): List of prediction results
    """
    # Prepare data
    class_names = ['Positive', 'Neutral', 'Negative']
    categories = ['Positive Sentences', 'Neutral Sentences', 'Negative Sentences']
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # Plot each sentence type
    for i, category_start in enumerate([0, 3, 6]):
        for j, sentence_idx in enumerate(range(category_start, category_start + 3)):
            ax = axes[j, i]
            
            # Get data for this sentence
            result = results[sentence_idx]
            sentence = sentences[sentence_idx]
            
            # Set up bar positions
            x = range(3)
            width = 0.4
            
            # Plot probabilities
            bars1 = ax.bar([p - width/2 for p in x], result['text_probs'] * 100, width, label='Text Model')
            bars2 = ax.bar([p + width/2 for p in x], result['fusion_probs'] * 100, width, label='Fusion Model')
            
            # Highlight predicted class
            predicted_class = result['fusion_class']
            bars2[predicted_class].set_edgecolor('black')
            bars2[predicted_class].set_linewidth(2)
            
            # Set labels
            ax.set_title(f"{categories[i].split()[0]} Example {j+1}")
            ax.set_xticks(x)
            ax.set_xticklabels(['Pos', 'Neu', 'Neg'])
            ax.set_ylim(0, 100)
            
            # Add sentence as text
            if j == 0:
                ax.annotate(categories[i], xy=(0.5, 1.15), xycoords='axes fraction', fontsize=14,
                           ha='center', va='center', fontweight='bold')
            
            # Only add legend to first plot
            if i == 0 and j == 0:
                ax.legend(loc='upper right')
                
            # Add probability values on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 5:  # Only show if more than 5%
                        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                                f'{height:.1f}%', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, top=0.9)
    plt.suptitle("Sentiment Analysis Probability Distributions", fontsize=16, y=0.98)
    
    # Save or show
    plt.savefig("sentiment_probabilities.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test sentiment model with predefined sentences")
    parser.add_argument('--model_path', type=str, default='improved_checkpoints/best_model.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--force_cpu', action='store_true', default=True,
                        help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    test_sentiment_model(args.model_path, args.force_cpu) 
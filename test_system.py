import os
import torch
import argparse
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms

from model import MultimodalSentimentModel
from data_utils import TwitterSentimentDataset

def test_model_loading():
    """Test that the model can be initialized properly"""
    print("Testing model initialization...")
    
    # Initialize model
    model = MultimodalSentimentModel(
        num_classes=3,
        text_dropout=0.1,
        image_dropout=0.1,
        fusion_dropout=0.1,
        freeze_bert=True,
        freeze_resnet=True,
        fusion_method="concat"
    )
    
    print("✓ Model initialized successfully!")
    return model

def test_dataset_loading(data_path, images_dir):
    """Test that the dataset can be loaded properly"""
    print(f"\nTesting dataset loading with:")
    print(f"- Data path: {data_path}")
    print(f"- Images directory: {images_dir}")
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file {data_path} does not exist!")
        return None
    
    if not os.path.exists(images_dir):
        print(f"❌ Error: Images directory {images_dir} does not exist!")
        return None
        
    try:
        # Try to create dataset
        dataset = TwitterSentimentDataset(data_path, images_dir, max_length=128)
        print(f"✓ Dataset loaded successfully with {len(dataset)} samples!")
        
        # Check a sample
        sample = dataset[0]
        print(f"\nSample data:")
        print(f"- Text: {sample['text'][:50]}..." if len(sample['text']) > 50 else f"- Text: {sample['text']}")
        print(f"- Image shape: {sample['image'].shape}")
        print(f"- Label: {sample['label'].item()}")
        
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def run_simple_inference(model, text, image_path):
    """Run a simple inference test"""
    print(f"\nTesting inference with:")
    print(f"- Text: {text}")
    print(f"- Image: {image_path}")
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file {image_path} does not exist!")
        return
    
    try:
        # Tokenize text
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
        
        # Process image
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
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            
            # Get predictions
            fusion_probs = torch.nn.functional.softmax(outputs['fusion_logits'], dim=1)
            
            # Get predicted class
            fusion_class = torch.argmax(fusion_probs, dim=1).item()
        
        # Map class indices to labels
        class_names = ['Positive', 'Neutral', 'Negative']
        
        print(f"\nPrediction: {class_names[fusion_class]}")
        print(f"Probabilities:")
        for i, name in enumerate(class_names):
            print(f"  {name}: {fusion_probs[0, i].item() * 100:.2f}%")
            
        print("\n✓ Inference test completed successfully!")
    except Exception as e:
        print(f"❌ Error during inference: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Multimodal Sentiment Analysis System')
    
    parser.add_argument('--data_path', type=str, default='dataset/LabeledText.xlsx',
                      help='Path to the labeled text Excel file')
    parser.add_argument('--images_dir', type=str, default='dataset',
                      help='Directory containing the image folders')
    parser.add_argument('--test_text', type=str, default='I love this beautiful day!',
                      help='Text to use for inference test')
    parser.add_argument('--test_image', type=str, default=None,
                      help='Image to use for inference test')
    
    args = parser.parse_args()
    
    print("=== Multimodal Sentiment Analysis System Test ===\n")
    
    # Test model loading
    model = test_model_loading()
    
    # Test dataset loading
    dataset = test_dataset_loading(args.data_path, args.images_dir)
    
    # Run simple inference if dataset is loaded successfully
    if dataset is not None and model is not None:
        # If no test image is provided, use the first image from the dataset
        test_image = args.test_image
        if test_image is None:
            # Try to find a sample image in the dataset
            for folder in ['Positive', 'Neutral', 'Negative']:
                image_dir = os.path.join(args.images_dir, 'Images', folder)
                if os.path.exists(image_dir):
                    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
                    if images:
                        test_image = os.path.join(image_dir, images[0])
                        break
        
        if test_image is not None:
            run_simple_inference(model, args.test_text, test_image)
        else:
            print("\n❌ No test image found. Skipping inference test.")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 
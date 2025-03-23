import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer, logging
import openpyxl
import glob

# Set transformers logging to error only
logging.set_verbosity_error()


class TwitterSentimentDataset(Dataset):
    """
    Dataset class for Twitter sentiment analysis with images
    """
    def __init__(self, text_data_path, images_dir, transform=None, max_length=128, text_only_mode=False):
        """
        Initialize the dataset
        
        Args:
            text_data_path (str): Path to the labeled text Excel file
            images_dir (str): Directory containing the image folders
            transform (callable, optional): Transform to apply to images
            max_length (int): Maximum sequence length for BERT tokenizer
            text_only_mode (bool): If True, use only text data and create dummy images
        """
        # Load the data
        self.text_data = pd.read_excel(text_data_path)
        print(f"Loaded dataset with columns: {self.text_data.columns.tolist()}")
        self.images_dir = images_dir
        self.max_length = max_length
        self.text_only_mode = text_only_mode
        
        # Create column mappings from actual columns to expected ones
        self.column_mapping = {
            'image_id_col': 'File Name',  # Column containing image IDs
            'text_col': 'Caption',        # Column containing text data
            'sentiment_col': 'LABEL'      # Column containing sentiment labels
        }
        
        # Validate required columns exist
        for col_key, col_name in self.column_mapping.items():
            if col_name not in self.text_data.columns:
                raise ValueError(f"Required column '{col_name}' not found in dataset. Available columns: {self.text_data.columns.tolist()}")
        
        # Set up image transformation
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
            
        # Set up BERT tokenizer
        try:
            # Try to load the tokenizer with internet connection
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Warning: {e}")
            print("Attempting to load tokenizer from local cache...")
            try:
                # Try with offline mode
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
            except Exception as local_e:
                print(f"Error loading BERT tokenizer: {local_e}")
                raise RuntimeError("Could not load BERT tokenizer. Please ensure internet connectivity or pre-downloaded models.")
        
        # Map sentiment labels to numeric values
        self.sentiment_map = {
            'positive': 0,
            'neutral': 1,
            'negative': 2,
            'POSITIVE': 0,
            'NEUTRAL': 1,
            'NEGATIVE': 2,
            'Positive': 0,
            'Neutral': 1,
            'Negative': 2,
            # Add any other formats that might be in your data
        }
        
        # Create a mapping of image IDs to actual file paths to speed up image loading
        self.image_path_map = {}
        
        if not self.text_only_mode:
            print("Initializing image path mapping...")
            # Scan through image directories and find all image files
            for sentiment in ['positive', 'Neutral', 'Negative']:
                sentiment_path = os.path.join(self.images_dir, 'Images', sentiment)
                if os.path.exists(sentiment_path):
                    image_files = glob.glob(os.path.join(sentiment_path, "*.jpg"))
                    for img_path in image_files:
                        # Extract the image ID (filename without extension)
                        img_id = os.path.basename(img_path).split('.')[0]
                        self.image_path_map[img_id] = img_path
            
            print(f"Found {len(self.image_path_map)} images in the dataset")
            
            # Clean up data entries that don't have matching images
            if len(self.image_path_map) > 0:
                # Convert numeric File Name values to strings
                self.text_data[self.column_mapping['image_id_col']] = self.text_data[self.column_mapping['image_id_col']].astype(str)
                
                # Check if we need to strip .txt or other extensions
                sample_ids = self.text_data[self.column_mapping['image_id_col']].values[:10]
                if any('.txt' in id for id in sample_ids):
                    print("Detected .txt extensions in image IDs, stripping them...")
                    self.text_data[self.column_mapping['image_id_col']] = self.text_data[self.column_mapping['image_id_col']].apply(
                        lambda x: x.replace('.txt', '') if isinstance(x, str) and '.txt' in x else x
                    )
        
        # Print sample rows for debugging
        if len(self.text_data) > 0:
            print(f"Sample row: {self.text_data.iloc[0].to_dict()}")
            
            # Check sentiments in dataset
            sentiments = self.text_data[self.column_mapping['sentiment_col']].unique()
            print(f"Unique sentiment values in dataset: {sentiments}")
            
            # Check if sentiments are valid
            invalid_sentiments = []
            for s in sentiments:
                if str(s).lower() not in self.sentiment_map and str(s).upper() not in self.sentiment_map:
                    invalid_sentiments.append(s)
            
            if invalid_sentiments:
                print(f"WARNING: Found invalid sentiment values: {invalid_sentiments}")
                print(f"Valid sentiment values are: {list(self.sentiment_map.keys())}")
        
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        # Get text and related info
        row = self.text_data.iloc[idx]
        text = str(row[self.column_mapping['text_col']])
        image_id = str(row[self.column_mapping['image_id_col']])
        sentiment_raw = str(row[self.column_mapping['sentiment_col']])
        
        # Convert sentiment to lowercase for standardization
        sentiment = sentiment_raw.lower()
        
        # Debug info for the first few items
        if idx < 3:
            print(f"Processing item {idx}:")
            print(f"  - Image ID: {image_id}")
            print(f"  - Text: {text[:30]}..." if len(text) > 30 else f"  - Text: {text}")
            print(f"  - Raw sentiment: {sentiment_raw}")
            print(f"  - Processed sentiment: {sentiment}")
        
        # Tokenize text for BERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Load image or create dummy image
        image_loaded = False
        
        if not self.text_only_mode:
            # Try direct mapping first
            if image_id in self.image_path_map:
                image_path = self.image_path_map[image_id]
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                    image_loaded = True
                    if idx < 3:
                        print(f"  - Successfully loaded image from: {image_path}")
                except Exception as e:
                    if idx < 3:
                        print(f"  - Error loading mapped image {image_path}: {e}")
            
            # If not found or failed to load, try searching for the image
            if not image_loaded:
                for sentiment_folder in ['positive', 'Neutral', 'Negative']:
                    # Try different variations of the filename
                    potential_paths = [
                        os.path.join(self.images_dir, 'Images', sentiment_folder, f"{image_id}.jpg"),
                        os.path.join(self.images_dir, 'Images', sentiment_folder, f"{image_id.split('.')[0]}.jpg"),
                    ]
                    
                    for path in potential_paths:
                        if os.path.exists(path):
                            try:
                                image = Image.open(path).convert('RGB')
                                image = self.transform(image)
                                image_loaded = True
                                # Add to the mapping for future use
                                self.image_path_map[image_id] = path
                                if idx < 3:
                                    print(f"  - Found and loaded image from: {path}")
                                break
                            except Exception as e:
                                if idx < 3:
                                    print(f"  - Error loading image {path}: {e}")
                    
                    if image_loaded:
                        break
        
        # If we couldn't load an image or in text_only_mode, create a dummy image
        if not image_loaded:
            image = torch.zeros(3, 224, 224)
            if idx < 3 and not self.text_only_mode:
                print(f"  - Using dummy image tensor (image not found)")
        
        # Check if sentiment key exists in mapping, use default if not
        if sentiment not in self.sentiment_map:
            if idx < 10:
                print(f"Warning: Sentiment '{sentiment}' not found in mapping. Using neutral sentiment.")
            # Default to neutral if sentiment is unknown
            sentiment_index = 1
        else:
            sentiment_index = self.sentiment_map[sentiment]
        
        # Create label tensor
        label = torch.tensor(sentiment_index)
        
        return {
            'text': text,
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }


def get_data_loaders(text_data_path, images_dir, batch_size=16, val_split=0.15, test_split=0.15, 
                    random_state=42, max_length=128, text_only_mode=False):
    """
    Create train, validation, and test data loaders
    
    Args:
        text_data_path (str): Path to the labeled text Excel file
        images_dir (str): Directory containing the image folders
        batch_size (int): Batch size for data loaders
        val_split (float): Proportion of data for validation
        test_split (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        max_length (int): Maximum sequence length for BERT tokenizer
        text_only_mode (bool): If True, use only text data and create dummy images
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = TwitterSentimentDataset(text_data_path, images_dir, max_length=max_length, text_only_mode=text_only_mode)
    
    # Split dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    test_size = int(np.floor(test_split * dataset_size))
    val_size = int(np.floor(val_split * dataset_size))
    train_size = dataset_size - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create samplers
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler
    )
    
    test_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler
    )
    
    return train_loader, val_loader, test_loader


def preprocess_excel_data(excel_path):
    """
    Preprocess the Excel data by cleaning text, mapping image IDs, etc.
    
    Args:
        excel_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: Processed dataframe
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Ensure required columns exist
    required_columns = ['Caption', 'File Name', 'LABEL']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in data. Creating empty column.")
            df[col] = ""
    
    # Clean text
    df['Caption'] = df['Caption'].astype(str).apply(lambda x: x.strip())
    
    # Handle potential NaN values
    df = df.fillna('')
    
    # Ensure sentiment labels are standardized
    sentiment_map = {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
        'NEUTRAL': 'neutral',
        'POS': 'positive',
        'NEG': 'negative',
        'NEU': 'neutral'
    }
    
    df['LABEL'] = df['LABEL'].astype(str).apply(
        lambda x: sentiment_map.get(x.upper(), x.lower())
    )
    
    # Filter out rows with invalid sentiments
    valid_sentiments = ['positive', 'negative', 'neutral', 
                        'POSITIVE', 'NEGATIVE', 'NEUTRAL', 
                        'Positive', 'Negative', 'Neutral']
    df = df[df['LABEL'].isin(valid_sentiments)]
    
    # Clean up image IDs if needed (remove .txt extensions)
    df['File Name'] = df['File Name'].astype(str).apply(
        lambda x: x.replace('.txt', '') if isinstance(x, str) and '.txt' in x else x
    )
    
    return df
import torch
import torch.nn as nn
from transformers import BertModel, logging

# Set transformers logging to error only
logging.set_verbosity_error()


class TextSentimentModel(nn.Module):
    """
    Text-based sentiment analysis model using BERT
    """
    def __init__(self, num_classes=3, dropout=0.1, freeze_bert=False):
        """
        Initialize the model
        
        Args:
            num_classes (int): Number of sentiment classes
            dropout (float): Dropout probability
            freeze_bert (bool): Whether to freeze BERT parameters
        """
        super(TextSentimentModel, self).__init__()
        
        # Load pre-trained BERT model
        try:
            # Try to load the model with internet connection
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"Warning: {e}")
            print("Attempting to load from local cache or failing gracefully...")
            try:
                # Try with offline mode
                self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
            except Exception as local_e:
                print(f"Error loading BERT model: {local_e}")
                print("Creating BERT model with default configuration instead.")
                from transformers import BertConfig
                config = BertConfig()
                self.bert = BertModel(config)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # BERT output dimension
        self.bert_output_dim = self.bert.config.hidden_size
        
        # Classifier layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.bert_output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            tuple: (logits, features)
                - logits (torch.Tensor): Class logits
                - features (torch.Tensor): Extracted text features for fusion
        """
        # Pass inputs through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        x = self.dropout(pooled_output)
        
        # Pass through classifier layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        features = self.fc2(x)
        x = self.relu(features)
        x = self.dropout(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits, features
    
    def extract_features(self, input_ids, attention_mask):
        """
        Extract text features only, without classification
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Extracted text features
        """
        # Pass inputs through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply layers up to feature extraction
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        features = self.fc2(x)
        
        return features 
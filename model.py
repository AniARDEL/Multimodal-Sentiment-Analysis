import torch
import torch.nn as nn

from text_model import TextSentimentModel
from image_model import ImageSentimentModel
from fusion_model import FeatureFusionModel


class MultimodalSentimentModel(nn.Module):
    """
    Complete multimodal sentiment analysis model integrating text and image components
    """
    def __init__(self, num_classes=3, text_dropout=0.1, image_dropout=0.1, fusion_dropout=0.1, 
                 freeze_bert=True, freeze_resnet=True, fusion_method='concat'):
        """
        Initialize the model
        
        Args:
            num_classes (int): Number of sentiment classes
            text_dropout (float): Dropout probability for text model
            image_dropout (float): Dropout probability for image model
            fusion_dropout (float): Dropout probability for fusion model
            freeze_bert (bool): Whether to freeze BERT parameters
            freeze_resnet (bool): Whether to freeze ResNet parameters
            fusion_method (str): Method for fusing features ('concat' or 'attention')
        """
        super(MultimodalSentimentModel, self).__init__()
        
        # Initialize component models
        self.text_model = TextSentimentModel(num_classes=num_classes, dropout=text_dropout, freeze_bert=freeze_bert)
        self.image_model = ImageSentimentModel(num_classes=num_classes, dropout=image_dropout, freeze_backbone=freeze_resnet)
        
        # Feature dimensions from each model
        text_feature_dim = 256  # The output dimension from TextSentimentModel's feature extractor
        image_feature_dim = 256  # The output dimension from ImageSentimentModel's feature extractor
        
        # Initialize fusion model
        self.fusion_model = FeatureFusionModel(
            text_feature_dim=text_feature_dim,
            image_feature_dim=image_feature_dim,
            hidden_dim=512,
            num_classes=num_classes,
            dropout=fusion_dropout,
            fusion_method=fusion_method
        )
        
    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): Token IDs for text
            attention_mask (torch.Tensor): Attention mask for text
            images (torch.Tensor): Input images
            
        Returns:
            dict: Dictionary containing various outputs
                - fusion_logits: Final fused prediction logits
                - text_logits: Text-only prediction logits
                - image_logits: Image-only prediction logits
                - text_features: Extracted text features
                - image_features: Extracted image features
        """
        # Process text input
        text_logits, text_features = self.text_model(input_ids, attention_mask)
        
        # Process image input
        image_logits, image_features = self.image_model(images)
        
        # Fuse features and get final prediction
        fusion_logits = self.fusion_model(text_features, image_features)
        
        return {
            'fusion_logits': fusion_logits,
            'text_logits': text_logits,
            'image_logits': image_logits,
            'text_features': text_features,
            'image_features': image_features
        }
        
    def predict(self, input_ids, attention_mask, images):
        """
        Make predictions
        
        Args:
            input_ids (torch.Tensor): Token IDs for text
            attention_mask (torch.Tensor): Attention mask for text
            images (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        outputs = self.forward(input_ids, attention_mask, images)
        _, predictions = torch.max(outputs['fusion_logits'], dim=1)
        return predictions
import torch
import torch.nn as nn


class FeatureFusionModel(nn.Module):
    """
    Model for fusing text and image features for sentiment analysis
    """
    def __init__(self, text_feature_dim=256, image_feature_dim=256, hidden_dim=512, num_classes=3, dropout=0.1, fusion_method='concat'):
        """
        Initialize the model
        
        Args:
            text_feature_dim (int): Dimension of text features
            image_feature_dim (int): Dimension of image features
            hidden_dim (int): Dimension of hidden layer
            num_classes (int): Number of sentiment classes
            dropout (float): Dropout probability
            fusion_method (str): Method for fusing features ('concat' or 'attention')
        """
        super(FeatureFusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            # Concatenation fusion
            self.fusion_dim = text_feature_dim + image_feature_dim
            self.fusion = lambda x, y: torch.cat((x, y), dim=1)
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.fusion_dim = text_feature_dim
            self.attention = nn.Sequential(
                nn.Linear(text_feature_dim + image_feature_dim, text_feature_dim),
                nn.Tanh(),
                nn.Linear(text_feature_dim, 1),
                nn.Softmax(dim=1)
            )
            self.fusion = self._attention_fusion
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Classifier layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.fusion_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def _attention_fusion(self, text_features, image_features):
        """
        Attention-based fusion of text and image features
        
        Args:
            text_features (torch.Tensor): Text features
            image_features (torch.Tensor): Image features
            
        Returns:
            torch.Tensor: Fused features
        """
        # Concatenate features for attention calculation
        combined = torch.cat((text_features, image_features), dim=1)
        
        # Compute attention weights
        attention_weights = self.attention(combined)
        
        # Apply attention to features
        attended_text = text_features * attention_weights
        attended_image = image_features * (1 - attention_weights)
        
        return attended_text + attended_image
        
    def forward(self, text_features, image_features):
        """
        Forward pass
        
        Args:
            text_features (torch.Tensor): Features from text model
            image_features (torch.Tensor): Features from image model
            
        Returns:
            torch.Tensor: Class logits
        """
        # Fuse features
        fused_features = self.fusion(text_features, image_features)
        
        # Apply classifier layers
        x = self.dropout(fused_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final classification
        logits = self.fc2(x)
        
        return logits 
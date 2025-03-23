import torch
import torch.nn as nn
import torchvision.models as models


class ImageSentimentModel(nn.Module):
    """
    Image-based sentiment analysis model using ResNet50
    """
    def __init__(self, num_classes=3, dropout=0.1, freeze_backbone=True):
        """
        Initialize the model
        
        Args:
            num_classes (int): Number of sentiment classes
            dropout (float): Dropout probability
            freeze_backbone (bool): Whether to freeze ResNet parameters
        """
        super(ImageSentimentModel, self).__init__()
        
        # Load pre-trained ResNet50 model
        try:
            # Try the newer method first
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except TypeError:
            # Fall back to older method if needed
            self.resnet = models.resnet50(pretrained=True)
        
        # Freeze ResNet parameters if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # ResNet50 output dimension
        self.resnet_output_dim = 2048
        
        # Classifier layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.resnet_output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images (torch.Tensor): Input images
            
        Returns:
            tuple: (logits, features)
                - logits (torch.Tensor): Class logits
                - features (torch.Tensor): Extracted image features for fusion
        """
        # Pass inputs through ResNet
        x = self.features(images)
        x = torch.flatten(x, 1)
        
        # Apply dropout
        x = self.dropout(x)
        
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
    
    def extract_features(self, images):
        """
        Extract image features only, without classification
        
        Args:
            images (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Extracted image features
        """
        # Pass inputs through ResNet
        x = self.features(images)
        x = torch.flatten(x, 1)
        
        # Apply layers up to feature extraction
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        features = self.fc2(x)
        
        return features 
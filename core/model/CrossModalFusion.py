import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionFusion(nn.Module):
    """
    跨模态注意力融合模块
    替换简单的torch.cat，学习模态间动态依赖关系
    """
    
    def __init__(self, input_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Multi-head attention for cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Modal importance weights (learnable)
        self.modal_weights = nn.Parameter(torch.ones(3))  # metric, log, trace
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modal_features):
        """
        Args:
            modal_features: dict with keys ['metric', 'log', 'trace']
                           Each value: [batch_size, feature_dim] 
        
        Returns:
            fused_features: [batch_size, feature_dim] - same dimension as input
        """
        
        # Stack modalities: [batch_size, 3, feature_dim]
        modalities = ['metric', 'log', 'trace']
        features = torch.stack([modal_features[mod] for mod in modalities], dim=1)
        
        # Cross-modal attention
        attended_features, attention_weights = self.cross_attention(
            query=features,
            key=features, 
            value=features
        )
        
        # Modal importance weighting
        modal_weights = torch.softmax(self.modal_weights, dim=0)  # [3]
        weighted_features = attended_features * modal_weights.view(1, -1, 1)
        
        # Residual connection + Layer norm
        output_features = self.layer_norm(weighted_features + features)
        
        # Weighted fusion to single feature
        fused_features = torch.sum(output_features * modal_weights.view(1, -1, 1), dim=1)
        fused_features = self.dropout(fused_features)
        
        return fused_features

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
        
    def forward(self, modal_features, modality_masks=None):
        """
        Args:
            modal_features: dict with keys from ['metric', 'log', 'trace']
                           Each value: [batch_size, feature_dim]
            modality_masks: [batch_size, 3] tensor indicating available modalities per sample
                           If None, assumes all modalities are available for all samples
        
        Returns:
            fused_features: [batch_size, feature_dim] - same dimension as input
        """
        
        modalities = ['metric', 'log', 'trace']
        
        # 判断是图级特征还是节点级特征
        feature_first_dim = next(iter(modal_features.values())).shape[0]
        
        if modality_masks is not None:
            # 有模态掩码，用掩码的第一个维度作为实际的样本数
            actual_batch_size = modality_masks.shape[0]
            
            # 检查特征维度是否与掩码匹配
            if feature_first_dim == actual_batch_size:
                # 图级特征：特征数量 = 图数量 = 掩码数量
                is_node_level = False
                batch_size = actual_batch_size
            else:
                # 节点级特征：特征数量 = 所有节点数 > 图数量
                is_node_level = True
                batch_size = actual_batch_size
        else:
            # 没有模态掩码，假设是图级特征且所有模态都可用
            is_node_level = False
            batch_size = feature_first_dim
            modality_masks = torch.ones(batch_size, 3, dtype=torch.bool, 
                                       device=next(iter(modal_features.values())).device)
        
        if is_node_level:
            # 节点级特征：对所有节点统一应用可用模态（简化处理）
            # 取所有图中可用的模态的并集
            global_mask = modality_masks.any(dim=0)  # [3] - 任何图中使用的模态
            
            # 只保留全局可用的模态
            available_features = []
            available_weights = []
            
            for i, mod in enumerate(modalities):
                if mod in modal_features and global_mask[i]:
                    available_features.append(modal_features[mod])  # [total_nodes, dim]
                    available_weights.append(self.modal_weights[i])
            
            if len(available_features) == 0:
                # 无可用模态
                return torch.zeros_like(next(iter(modal_features.values())))
            elif len(available_features) == 1:
                # 单模态
                return available_features[0]
            else:
                # 多模态融合（所有节点使用相同的融合方式）
                features = torch.stack(available_features, dim=1)  # [total_nodes, num_modalities, dim]
                
                # 重新调整为适合attention的形状
                total_nodes, num_modalities, feature_dim = features.shape
                features = features.view(total_nodes, num_modalities, feature_dim)
                
                # 跨模态注意力
                attended_features, _ = self.cross_attention(
                    query=features, 
                    key=features, 
                    value=features
                )
                
                # 计算模态权重
                available_weights = torch.stack(available_weights)
                modal_weights = torch.softmax(available_weights, dim=0)
                
                # 加权融合
                weighted_features = attended_features * modal_weights.view(1, -1, 1)
                output_features = self.layer_norm(weighted_features + features)
                
                # 最终融合
                fused_features = torch.sum(output_features * modal_weights.view(1, -1, 1), dim=1)
                return self.dropout(fused_features)
        
        else:
            # 图级特征：针对每个样本单独处理模态融合
            fused_results = []
            
            for sample_idx in range(batch_size):
                sample_mask = modality_masks[sample_idx]  # [3] - 当前样本的模态掩码
                
                # 收集当前样本的可用模态
                available_features = []
                available_weights = []
                
                for i, mod in enumerate(modalities):
                    if mod in modal_features and sample_mask[i]:  # 样本级检查
                        available_features.append(modal_features[mod][sample_idx:sample_idx+1])  # [1, dim]
                        available_weights.append(self.modal_weights[i])
                
                # 处理当前样本的模态融合
                if len(available_features) == 0:
                    # 空模态情况：返回零向量
                    sample_result = torch.zeros(1, self.input_dim, device=modality_masks.device)
                elif len(available_features) == 1:
                    # 单模态情况：直接返回
                    sample_result = available_features[0]
                else:
                    # 多模态融合
                    features = torch.stack(available_features, dim=1)  # [1, num_modalities, feature_dim]
                    
                    # 跨模态注意力
                    attended_features, _ = self.cross_attention(
                        query=features, 
                        key=features, 
                        value=features
                    )
                    
                    # 计算模态权重
                    available_weights = torch.stack(available_weights)  # [num_modalities]
                    modal_weights = torch.softmax(available_weights, dim=0)  # [num_modalities]
                    
                    # 加权融合 + 残差连接
                    weighted_features = attended_features * modal_weights.view(1, -1, 1)  # [1, num_modalities, feature_dim]
                    output_features = self.layer_norm(weighted_features + features)  # 残差连接
                    
                    # 最终融合
                    sample_result = torch.sum(output_features * modal_weights.view(1, -1, 1), dim=1)  # [1, feature_dim]
                    sample_result = self.dropout(sample_result)
                
                fused_results.append(sample_result)
            
            # 合并所有样本的结果
            return torch.cat(fused_results, dim=0)  # [batch_size, feature_dim]

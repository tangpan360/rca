import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificModalAttention(nn.Module):
    """
    任务特定的模态注意力机制
    FTI和RCL任务使用独立的可学习查询向量
    """
    def __init__(self, modal_dim, num_heads=4, dropout=0.1, task_type="fti"):
        super(TaskSpecificModalAttention, self).__init__()
        
        self.task_type = task_type
        self.modal_dim = modal_dim
        self.num_heads = num_heads
        
        # 可学习的任务查询向量（核心改进）
        self.task_query = nn.Parameter(torch.empty(1, modal_dim))
        nn.init.xavier_normal_(self.task_query)
        
        # 简化的查询投影
        self.query_proj = nn.Sequential(
            nn.Linear(modal_dim, modal_dim),
            nn.LayerNorm(modal_dim),
            nn.GELU()
        )
        
        # Key/Value投影
        self.key_proj = nn.Linear(modal_dim, modal_dim)
        self.value_proj = nn.Linear(modal_dim, modal_dim)
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=modal_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 增强的输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(modal_dim, modal_dim),
            nn.LayerNorm(modal_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, modal_features, context_features=None):
        """
        基于可学习任务查询的注意力计算
        
        Args:
            modal_features: [batch_size, num_modals, modal_dim] 堆叠的模态特征
            context_features: 保留参数以兼容旧接口（未使用）
        
        Returns:
            fused_features: [batch_size, modal_dim] 融合后的特征
            attention_weights: [batch_size, num_heads, 1, num_modals] 注意力权重
        """
        batch_size, num_modals, modal_dim = modal_features.shape
        
        # 使用可学习的任务查询（避免循环依赖）
        query = self.task_query.expand(batch_size, -1)  # [batch_size, modal_dim]
        query = self.query_proj(query).unsqueeze(1)  # [batch_size, 1, modal_dim]
        
        # Key/Value变换
        key_features = self.key_proj(modal_features)    # [batch_size, num_modals, modal_dim]
        value_features = self.value_proj(modal_features)  # [batch_size, num_modals, modal_dim]
        
        # 多头注意力计算
        attn_output, attn_weights = self.attention(
            query=query,                    # [batch_size, 1, modal_dim]
            key=key_features,              # [batch_size, num_modals, modal_dim]
            value=value_features,          # [batch_size, num_modals, modal_dim]
            need_weights=True
        )
        
        # 输出投影
        fused_features = self.output_proj(attn_output.squeeze(1))  # [batch_size, modal_dim]
        
        return fused_features, attn_weights


class AdaptiveModalFusion(nn.Module):
    """
    简化的多模态融合模块 - 两种融合策略，统一32维输出
    """
    def __init__(self, modal_dim, num_heads=4, dropout=0.1, fusion_mode="adaptive"):
        super(AdaptiveModalFusion, self).__init__()
        
        self.fusion_mode = fusion_mode
        self.modal_dim = modal_dim
        
        if fusion_mode == "average":
            # 简单平均融合
            pass
            
        elif fusion_mode == "adaptive":
            # 自适应权重融合 - 任务特定的注意力机制
            self.fti_attention = TaskSpecificModalAttention(
                modal_dim=modal_dim,
                num_heads=num_heads,
                dropout=dropout,
                task_type="fti"
            )
            
            self.rcl_attention = TaskSpecificModalAttention(
                modal_dim=modal_dim,
                num_heads=num_heads,
                dropout=dropout,
                task_type="rcl"
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}. Use 'average' or 'adaptive'.")
    
    def forward(self, modal_fs, modal_es, used_modalities):
        """
        公平对比的多模态融合
        
        Args:
            modal_fs: dict of {modality: [batch_size, modal_dim]} 图级特征
            modal_es: dict of {modality: [num_nodes, modal_dim]} 节点级特征
            used_modalities: list of modality names
        
        Returns:
            f_fused: [batch_size, modal_dim] FTI任务特征 (统一32维输出)
            e_fused: [num_nodes, modal_dim] RCL任务特征 (统一32维输出)
            fusion_info: dict 融合过程信息
        """
        fusion_info = {}
        
        # 步骤1: 将模态特征堆叠成统一格式
        f_stack = torch.stack([modal_fs[mod] for mod in used_modalities], dim=1)  # [B, M, D]
        e_stack = torch.stack([modal_es[mod] for mod in used_modalities], dim=1)  # [N, M, D]
        
        if self.fusion_mode == "average":
            # 简单平均融合
            f_fused = f_stack.mean(dim=1)  # [B, D]
            e_fused = e_stack.mean(dim=1)  # [N, D]
            fusion_info['fusion_type'] = 'simple_average'
            
        elif self.fusion_mode == "adaptive":
            # 自适应权重融合（使用可学习的任务查询）
            f_fused, fti_attn = self.fti_attention(f_stack)  # [B, D]
            e_fused, rcl_attn = self.rcl_attention(e_stack)  # [N, D]
            
            fusion_info['fti_attention'] = fti_attn
            fusion_info['rcl_attention'] = rcl_attn  
            fusion_info['fusion_type'] = 'adaptive_attention'
            
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}. Use 'average' or 'adaptive'.")
        
        # 两种模式都输出相同维度: [B, D] 和 [N, D]
        return f_fused, e_fused, fusion_info
    
    def get_modal_importance(self, attention_weights, modalities):
        """
        从注意力权重中提取模态重要性分数
        
        Args:
            attention_weights: [batch_size, num_heads, 1, num_modals] 
            modalities: list of modality names
            
        Returns:
            dict: {modality: importance_score}
        """
        if attention_weights is None:
            return {}
        
        # 计算平均注意力权重
        avg_weights = attention_weights.mean(dim=1).squeeze(1)  # [batch_size, num_modals]
        avg_weights = avg_weights.mean(dim=0)  # [num_modals] - 跨batch平均
        
        # 创建模态重要性字典
        importance = {}
        for i, modality in enumerate(modalities):
            if i < len(avg_weights):
                importance[modality] = avg_weights[i].item()
        
        return importance


class ModalAttentionVisualizer:
    """
    注意力权重可视化工具
    """
    @staticmethod
    def plot_attention_comparison(fti_importance, rcl_importance, modalities, save_path=None):
        """
        对比FTI和RCL任务的模态注意力权重
        
        Args:
            fti_importance: dict of {modality: weight}
            rcl_importance: dict of {modality: weight}  
            modalities: list of modality names
            save_path: str, optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            x = np.arange(len(modalities))
            width = 0.35
            
            fti_weights = [fti_importance.get(mod, 0) for mod in modalities]
            rcl_weights = [rcl_importance.get(mod, 0) for mod in modalities]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar(x - width/2, fti_weights, width, label='FTI Task', alpha=0.8)
            rects2 = ax.bar(x + width/2, rcl_weights, width, label='RCL Task', alpha=0.8)
            
            ax.set_xlabel('Modalities')
            ax.set_ylabel('Attention Weight')
            ax.set_title('Modal Attention Weights Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([mod.capitalize() for mod in modalities])
            ax.legend()
            
            # 添加数值标签
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Please install it for visualization.")
        except Exception as e:
            print(f"Visualization error: {e}")
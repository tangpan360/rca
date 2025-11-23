import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificModalAttention(nn.Module):
    """
    任务特定的模态注意力机制
    为FTI和RCL任务分别设计不同的模态融合策略
    
    修复版本：
    1. 正确的维度计算
    2. 真正的任务特定性
    3. 清晰的语义表达
    """
    def __init__(self, modal_dim, num_heads=4, dropout=0.1, task_type="fti"):
        super(TaskSpecificModalAttention, self).__init__()
        
        self.task_type = task_type
        self.modal_dim = modal_dim
        self.num_heads = num_heads
        
        # 任务特定的查询变换 - 增强差异化设计
        if task_type == "fti":
            # FTI关注全局故障模式：使用更复杂的变换捕获跨模态关系
            self.query_proj = nn.Sequential(
                nn.Linear(modal_dim, modal_dim * 2),
                nn.LayerNorm(modal_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(modal_dim * 2, modal_dim),
                nn.LayerNorm(modal_dim),
                nn.Tanh()  # 使用tanh激活，适合全局特征学习
            )
            # FTI专用的key/value投影（可选）
            self.key_proj = nn.Linear(modal_dim, modal_dim)
            self.value_proj = nn.Linear(modal_dim, modal_dim)
            
        else:  # rcl
            # RCL关注局部异常信号：使用更直接的变换保持细节敏感性
            self.query_proj = nn.Sequential(
                nn.Linear(modal_dim, modal_dim),
                nn.LayerNorm(modal_dim), 
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(modal_dim, modal_dim),
                nn.Sigmoid()  # 使用sigmoid激活，适合异常检测
            )
            # RCL使用恒等映射保持原始特征
            self.key_proj = nn.Identity()
            self.value_proj = nn.Identity()
        
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=modal_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出投影和标准化
        self.output_proj = nn.Sequential(
            nn.Linear(modal_dim, modal_dim),
            nn.LayerNorm(modal_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, modal_features, context_features):
        """
        增强的任务特定注意力计算
        
        Args:
            modal_features: [batch_size, num_modals, modal_dim] 堆叠的模态特征
            context_features: [batch_size, modal_dim] 任务上下文特征
        
        Returns:
            fused_features: [batch_size, modal_dim] 融合后的特征
            attention_weights: [batch_size, num_heads, 1, num_modals] 注意力权重
        """
        batch_size, num_modals, modal_dim = modal_features.shape
        
        # 任务特定的查询变换
        query = self.query_proj(context_features).unsqueeze(1)  # [batch_size, 1, modal_dim]
        
        # 任务特定的key/value变换
        # FTI: 使用学习的投影捕获跨模态关系
        # RCL: 使用恒等映射保持原始细节
        key_features = self.key_proj(modal_features.reshape(-1, modal_dim)).reshape(batch_size, num_modals, modal_dim)
        value_features = self.value_proj(modal_features.reshape(-1, modal_dim)).reshape(batch_size, num_modals, modal_dim)
        
        # 多头注意力计算
        attn_output, attn_weights = self.attention(
            query=query,                    # [batch_size, 1, modal_dim]
            key=key_features,              # [batch_size, num_modals, modal_dim] (任务特定变换)
            value=value_features,          # [batch_size, num_modals, modal_dim] (任务特定变换)
            need_weights=True
        )
        
        # 输出投影
        fused_features = self.output_proj(attn_output.squeeze(1))  # [batch_size, modal_dim]
        
        # 任务特定的残差连接策略
        if self.task_type == "fti":
            # FTI: 较强的残差连接，保持全局信息
            fused_features = fused_features + 0.5 * context_features
        else:  # rcl
            # RCL: 较弱的残差连接，突出注意力结果
            fused_features = fused_features + 0.2 * context_features
        
        return fused_features, attn_weights


class AdaptiveModalFusion(nn.Module):
    """
    简化的多模态融合模块 - 三种融合策略，统一32维输出
    """
    def __init__(self, modal_dim, num_heads=4, dropout=0.1, fusion_mode="uniform", max_modalities=3):
        super(AdaptiveModalFusion, self).__init__()
        
        self.fusion_mode = fusion_mode  # "average", "uniform", 或 "adaptive"
        self.modal_dim = modal_dim
        self.max_modalities = max_modalities
        
        if fusion_mode == "average":
            # 简单平均融合 - 最简单的baseline，无需参数
            pass
            
        elif fusion_mode == "uniform":
            # 可学习固定权重融合
            self.fti_uniform_weights = nn.Parameter(torch.ones(max_modalities) / max_modalities)
            self.rcl_uniform_weights = nn.Parameter(torch.ones(max_modalities) / max_modalities)
            
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
            # 简单平均融合 - 最简单的baseline
            f_fused = f_stack.mean(dim=1)  # [B, D]
            e_fused = e_stack.mean(dim=1)  # [N, D]
            fusion_info['fusion_type'] = 'simple_average'
            
        elif self.fusion_mode == "uniform":
            # 可学习固定权重融合
            num_used = len(used_modalities)
            fti_weights = F.softmax(self.fti_uniform_weights[:num_used], dim=0)  # [M]
            rcl_weights = F.softmax(self.rcl_uniform_weights[:num_used], dim=0)  # [M]
            
            f_fused = torch.sum(f_stack * fti_weights.view(1, -1, 1), dim=1)  # [B, D]
            e_fused = torch.sum(e_stack * rcl_weights.view(1, -1, 1), dim=1)  # [N, D]
            
            fusion_info['fti_weights'] = fti_weights
            fusion_info['rcl_weights'] = rcl_weights
            fusion_info['fusion_type'] = 'uniform_weighted_average'
            
        elif self.fusion_mode == "adaptive":
            # 自适应权重融合
            f_context = f_stack.mean(dim=1)  # [B, D] - 全局上下文
            e_context = e_stack.mean(dim=1)  # [N, D] - 局部上下文
            
            f_fused, fti_attn = self.fti_attention(f_stack, f_context)  # [B, D]
            e_fused, rcl_attn = self.rcl_attention(e_stack, e_context)  # [N, D]
            
            fusion_info['fti_attention'] = fti_attn
            fusion_info['rcl_attention'] = rcl_attn  
            fusion_info['fusion_type'] = 'adaptive_attention'
            
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        
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
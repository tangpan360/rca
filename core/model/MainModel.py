import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.GraphEncoder import GraphEncoder
from core.model.ModalEncoder import MultiModalEncoder
from core.model.TaskSpecificAttention import AdaptiveModalFusion


class MainModel(nn.Module):
    """
    多模态故障诊断主模型
    流程: 原始时序数据 -> 多模态编码器 -> 图网络 -> 诊断输出
    """
    def __init__(self, config: Config):
        super(MainModel, self).__init__()
        
        self.config = config
        
        # 多模态编码器（将原始数据编码为固定维度）
        self.modal_encoder = MultiModalEncoder(
            output_dim=config.feature_embedding_dim,
            metric_channels=config.metric_channels,
            log_dim=config.log_dim,
            seq_len=config.seq_len
        )
        
        # 图编码器（每个模态一个）
        self.graph_encoders = nn.ModuleDict()
        for modality in config.modalities:
            self.graph_encoders[modality] = GraphEncoder(
                feature_embedding_dim=config.feature_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop
            )

        # 简化的模态融合模块
        self.adaptive_fusion = AdaptiveModalFusion(
            modal_dim=config.graph_out,
            num_heads=getattr(config, 'attention_heads', 4),
            dropout=getattr(config, 'attention_dropout', 0.1),
            fusion_mode=config.fusion_mode
        )
        
        # 统一分类器：由于融合后所有模态组合都输出32维，使用统一分类器
        self.typeClassifier = Classifier(
            in_dim=config.graph_out,
            hiddens=config.linear_hidden,
            out_dim=config.ft_num
        )
        self.locator = Voter(
            config.graph_out,
            hiddens=config.linear_hidden,
            out_dim=1
        )

    def forward(self, batch_graphs, active_modalities=None):
        # 确定使用的模态
        if self.config.use_partial_modalities and active_modalities is not None:
            # 部分模态模式：只使用指定的模态
            used_modalities = active_modalities
        else:
            # 默认模式：使用所有配置的模态
            used_modalities = self.config.modalities
        
        # 步骤1: 使用多模态编码器处理原始数据
        metric_raw = batch_graphs.ndata['metric']  # [num_nodes, 20, 12]
        log_raw = batch_graphs.ndata['log']  # [num_nodes, 48]
        trace_raw = batch_graphs.ndata['trace']  # [num_nodes, 20, 2]
        
        metric_emb, log_emb, trace_emb = self.modal_encoder(metric_raw, log_raw, trace_raw)
        
        modal_embs = {
            'metric': metric_emb,
            'log': log_emb,
            'trace': trace_emb
        }
        
        # 步骤2: 使用图编码器处理（只处理使用的模态）
        fs, es = {}, {}
        
        for modality in used_modalities:
            if modality in self.graph_encoders:
                x_d = modal_embs[modality]
                f_d, e_d = self.graph_encoders[modality](batch_graphs, x_d)  # graph-level, node-level
                fs[modality] = f_d
                es[modality] = e_d

        # 步骤3: 多模态融合
        f, e, fusion_info = self.adaptive_fusion(fs, es, used_modalities)
        # 输出: f[B, 32], e[N, 32]

        # 步骤4: 故障诊断
        type_logit = self.typeClassifier(f)  # 故障类型识别
        root_logit = self.locator(e)  # 根因定位

        # 存储融合信息用于分析
        self._last_fusion_info = fusion_info
        
        return fs, es, root_logit, type_logit
    
    def get_fusion_info(self):
        """
        获取最后一次前向传播的融合信息
        用于模型分析和可视化
        
        Returns:
            dict: 包含融合权重和注意力信息
        """
        return getattr(self, '_last_fusion_info', {})
    
    def get_attention_info(self):
        """
        获取注意力权重信息 (向后兼容)
        
        Returns:
            dict: 包含FTI和RCL任务的注意力权重
        """
        fusion_info = self.get_fusion_info()
        attention_info = {}
        
        # 从fusion_info中提取attention信息
        if 'fti_attention' in fusion_info:
            attention_info['fti_attention'] = fusion_info['fti_attention']
        if 'rcl_attention' in fusion_info:
            attention_info['rcl_attention'] = fusion_info['rcl_attention']
            
        return attention_info
    
    def get_fusion_mode(self):
        """获取当前的融合模式"""
        return self.adaptive_fusion.fusion_mode
    
    def get_modal_importance_analysis(self, used_modalities):
        """
        获取模态重要性分析结果
        
        Args:
            used_modalities: list of modality names used in last forward pass
            
        Returns:
            dict: 包含FTI和RCL任务的模态重要性分析
        """
        attention_info = getattr(self, '_last_attention_info', {})
        
        if not attention_info or self.adaptive_fusion is None:
            return {'error': 'No attention information available'}
        
        analysis = {}
        
        # FTI任务的模态重要性
        if 'fti_attention' in attention_info:
            fti_importance = self.adaptive_fusion.get_modal_importance(
                attention_info['fti_attention'], used_modalities
            )
            analysis['fti_modal_importance'] = fti_importance
        
        # RCL任务的模态重要性
        if 'rcl_attention' in attention_info:
            rcl_importance = self.adaptive_fusion.get_modal_importance(
                attention_info['rcl_attention'], used_modalities
            )
            analysis['rcl_modal_importance'] = rcl_importance
        
        # 计算任务间的模态偏好差异
        if 'fti_modal_importance' in analysis and 'rcl_modal_importance' in analysis:
            fti_imp = analysis['fti_modal_importance']
            rcl_imp = analysis['rcl_modal_importance']
            
            differences = {}
            for modality in used_modalities:
                if modality in fti_imp and modality in rcl_imp:
                    differences[modality] = abs(fti_imp[modality] - rcl_imp[modality])
            
            analysis['task_preference_differences'] = differences
        
        return analysis

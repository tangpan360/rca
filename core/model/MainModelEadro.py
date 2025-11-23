import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.Encoder import Encoder
from core.model.EadroEncoder import EadroModalEncoder
from core.model.TaskSpecificAttention import AdaptiveModalFusion


class MainModelEadro(nn.Module):
    """
    集成Eadro编码器的TVDiag主模型
    流程: 原始时序数据 -> Eadro编码器 -> TVDiag图网络 -> 诊断输出
    """
    def __init__(self, config: Config):
        super(MainModelEadro, self).__init__()
        
        self.config = config
        
        # Eadro模态编码器（将原始数据编码为固定维度）
        self.eadro_encoder = EadroModalEncoder(output_dim=config.alert_embedding_dim)
        
        # TVDiag图编码器（每个模态一个）
        self.encoders = nn.ModuleDict()
        for modality in config.modalities:
            self.encoders[modality] = Encoder(
                alert_embedding_dim=config.alert_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop
            )

        # 支持动态模态组合的融合层
        if config.use_partial_modalities:
            # 为所有可能的模态组合创建分类器
            self.adaptive_classifiers = nn.ModuleDict()
            self.adaptive_locators = nn.ModuleDict()
            
            # 预创建常用的模态组合
            common_combinations = [
                ['metric'],
                ['log'], 
                ['trace'],
                ['metric', 'log'],
                ['metric', 'trace'],
                ['log', 'trace'],
                ['metric', 'log', 'trace']
            ]
            
            for combination in common_combinations:
                combo_key = '_'.join(sorted(combination))
                
                # 新架构：所有模态组合都使用相同的32维输出
                fusion_output_dim = config.graph_out  # 32维统一输出
                
                self.adaptive_classifiers[combo_key] = Classifier(
                    in_dim=fusion_output_dim,
                    hiddens=config.linear_hidden,
                    out_dim=config.ft_num
                )
                self.adaptive_locators[combo_key] = Voter(
                    fusion_output_dim,
                    hiddens=config.linear_hidden,
                    out_dim=1
                )
        else:
            # 原来的固定三模态融合 - 更新为32维输出适配
            if hasattr(config, 'use_adaptive_fusion') and config.use_adaptive_fusion:
                # 如果启用了新融合但没有自适应融合对象，使用32维分类器
                fusion_dim = config.graph_out  # 32维
            else:
                # 传统concatenation模式
                fusion_dim = len(config.modalities) * config.graph_out  # 96维

            self.locator = Voter(fusion_dim,
                                 hiddens=config.linear_hidden,
                                 out_dim=1)
            self.typeClassifier = Classifier(in_dim=fusion_dim,
                                           hiddens=config.linear_hidden,
                                           out_dim=config.ft_num)
        
        # 自适应模态融合模块
        if hasattr(config, 'use_adaptive_fusion') and config.use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveModalFusion(
                modal_dim=config.graph_out,
                num_heads=getattr(config, 'attention_heads', 4),
                dropout=getattr(config, 'attention_dropout', 0.1),
                fusion_mode=getattr(config, 'fusion_mode', 'adaptive'),  # 默认adaptive
                max_modalities=len(config.modalities)
            )
            
            # 新架构：uniform vs adaptive融合，统一32维输出确保公平对比
            fusion_output_dim = config.graph_out  # 统一使用32维输出
            
            # 所有融合模式使用相同维度的分类器
            if not config.use_partial_modalities:
                self.adaptive_typeClassifier = Classifier(
                    in_dim=fusion_output_dim,
                    hiddens=config.linear_hidden,
                    out_dim=config.ft_num
                )
                self.adaptive_locator = Voter(
                    fusion_output_dim,
                    hiddens=config.linear_hidden,
                    out_dim=1
                )
        else:
            self.adaptive_fusion = None
            
        # 为传统concatenation模式添加投影层(如果需要)
        # 这个逻辑放在adaptive_fusion创建之后
        if (hasattr(config, 'use_adaptive_fusion') and config.use_adaptive_fusion and 
            self.adaptive_fusion is None):
            # 启用了融合但没有adaptive_fusion对象时，添加投影层
            concat_dim = len(config.modalities) * config.graph_out  # 96维
            target_dim = config.graph_out  # 32维
            self.concat_proj_f = nn.Linear(concat_dim, target_dim)
            self.concat_proj_e = nn.Linear(concat_dim, target_dim)

    def forward(self, batch_graphs, active_modalities=None):
        # 确定使用的模态
        if self.config.use_partial_modalities and active_modalities is not None:
            # 部分模态模式：只使用指定的模态
            used_modalities = active_modalities
        else:
            # 默认模式：使用所有配置的模态
            used_modalities = self.config.modalities
        
        # 步骤1: 使用Eadro编码器处理原始数据
        metric_raw = batch_graphs.ndata['metric']  # [num_nodes, 20, 12]
        log_raw = batch_graphs.ndata['log']  # [num_nodes, 48]
        trace_raw = batch_graphs.ndata['trace']  # [num_nodes, 20, 1]
        
        metric_emb, log_emb, trace_emb = self.eadro_encoder(metric_raw, log_raw, trace_raw)
        
        modal_embs = {
            'metric': metric_emb,
            'log': log_emb,
            'trace': trace_emb
        }
        
        # 步骤2: 使用TVDiag图编码器处理（只处理使用的模态）
        fs, es = {}, {}
        
        for modality in used_modalities:
            if modality in self.encoders:
                x_d = modal_embs[modality]
                f_d, e_d = self.encoders[modality](batch_graphs, x_d)  # graph-level, node-level
                fs[modality] = f_d
                es[modality] = e_d

        # 步骤3: 多模态融合 (新版本: 统一32维输出)
        fusion_info = {}
        
        if self.adaptive_fusion is not None and self.config.use_adaptive_fusion:
            # 使用新的融合策略: uniform vs adaptive
            f, e, fusion_info = self.adaptive_fusion(fs, es, used_modalities)
            # 输出: f[B, 32], e[N, 32]
            
        else:
            # 传统方式或禁用自适应融合时的处理
            if hasattr(self.config, 'use_adaptive_fusion') and self.config.use_adaptive_fusion:
                # 启用了新融合但没有adaptive_fusion对象：使用简单平均
                f_stack = torch.stack([fs[mod] for mod in used_modalities], dim=1)
                e_stack = torch.stack([es[mod] for mod in used_modalities], dim=1)
                f = f_stack.mean(dim=1)  # [B, 32] - 简单平均
                e = e_stack.mean(dim=1)  # [N, 32] - 简单平均
                fusion_info['fusion_type'] = 'simple_average'
                
            else:
                # 完全传统的concatenation模式
                max_modalities = len(self.config.modalities)
                used_f = [fs[mod] for mod in used_modalities]
                used_e = [es[mod] for mod in used_modalities]
                
                # 填充缺失模态
                while len(used_f) < max_modalities:
                    device = used_f[0].device
                    dtype = used_f[0].dtype
                    zero_f = torch.zeros(used_f[0].shape[0], used_f[0].shape[1], device=device, dtype=dtype)
                    zero_e = torch.zeros(used_e[0].shape[0], used_e[0].shape[1], device=device, dtype=dtype)
                    used_f.append(zero_f)
                    used_e.append(zero_e)
                
                f = torch.cat(used_f, dim=1)  # [B, 96]
                e = torch.cat(used_e, dim=1)  # [N, 96]
                
                # 如果有投影层，投影到32维
                if hasattr(self, 'concat_proj_f'):
                    f = self.concat_proj_f(f)  # [B, 96] -> [B, 32]
                    e = self.concat_proj_e(e)  # [N, 96] -> [N, 32]
                
                fusion_info['fusion_type'] = 'traditional_concat'

        # 步骤4: 故障诊断（动态选择分类器）
        if self.config.use_partial_modalities:
            combo_key = '_'.join(sorted(used_modalities))
            
            if combo_key in self.adaptive_classifiers:
                type_logit = self.adaptive_classifiers[combo_key](f)
                root_logit = self.adaptive_locators[combo_key](e)
            else:
                # 如果没有预定义的组合，使用第一个可用的
                first_key = list(self.adaptive_classifiers.keys())[0]
                type_logit = self.adaptive_classifiers[first_key](f)
                root_logit = self.adaptive_locators[first_key](e)
        else:
            # 选择分类器：优先使用自适应分类器
            if (self.adaptive_fusion is not None and 
                hasattr(self, 'adaptive_typeClassifier') and 
                self.config.use_adaptive_fusion):
                type_logit = self.adaptive_typeClassifier(f)  # 故障类型识别
                root_logit = self.adaptive_locator(e)  # 根因定位
            else:
                # 使用原来的固定分类器
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
        """
        获取当前使用的融合模式
        
        Returns:
            str: 融合模式 ('concat', 'attention')
        """
        if self.adaptive_fusion is not None:
            return self.adaptive_fusion.fusion_mode
        else:
            return 'concat'
    
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



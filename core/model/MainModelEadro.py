import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.Encoder import Encoder
from core.model.EadroEncoder import EadroModalEncoder
from core.model.CrossModalFusion import CrossModalAttentionFusion


class MainModelEadro(nn.Module):
    """
    集成Eadro编码器的TVDiag主模型
    流程: 原始时序数据 -> Eadro编码器 -> TVDiag图网络 -> 诊断输出
    """
    def __init__(self, config: Config):
        super(MainModelEadro, self).__init__()
        
        # 保存配置参数
        self.use_cross_modal_attention = getattr(config, 'use_cross_modal_attention', True)
        
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

        # 跨模态注意力融合模块（仅在启用时创建）
        if self.use_cross_modal_attention:
            self.cross_modal_fusion_graph = CrossModalAttentionFusion(
                input_dim=config.graph_out,
                num_heads=getattr(config, 'attention_heads', 4),
                dropout=getattr(config, 'attention_dropout', 0.1)
            )
            self.cross_modal_fusion_node = CrossModalAttentionFusion(
                input_dim=config.graph_out,
                num_heads=getattr(config, 'attention_heads', 4),
                dropout=getattr(config, 'attention_dropout', 0.1)
            )
        
        # 动态计算分类器输入维度
        if self.use_cross_modal_attention:
            # 注意力融合模式: 输出维度保持为32
            fti_fuse_dim = config.graph_out
            rcl_fuse_dim = config.graph_out
        else:
            # 传统concatenation模式: 输出维度为3*32=96
            fti_fuse_dim = len(config.modalities) * config.graph_out
            rcl_fuse_dim = len(config.modalities) * config.graph_out

        self.locator = Voter(rcl_fuse_dim,
                             hiddens=config.linear_hidden,
                             out_dim=1)
        self.typeClassifier = Classifier(in_dim=fti_fuse_dim,
                                         hiddens=config.linear_hidden,
                                         out_dim=config.ft_num)

    def forward(self, batch_graphs):
        # 步骤1: 使用Eadro编码器处理原始数据
        metric_raw = batch_graphs.ndata['metric']  # [num_nodes, 20, 12]
        log_raw = batch_graphs.ndata['log']  # [num_nodes, 40]
        trace_raw = batch_graphs.ndata['trace']  # [num_nodes, 20, 1]
        
        metric_emb, log_emb, trace_emb = self.eadro_encoder(metric_raw, log_raw, trace_raw)
        
        # 步骤2: 使用TVDiag图编码器处理
        fs, es = {}, {}
        modal_embs = {
            'metric': metric_emb,
            'log': log_emb,
            'trace': trace_emb
        }
        
        for modality, encoder in self.encoders.items():
            x_d = modal_embs[modality]
            f_d, e_d = encoder(batch_graphs, x_d)  # graph-level, node-level
            fs[modality] = f_d
            es[modality] = e_d

        # 步骤3: 多模态融合（根据配置选择融合策略）
        if self.use_cross_modal_attention:
            # 使用跨模态注意力融合
            f = self.cross_modal_fusion_graph(fs)
            e = self.cross_modal_fusion_node(es)
        else:
            # 使用传统concatenation融合（baseline）
            f = torch.cat(tuple(fs.values()), dim=1)
            e = torch.cat(list(es.values()), dim=1)

        # 步骤4: 故障诊断
        type_logit = self.typeClassifier(f)  # 故障类型识别
        root_logit = self.locator(e)  # 根因定位

        return fs, es, root_logit, type_logit



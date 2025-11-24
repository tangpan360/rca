import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.Encoder import Encoder
from core.model.EadroEncoder import EadroModalEncoder


class MainModelEadro(nn.Module):
    """
    集成Eadro编码器的TVDiag主模型
    流程: 原始时序数据 -> Eadro编码器 -> TVDiag图网络 -> 诊断输出
    """
    def __init__(self, config: Config):
        super(MainModelEadro, self).__init__()
        
        self.config = config
        
        # Eadro模态编码器（将原始数据编码为固定维度）
        self.eadro_encoder = EadroModalEncoder(output_dim=config.feature_embedding_dim)
        
        # TVDiag图编码器（每个模态一个）
        self.encoders = nn.ModuleDict()
        for modality in config.modalities:
            self.encoders[modality] = Encoder(
                feature_embedding_dim=config.feature_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop
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

        # 步骤3: 多模态融合（简单平均）
        f_stack = torch.stack([fs[mod] for mod in used_modalities], dim=1)  # [B, M, D]
        e_stack = torch.stack([es[mod] for mod in used_modalities], dim=1)  # [N, M, D]
        
        f = f_stack.mean(dim=1)  # [B, D] - 图级特征平均融合
        e = e_stack.mean(dim=1)  # [N, D] - 节点级特征平均融合

        # 步骤4: 故障诊断
        type_logit = self.typeClassifier(f)  # 故障类型识别
        root_logit = self.locator(e)  # 根因定位
        
        return fs, es, root_logit, type_logit



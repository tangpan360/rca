import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.Encoder import Encoder
from core.model.EadroEncoder import EadroModalEncoder
from core.model.KNNModalityImputation import KNNModalityImputation


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
        
        # k-NN模态填补器（可选）
        self.knn_imputer = None
        if config.use_knn_imputation:
            self.knn_imputer = KNNModalityImputation(config)
        
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
                fuse_dim = len(combination) * config.graph_out
                
                self.adaptive_classifiers[combo_key] = Classifier(
                    in_dim=fuse_dim,
                    hiddens=config.linear_hidden,
                    out_dim=config.ft_num
                )
                self.adaptive_locators[combo_key] = Voter(
                    fuse_dim,
                    hiddens=config.linear_hidden,
                    out_dim=1
                )
        else:
            # 原来的固定三模态融合
            fti_fuse_dim = len(config.modalities) * config.graph_out
            rcl_fuse_dim = len(config.modalities) * config.graph_out

            self.locator = Voter(rcl_fuse_dim,
                                 hiddens=config.linear_hidden,
                                 out_dim=1)
            self.typeClassifier = Classifier(in_dim=fti_fuse_dim,
                                           hiddens=config.linear_hidden,
                                           out_dim=config.ft_num)

    def forward(self, batch_graphs, missing_modalities=None, active_modalities=None):
        # 确定使用的模态
        if self.config.use_partial_modalities and active_modalities is not None:
            # 部分模态模式：只使用指定的模态
            used_modalities = active_modalities
        else:
            # 默认模式：使用所有配置的模态
            used_modalities = self.config.modalities
        
        # 步骤1: 使用Eadro编码器处理原始数据
        metric_raw = batch_graphs.ndata['metric']  # [num_nodes, 20, 12]
        log_raw = batch_graphs.ndata['log']  # [num_nodes, 40]
        trace_raw = batch_graphs.ndata['trace']  # [num_nodes, 20, 1]
        
        metric_emb, log_emb, trace_emb = self.eadro_encoder(metric_raw, log_raw, trace_raw)
        
        modal_embs = {
            'metric': metric_emb,
            'log': log_emb,
            'trace': trace_emb
        }
        
        # 步骤2: k-NN模态填补（如果启用且有缺失模态）
        if self.config.use_knn_imputation and self.knn_imputer and missing_modalities:
            modal_embs = self.knn_imputer.impute(modal_embs, missing_modalities)
        
        # 步骤3: 使用TVDiag图编码器处理（只处理使用的模态）
        fs, es = {}, {}
        
        for modality in used_modalities:
            if modality in self.encoders:
                x_d = modal_embs[modality]
                f_d, e_d = self.encoders[modality](batch_graphs, x_d)  # graph-level, node-level
                fs[modality] = f_d
                es[modality] = e_d

        # 步骤4: 多模态融合
        f = torch.cat([fs[mod] for mod in used_modalities], dim=1)
        e = torch.cat([es[mod] for mod in used_modalities], dim=1)

        # 步骤5: 故障诊断（动态选择分类器）
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
            # 原来的固定分类器
            type_logit = self.typeClassifier(f)  # 故障类型识别
            root_logit = self.locator(e)  # 根因定位

        return fs, es, root_logit, type_logit
    
    def setup_knn_imputer(self, logger):
        """设置k-NN填补器（加载向量库）"""
        if self.config.use_knn_imputation and self.knn_imputer:
            success = self.knn_imputer.load_vector_database(logger)
            if not success:
                logger.warning("k-NN向量库加载失败，模态填补功能不可用")
                self.knn_imputer = None
    
    def build_knn_database(self, train_dataloader, device, logger):
        """构建k-NN向量库（训练完成后调用）"""
        if self.config.use_knn_imputation and self.knn_imputer:
            self.knn_imputer.build_vector_database(self, train_dataloader, device, logger)



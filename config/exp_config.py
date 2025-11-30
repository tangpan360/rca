from utils.template_utils import get_log_template_count

class Config:
    def __init__(self, dataset) -> None:
        # base config
        self.dataset = dataset
        self.gpu_device = '0'
        self.seed = 2

        self.modalities = ['metric', 'trace', 'log']
        
        # 部分模态训练/测试配置
        self.use_partial_modalities = False  # 是否启用部分模态功能
        # self.training_modalities = ['metric', 'trace', 'log']  # 训练时使用的模态
        # self.testing_modalities = ['metric', 'trace', 'log']   # 测试时使用的模态
        self.training_modalities = ['trace', 'log']  # 训练时使用的模态
        self.testing_modalities = ['trace', 'log']   # 测试时使用的模态
        # self.training_modalities = ['metric', 'log']  # 训练时使用的模态
        # self.testing_modalities = ['metric', 'log']   # 测试时使用的模态
        # self.training_modalities = ['metric', 'trace']  # 训练时使用的模态
        # self.testing_modalities = ['metric', 'trace']   # 测试时使用的模态
        # self.training_modalities = ['metric']  # 训练时使用的模态
        # self.testing_modalities = ['metric']   # 测试时使用的模态
        # self.training_modalities = ['trace']  # 训练时使用的模态
        # self.testing_modalities = ['trace']   # 测试时使用的模态
        # self.training_modalities = ['log']  # 训练时使用的模态
        # self.testing_modalities = ['log']   # 测试时使用的模态

        # TVDiag modules
        self.aug_percent = 0.2  # 删除节点比例
        self.aug_times = 5       # 启用数据增强，每个样本生成5个增强样本
        self.dynamic_weight = True
        
        # 参数化重要性感知增强配置
        self.use_degree = True    # 是否考虑节点度数（入度+出度）重要性
        self.use_distance = True  # 是否考虑距离根因节点的距离重要性
        
        # 模型权重选择配置
        self.use_best_model = True  # True: 使用验证集最优权重, False: 使用最后权重

        # model config
        self.batch_size = 8
        self.epochs = 500
        self.feature_embedding_dim = 128  # Eadro编码器输出的特征维度
        self.graph_hidden_dim = 64
        self.graph_out = 32
        self.graph_layers = 2
        self.linear_hidden = [64]
        self.lr = 0.001
        self.weight_decay = 0.0001
        
        # 模态融合配置
        self.fusion_mode = "adaptive"       # 融合模式: "average" | "adaptive"
        self.attention_heads = 4            # 注意力头数
        self.attention_dropout = 0.1        # 注意力dropout率        

        if self.dataset == 'gaia':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 5
            self.aggregator = 'mean'
            # Gaia数据集维度配置
            self.metric_channels = 12
            self.log_dim = get_log_template_count('gaia')
            self.seq_len = 20
            # Gaia数据集路径配置
            self.dataset_path = "./preprocess/processed_data/gaia/dataset.pkl"
            self.nodes_path = "./preprocess/processed_data/gaia/graph/nodes_dynamic_no_influence.json"
            self.edges_path = "./preprocess/processed_data/gaia/graph/edges_dynamic_no_influence.json"
        elif self.dataset == 'sn':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 12  # SN有12个服务
            self.aggregator = 'mean'
            self.batch_size = 8
            # SN数据集维度配置
            self.metric_channels = 7
            self.log_dim = get_log_template_count('sn')
            self.seq_len = 10
            # SN数据集路径配置
            self.dataset_path = "./preprocess/processed_data/sn/dataset.pkl"
            self.nodes_path = "./preprocess/processed_data/sn/graph/nodes_predefined_static_no_influence.json"
            self.edges_path = "./preprocess/processed_data/sn/graph/edges_predefined_static_no_influence.json"
        elif self.dataset == 'tt':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 27  # TT有27个服务
            self.aggregator = 'mean'
            self.batch_size = 8
            # TT数据集维度配置
            self.metric_channels = 7
            self.log_dim = get_log_template_count('tt')
            self.seq_len = 10
            # TT数据集路径配置
            self.dataset_path = "./preprocess/processed_data/tt/dataset.pkl"
            self.nodes_path = "./preprocess/processed_data/tt/graph/nodes_predefined_static_no_influence.json"
            self.edges_path = "./preprocess/processed_data/tt/graph/edges_predefined_static_no_influence.json"
        else:
            raise NotImplementedError()
    
    def print_configs(self, logger):
        for attr, value in vars(self).items():
            logger.info(f"{attr}: {value}")
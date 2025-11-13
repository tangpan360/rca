class Config:
    def __init__(self, dataset) -> None:
        # base config
        self.dataset = dataset
        self.reconstruct = False
        self.log_step = 20
        self.gpu_device = '0'

        self.modalities = ['metric', 'trace', 'log']
        
        # alert config
        self.metric_direction = True
        self.trace_op = True
        self.trace_ab_type = True

        # TVDiag modules
        self.aug_percent = 0.2  # 删除节点比例
        self.aug_times = 5       # 启用数据增强，每个样本生成5个增强样本
        self.dynamic_weight = True
        
        # 参数化重要性感知增强配置
        self.use_degree = True    # 是否考虑节点度数（入度+出度）重要性
        self.use_distance = True  # 是否考虑距离根因节点的距离重要性

        # 跨模态注意力融合配置
        self.use_cross_modal_attention = True
        self.attention_heads = 4      # 注意力头数
        self.attention_dropout = 0.1  # 注意力dropout

        # model config
        self.batch_size = 512
        self.epochs = 500
        self.alert_embedding_dim = 128
        self.graph_hidden_dim = 64
        self.graph_out = 32
        self.graph_layers = 2
        self.linear_hidden = [64]
        self.lr = 0.001
        self.weight_decay = 0.0001

        if self.dataset == 'gaia':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 5
            self.aggregator = 'mean'
        elif self.dataset == 'aiops22':
            if not self.trace_op:
                self.lr = 0.01
            self.feat_drop = 0.1
            self.batch_size = 128
            self.patience =20
            self.ft_num = 9
            self.aggregator = 'mean'
        elif self.dataset == 'sockshop':
            self.feat_drop = 0
            self.aug_percent = 0.4
            self.batch_size = 128
            self.patience =10
            self.ft_num = 7
            self.aggregator = 'mean'
        elif self.dataset == 'hotel':
            self.feat_drop = 0.3
            self.aug_percent = 0.2
            self.batch_size = 128
            self.patience =10
            self.ft_num = 5
            self.graph_layers=2
            self.aggregator = 'mean'
        else:
            raise NotImplementedError()
    
    def print_configs(self, logger):
        for attr, value in vars(self).items():
            logger.info(f"{attr}: {value}")
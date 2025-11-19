class Config:
    def __init__(self, dataset) -> None:
        # base config
        self.dataset = dataset
        self.reconstruct = False
        self.log_step = 20
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
        
        # 模型权重选择配置
        self.use_best_model = True  # True: 使用验证集最优权重, False: 使用最后权重
        
        # k-NN模态填补配置
        self.use_knn_imputation = False  # 是否启用k-NN模态填补（消融实验）
        self.knn_k = 3                   # k-NN中的k值，默认使用top-3相似样本
        self.knn_similarity_metric = 'cosine'  # 相似度度量: 'cosine', 'euclidean'
        self.knn_strategy = 'average'    # 多模态相似度组合策略: 'average', 'weighted'
        self.vector_db_path = 'vector_database.pt'  # 向量库保存路径

        # model config
        self.batch_size = 8
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
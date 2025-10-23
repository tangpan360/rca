import os
import pickle
import torch
import numpy as np
import dgl
from core.multimodal_dataset import MultiModalDataSet
from core.aug import aug_drop_node
from config.exp_config import Config


class DatasetProcess:
    """加载预处理的dataset.pkl并构建图数据集"""
    
    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.dataset_path = "./preprocess/processed_data/dataset.pkl"
        
    def process(self):
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        # 加载数据
        with open(self.dataset_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.logger.info(f"Loaded {len(data_dict)} samples")
        
        # 加载真实的拓扑数据（nodes和edges）
        import json
        nodes_path = f"./data/{self.config.dataset}/raw/nodes.json"
        edges_path = f"./data/{self.config.dataset}/raw/edges.json"
        
        with open(nodes_path, 'r') as f:
            nodes_dict = json.load(f)
        with open(edges_path, 'r') as f:
            edges_dict = json.load(f)
        
        self.logger.info(f"Loaded real topology: {len(edges_dict['0'])} edges per sample")
        
        # 构建标签映射
        all_services = set()
        all_types = set()
        for sample in data_dict.values():
            all_services.add(sample['fault_service'])
            all_types.add(sample['fault_type'])
        
        all_services = sorted(list(all_services))
        all_types = sorted(list(all_types))
        
        service2idx = {s: i for i, s in enumerate(all_services)}
        type2idx = {t: i for i, t in enumerate(all_types)}
        
        self.logger.info(f"Services: {all_services}")
        self.logger.info(f"Fault types: {all_types}")
        
        # 构建数据集
        train_data, test_data = MultiModalDataSet(), MultiModalDataSet()
        
        for sample_id, sample in data_dict.items():
            metric_data = sample['metric_data']  # [10, 20, 12]
            log_data = sample['log_data']  # [10, 40]
            trace_data = sample['trace_data']  # [10, 20, 1]
            
            fault_service = sample['fault_service']
            fault_type = sample['fault_type']
            data_type = sample['data_type']
            
            global_root_id = service2idx[fault_service]
            failure_type_id = type2idx[fault_type]
            
            # 使用该样本的真实拓扑
            sample_nodes = nodes_dict[str(sample_id)]
            sample_edges = edges_dict[str(sample_id)]
            
            if data_type == 'train':
                train_data.add_data(
                    metric_Xs=metric_data,
                    trace_Xs=trace_data,
                    log_Xs=log_data,
                    global_root_id=global_root_id,
                    failure_type_id=failure_type_id,
                    local_root=fault_service,
                    nodes=sample_nodes,
                    edges=sample_edges
                )
            else:
                test_data.add_data(
                    metric_Xs=metric_data,
                    trace_Xs=trace_data,
                    log_Xs=log_data,
                    global_root_id=global_root_id,
                    failure_type_id=failure_type_id,
                    local_root=fault_service,
                    nodes=sample_nodes,
                    edges=sample_edges
                )
        
        # 数据增强
        aug_data = []
        if self.config.aug_times > 0:
            self.logger.info(f"Generating {self.config.aug_times} augmented samples per training sample")
            for time in range(self.config.aug_times):
                for (graph, labels) in train_data:
                    root = graph.ndata['root'].tolist().index(1)
                    aug_graph = aug_drop_node(graph, root, drop_percent=self.config.aug_percent)
                    aug_data.append((aug_graph, labels))
        
        self.logger.info(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}, Aug samples: {len(aug_data)}")
        
        return train_data, aug_data, test_data


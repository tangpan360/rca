import os
import pickle
import torch
import numpy as np
import pandas as pd
import dgl
from core.multimodal_dataset import MultiModalDataSet
from core.aug import aug_drop_node, aug_importance_aware_drop
from config.exp_config import Config
from sklearn.model_selection import train_test_split


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
        nodes_path = "./preprocess/processed_data/nodes.json"
        edges_path = "./preprocess/processed_data/edges.json"
        
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
        
        # 构建数据集（先收集所有训练样本，然后划分train/val）
        train_val_samples = []
        train_val_fault_types = []  # 用于分层抽样
        test_data = MultiModalDataSet()
        
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
            
            sample_data = {
                'metric_Xs': metric_data,
                'trace_Xs': trace_data,
                'log_Xs': log_data,
                'global_root_id': global_root_id,
                'failure_type_id': failure_type_id,
                'local_root': fault_service,
                'nodes': sample_nodes,
                'edges': sample_edges
            }
            
            if data_type == 'train':
                train_val_samples.append(sample_data)
                train_val_fault_types.append(fault_type)  # 记录故障类型
            else:
                test_data.add_data(**sample_data)        
        
        train_samples, val_samples, train_types, val_types = train_test_split(
            train_val_samples,
            train_val_fault_types,
            test_size=0.3,  # 30%作为验证集
            random_state=42,  # 固定随机种子，保证可复现
            stratify=train_val_fault_types  # 按故障类型分层抽样
        )
        
        self.logger.info(f"Split with stratification by fault type:")
        self.logger.info(f"  Train fault types distribution: {dict(pd.Series(train_types).value_counts())}")
        self.logger.info(f"  Val fault types distribution: {dict(pd.Series(val_types).value_counts())}")
        
        train_data = MultiModalDataSet()
        val_data = MultiModalDataSet()
        
        for sample_data in train_samples:
            train_data.add_data(**sample_data)
        
        for sample_data in val_samples:
            val_data.add_data(**sample_data)
        
        # 数据增强（只对训练集进行增强）
        aug_data = []
        if self.config.aug_times > 0:
            # 确定使用的增强策略
            use_degree = getattr(self.config, 'use_degree', True)
            use_distance = getattr(self.config, 'use_distance', True)
            
            # 策略描述
            if use_degree and use_distance:
                strategy_name = "度数+距离综合重要性感知增强"
            elif use_degree and not use_distance:
                strategy_name = "仅度数重要性感知增强"
            elif not use_degree and use_distance:
                strategy_name = "仅距离重要性感知增强"
            else:
                strategy_name = "传统随机增强 (fallback)"
            
            self.logger.info(f"数据增强策略: {strategy_name}")
            self.logger.info(f"  - 使用度数重要性: {use_degree}")
            self.logger.info(f"  - 使用距离重要性: {use_distance}")
            self.logger.info(f"Generating {self.config.aug_times} augmented samples per training sample")
            
            for time in range(self.config.aug_times):
                for (graph, labels) in train_data:
                    root = graph.ndata['root'].tolist().index(1)
                    
                    # 使用参数化重要性感知增强
                    if use_degree or use_distance:
                        aug_graph = aug_importance_aware_drop(
                            graph, 
                            root, 
                            drop_percent=self.config.aug_percent,
                            use_degree=use_degree,
                            use_distance=use_distance
                        )
                    else:
                        # 两个参数都为False时，使用随机增强
                        aug_graph = aug_drop_node(graph, root, drop_percent=self.config.aug_percent)
                    
                    aug_data.append((aug_graph, labels))
        
        self.logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}, Aug samples: {len(aug_data)}")
        
        return train_data, val_data, aug_data, test_data


import torch
import torch.nn.functional as F
import os
from typing import Dict, List, Optional, Tuple


class KNNModalityImputation:
    """
    基于k-NN的模态填补模块
    - 简洁高效的实现
    - 支持多种相似度度量
    - 可通过配置文件控制
    """
    
    def __init__(self, config):
        self.config = config
        self.k = config.knn_k
        self.similarity_metric = config.knn_similarity_metric
        self.strategy = config.knn_strategy
        self.vector_db_path = config.vector_db_path
        
        # 向量库（训练完成后构建）
        self.vector_db = None
        self.is_built = False
        
        # 模态权重（weighted策略使用）
        self.modality_weights = {
            'metric': 0.4,
            'log': 0.3, 
            'trace': 0.3
        }
    
    def build_vector_database(self, model, train_dataloader, device, logger):
        """
        构建向量库（训练完成后调用）
        """
        if self.is_built:
            logger.info("向量库已存在，跳过构建")
            return
            
        logger.info(f"开始构建k-NN向量库，k={self.k}")
        
        model.eval()
        vector_db = {
            'metric': [],
            'log': [],
            'trace': []
        }
        
        with torch.no_grad():
            for batch_idx, (batch_graphs, _) in enumerate(train_dataloader):
                batch_graphs = batch_graphs.to(device)
                
                # 获取Eadro编码后的嵌入
                metric_raw = batch_graphs.ndata['metric']
                log_raw = batch_graphs.ndata['log']
                trace_raw = batch_graphs.ndata['trace']
                
                metric_emb, log_emb, trace_emb = model.eadro_encoder(
                    metric_raw, log_raw, trace_raw
                )
                
                # 按图分割嵌入（每个图有固定数量的服务节点）
                num_nodes_list = batch_graphs.batch_num_nodes()
                start_idx = 0
                
                for num_nodes in num_nodes_list:
                    end_idx = start_idx + num_nodes
                    
                    # 提取单个图的嵌入
                    graph_metric = metric_emb[start_idx:end_idx]  # [num_services, 128]
                    graph_log = log_emb[start_idx:end_idx]        # [num_services, 128]
                    graph_trace = trace_emb[start_idx:end_idx]    # [num_services, 128]
                    
                    # 存储到向量库
                    vector_db['metric'].append(graph_metric.cpu())
                    vector_db['log'].append(graph_log.cpu())
                    vector_db['trace'].append(graph_trace.cpu())
                    
                    start_idx = end_idx
                
                if batch_idx % 50 == 0:
                    logger.info(f"已处理 {batch_idx} 个batch")
        
        # 转换为张量：[num_samples, num_services, 128]
        for modality in ['metric', 'log', 'trace']:
            vector_db[modality] = torch.stack(vector_db[modality], dim=0)
            logger.info(f"{modality} 向量库: {vector_db[modality].shape}")
        
        # 保存向量库
        torch.save(vector_db, self.vector_db_path)
        logger.info(f"向量库已保存: {self.vector_db_path}")
        
        self.vector_db = vector_db
        self.is_built = True
    
    def load_vector_database(self, logger):
        """加载预构建的向量库"""
        if not os.path.exists(self.vector_db_path):
            logger.warning(f"向量库不存在: {self.vector_db_path}")
            return False
            
        self.vector_db = torch.load(self.vector_db_path, map_location='cpu')
        self.is_built = True
        logger.info(f"向量库已加载: {self.vector_db_path}")
        
        for mod in ['metric', 'log', 'trace']:
            logger.info(f"  {mod}: {self.vector_db[mod].shape}")
        
        return True
    
    def compute_similarity(self, query_emb: torch.Tensor, db_embs: torch.Tensor) -> torch.Tensor:
        """
        计算相似度
        Args:
            query_emb: [num_services, 128] 查询嵌入
            db_embs: [num_samples, num_services, 128] 数据库嵌入
        Returns:
            similarities: [num_samples] 相似度分数  
        """
        # 使用平均池化处理不同大小的图，避免形状不匹配
        # 查询图聚合: [num_services, 128] -> [128]
        query_pooled = torch.mean(query_emb, dim=0)  # [128]
        
        # 数据库图聚合: [num_samples, num_services, 128] -> [num_samples, 128]
        db_pooled = torch.mean(db_embs, dim=1)  # [num_samples, 128]
        
        if self.similarity_metric == 'cosine':
            # 余弦相似度
            query_norm = F.normalize(query_pooled.unsqueeze(0), p=2, dim=1)  # [1, 128]
            db_norm = F.normalize(db_pooled, p=2, dim=1)  # [num_samples, 128]
            similarities = torch.mm(query_norm, db_norm.t()).squeeze()  # [num_samples]
            
        elif self.similarity_metric == 'euclidean':
            # 欧氏距离转相似度
            distances = torch.norm(db_pooled - query_pooled.unsqueeze(0), p=2, dim=1)  # [num_samples]
            similarities = 1.0 / (1.0 + distances)
        
        else:
            raise ValueError(f"不支持的相似度度量: {self.similarity_metric}")
        
        return similarities
    
    def find_top_k_similar(self, query_embs: Dict[str, torch.Tensor], 
                          available_modalities: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        找到最相似的k个样本
        """
        modality_similarities = {}
        
        # 计算每个可用模态的相似度
        for modality in available_modalities:
            similarities = self.compute_similarity(
                query_embs[modality],
                self.vector_db[modality]
            )
            modality_similarities[modality] = similarities
        
        # 组合多模态相似度
        if self.strategy == 'average':
            # 简单平均
            combined_similarities = torch.stack(list(modality_similarities.values())).mean(dim=0)
            
        elif self.strategy == 'weighted':  
            # 加权平均
            weighted_sum = 0.0
            total_weight = 0.0
            for modality in available_modalities:
                weight = self.modality_weights[modality]
                weighted_sum += weight * modality_similarities[modality]
                total_weight += weight
            combined_similarities = weighted_sum / total_weight
            
        else:
            raise ValueError(f"不支持的组合策略: {self.strategy}")
        
        # 找top-k
        top_k_similarities, top_k_indices = torch.topk(
            combined_similarities,
            k=min(self.k, len(combined_similarities)),
            largest=True
        )
        
        return top_k_indices, top_k_similarities
    
    def impute(self, modal_embeddings: Dict[str, torch.Tensor], 
               missing_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """
        k-NN模态填补主函数
        """
        if not missing_modalities or not self.is_built:
            return modal_embeddings
        
        # 确保向量库在正确设备上
        device = next(iter(modal_embeddings.values())).device
        for modality in self.vector_db:
            self.vector_db[modality] = self.vector_db[modality].to(device)
        
        completed_embeddings = modal_embeddings.copy()
        
        # 确定可用模态
        available_modalities = [
            mod for mod in modal_embeddings.keys()
            if mod not in missing_modalities
        ]
        
        if not available_modalities:
            # 所有模态都缺失，使用零填充
            for mod in missing_modalities:
                completed_embeddings[mod] = torch.zeros_like(
                    list(modal_embeddings.values())[0] if modal_embeddings 
                    else torch.zeros(10, 128, device=device)
                )
            return completed_embeddings
        
        # 找相似样本
        top_k_indices, similarities = self.find_top_k_similar(
            modal_embeddings, available_modalities
        )
        
        # 填补每个缺失模态
        for missing_mod in missing_modalities:
            # 获取top-k样本的嵌入
            top_k_embs = self.vector_db[missing_mod][top_k_indices]  # [k, db_num_services, 128]
            
            # 基于相似度加权平均得到图级表示
            weights = F.softmax(similarities, dim=0)  # [k]
            
            # 先对每个样本进行图级聚合，再加权平均
            top_k_graph_level = torch.mean(top_k_embs, dim=1)  # [k, 128] 
            weighted_graph_repr = (top_k_graph_level * weights.unsqueeze(1)).sum(dim=0)  # [128]
            
            # 获取查询图的节点数，扩展图级表示到节点级
            query_num_nodes = list(modal_embeddings.values())[0].size(0)  # 查询图的节点数
            imputed_emb = weighted_graph_repr.unsqueeze(0).expand(query_num_nodes, -1)  # [query_num_nodes, 128]
            
            completed_embeddings[missing_mod] = imputed_emb
        
        return completed_embeddings


def build_vector_database_standalone(model_path: str, config, train_dataloader, logger):
    """
    独立构建向量库的函数（可在训练完成后单独调用）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载训练好的模型
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # 创建k-NN填补器并构建向量库
    knn_imputer = KNNModalityImputation(config)
    knn_imputer.build_vector_database(model, train_dataloader, device, logger)
    
    return knn_imputer

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import dgl
import numpy as np

from core.ita import cal_task_affinity
from core.loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from core.model.MainModelEadro import MainModelEadro
from helper.eval import *
from helper.early_stop import EarlyStopping
from helper.Result import Result
from config.exp_config import Config


class TVDiagEadro(object):
    """集成Eadro编码器的TVDiag训练和评估框架"""

    def __init__(self, config: Config, logger, log_dir: str):
        self.config = config
        self.logger = logger
        os.makedirs(log_dir, exist_ok=True)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Currently using GPU {}".format(config.gpu_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device
            self.device = 'cuda'
        else:
            logger.info("Currently using CPU (GPU is highly recommended)")
            self.device = 'cpu'

        self.result = Result()
        self.writer = SummaryWriter(log_dir)
        self.printParams()

    def printParams(self):
        self.config.print_configs(self.logger)

    def train(self, train_data, val_data, aug_data):
        model = MainModelEadro(self.config).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        awl = AutomaticWeightedLoss(2)  # 只有2个损失：l_rcl 和 l_fti

        self.logger.info(model)
        self.logger.info(f"Start training for {self.config.epochs} epochs.")
        
        train_times = []
        Z_r2fs, Z_f2rs = [], []
        
        earlyStop = EarlyStopping(patience=self.config.patience)
        best_model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_best.pt')
        
        for epoch in range(self.config.epochs):
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss, epoch_rcl_l, epoch_fti_l = 0, 0, 0
            rcl_results = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [],"HR@5": [], "MRR@3": []}
            fti_results = {'pre':[], 'rec':[], 'f1':[]}

            train_dl = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate)
            for batch_graphs, batch_labels in train_dl:
                batch_graphs = batch_graphs
                instance_labels = batch_labels[:, 0]
                type_labels = batch_labels[:, 1]

                if self.config.aug_times > 0:
                    raw_graphs = dgl.unbatch(batch_graphs)
                    aug_graphs, aug_labels = map(list, zip(*random.sample(aug_data, len(raw_graphs))))
                    batch_graphs = dgl.batch(raw_graphs + aug_graphs)
                    instance_labels = torch.hstack((instance_labels, torch.tensor(aug_labels)[:,0].flatten()))
                    type_labels = torch.hstack((type_labels, torch.tensor(aug_labels)[:,1].flatten()))

                batch_graphs = batch_graphs.to(self.device)
                instance_labels = instance_labels.to(self.device)
                type_labels = type_labels.to(self.device)

                opt.zero_grad()
                
                # 多场景数据扩展或普通训练
                if (getattr(self.config, 'use_modality_dropout', False) and 
                    getattr(self.config, 'modality_dropout_mode', 'random') == 'multi_scenario' and
                    getattr(self.config, 'use_cross_modal_attention', False)):
                    # 多场景训练：将batch扩展为4种模态配置的混合batch
                    expanded_graphs, expanded_type_labels = self._expand_batch_multi_scenario(batch_graphs, type_labels)
                    fs, es, root_logit, type_logit = model(expanded_graphs)
                    l_rcl = self.cal_rcl_loss(root_logit, expanded_graphs)
                    l_fti = F.cross_entropy(type_logit, expanded_type_labels)
                else:
                    # 普通训练
                    fs, es, root_logit, type_logit = model(batch_graphs)
                    l_rcl = self.cal_rcl_loss(root_logit, batch_graphs)
                    l_fti = F.cross_entropy(type_logit, type_labels)
                
                if self.config.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti)
                else:
                    total_loss = l_rcl + l_fti

                total_loss.backward()
                opt.step()
                
                epoch_loss += total_loss.detach().item()
                epoch_rcl_l += l_rcl.detach().item()
                epoch_fti_l += l_fti.detach().item()

                # 在多场景模式下，使用原始batch进行评估
                if (getattr(self.config, 'use_modality_dropout', False) and 
                    getattr(self.config, 'modality_dropout_mode', 'random') == 'multi_scenario' and
                    getattr(self.config, 'use_cross_modal_attention', False)):
                    # 多场景模式：使用原始batch（完整模态）进行评估
                    _, _, eval_root_logit, eval_type_logit = model(batch_graphs)
                    rcl_res = RCA_eval(eval_root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                    fti_res = FTI_eval(eval_type_logit, type_labels)
                else:
                    # 普通模式：使用训练的预测结果
                    rcl_res = RCA_eval(root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                    fti_res = FTI_eval(type_logit, type_labels)
                
                [rcl_results[key].append(value) for key, value in rcl_res.items()]
                [fti_results[key].append(value) for key, value in fti_res.items()]
                n_iter += 1
                
            mean_epoch_loss = epoch_loss / n_iter
            mean_rcl_loss = epoch_rcl_l / n_iter
            mean_fti_loss = epoch_fti_l / n_iter
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            train_times.append(time_per_epoch)
            
            self.logger.info("Epoch {} done. Loss: {:.3f}, Time per epoch: {:.3f}[s]"
                         .format(epoch, mean_epoch_loss, time_per_epoch))

            for k, v in rcl_results.items():
                rcl_results[k] = np.mean(v)
            for k, v in fti_results.items():
                fti_results[k] = np.mean(v)
                
            self.writer.add_scalar('loss/train_total_loss', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('train/HR@3', rcl_results['HR@3'], global_step=epoch)
            self.writer.add_scalar('train/f1-score', fti_results['f1'], global_step=epoch)

            # 在验证集上评估
            val_loss, val_rcl, val_fti = self._validate(model, val_data)
            
            self.writer.add_scalar('loss/val_total_loss', val_loss, global_step=epoch)
            self.writer.add_scalar('val/HR@3', val_rcl['HR@3'], global_step=epoch)
            self.writer.add_scalar('val/f1-score', val_fti['f1'], global_step=epoch)
            
            self.logger.info(f"Val Loss: {val_loss:.3f}, Val HR@3: {val_rcl['HR@3']:.3%}, Val F1: {val_fti['f1']:.3%}")

            # 早停判断（基于验证集loss）
            stop, is_best = earlyStop.should_stop(val_loss, epoch)
            
            if is_best:
                # 保存最优模型
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'val_loss': val_loss,
                }
                torch.save(state, best_model_path)
                self.logger.info(f"✓ Best model saved at epoch {epoch} with val_loss: {val_loss:.3f}")
            
            if stop:
                self.logger.info(f"Early stop at epoch {epoch} due to no improvement on validation set.")
                break

        # 保存最终模型（最后一轮）
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        torch.save(state, os.path.join(self.writer.log_dir, 'TVDiagEadro_last.pt'))
        self.result.set_train_efficiency(train_times)
        self.logger.info("Training has finished.")
        self.logger.info(f"Best model saved at: {best_model_path}")

    def _validate(self, model, val_data):
        """
        在验证集上评估模型
        
        Returns:
            tuple: (val_loss, rcl_results, fti_results)
        """
        model.eval()
        val_loss, val_rcl_l, val_fti_l = 0, 0, 0
        rcl_results = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [], "HR@5": [], "MRR@3": []}
        fti_results = {'pre': [], 'rec': [], 'f1': []}
        n_iter = 0
        
        val_dl = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate)
        
        with torch.no_grad():
            for batch_graphs, batch_labels in val_dl:
                batch_graphs = batch_graphs.to(self.device)
                instance_labels = batch_labels[:, 0].to(self.device)
                type_labels = batch_labels[:, 1].to(self.device)
                
                fs, es, root_logit, type_logit = model(batch_graphs)
                
                # 只计算主任务损失
                l_rcl = self.cal_rcl_loss(root_logit, batch_graphs)
                l_fti = F.cross_entropy(type_logit, type_labels)
                
                total_loss = l_rcl + l_fti
                
                val_loss += total_loss.detach().item()
                val_rcl_l += l_rcl.detach().item()
                val_fti_l += l_fti.detach().item()
                
                rcl_res = RCA_eval(root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                fti_res = FTI_eval(type_logit, type_labels)
                [rcl_results[key].append(value) for key, value in rcl_res.items()]
                [fti_results[key].append(value) for key, value in fti_res.items()]
                n_iter += 1
        
        mean_val_loss = val_loss / n_iter
        for k, v in rcl_results.items():
            rcl_results[k] = np.mean(v)
        for k, v in fti_results.items():
            fti_results[k] = np.mean(v)
        
        model.train()  # 恢复训练模式
        return mean_val_loss, rcl_results, fti_results

    def evaluate(self, test_data, model=None):
        if model is None:
            # 加载最优模型权重
            best_model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_best.pt')
            if os.path.exists(best_model_path):
                self.logger.info(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path)
            else:
                # 如果没有最优模型，加载最后的模型
                self.logger.info("Best model not found, loading last model")
                checkpoint = torch.load(os.path.join(self.writer.log_dir, 'TVDiagEadro_last.pt'))
            
            model = MainModelEadro(self.config).to(self.device)
            model.load_state_dict(checkpoint['model'])
            self.logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
       
        model.eval()
        root_logits, type_logits = [], []
        roots, types = [], []
        inference_times = []
        num_node_list = []
        
        for data in test_data:
            graph = data[0].to(self.device)
            failure_type = data[1][1]
            roots.append(graph.ndata['root'])
            types.append(failure_type)
            num_node_list.append(graph.num_nodes())
        
            start_time = time.time()
            with torch.no_grad():
                _, _, root_logit, type_logit = model(graph)
                root_logits.append(root_logit.flatten())
                type_logits.append(type_logit.flatten())
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
        root_logits = torch.hstack(root_logits).cpu()
        type_logits = torch.vstack(type_logits).cpu()
        roots = torch.hstack(roots)
        types = torch.tensor(types)

        rcl_res = RCA_eval(root_logits, num_node_list, roots)
        fti_res = FTI_eval(type_logits, types)
        self.result.set_performance(rcl_res, fti_res)
        self.result.set_inference_efficiency(inference_times)

        avg_3 = np.mean([rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3']])

        self.logger.info("[Root localization] HR@1: {:.3%}, HR@2: {:.3%}, HR@3: {:.3%}, HR@4: {:.3%}, HR@5: {:.3%}, avg@3: {:.3f}, MRR@3: {:.3f}"\
            .format(rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3'], rcl_res['HR@4'], rcl_res['HR@5'] , avg_3, rcl_res['MRR@3']))
        self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}"\
            .format(fti_res['pre'], fti_res['rec'], fti_res['f1']))
        self.logger.info(f"The average test time is {np.mean(inference_times)}[s]")

        return self.result

    def cal_rcl_loss(self, root_logit, batch_graphs):        
        num_nodes_list = batch_graphs.batch_num_nodes()
        total_loss = None
        
        start_idx = 0
        for idx, num_nodes in enumerate(num_nodes_list):
            end_idx = start_idx + num_nodes
            node_logits = root_logit[start_idx : end_idx].reshape(1, -1)
            root = batch_graphs.ndata["root"][start_idx : end_idx].tolist().index(1)
            loss = F.cross_entropy(node_logits, torch.LongTensor([root]).view(1).to(self.device))
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            start_idx += num_nodes
            
        l_rcl = total_loss / len(num_nodes_list)
        return l_rcl

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

    def _expand_batch_multi_scenario(self, batch_graphs, type_labels):
        """
        多场景batch扩展：将原始batch扩展为1.5x大小的混合batch
        
        比例分配：
        - 完整模态：N个样本（保持原有数量）
        - 缺失模态：N/2个样本（总共，平均分配给3种缺失情况）
        - 总样本数：1.5N
        
        Args:
            batch_graphs: 原始batch图数据
            type_labels: 原始类型标签
            
        Returns:
            expanded_graphs: 扩展后的图数据（1.5x大小）
            expanded_type_labels: 扩展后的类型标签（1.5x大小）
        """
        
        # 解batch获取单个图
        graph_list = dgl.unbatch(batch_graphs)
        original_batch_size = len(graph_list)
        
        # 如果启用了数据增强，batch包含原始数据+增强数据，需要只从原始数据中选择
        if self.config.aug_times > 0:
            # batch结构：前半部分是原始数据，后半部分是增强数据
            original_data_size = original_batch_size // 2
            original_graphs = graph_list[:original_data_size]  # 只取原始数据
            original_labels = type_labels[:original_data_size]
            print(f"   🔍 检测到数据增强: batch={original_batch_size}, 仅使用原始数据={original_data_size}")
        else:
            # 没有数据增强，整个batch都是原始数据
            original_graphs = graph_list
            original_labels = type_labels
            original_data_size = original_batch_size
        
        # 计算各种配置的样本数量（基于原始数据大小）
        full_modality_count = original_data_size  # 完整模态样本数
        missing_ratio = getattr(self.config, 'missing_modality_ratio', 0.5)  # 缺失模态比例
        missing_modality_total = int(original_data_size * missing_ratio)  # 缺失模态总数
        missing_per_type = missing_modality_total // 3  # 每种缺失类型的样本数
        
        # 处理不能整除的情况，优先分配给缺metric
        remaining = missing_modality_total - missing_per_type * 3
        missing_metric_count = missing_per_type + remaining
        missing_log_count = missing_per_type
        missing_trace_count = missing_per_type
        
        print(f"   📊 缺失模态比例: {missing_ratio:.1f} (总缺失={missing_modality_total})")
        print(f"   📊 样本分配: 完整={full_modality_count}, 缺metric={missing_metric_count}, 缺log={missing_log_count}, 缺trace={missing_trace_count}")
        
        # 定义模态配置和对应数量
        config_specs = [
            ({'metric': True, 'log': True, 'trace': True}, full_modality_count),    # 完整
            ({'metric': False, 'log': True, 'trace': True}, missing_metric_count),  # 缺metric
            ({'metric': True, 'log': False, 'trace': True}, missing_log_count),     # 缺log  
            ({'metric': True, 'log': True, 'trace': False}, missing_trace_count)    # 缺trace
        ]
        
        expanded_graphs = []
        expanded_labels = []
        modality_masks = []  # 收集所有的模态掩码
        
        # 为每种配置生成对应数量的样本
        for config_idx, (config, count) in enumerate(config_specs):
            is_full_modality = (config_idx == 0)  # 第一个配置是完整模态
            
            for i in range(count):
                if is_full_modality:
                    # 完整模态：顺序使用所有原始数据（确保覆盖完整）
                    graph_idx = i % original_data_size
                else:
                    # 缺失模态：随机选择（增加训练多样性）
                    graph_idx = random.randint(0, original_data_size - 1)
                
                graph = original_graphs[graph_idx]
                
                # 复制图
                new_graph = graph.clone()
                expanded_graphs.append(new_graph)
                expanded_labels.append(original_labels[graph_idx].item())
                
                # 收集模态掩码
                mask = torch.tensor([
                    config['metric'], config['log'], config['trace']
                ], dtype=torch.bool).to(graph.device)
                modality_masks.append(mask)
        
        # 重新batch化
        expanded_batch_graphs = dgl.batch(expanded_graphs)
        expanded_type_labels = torch.tensor(expanded_labels, dtype=original_labels.dtype).to(original_labels.device)
        
        # 将模态掩码存储在batch图的属性中
        expanded_batch_graphs.modality_masks = torch.stack(modality_masks)  # [batch_size, 3]
        
        total_samples = len(expanded_labels)
        print(f"   ✅ Batch扩展: 原始数据{original_data_size} → {total_samples} (扩展倍数: {total_samples/original_data_size:.1f}x)")
        
        return expanded_batch_graphs, expanded_type_labels



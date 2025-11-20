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
                
                # 确定训练时使用的模态
                active_modalities = None
                if self.config.use_partial_modalities:
                    active_modalities = self.config.training_modalities
                
                fs, es, root_logit, type_logit = model(batch_graphs, active_modalities=active_modalities)

                # 只保留主任务损失
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
        
        # 构建k-NN向量库（如果启用）
        if self.config.use_knn_imputation:
            self.logger.info("构建k-NN向量库...")
            model.build_knn_database(train_dl, self.device, self.logger)
            self.logger.info("k-NN向量库构建完成")
            
            # k-NN微调训练（如果启用）
            if self.config.enable_knn_finetune:
                self.logger.info("=" * 60)
                self.logger.info("开始k-NN微调训练...")
                self.logger.info("=" * 60)
                self._knn_finetune(model, train_data, val_data)

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
                
                # 确定验证时使用的模态
                active_modalities = None
                if self.config.use_partial_modalities:
                    active_modalities = self.config.testing_modalities
                
                fs, es, root_logit, type_logit = model(batch_graphs, active_modalities=active_modalities)
                
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
            if self.config.use_best_model:
                model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_best.pt')
                if not os.path.exists(model_path):
                    model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_last.pt')
                    self.logger.info("Best model not found, fallback to last model")
            else:
                # 加载最后的模型
                model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_last.pt')
            
            self.logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path)
            
            model = MainModelEadro(self.config).to(self.device)
            model.load_state_dict(checkpoint['model'])
            self.logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
        
        # 设置k-NN填补器（如果启用）
        if self.config.use_knn_imputation:
            model.setup_knn_imputer(self.logger)
       
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
                # 确定测试时使用的模态
                active_modalities = None
                if self.config.use_partial_modalities:
                    active_modalities = self.config.testing_modalities
                
                _, _, root_logit, type_logit = model(graph, active_modalities=active_modalities)
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
        # 处理普通格式 (graph, labels) 和微调格式 (graph, labels, missing_modalities)
        if len(samples[0]) == 2:
            # 普通格式
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            batched_labels = torch.tensor(labels)
            return batched_graph, batched_labels
        else:
            # 微调格式，包含缺失模态信息
            graphs, labels, missing_info = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            batched_labels = torch.tensor(labels)
            # 将缺失模态信息作为额外返回值
            return batched_graph, batched_labels, missing_info
    
    def evaluate_with_missing_modalities(self, test_data, missing_modalities=None, model=None):
        """
        测试模态缺失场景下的模型性能
        Args:
            test_data: 测试数据
            missing_modalities: 缺失的模态列表，如 ['log'] 或 ['metric', 'trace']
            model: 可选的模型实例
        """
        if model is None:
            # 加载模型（与evaluate函数相同逻辑）
            if self.config.use_best_model:
                model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_best.pt')
                if not os.path.exists(model_path):
                    model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_last.pt')
                    self.logger.info("Best model not found, fallback to last model")
            else:
                model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_last.pt')
            
            self.logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path)
            
            model = MainModelEadro(self.config).to(self.device)
            model.load_state_dict(checkpoint['model'])
            self.logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
            
            # 设置k-NN填补器（如果启用）
            if self.config.use_knn_imputation:
                model.setup_knn_imputer(self.logger)
        
        if missing_modalities:
            self.logger.info(f"测试场景：缺失模态 {missing_modalities}")
            if self.config.use_knn_imputation:
                self.logger.info(f"使用k-NN填补，k={self.config.knn_k}, 相似度={self.config.knn_similarity_metric}")
            else:
                self.logger.info("未启用k-NN填补，将使用原始模态数据")
        else:
            self.logger.info("测试场景：完整模态")
        
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
                # 确定测试时使用的模态
                active_modalities = None
                if self.config.use_partial_modalities:
                    active_modalities = self.config.testing_modalities
                
                # 传入缺失模态信息和使用模态信息
                _, _, root_logit, type_logit = model(graph, missing_modalities, active_modalities)
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
        
        avg_3 = np.mean([rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3']])
        
        # 打印结果
        missing_str = f"[缺失{missing_modalities}]" if missing_modalities else "[完整模态]"
        self.logger.info(f"{missing_str} [Root localization] HR@1: {rcl_res['HR@1']:.3%}, HR@2: {rcl_res['HR@2']:.3%}, HR@3: {rcl_res['HR@3']:.3%}, HR@4: {rcl_res['HR@4']:.3%}, HR@5: {rcl_res['HR@5']:.3%}, avg@3: {avg_3:.3f}, MRR@3: {rcl_res['MRR@3']:.3f}")
        self.logger.info(f"{missing_str} [Failure type classification] precision: {fti_res['pre']:.3%}, recall: {fti_res['rec']:.3%}, f1-score: {fti_res['f1']:.3%}")
        self.logger.info(f"{missing_str} The average test time is {np.mean(inference_times):.4f}[s]")
        
        return rcl_res, fti_res, np.mean(inference_times)

    def _knn_finetune(self, model, train_data, val_data):
        """k-NN微调训练主流程"""
        
        # 1. 构造微调数据集
        finetune_data = self._create_finetune_dataset(train_data)
        
        # 2. 设置k-NN填补器
        model.setup_knn_imputer(self.logger)
        
        # 3. 冻结指定参数
        self._freeze_parameters(model)
        
        # 4. 创建微调优化器
        finetune_lr = self.config.lr * self.config.finetune_lr_ratio
        finetune_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=finetune_lr,
            weight_decay=self.config.weight_decay
        )
        
        # 5. 创建数据加载器
        finetune_dl = DataLoader(finetune_data, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate)
        
        # 6. 微调训练循环
        self._finetune_training_loop(model, finetune_dl, val_data, finetune_optimizer)
        
        self.logger.info("k-NN微调训练完成")

    def _create_finetune_dataset(self, train_data):
        """构造微调数据集: 50%原始 + 50%模拟缺失"""
        import random
        import copy
        
        self.logger.info("构造k-NN微调数据集...")
        
        # 随机打乱并划分
        shuffled_data = list(train_data)
        random.shuffle(shuffled_data)
        mid_point = len(shuffled_data) // 2
        
        # 50% 原始完整数据
        complete_data = shuffled_data[:mid_point]
        
        # 50% 模拟缺失数据，平均分配给三种模态
        incomplete_data = shuffled_data[mid_point:]
        modalities = ['metric', 'log', 'trace']
        num_per_modality = len(incomplete_data) // 3
        
        knn_augmented_data = []
        
        for i, modality in enumerate(modalities):
            start_idx = i * num_per_modality
            end_idx = (i + 1) * num_per_modality if i < 2 else len(incomplete_data)
            
            modality_data = incomplete_data[start_idx:end_idx]
            
            for sample in modality_data:
                # 创建带缺失标记的样本
                augmented_sample = copy.deepcopy(sample)
                # 为样本添加缺失模态信息（通过添加属性）
                if not hasattr(augmented_sample, 'missing_modalities'):
                    augmented_sample = (augmented_sample[0], augmented_sample[1], [modality])
                else:
                    # 如果已经是三元组，修改第三个元素
                    augmented_sample = (augmented_sample[0], augmented_sample[1], [modality])
                knn_augmented_data.append(augmented_sample)
        
        # 为完整数据添加空缺失列表
        complete_data_with_missing = []
        for sample in complete_data:
            if len(sample) == 2:  # 原始格式
                complete_data_with_missing.append((sample[0], sample[1], []))
            else:
                complete_data_with_missing.append(sample)
        
        # 合并数据集
        finetune_dataset = complete_data_with_missing + knn_augmented_data
        random.shuffle(finetune_dataset)
        
        self.logger.info(f"微调数据集构成:")
        self.logger.info(f"  - 完整数据: {len(complete_data)} 样本")
        self.logger.info(f"  - k-NN补全数据: {len(knn_augmented_data)} 样本")
        self.logger.info(f"    - 缺失metric: {len([x for x in knn_augmented_data if 'metric' in x[2]])} 样本")
        self.logger.info(f"    - 缺失log: {len([x for x in knn_augmented_data if 'log' in x[2]])} 样本")
        self.logger.info(f"    - 缺失trace: {len([x for x in knn_augmented_data if 'trace' in x[2]])} 样本")
        self.logger.info(f"  - 总计: {len(finetune_dataset)} 样本")
        
        return finetune_dataset

    def _freeze_parameters(self, model):
        """冻结指定的模型参数"""
        
        frozen_params = 0
        total_params = 0
        
        # 冻结Eadro编码器
        if self.config.freeze_eadro_encoder:
            for param in model.eadro_encoder.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            self.logger.info("已冻结 eadro_encoder 参数")
        
        # 统计参数数量
        for param in model.parameters():
            total_params += param.numel()
        
        trainable_params = total_params - frozen_params
        
        self.logger.info(f"参数冻结统计:")
        self.logger.info(f"  - 总参数: {total_params:,}")
        self.logger.info(f"  - 冻结参数: {frozen_params:,}")
        self.logger.info(f"  - 可训练参数: {trainable_params:,}")
        self.logger.info(f"  - 可训练比例: {trainable_params/total_params:.1%}")

    def _finetune_training_loop(self, model, finetune_dl, val_data, optimizer):
        """微调训练循环"""
        import torch.nn.functional as F
        
        # 早停策略
        early_stop = EarlyStopping(patience=self.config.finetune_patience)
        
        # 保存路径
        finetune_model_path = os.path.join(self.writer.log_dir, 'TVDiagEadro_finetuned.pt')
        
        model.train()
        
        for epoch in range(self.config.finetune_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_data in finetune_dl:
                # 处理batch数据格式
                if len(batch_data) == 3:  # 包含缺失模态信息
                    batch_graphs, batch_labels, missing_modalities_batch = batch_data
                    missing_modalities = missing_modalities_batch[0] if missing_modalities_batch[0] else None
                else:
                    batch_graphs, batch_labels = batch_data
                    missing_modalities = None
                
                batch_graphs = batch_graphs.to(self.device)
                instance_labels = batch_labels[:, 0].to(self.device)
                type_labels = batch_labels[:, 1].to(self.device)
                
                optimizer.zero_grad()
                
                fs, es, root_logit, type_logit = model(batch_graphs, missing_modalities)
                
                # 计算损失
                l_rcl = self.cal_rcl_loss(root_logit, batch_graphs)
                l_fti = F.cross_entropy(type_logit, type_labels)
                
                if self.config.dynamic_weight:
                    # 这里简化处理，直接相加
                    total_loss = l_rcl + l_fti
                else:
                    total_loss = l_rcl + l_fti
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.detach().item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            
            # 验证
            val_loss, val_rcl, val_fti = self._validate(model, val_data)
            
            self.logger.info(f"微调 Epoch {epoch+1}/{self.config.finetune_epochs}: "
                           f"Train Loss: {avg_loss:.3f}, Val Loss: {val_loss:.3f}, "
                           f"Val HR@3: {val_rcl['HR@3']:.3%}, Val F1: {val_fti['f1']:.3%}")
            
            # 早停判断
            stop, is_best = early_stop.should_stop(val_loss, epoch)
            
            if is_best:
                # 保存微调后的最优模型
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'is_finetuned': True
                }
                torch.save(state, finetune_model_path)
                self.logger.info(f"✓ 微调最优模型保存于 epoch {epoch+1}")
            
            if stop:
                self.logger.info(f"微调在 epoch {epoch+1} 提前停止")
                break



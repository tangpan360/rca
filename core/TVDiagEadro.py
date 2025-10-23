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
from core.loss.UnsupervisedContrastiveLoss import UspConLoss
from core.loss.SupervisedContrastiveLoss import SupConLoss
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

    def train(self, train_data, aug_data):
        model = MainModelEadro(self.config).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        awl = AutomaticWeightedLoss(4)
        supConLoss = SupConLoss(self.config.temperature, self.device).to(self.device)
        uspConLoss = UspConLoss(self.config.temperature, self.device).to(self.device)

        self.logger.info(model)
        self.logger.info(f"Start training for {self.config.epochs} epochs.")
        
        train_times = []
        Z_r2fs, Z_f2rs = [], []
        
        earlyStop = EarlyStopping(patience=self.config.patience)
        for epoch in range(self.config.epochs):
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss, epoch_con_l, epoch_rcl_l, epoch_fti_l = 0, 0, 0, 0
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
                fs, es, root_logit, type_logit = model(batch_graphs)

                # Task-oriented learning
                l_to, l_cm = torch.tensor(0.0, dtype=torch.float32, device=self.device), \
                    torch.tensor(0.0, dtype=torch.float32, device=self.device)
                modalities = self.config.modalities
                if self.config.TO:
                    if 'metric' in modalities:
                        l_to += supConLoss(fs['metric'], instance_labels)
                    if 'log' in modalities:
                        l_to += supConLoss(fs['log'], type_labels)
                    if 'trace' in modalities:
                        l_to += supConLoss(fs['trace'], instance_labels)

                # Cross-modal association
                if self.config.CM:
                    if len(modalities) >= 2 and 'metric' in modalities:
                        left_modalies = [modality for modality in modalities if modality != 'metric']
                        for modality in left_modalies:
                            l_cm += uspConLoss(fs['metric'], fs[modality])
                
                sigma = self.config.contrastive_loss_scale
                l_con = sigma * (l_to + l_cm)
                l_rcl = self.cal_rcl_loss(root_logit, batch_graphs)
                l_fti = F.cross_entropy(type_logit, type_labels)
                
                if self.config.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti, sigma*l_to, sigma*l_cm)
                else:
                    total_loss = l_con + l_rcl + l_fti

                total_loss.backward()
                opt.step()
                
                epoch_loss += total_loss.detach().item()
                epoch_con_l += l_con.detach().item()
                epoch_rcl_l += l_rcl.detach().item()
                epoch_fti_l += l_fti.detach().item()

                rcl_res = RCA_eval(root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                fti_res = FTI_eval(type_logit, type_labels)
                [rcl_results[key].append(value) for key, value in rcl_res.items()]
                [fti_results[key].append(value) for key, value in fti_res.items()]
                n_iter += 1
                
            mean_epoch_loss = epoch_loss / n_iter
            mean_con_loss = epoch_con_l / n_iter
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
                
            self.writer.add_scalar('loss/mean total loss', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('train/HR@3', rcl_results['HR@3'], global_step=epoch)
            self.writer.add_scalar('train/f1-score', fti_results['f1'], global_step=epoch)

            stop = earlyStop.should_stop(mean_epoch_loss, epoch)
            if stop:
                self.logger.info(f"Early stop at epoch {epoch} due to lack of improvement.")
                break

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        torch.save(state, os.path.join(self.writer.log_dir, 'TVDiagEadro.pt'))
        self.result.set_train_efficiency(train_times)
        self.logger.info("Training has finished.")

    def evaluate(self, test_data, model=None):
        if model is None:
            checkpoint = torch.load(os.path.join(self.writer.log_dir, 'TVDiagEadro.pt'))
            model = MainModelEadro(self.config).to(self.device)
            model.load_state_dict(checkpoint['model'])
       
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



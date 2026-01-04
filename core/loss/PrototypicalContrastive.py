import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalContrastiveLoss(nn.Module):
    """
    原型引导的任务特定对比学习
    """
    
    def __init__(self, 
                 num_fti_classes,
                 num_rcl_classes,
                 feature_dim=32,
                 temperature=0.3,
                 initial_momentum=0.5,
                 final_momentum=0.9,
                 warmup_epochs=3,
                 device='cuda'):
        super().__init__()
        
        self.num_fti_classes = num_fti_classes
        self.num_rcl_classes = num_rcl_classes
        self.temperature = temperature
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.current_epoch = 0
        
        # 初始化原型（register_buffer会自动保存/加载）
        self.register_buffer(
            'prototypes_fti',
            F.normalize(torch.randn(num_fti_classes, feature_dim), dim=1).to(device)
        )
        self.register_buffer(
            'prototypes_rcl',
            F.normalize(torch.randn(num_rcl_classes, feature_dim), dim=1).to(device)
        )
    
    def set_epoch(self, epoch):
        """设置当前epoch（用于自适应动量）"""
        self.current_epoch = epoch
    
    def get_current_momentum(self):
        """自适应动量：前期快速学习，后期稳定收敛"""
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            return self.initial_momentum + \
                   (self.final_momentum - self.initial_momentum) * progress
        return self.final_momentum
    
    def forward(self, f_fti, e_rcl, type_labels, node_root_labels):
        """
        Args:
            f_fti: [B, D] - FTI任务的图级特征
            e_rcl: [N, D] - RCL任务的节点级特征
            type_labels: [B] - 故障类型标签
            node_root_labels: [N] - 节点根因标签
        Returns:
            loss_fti, loss_rcl: 两个任务的对比损失
        """
        # 更新原型
        self._update_prototypes(f_fti, type_labels, self.prototypes_fti, self.num_fti_classes)
        self._update_prototypes(e_rcl, node_root_labels, self.prototypes_rcl, self.num_rcl_classes)
        
        # 计算原型对比损失
        loss_fti = self._prototype_loss(f_fti, type_labels, self.prototypes_fti)
        loss_rcl = self._prototype_loss(e_rcl, node_root_labels, self.prototypes_rcl)
        
        return loss_fti, loss_rcl
    
    def _update_prototypes(self, features, labels, prototypes, num_classes):
        """
        动量更新原型
        
        使用指数移动平均(EMA)更新原型向量，跨batch积累全局知识。
        动量系数通过get_current_momentum()自适应调整（针对早停场景优化）。
        """
        features = F.normalize(features, dim=1)
        momentum = self.get_current_momentum()
        
        with torch.no_grad():
            for c in range(num_classes):
                mask = (labels == c)
                
                if mask.sum() > 0:
                    # 计算当前batch中类别c的平均特征
                    class_mean = features[mask].mean(dim=0)
                    class_mean = F.normalize(class_mean.unsqueeze(0), dim=1).squeeze(0)
                    
                    # 标准EMA更新：prototype_new = m * prototype_old + (1-m) * class_mean
                    prototypes[c] = momentum * prototypes[c] + (1 - momentum) * class_mean
                    
                    # 重新归一化到单位球面
                    prototypes[c] = F.normalize(prototypes[c].unsqueeze(0), dim=1).squeeze(0)
    
    def _prototype_loss(self, features, labels, prototypes):
        """计算样本-原型对比损失"""
        # 过滤无效标签（-1表示忽略）
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # 归一化并计算相似度
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, prototypes.T) / self.temperature
        
        # 交叉熵损失：让样本向其类别原型靠近
        return F.cross_entropy(logits, labels)
    
    def get_prototype_info(self):
        """获取原型信息（用于分析）"""
        with torch.no_grad():
            # 计算原型间余弦相似度
            sim_fti = torch.matmul(self.prototypes_fti, self.prototypes_fti.T)
            sim_rcl = torch.matmul(self.prototypes_rcl, self.prototypes_rcl.T)
            
            # 计算类间平均距离（去除对角线）
            mask_fti = 1 - torch.eye(self.num_fti_classes, device=self.device)
            mask_rcl = 1 - torch.eye(self.num_rcl_classes, device=self.device)
            
            avg_sim_fti = (sim_fti * mask_fti).sum() / mask_fti.sum()
            avg_sim_rcl = (sim_rcl * mask_rcl).sum() / mask_rcl.sum()
            
            return {
                'fti_inter_similarity': avg_sim_fti.item(),
                'rcl_inter_similarity': avg_sim_rcl.item(),
                'prototypes_fti': self.prototypes_fti.cpu().numpy(),
                'prototypes_rcl': self.prototypes_rcl.cpu().numpy(),
            }


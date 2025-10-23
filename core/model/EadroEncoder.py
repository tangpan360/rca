import torch
from torch import nn
import torch.nn.functional as F


class MetricEncoder(nn.Module):
    """Eadro风格的Metric编码器: 1D CNN处理时序数据"""
    def __init__(self, input_channels=12, seq_len=20, output_dim=128):
        super(MetricEncoder, self).__init__()
        # 输入: [batch, 20, 12] -> [batch, 12, 20]
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # x: [batch, 20, 12]
        x = x.transpose(1, 2)  # [batch, 12, 20]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # [batch, 128]
        x = self.fc(x)  # [batch, output_dim]
        return x


class LogEncoder(nn.Module):
    """Eadro风格的Log编码器: MLP处理template统计"""
    def __init__(self, input_dim=40, output_dim=128):
        super(LogEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # x: [batch, 40]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # [batch, output_dim]
        return x


class TraceEncoder(nn.Module):
    """Eadro风格的Trace编码器: 1D CNN处理duration时序"""
    def __init__(self, seq_len=20, output_dim=128):
        super(TraceEncoder, self).__init__()
        # 输入: [batch, 20, 1]
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # x: [batch, 20, 1]
        x = x.transpose(1, 2)  # [batch, 1, 20]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # [batch, 64]
        x = self.fc(x)  # [batch, output_dim]
        return x


class EadroModalEncoder(nn.Module):
    """
    Eadro风格的多模态编码器
    将原始时序数据编码为固定维度的embedding，然后接入TVDiag的图网络
    """
    def __init__(self, output_dim=128):
        super(EadroModalEncoder, self).__init__()
        self.metric_encoder = MetricEncoder(output_dim=output_dim)
        self.log_encoder = LogEncoder(output_dim=output_dim)
        self.trace_encoder = TraceEncoder(output_dim=output_dim)
        
    def forward(self, metric_data, log_data, trace_data):
        """
        Args:
            metric_data: [batch, 20, 12]
            log_data: [batch, 40]
            trace_data: [batch, 20, 1]
        Returns:
            metric_emb: [batch, output_dim]
            log_emb: [batch, output_dim]
            trace_emb: [batch, output_dim]
        """
        metric_emb = self.metric_encoder(metric_data)
        log_emb = self.log_encoder(log_data)
        trace_emb = self.trace_encoder(trace_data)
        return metric_emb, log_emb, trace_emb

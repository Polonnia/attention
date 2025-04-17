import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

def get_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    生成序列的填充掩码（Padding Mask）
    
    参数：
        seq: 输入序列 [batch_size, seq_len]
        pad_idx: 填充token的索引
    
    返回：
        mask: 掩码矩阵 [batch_size, 1, seq_len]，pad_idx位置为0，其余为1
    """
    return (seq != pad_idx).unsqueeze(1)  # 添加维度用于广播

def get_subsequent_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    生成因果掩码（Causal Mask），防止解码器看到未来信息
    
    参数：
        seq: 输入序列 [batch_size, seq_len]
    
    返回：
        subsequent_mask: 下三角矩阵 [seq_len, seq_len]
    """
    sz_b, seq_len = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def position_encoding_init(n_position: int, d_model: int) -> torch.Tensor:
    """
    生成正弦位置编码表（非学习型）
    
    参数：
        n_position: 最大位置长度
        d_model: 模型维度
    
    返回：
        encodings: 位置编码矩阵 [n_position, d_model]
    """
    position = torch.arange(n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    encodings = torch.zeros(n_position, d_model)
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    return encodings

class ScheduledOptim:
    """
    带warmup的学习率调度器（适配Transformer训练）
    实现线性warmup和逆平方根衰减
    """
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 d_model: int, n_warmup_steps: int, lr_mul: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.lr_mul = lr_mul
        self.n_steps = 0

    def step_and_update_lr(self):
        """执行优化步骤并更新学习率"""
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        """计算学习率缩放因子"""
        d_model = self.d_model
        n_steps, n_warmup = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup ** (-1.5))

    def _update_learning_rate(self):
        """更新学习率"""
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def label_smoothing_kl_loss(pred: torch.Tensor, gold: torch.Tensor, 
                           pad_idx: int, smoothing: float = 0.1) -> torch.Tensor:
    """
    Label Smoothing的KL散度损失计算
    
    参数：
        pred: 模型预测分布 [batch_size * seq_len, n_vocab]
        gold: 真实标签 [batch_size * seq_len]
        pad_idx: 填充token索引
        smoothing: 平滑系数（通常0.1）
    
    返回：
        loss: 标量损失值
    """
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    
    log_prb = F.log_softmax(pred, dim=1)
    non_pad_mask = gold.ne(pad_idx)
    loss = -(one_hot * log_prb).sum(dim=1)
    return loss.masked_select(non_pad_mask).sum()

def count_parameters(model: nn.Module) -> int:
    """统计模型的可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    config: dict
):
    """保存训练检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, path)

def load_checkpoint(path: str, device: torch.device) -> Tuple:
    """加载训练检查点"""
    checkpoint = torch.load(path, map_location=device)
    return (
        checkpoint['model_state_dict'],
        checkpoint['optimizer_state_dict'],
        checkpoint['epoch'],
        checkpoint['config']
    )

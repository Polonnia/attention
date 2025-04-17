import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    支持自注意力和编码器-解码器注意力两种模式
    
    参数：
        n_head: 注意力头数
        d_model: 模型维度
        d_k: 每个头的键/查询维度
        d_v: 每个头的值维度
        dropout: Dropout概率
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 线性投影矩阵
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        # 正则化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        参数：
            q: 查询向量 [batch_size, len_q, d_model]
            k: 键向量 [batch_size, len_k, d_model]
            v: 值向量 [batch_size, len_v, d_model]
            mask: 掩码矩阵 [batch_size, len_q, len_k]
        返回：
            output: 注意力输出 [batch_size, len_q, d_model]
        """
        residual = q
        batch_size, len_q = q.size(0), q.size(1)
        
        # 1. 线性投影并分头
        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k).transpose(1, 2)  # [b, h, len_q, d_k]
        k = self.w_ks(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)     # [b, h, len_k, d_k]
        v = self.w_vs(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)     # [b, h, len_v, d_v]
        
        # 2. 计算缩放点积注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [b, h, len_q, len_k]
        
        # 3. 应用掩码（因果掩码或填充掩码）
        if mask is not None:
            mask = mask.unsqueeze(1)  # 广播到所有注意力头 [b, 1, len_q, len_k]
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # 4. 计算注意力权重
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 5. 加权求和
        output = torch.matmul(attn, v)  # [b, h, len_q, d_v]
        
        # 6. 合并多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        
        # 7. 残差连接和层归一化
        return self.layer_norm(output + residual)


class PositionwiseFeedForward(nn.Module):
    """
    位置逐点前馈网络 (Position-wise FFN)
    结构：Linear → GELU → Dropout → Linear → Residual
    
    参数：
        d_in: 输入维度
        d_hid: 隐藏层维度（通常为4*d_model）
        dropout: Dropout概率
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数：
            x: 输入张量 [batch_size, seq_len, d_model]
        返回：
            output: 变换后的张量 [batch_size, seq_len, d_model]
        """
        residual = x
        output = self.w_2(F.gelu(self.w_1(x)))  # 使用GELU激活函数
        return self.layer_norm(self.dropout(output) + residual)


class PositionalEncoding(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)
    为输入序列注入位置信息
    
    参数：
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout概率
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数：
            x: 输入张量 [batch_size, seq_len, d_model]
        返回：
            output: 加入位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

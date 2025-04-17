class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    支持自注意力和编码器-解码器注意力两种模式
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        # 线性变换矩阵
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        # 输出投影
        self.fc = nn.Linear(n_head * d_v, d_model)
        
        # 正则化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        """
        参数：
            q: 查询向量 [batch_size, len_q, d_model]
            k: 键向量 [batch_size, len_k, d_model]
            v: 值向量 [batch_size, len_v, d_model]
            mask: 掩码矩阵 [batch_size, len_q, len_k]
        """
        residual = q
        batch_size = q.size(0)
        
        # 1. 线性投影并分头
        q = self.w_qs(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # [b, h, len_q, d_k]
        k = self.w_ks(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # [b, h, len_k, d_k]
        v = self.w_vs(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)  # [b, h, len_v, d_v]
        
        # 2. 计算缩放点积注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [b, h, len_q, len_k]
        
        # 3. 应用掩码（如因果掩码或填充掩码）
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # 4. 计算注意力权重
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 5. 加权求和
        output = torch.matmul(attn, v)  # [b, h, len_q, d_v]
        
        # 6. 合并多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        output = self.dropout(self.fc(output))
        
        # 7. 残差连接
        return output + residual

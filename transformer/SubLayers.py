class MultiHeadAttention(nn.Module):
    """适配原训练脚本的注意力实现"""
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        residual = q
        batch_size = q.size(0)
        
        # 分头处理
        q = self.w_qs(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        output = self.dropout(self.fc(output))
        return output + residual

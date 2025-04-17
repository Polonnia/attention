def get_pad_mask(seq, pad_idx):
    """
    生成填充掩码（padding mask）
    返回：掩码矩阵 [batch_size, 1, seq_len]，pad_idx位置为0
    """
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    """
    生成因果掩码（防止解码器看到未来信息）
    返回：下三角矩阵 [seq_len, seq_len]
    """
    sz = seq.size(1)
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, sz, sz), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

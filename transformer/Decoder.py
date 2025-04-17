import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Decoder(nn.Module):
    """
    Transformer解码器完整实现
    结构：N x (掩码自注意力 → 编码器-解码器注意力 → 前馈网络)
    
    参数：
        n_layers: 解码器层数
        d_model: 模型维度
        n_head: 注意力头数
        d_k: 每个注意力头的键/查询维度
        d_v: 每个注意力头的值维度
        d_inner: FFN隐藏层维度
        dropout: Dropout概率
    """
    def __init__(self, n_layers, d_model, n_head, d_k, d_v, d_inner, dropout=0.1):
        super().__init__()
        
        # 堆叠多个DecoderLayer
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_k, d_v, d_inner, dropout)
            for _ in range(n_layers)
        ])
        
        # 最终层归一化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, trg_seq, enc_output, trg_mask=None, src_mask=None):
        """
        前向传播
        
        参数：
            trg_seq: 目标序列 [batch_size, trg_len, d_model]
            enc_output: 编码器输出 [batch_size, src_len, d_model] 
            trg_mask: 目标序列掩码 [batch_size, trg_len, trg_len]
            src_mask: 源序列掩码 [batch_size, 1, src_len]
            
        返回：
            output: 解码结果 [batch_size, trg_len, d_model]
        """
        output = trg_seq
        
        # 逐层处理
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output, 
                enc_output,
                trg_mask=trg_mask,
                src_mask=src_mask
            )
            
        return self.layer_norm(output)


class DecoderLayer(nn.Module):
    """单层解码器实现"""
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, dropout=0.1):
        super().__init__()
        
        # 组件初始化
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, trg_mask=None, src_mask=None):
        """
        单层解码流程：
        1. 掩码自注意力（处理目标序列）
        2. 编码器-解码器注意力（融合编码器信息）
        3. 前馈网络
        每个步骤后接残差连接和层归一化
        """
        # 第一子层：掩码自注意力
        residual = dec_input
        output = self.norm1(dec_input)
        output = self.self_attn(output, output, output, mask=trg_mask)
        output = residual + self.dropout1(output)
        
        # 第二子层：编码器-解码器注意力
        residual = output
        output = self.norm2(output)
        output = self.enc_attn(output, enc_output, enc_output, mask=src_mask)
        output = residual + self.dropout2(output)
        
        # 第三子层：前馈网络
        residual = output
        output = self.norm3(output)
        output = self.pos_ffn(output)
        output = residual + self.dropout3(output)
        
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Decoder(nn.Module):
    """兼容原训练脚本的Decoder实现"""
    def __init__(self, n_layers, d_model, n_head, d_k, d_v, d_inner, dropout=0.1):
        super().__init__()
        
        # 保持与原脚本参数命名一致
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_k, d_v, d_inner, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, trg_seq, enc_output, trg_mask=None, src_mask=None):
        output = trg_seq
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output, 
                enc_output,
                trg_mask=trg_mask,
                src_mask=src_mask
            )
        return self.layer_norm(output)

class DecoderLayer(nn.Module):
    """兼容原训练脚本的单层Decoder"""
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, dropout=0.1):
        super().__init__()
        
        # 自注意力
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 编码器-解码器注意力
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 前馈网络
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, trg_mask=None, src_mask=None):
        # 自注意力
        residual = dec_input
        output = self.norm1(dec_input)
        output = self.self_attn(output, output, output, mask=trg_mask)
        output = residual + self.dropout1(output)
        
        # 编码器-解码器注意力
        residual = output
        output = self.norm2(output)
        output = self.enc_attn(output, enc_output, enc_output, mask=src_mask)
        output = residual + self.dropout2(output)
        
        # 前馈网络
        residual = output
        output = self.norm3(output)
        output = self.pos_ffn(output)
        output = residual + self.dropout3(output)
        
        return output

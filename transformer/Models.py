class Transformer(nn.Module):
    def __init__(self, ...):
        # ...其他初始化...
        self.decoder = Decoder(
            n_layers=opt.n_layers,
            d_model=opt.d_model,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_inner=opt.d_inner_hid,
            dropout=opt.dropout
        )

    def forward(self, src_seq, trg_seq):
        # ...编码器部分...
        
        # 生成解码器掩码
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & \
                   get_subsequent_mask(trg_seq)
                   
        dec_output = self.decoder(
            trg_seq, 
            enc_output,
            trg_mask=trg_mask,
            src_mask=get_pad_mask(src_seq, self.src_pad_idx)
        )
        
        return dec_output

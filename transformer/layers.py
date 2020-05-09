import torch.nn as nn
from transformer.modules import SublayerConnection
from transformer.sublayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        output = self.sublayers[0](x, lambda _x: self.self_attn(_x, _x, _x, mask=mask))
        return self.sublayers[1](output, self.feed_forward)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ffn, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.context_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_output, context_mask, trg_mask):
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, mask=trg_mask))
        x = self.sublayer[1](x, lambda _x: self.context_attn(_x, enc_output, enc_output, context_mask))
        return self.sublayer[2](x, self.feed_forward)
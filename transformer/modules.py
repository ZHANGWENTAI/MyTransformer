import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def scale_dot_product_attention(q, k, v, dropout=None, mask=None):
    """
    Args:
        q: [BATCH_SIZE, n_heads, seq_len, d_q]
        k: [BATCH_SIZE, n_heads, seq_len, d_k]
        v: [BATCH_SIZE, n_heads, seq_len, d_v]
        dropout: a dropout layer
        mask: [BATCH_SIZE, 1, seq_len, seq_len] ByteTensor,
              in Encoder: mask out all the attention relate to <pad>
              in Decoder: mask out all the attention relate to <pad> and the subsequence position
    return:
        output: [BATCH_SIZE, n_heads, seq_len, d_v]
        attention: [BATCH_SIZE, n_heads, seq_len, seq_len]
    """
    d_k = k.size(-1)
    attention = torch.matmul(q, k.transpose(2, 3))  # torch.matmul support broadcast mechanism
    attention *= (d_k ** -0.5)
    if mask is not None:
        attention = attention.masked_fill_(mask == 0, -np.inf)

    attention = F.softmax(attention, dim=-1)

    if dropout is not None:
        attention = dropout(attention)

    output = torch.matmul(attention, v)
    return output, attention


class SublayerConnection(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer):
        return self.norm(x + self.dropout(layer(x)))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_seq_len=100):
        super().__init__()

        position_encoding = np.array([
            [pos / (10000 ** (2.0 * (i // 2) / d_model)) for i in range(d_model)]
            for pos in range(max_seq_len)
        ])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # self.position_encoding: of shape [1, max_seq_len, d_model]
        self.position_encoding = torch.tensor(position_encoding, device='cuda').float().unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: [BATCH_SIZE, seq_len, d_model]

        return:
            output: [BATCH_SIZE, seq_len, d_model]
        """
        x = x + torch.tensor(self.position_encoding[:, :x.size(1)], requires_grad=False)
        output = self.dropout(x)
        return output


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * (self.d_model ** 0.5)


class Projector(nn.Module):
    """generator probability distribution from word vector."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
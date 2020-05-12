import torch.nn as nn
import torch.nn.functional as F
from transformer.modules import scale_dot_product_attention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: 输入的维度
            n_heads: 注意力头的个数
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-9)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, seq_len, d_model]
            k: [batch_size, seq_len, d_model]
            v: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] ByteTensor

         Return:
            output: [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)

        # self.w_q(q) 这里会把 q 转化成 [batch_size * seq_len, d_model] 再相乘
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            # use mask in all heads
            mask = mask.unsqueeze(1)

        output = scale_dot_product_attention(q, k, v, dropout=self.dropout, mask=mask)

        # concat
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.n_heads)
        output = self.dropout(self.w_o(output))

        return output



class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x: [batch_size, seq_len, d_model]
        :return output: [batch_size, seq_len, d_model]
        '''

        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)

        return output
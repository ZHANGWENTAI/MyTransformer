import torch.nn as nn
from transformer.layers import EncoderLayer, DecoderLayer
from transformer.modules import Projector, Embeddings, PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, d_ffn, n_heads,
                 max_seq_len, src_vocab_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_emb = Embeddings(d_model, src_vocab_size)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)
        self.dropout_emb = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_mask):
        """
        :param enc_inputs: [batch_size, seq_len]
        :param enc_mask: [batch_size, 1, seq_len]
        :return:
        """
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_enc(enc_outputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_mask)

        return enc_outputs


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, n_heads,
                 max_seq_len, trg_vocab_size, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.trg_emb = Embeddings(d_model, trg_vocab_size)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs, context_mask, trg_mask):
        dec_outputs = self.trg_emb(dec_inputs)
        dec_outputs = self.pos_enc(dec_outputs)
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, context_mask, trg_mask)

        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.encoder = Encoder(opt["n_layers"], opt["d_model"], opt["d_ffn"], opt["n_heads"],
                               opt["max_src_seq_len"], opt["src_vocab_size"], opt["dropout"])
        self.decoder = Decoder(opt["n_layers"], opt['d_model'], opt['d_ffn'], opt['n_heads'],
                               opt['max_trg_seq_len'], opt['trg_vocab_size'], opt['dropout'])
        self.trg_proj = Projector(opt['d_model'], opt['trg_vocab_size'])



    def forward(self, src, trg, src_mask, trg_mask):
        return self.project(
            self.decode(self.encode(src, src_mask), src_mask, trg, trg_mask)
        )

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, enc_output, src_mask, trg, trg_mask):
        return self.decoder(trg, enc_output, src_mask, trg_mask)

    def project(self, dec_output):
        return self.trg_proj(dec_output)
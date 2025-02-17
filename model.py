# model.py
import math
import torch
import torch.nn as nn
import numpy as np

from config import (
    D_MODEL, D_FF, D_K, D_V,
    N_LAYERS, N_HEADS, TGT_LEN,
    USE_CUDA
)

# --------------------
# Positional Encoding
# --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #  register_buffer确保模型在eval和cuda时可以使用这个pe，但不会被作为可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 这里将填充为0的部分mask掉
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# --------------------
# Multi-Head Attention
# --------------------
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.use_cuda = USE_CUDA
        device = torch.device("cuda" if self.use_cuda else "cpu")

        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS, bias=False)
        self.fc = nn.Linear(N_HEADS * D_V, D_MODEL, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)

        # Q/K/V 线性变换后，拆分成多头
        Q = self.W_Q(input_Q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)

        # attn_mask 也要扩充到 [batch_size, N_HEADS, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, N_HEADS, 1, 1)

        # 计算注意力
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 还原多头
        context = context.transpose(1, 2).reshape(batch_size, -1, N_HEADS * D_V)
        output = self.fc(context)
        # 残差连接 + LayerNorm
        return nn.LayerNorm(D_MODEL).to(input_Q.device)(output + residual), attn


# --------------------
# FeedForward
# --------------------
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.use_cuda = USE_CUDA
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, bias=False),
            nn.ReLU(),
            nn.Linear(D_FF, D_MODEL, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(D_MODEL).to(inputs.device)(output + residual)


# --------------------
# Encoder
# --------------------
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        # 位置编码要求 [seq_len, batch_size, d_model]，所以需要transpose
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# --------------------
# Decoder
# --------------------
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.tgt_len = TGT_LEN

    def forward(self, dec_inputs):
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1)
        # 这里的mask先用全 False，即不额外mask
        dec_self_attn_pad_mask = torch.zeros(
            (dec_inputs.shape[0], self.tgt_len, self.tgt_len),
            dtype=torch.bool,
            device=dec_inputs.device
        )
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
        return dec_outputs, dec_self_attns


# --------------------
# Transformer
# --------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super(Transformer, self).__init__()
        self.pep_encoder = Encoder(vocab_size)
        self.hla_encoder = Encoder(vocab_size)
        self.decoder = Decoder()

        self.projection = nn.Sequential(
            nn.Linear(TGT_LEN * D_MODEL, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )

    def forward(self, pep_inputs, hla_inputs):
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)
        # 拼接
        enc_outputs = torch.cat((pep_enc_outputs, hla_enc_outputs), dim=1)

        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        # flatten
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), pep_enc_self_attns, hla_enc_self_attns, dec_self_attns

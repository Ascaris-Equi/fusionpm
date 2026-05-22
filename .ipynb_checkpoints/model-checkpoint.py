# model.py
# TransPHLA backbone + Cross-Attention + Masked Language Modeling auxiliary head.
import math
import torch
import torch.nn as nn
import numpy as np

# ---- hyperparameters (TransPHLA-aligned) ----
PEP_MAX_LEN = 15
HLA_MAX_LEN = 34
TGT_LEN = PEP_MAX_LEN + HLA_MAX_LEN          # 49
D_MODEL = 64
D_FF = 512
D_K = 64
D_V = 64
N_LAYERS = 1
N_HEADS = 9


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        return self.dropout(x + self.pe[:x.size(0), :])


def get_attn_pad_mask(seq_q, seq_k):
    # seq_q [B, Lq], seq_k [B, Lk]; mask pad (=0) in seq_k.
    b, lq = seq_q.size()
    _, lk = seq_k.size()
    return seq_k.data.eq(0).unsqueeze(1).expand(b, lq, lk)


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask):
        s = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(D_K)
        s = s.masked_fill(mask, -1e9)
        a = nn.Softmax(dim=-1)(s)
        return torch.matmul(a, V), a


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS, bias=False)
        self.fc = nn.Linear(N_HEADS * D_V, D_MODEL, bias=False)
        self.ln = nn.LayerNorm(D_MODEL)

    def forward(self, q_in, k_in, v_in, mask):
        residual, b = q_in, q_in.size(0)
        Q = self.W_Q(q_in).view(b, -1, N_HEADS, D_K).transpose(1, 2)
        K = self.W_K(k_in).view(b, -1, N_HEADS, D_K).transpose(1, 2)
        V = self.W_V(v_in).view(b, -1, N_HEADS, D_V).transpose(1, 2)
        mask = mask.unsqueeze(1).repeat(1, N_HEADS, 1, 1)
        ctx, a = ScaledDotProductAttention()(Q, K, V, mask)
        ctx = ctx.transpose(1, 2).reshape(b, -1, N_HEADS * D_V)
        return self.ln(self.fc(ctx) + residual), a


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, bias=False),
            nn.ReLU(),
            nn.Linear(D_FF, D_MODEL, bias=False),
        )
        self.ln = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        return self.ln(self.fc(x) + x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, x, mask):
        x, a = self.enc_self_attn(x, x, x, mask)
        return self.pos_ffn(x), a


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])

    def forward(self, x):
        out = self.src_emb(x)
        out = self.pos_emb(out.transpose(0, 1)).transpose(0, 1)
        mask = get_attn_pad_mask(x, x)
        atts = []
        for L in self.layers:
            out, a = L(out, mask)
            atts.append(a)
        return out, atts


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention: pep <-> hla."""
    def __init__(self):
        super().__init__()
        self.pep_to_hla = MultiHeadAttention()
        self.hla_to_pep = MultiHeadAttention()
        self.pep_ffn = PoswiseFeedForwardNet()
        self.hla_ffn = PoswiseFeedForwardNet()

    def forward(self, pep_enc, hla_enc, pep_in, hla_in):
        m_p2h = get_attn_pad_mask(pep_in, hla_in)
        m_h2p = get_attn_pad_mask(hla_in, pep_in)
        pep_c, a_p2h = self.pep_to_hla(pep_enc, hla_enc, hla_enc, m_p2h)
        hla_c, a_h2p = self.hla_to_pep(hla_enc, pep_enc, pep_enc, m_h2p)
        return self.pep_ffn(pep_c), self.hla_ffn(hla_c), a_p2h, a_h2p


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, x, mask):
        x, a = self.dec_self_attn(x, x, x, mask)
        return self.pos_ffn(x), a


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.tgt_len = TGT_LEN

    def forward(self, x):
        out = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        mask = torch.zeros((x.shape[0], self.tgt_len, self.tgt_len),
                           dtype=torch.bool, device=x.device)
        atts = []
        for L in self.layers:
            out, a = L(out, mask)
            atts.append(a)
        return out, atts


class FusionPM(nn.Module):
    """
    TransPHLA-style two-encoder fusion + Cross-Attention bridge + Decoder + MLM head.
    forward(pep, hla)              -> logits [B, 2]
    forward(pep, hla, mlm=True)    -> (logits, mlm_logits [B, PEP_MAX_LEN, V])
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.pep_encoder = Encoder(vocab_size)
        self.hla_encoder = Encoder(vocab_size)
        self.cross = CrossAttentionBlock()
        self.decoder = Decoder()
        self.projection = nn.Sequential(
            nn.Linear(TGT_LEN * D_MODEL, 256), nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64), nn.ReLU(True),
            nn.Linear(64, 2),
        )
        # MLM head (only used in training)
        self.mlm_head = nn.Linear(D_MODEL, vocab_size)

    def encode(self, pep, hla):
        pep_e, _ = self.pep_encoder(pep)
        hla_e, _ = self.hla_encoder(hla)
        pep_c, hla_c, _, _ = self.cross(pep_e, hla_e, pep, hla)
        # residual fuse
        pep_f = pep_e + pep_c
        hla_f = hla_e + hla_c
        return pep_f, hla_f

    def forward(self, pep, hla, mlm=False):
        pep_f, hla_f = self.encode(pep, hla)
        x = torch.cat((pep_f, hla_f), dim=1)         # [B, TGT_LEN, D]
        d, _ = self.decoder(x)
        d_flat = d.reshape(d.shape[0], -1)
        logits = self.projection(d_flat)
        if mlm:
            mlm_logits = self.mlm_head(pep_f)        # predict pep AA from fused pep repr
            return logits, mlm_logits
        return logits


def apply_mlm_mask(pep_inputs, mask_rate=0.15, pad_idx=0):
    """In-place style mask for peptide tokens.
       Returns (masked_inputs, mask_bool) where mask_bool=True marks positions to predict.
    """
    valid = pep_inputs.ne(pad_idx)
    rand = torch.rand(pep_inputs.shape, device=pep_inputs.device)
    mask = (rand < mask_rate) & valid
    masked = pep_inputs.clone()
    masked[mask] = pad_idx
    return masked, mask
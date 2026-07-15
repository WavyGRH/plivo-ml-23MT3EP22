"""A small GPT in plain PyTorch.

Anything public on Config is saved into the checkpoint by train.py and restored
by evaluate.py, so architecture switches ride along automatically and a
checkpoint always rebuilds the model it was trained as.

Architecture options (all default to the baseline's behaviour):
  norm = "layer" | "rms"       RMSNorm drops LayerNorm's mean-subtraction+bias
  pos  = "learned" | "rope"    RoPE removes the pos_emb table entirely
  mlp  = "gelu" | "swiglu"     SwiGLU is param-matched at hidden = 8/3 * n_embd
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256      # byte-level tokenizer default
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = False
    init_std = 0.05       # baseline: one std for every tensor
    resid_scale = False   # scale residual projections by 1/sqrt(2*n_layer)
    norm = "layer"
    pos = "learned"
    mlp = "gelu"


class RMSNorm(nn.Module):
    """LayerNorm without the mean-subtraction or the bias: n params, not 2n."""
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


def make_norm(cfg):
    return RMSNorm(cfg.n_embd) if cfg.norm == "rms" else nn.LayerNorm(cfg.n_embd)


def rope_tables(block, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(block).float(), inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_dim). Rotates each (even, odd) channel pair by an
    angle proportional to absolute position, which makes attention scores depend
    on relative distance — no position parameters at all."""
    T = x.size(2)
    cos = cos[:T].to(x.dtype)[None, None, :, :]
    sin = sin[:T].to(x.dtype)[None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.use_rope = cfg.pos == "rope"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        if self.use_rope:
            cos, sin = rope_tables(cfg.block_size, cfg.n_embd // cfg.n_head)
            # buffers, not parameters: not trained, not counted, not saved
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    """Gated MLP. SwiGLU needs three matrices where GELU needs two, so the
    expansion factor must drop from 4x to 8/3 to keep the parameter count equal:
    3 * d * (8d/3) == 8d^2. Otherwise a SwiGLU "win" is just extra capacity.

    Rounded DOWN to a multiple of 8 (426.67 -> 424), which leaves SwiGLU 5,120
    params BELOW the GELU baseline rather than above it. Rounding up would have
    handed it +11,136 (+0.58%) and quietly contaminated the comparison; erring
    low means any measured gain is attributable to the gating, not the budget."""
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden -= hidden % 8               # round DOWN to a multiple of 8
        self.gate = nn.Linear(cfg.n_embd, hidden)
        self.up = nn.Linear(cfg.n_embd, hidden)
        self.down = nn.Linear(hidden, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        if cfg.mlp == "swiglu":
            self.mlp = SwiGLU(cfg)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_rope = cfg.pos == "rope"
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        if getattr(cfg, "resid_scale", False):
            # GPT-2 style: shrink what each block adds back to the residual
            # stream, so the stream's variance does not grow with depth.
            scale = 1.0 / math.sqrt(2 * cfg.n_layer)
            for blk in self.blocks:
                nn.init.normal_(blk.attn.proj.weight, 0.0, cfg.init_std * scale)
                last = blk.mlp.down if cfg.mlp == "swiglu" else blk.mlp[2]
                nn.init.normal_(last.weight, 0.0, cfg.init_std * scale)

    def _init(self, m):
        std = getattr(self.cfg, "init_std", 0.05)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if not self.use_rope:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

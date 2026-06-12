"""MLB model: v2's transformer internals with baseball state channels.

The attention Block and KVCache are imported straight from v2/sim/model.py —
one transformer, two sports — so the architectures cannot drift. Only the
additive state channels differ: baseball's are (score diff, inning, half,
outs, balls, strikes, bases), computed by Replay from the token prefix only,
so they can never leak the token being predicted.
"""

import importlib.util
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_v2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "v2", "sim", "model.py")
_spec = importlib.util.spec_from_file_location("v2_sim_model", _v2_path)
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)

Block = _v2.Block
KVCache = _v2.KVCache
PAD = _v2.PAD
pick_device = _v2.pick_device

MAX_DIFF = 15    # score diff clipped to +/-15 runs -> 31 buckets
MAX_INNING = 12  # everything later shares the "deep extras" bucket

# Channel vocab sizes, in Replay.channels tuple order:
# (diff, inning, half, outs, balls, strikes, bases)
CHANNEL_SIZES = (2 * MAX_DIFF + 1, MAX_INNING + 1, 2, 4, 4, 3, 8)
N_CHANNELS = len(CHANNEL_SIZES)


def bucketize_channels(channels):
    """Replay.channels [(diff, inning, half, outs, balls, strikes, bases),
    ...] -> per-channel index tensors, clipped into CHANNEL_SIZES."""
    cols = list(zip(*channels))
    out = []
    for j, (col, size) in enumerate(zip(cols, CHANNEL_SIZES)):
        if j == 0:  # score diff is the only signed channel
            col = [max(-MAX_DIFF, min(MAX_DIFF, v)) + MAX_DIFF for v in col]
        else:
            col = [min(v, size - 1) for v in col]
        out.append(torch.tensor(col))
    return tuple(out)


@dataclass
class Config:
    vocab_size: int
    d_model: int = 256
    n_head: int = 8
    n_layer: int = 6
    max_len: int = 3712  # longest 2024 game is 3,664 tokens
    dropout: float = 0.1


class MLBEventGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.chan_emb = nn.ModuleList(
            nn.Embedding(n, cfg.d_model) for n in CHANNEL_SIZES)

        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        for emb in (self.tok_emb, self.pos_emb, *self.chan_emb):
            nn.init.normal_(emb.weight, std=0.02)

    def embed(self, ids, channels, pos):
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for emb, c in zip(self.chan_emb, channels):
            x = x + emb(c)
        return x

    def forward(self, ids, channels, targets=None):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device)
        x = self.embed(ids, channels, pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                                   targets.reshape(-1), ignore_index=PAD)
        return logits, loss

    def prime(self, ids, channels, cache: KVCache):
        """Run a (right-padded) prefix once, filling the cache from 0."""
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embed(ids, channels, pos)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer=i)
        cache.t += ids.shape[1]
        return self.head(self.ln_f(x))

    def step(self, ids, channels, pos, cache: KVCache, key_valid):
        """One generation step: ids is (B, 1). Returns next-token logits."""
        x = self.embed(ids, channels, pos)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer=i, key_valid=key_valid)
        cache.t += 1
        return self.head(self.ln_f(x))[:, -1]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

"""A small GPT over play-by-play tokens, with additive game-state channels.

The model predicts the next token given the game so far. Three state channels
(score difference, period, clock) are added to the token embeddings — the same
mechanism as positional embeddings — so the model never has to do long-range
arithmetic to know the score. Channels are computed by Replay from the prefix
only, so they can never leak the token being predicted.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0  # vocab index of [PAD]

# State-channel bucket sizes (see bucketize_channels)
MAX_DIFF = 30   # score diff clipped to +/-30 -> 61 buckets
MAX_PERIOD = 8  # regulation + up to 4 OTs
CLOCK_BUCKETS = 13  # remaining seconds // 60 -> 0..12 minutes


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


@dataclass
class Config:
    vocab_size: int
    d_model: int = 256
    n_head: int = 8
    n_layer: int = 6
    max_len: int = 3328  # longest game across six seasons is 3,124 tokens (4OT)
    dropout: float = 0.1
    tied_head: bool = True  # untied (#10): output head decoupled from tok_emb
    lineup_channel: bool = False  # (#10): on-floor five per side, added like
    #                               the other channels (away/home projections)
    n_seasons: int = 0  # (#7) > 0 enables season conditioning: a global
    #                     season channel plus per-(token, season) vintage
    #                     deltas e_tok + delta, zero-init and weight-decayed
    #                     so a player's vector only forks where seasons
    #                     genuinely differ


def bucketize_channels(channels):
    """Replay channels [(score_diff, period, clock_s, possession), ...] -> index tensors."""
    diff = torch.tensor([max(-MAX_DIFF, min(MAX_DIFF, d)) + MAX_DIFF
                         for d, _, _, _ in channels])
    period = torch.tensor([min(p, MAX_PERIOD) for _, p, _, _ in channels])
    clock = torch.tensor([min(max(c, 0) // 60, CLOCK_BUCKETS - 1)
                          for _, _, c, _ in channels])
    poss = torch.tensor([p for _, _, _, p in channels])  # 0 unknown / 1 away / 2 home
    return diff, period, clock, poss


class KVCache:
    """Preallocated per-layer KV buffers for generation.

    Concatenating a fresh (k, v) every step allocates a new, slightly larger
    tensor each time — the caching allocator can never reuse a block, reserved
    memory balloons past dedicated VRAM, and Windows silently spills to shared
    memory (10x slowdown at batch 32, OOM at 128). One fixed buffer written in
    place keeps memory flat: ~1.3GB at batch 64.
    """

    def __init__(self, cfg: Config, batch: int, device, dtype):
        head_dim = cfg.d_model // cfg.n_head
        shape = (cfg.n_layer, batch, cfg.n_head, cfg.max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.t = 0  # filled positions


class Block(nn.Module):
    """Pre-norm transformer block on F.scaled_dot_product_attention.

    Flash attention never materializes the L x L matrix (PyTorch's built-in
    TransformerEncoder does once you hand it explicit masks, which costs
    gigabytes at 2,300-token games). No padding mask is needed: sequences are
    right-padded and attention is causal, so real tokens can never attend to
    [PAD], and [PAD] targets are excluded from the loss.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None, layer=0, key_valid=None):
        """cache/layer: KVCache buffers to read+write during generation.
        key_valid: (B, keys) bool marking real (non-padding) cache entries."""
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        if cache is not None:
            t = cache.t
            cache.k[layer][:, :, t:t + L] = k
            cache.v[layer][:, :, t:t + L] = v
            k = cache.k[layer][:, :, :t + L]
            v = cache.v[layer][:, :, :t + L]
        if key_valid is not None:  # generation step: q is the newest token only
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=key_valid[:, None, None, :])
        else:  # full sequence: training, or priming (cache.t == 0)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attn.transpose(1, 2).reshape(B, L, D)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class EventGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.diff_emb = nn.Embedding(2 * MAX_DIFF + 1, cfg.d_model)
        self.period_emb = nn.Embedding(MAX_PERIOD + 1, cfg.d_model)
        self.clock_emb = nn.Embedding(CLOCK_BUCKETS, cfg.d_model)
        self.poss_emb = nn.Embedding(3, cfg.d_model)  # unknown / away ball / home ball

        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tied_head:
            self.head.weight = self.tok_emb.weight  # weight tying
        if cfg.lineup_channel:
            # Zero-init so the channel starts as a no-op and training decides
            # how much on-floor context to inject.
            self.floor_away = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.floor_home = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            nn.init.zeros_(self.floor_away.weight)
            nn.init.zeros_(self.floor_home.weight)
        if cfg.n_seasons:
            self.season_emb = nn.Embedding(cfg.n_seasons, cfg.d_model)
            self.delta_emb = nn.Embedding(cfg.n_seasons * cfg.vocab_size,
                                          cfg.d_model)
            nn.init.normal_(self.season_emb.weight, std=0.02)
            nn.init.zeros_(self.delta_emb.weight)  # vintages start identical

        # GPT-style init: PyTorch's default embedding std of 1.0 produces huge
        # initial logits through the tied head.
        for emb in (self.tok_emb, self.pos_emb, self.diff_emb,
                    self.period_emb, self.clock_emb, self.poss_emb):
            nn.init.normal_(emb.weight, std=0.02)
        if not cfg.tied_head:
            nn.init.normal_(self.head.weight, std=0.02)

    def _tok_vec(self, ids, season):
        """Token embedding plus the per-(token, season) vintage delta.
        `season` must broadcast against ids; PAD positions get no delta so
        padding stays the zero vector."""
        e = self.tok_emb(ids)
        if self.cfg.n_seasons:
            d = self.delta_emb(season * self.cfg.vocab_size + ids)
            e = e + d * (ids != PAD).unsqueeze(-1)
        return e

    def _floor_vec(self, lineup, season):
        """lineup: (..., 10) vocab ids, away five then home five, PAD where the
        floor is unknown. PAD embeds to the zero vector (padding_idx), so the
        fixed /5 keeps absent players from shifting the mean."""
        e = self._tok_vec(lineup, season[..., None, None]
                          if season is not None else None)
        return (self.floor_away(e[..., :5, :].sum(-2) / 5)
                + self.floor_home(e[..., 5:, :].sum(-2) / 5))

    def forward(self, ids, diff, period, clock, poss, targets=None, lineup=None,
                season=None):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device)
        x = (self._tok_vec(ids, season[:, None] if season is not None else None)
             + self.pos_emb(pos)
             + self.diff_emb(diff) + self.period_emb(period)
             + self.clock_emb(clock) + self.poss_emb(poss))
        if self.cfg.n_seasons:
            x = x + self.season_emb(season)[:, None]
        if self.cfg.lineup_channel:
            x = x + self._floor_vec(lineup, season)

        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                                   targets.reshape(-1), ignore_index=PAD)
        return logits, loss

    def embed(self, ids, diff, period, clock, poss, pos, lineup=None,
              season=None):
        x = (self._tok_vec(ids, season[:, None] if season is not None else None)
             + self.pos_emb(pos)
             + self.diff_emb(diff) + self.period_emb(period)
             + self.clock_emb(clock) + self.poss_emb(poss))
        if self.cfg.n_seasons:
            x = x + self.season_emb(season)[:, None]
        if self.cfg.lineup_channel:
            x = x + self._floor_vec(lineup, season)
        return x

    def prime(self, ids, diff, period, clock, poss, cache: KVCache, lineup=None,
              season=None):
        """Run a (right-padded) prefix once, filling the cache from position 0.
        Padding keys enter the cache but are masked off by key_valid in step()."""
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embed(ids, diff, period, clock, poss, pos, lineup, season)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer=i)
        cache.t += ids.shape[1]
        return self.head(self.ln_f(x))

    def step(self, ids, diff, period, clock, poss, pos, cache: KVCache, key_valid,
             lineup=None, season=None):
        """One generation step: ids is (B, 1). Returns next-token logits."""
        x = self.embed(ids, diff, period, clock, poss, pos, lineup, season)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer=i, key_valid=key_valid)
        cache.t += 1
        return self.head(self.ln_f(x))[:, -1]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

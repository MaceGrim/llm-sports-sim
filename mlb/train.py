#!/usr/bin/env python3
"""Train MLBEventGPT on tokenized pitch-by-pitch.

  python train.py --smoke          # 60-step pipeline check (tiny model)
  python train.py                  # full training run

Split by date: train < --val-cutoff, validate after. Per-slot validation
losses are reported separately — the pitch slot (T:) is arsenal modeling,
the result slot (R:) is the batter's eye, E: is outcomes, B: is baserunner
advancement — because an aggregate perplexity hides which baseball skill
is failing. Mirrors v2/train.py (same masking and batching logic).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.model import (Config, MLBEventGPT, N_CHANNELS, PAD,
                       bucketize_channels, pick_device)
from sim.tokenizer import Replay, load_vocab

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dataset(val_cutoff: str):
    """Tokenized games -> (train, val) lists of
    (ids, *channels, given, blocked_players)."""
    vocab = load_vocab(os.path.join(HERE, "cache", "vocab.json"))
    index = {tok: i for i, tok in enumerate(vocab)}
    is_player = torch.tensor([t.startswith("P:") for t in vocab])

    train, val = [], []
    with open(os.path.join(HERE, "cache", "tokens.jsonl")) as f:
        for line in f:
            g = json.loads(line)
            tokens = g["tokens"]
            ids = torch.tensor([index[t] for t in tokens])
            channels = bucketize_channels(Replay(tokens).run().channels)
            # The header (matchup, lineups, pens) is user INPUT at simulation
            # time, never predicted — same reasoning as v2: grading roster
            # recitation turns player embeddings into team-membership tables.
            given = tokens.index("[HALF]") + 1
            # Legality mask: player-slot softmax restricted to this game's
            # header players (v2's fix for team-clustered embeddings).
            roster = [index[t] for t in tokens[:given] if t.startswith("P:")]
            blocked = is_player.clone()
            blocked[roster] = False
            example = (ids, *channels, given, blocked)
            (train if g["date"] < val_cutoff else val).append(example)
    return vocab, train, val


def mask_illegal_players(logits, targets, blocked, is_player):
    """Wherever the target is a player, block players not in this game."""
    player_slot = is_player[targets]  # (B, L)
    kill = player_slot[:, :, None] & blocked[:, None, :]  # (B, L, vocab)
    return logits.masked_fill(kill, float("-inf"))


def make_batch(examples, device):
    """Pad games to a common length; targets are inputs shifted by one, with
    the given header prefix excluded from the loss."""
    L = max(len(e[0]) for e in examples)
    n_seq = 1 + N_CHANNELS  # ids + channels
    batch = [torch.full((len(examples), L), PAD) for _ in range(n_seq)]
    for i, example in enumerate(examples):
        for j, seq in enumerate(example[:n_seq]):
            batch[j][i, :len(seq)] = seq
    batch = [t.to(device) for t in batch]
    ids, channels = batch[0], tuple(batch[1:])
    targets = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], PAD)], dim=1)
    for i, example in enumerate(examples):
        targets[i, :example[n_seq] - 1] = PAD  # conditioning, not prediction
    blocked = torch.stack([e[n_seq + 1] for e in examples]).to(device)
    return ids, channels, targets, blocked


@torch.no_grad()
def evaluate(model, val, vocab, device, batch_size, max_games=24):
    """Validation loss overall and per slot (pitch/result/event/bases)."""
    slots = {"pitch": "T:", "result": "R:", "event": "E:", "bases": "B:"}
    slot_masks = {name: torch.tensor([t.startswith(p) for t in vocab],
                                     device=device)
                  for name, p in slots.items()}
    is_player = torch.tensor([t.startswith("P:") for t in vocab], device=device)

    model.eval()
    sums = {name: [0.0, 0] for name in ["all", *slots]}
    for start in range(0, min(len(val), max_games), batch_size):
        ids, channels, targets, blocked = make_batch(
            val[start:start + batch_size], device)
        logits, _ = model(ids, channels)
        logits = mask_illegal_players(logits, targets, blocked, is_player)
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, len(vocab)), targets.reshape(-1),
            ignore_index=PAD, reduction="none").reshape(targets.shape)
        valid = targets != PAD
        for name in sums:
            mask = valid if name == "all" else slot_masks[name][targets] & valid
            sums[name][0] += losses[mask].sum().item()
            sums[name][1] += int(mask.sum())
    model.train()
    return {k: s / max(n, 1) for k, (s, n) in sums.items()}


def main():
    p = argparse.ArgumentParser(description="Train MLBEventGPT")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-cutoff", default="2024-08-15")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="override pick_device()")
    p.add_argument("--smoke", action="store_true", help="tiny model, 60 steps")
    args = p.parse_args()

    if args.smoke:
        args.steps, args.batch = 60, 4
        args.d_model, args.n_layer, args.n_head = 64, 2, 4
        args.val_every = 30

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device or pick_device()

    vocab, train, val = load_dataset(args.val_cutoff)
    is_player = torch.tensor([t.startswith("P:") for t in vocab], device=device)
    print(f"device={device}  train={len(train)} games  val={len(val)} games  "
          f"vocab={len(vocab)}", flush=True)

    cfg = Config(vocab_size=len(vocab), d_model=args.d_model,
                 n_layer=args.n_layer, n_head=args.n_head)
    model = MLBEventGPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    autocast = torch.autocast(device, dtype=torch.bfloat16,
                              enabled=device == "cuda")

    t0 = time.time()
    best_val = float("inf")
    out = os.path.join(HERE, "cache", "model.pt")
    for step in range(1, args.steps + 1):
        picks = rng.choice(len(train), size=args.batch, replace=False)
        ids, channels, targets, blocked = make_batch(
            [train[i] for i in picks], device)
        with autocast:
            logits, _ = model(ids, channels)
            logits = mask_illegal_players(logits, targets, blocked, is_player)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, len(vocab)), targets.reshape(-1),
                ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"step {step:>5}  train {loss.item():.3f}  "
                  f"({step / (time.time() - t0):.2f} steps/s)", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            with autocast:
                v = evaluate(model, val, vocab, device, args.batch)
            improved = v["all"] < best_val
            if improved:  # checkpoint the best-validation model, not the last
                best_val = v["all"]
                torch.save({"config": cfg.__dict__, "vocab": vocab, "step": step,
                            "state_dict": model.state_dict()}, out)
            print(f"step {step:>5}  VAL {v['all']:.3f}  pitch {v['pitch']:.3f}  "
                  f"result {v['result']:.3f}  event {v['event']:.3f}  "
                  f"bases {v['bases']:.3f}"
                  + ("  *saved*" if improved else ""), flush=True)

    print(f"best val {best_val:.3f} -> {out}")


if __name__ == "__main__":
    main()

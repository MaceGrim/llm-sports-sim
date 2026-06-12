#!/usr/bin/env python3
"""Train EventGPT on tokenized play-by-play.

  python train.py --smoke          # 60-step pipeline check (tiny model)
  python train.py                  # full training run

Split: regular-season games only (playoffs are distributionally different and
series violate independence); train < --val-cutoff date, validate after.
Per-slot validation losses are reported separately — the actor slot is usage
modeling, made/miss is shot-outcome calibration — because an aggregate
perplexity hides which basketball skill is failing.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.model import Config, EventGPT, bucketize_channels, PAD, pick_device
from sim.tokenizer import Replay, load_vocab

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dataset(val_cutoff: str):
    """Tokenized games -> (train, val) lists of
    (ids, diff, period, clock, given, blocked_players)."""
    vocab = load_vocab(os.path.join(HERE, "cache", "vocab.json"))
    index = {tok: i for i, tok in enumerate(vocab)}
    is_player = torch.tensor([t.startswith("P:") for t in vocab])

    train, val = [], []
    with open(os.path.join(HERE, "cache", "tokens.jsonl")) as f:
        for line in f:
            g = json.loads(line)
            if not g["game_id"].startswith("002"):  # regular season only
                continue
            tokens = g["tokens"]
            ids = torch.tensor([index[t] for t in tokens])
            channels = bucketize_channels(Replay(tokens).run().channels)
            # Rosters and Q1 starters are user INPUTS at simulation time, never
            # predicted. Grading the model on reciting them turns the player
            # embeddings into team-membership tables (probed: teammate@10=0.94).
            given = tokens.index("[START_Q]") + 11
            # Legality mask: player-slot softmax is restricted to this game's
            # rosters. Without it, every prediction pushes the ~520 absent
            # players away, and that suppression direction IS team identity —
            # the embeddings cluster by team instead of playstyle.
            first_dt = next(i for i, t in enumerate(tokens) if t.startswith("dt:"))
            roster = [index[t] for t in tokens[:first_dt] if t.startswith("P:")]
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
    the given prefix (header + Q1 starters) excluded from the loss."""
    L = max(len(e[0]) for e in examples)
    batch = [torch.full((len(examples), L), PAD) for _ in range(5)]
    for i, example in enumerate(examples):
        for j, seq in enumerate(example[:5]):
            batch[j][i, :len(seq)] = seq
    ids, diff, period, clock, poss = (t.to(device) for t in batch)
    targets = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], PAD)], dim=1)
    for i, example in enumerate(examples):
        targets[i, :example[5] - 1] = PAD  # conditioning, not prediction
    blocked = torch.stack([e[6] for e in examples]).to(device)
    return ids, diff, period, clock, poss, targets, blocked


@torch.no_grad()
def evaluate(model, val, vocab, device, batch_size, max_games=24):
    """Validation loss overall and per slot (actor / outcome / clock)."""
    is_player = torch.tensor([t.startswith("P:") for t in vocab], device=device)
    is_outcome = torch.tensor([t in ("made", "miss") for t in vocab], device=device)
    is_dt = torch.tensor([t.startswith("dt:") for t in vocab], device=device)

    model.eval()
    sums = {"all": [0.0, 0], "actor": [0.0, 0], "outcome": [0.0, 0], "dt": [0.0, 0]}
    for start in range(0, min(len(val), max_games), batch_size):
        ids, diff, period, clock, poss, targets, blocked = make_batch(
            val[start:start + batch_size], device)
        logits, _ = model(ids, diff, period, clock, poss)
        logits = mask_illegal_players(logits, targets, blocked, is_player)
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, len(vocab)), targets.reshape(-1),
            ignore_index=PAD, reduction="none").reshape(targets.shape)
        for name, mask in [("all", targets != PAD), ("actor", is_player[targets]),
                           ("outcome", is_outcome[targets]), ("dt", is_dt[targets])]:
            mask = mask & (targets != PAD)
            sums[name][0] += losses[mask].sum().item()
            sums[name][1] += int(mask.sum())
    model.train()
    return {k: s / max(n, 1) for k, (s, n) in sums.items()}


def main():
    p = argparse.ArgumentParser(description="Train EventGPT")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-cutoff", default="2023-03-01")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny model, 60 steps")
    args = p.parse_args()

    if args.smoke:
        args.steps, args.batch = 60, 4
        args.d_model, args.n_layer, args.n_head = 64, 2, 4
        args.val_every = 30

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()

    vocab, train, val = load_dataset(args.val_cutoff)
    is_player = torch.tensor([t.startswith("P:") for t in vocab], device=device)
    print(f"device={device}  train={len(train)} games  val={len(val)} games  "
          f"vocab={len(vocab)}")

    cfg = Config(vocab_size=len(vocab), d_model=args.d_model,
                 n_layer=args.n_layer, n_head=args.n_head)
    model = EventGPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # bf16 on GPU: halves memory (keeps us inside dedicated VRAM — Windows
    # silently spills to slow shared memory otherwise) and uses tensor cores.
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")

    t0 = time.time()
    best_val = float("inf")
    out = os.path.join(HERE, "cache", "model.pt")
    for step in range(1, args.steps + 1):
        picks = rng.choice(len(train), size=args.batch, replace=False)
        ids, diff, period, clock, poss, targets, blocked = make_batch(
            [train[i] for i in picks], device)
        with autocast:
            logits, _ = model(ids, diff, period, clock, poss)
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
                  f"({step / (time.time() - t0):.2f} steps/s)")
        if step % args.val_every == 0 or step == args.steps:
            with autocast:
                v = evaluate(model, val, vocab, device, args.batch)
            improved = v["all"] < best_val
            if improved:  # checkpoint the best-validation model, not the last
                best_val = v["all"]
                torch.save({"config": cfg.__dict__, "vocab": vocab, "step": step,
                            "state_dict": model.state_dict()}, out)
            print(f"step {step:>5}  VAL {v['all']:.3f}  "
                  f"actor {v['actor']:.3f}  outcome {v['outcome']:.3f}  "
                  f"dt {v['dt']:.3f}" + ("  *saved*" if improved else ""))

    print(f"best val {best_val:.3f} -> {out}")


if __name__ == "__main__":
    main()

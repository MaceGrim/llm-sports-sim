#!/usr/bin/env python3
"""Gate B verdict: transformer vs both baselines, paired bootstrap.

Gate B (Definition of GOOD): transformer Brier better than BOTH the
player-form baseline and the team-form baseline on the same 343 games,
with a paired bootstrap; margin MAE within +0.5 of the best baseline.
Vegas is printed as the reference ceiling, never a gate.

Run from v2/:  python test_scripts/gate_b.py
               [--transformer results/backtest_transformer_2025-03-01_s50.json]
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def per_game(rows):
    out = {}
    for r in rows:
        p = np.clip(r["home_win_prob"], 1e-6, 1 - 1e-6)
        w = float(r["home_won"])
        out[(r["date"], r["matchup"])] = {
            "brier": (p - w) ** 2,
            "abs_err": abs(r["margin_err"]),
            "pick": (p >= 0.5) == r["home_won"],
        }
    return out


def paired_boot(a, b, key, n=10_000, seed=0):
    """Bootstrap CI for mean(a[key] - b[key]) over common games."""
    diffs = np.array([a[k][key] - b[k][key] for k in a if k in b])
    rng = np.random.default_rng(seed)
    bs = rng.choice(diffs, size=(n, len(diffs))).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5)), len(diffs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transformer",
                   default=os.path.join(RESULTS,
                                        "backtest_transformer_2025-03-01_s50.json"))
    args = p.parse_args()

    def load(path, key):
        d = json.load(open(path))
        return per_game(d[key]), d["summary"]

    tf, tf_sum = load(args.transformer, "rows")
    refs = {
        "player-form": load(os.path.join(RESULTS,
                                         "backtest_player_2025-03-01.json"), "rows"),
        "team-form": load(os.path.join(RESULTS,
                                       "backtest_team_paired343.json"), "rows"),
        "vegas": load(os.path.join(RESULTS,
                                   "backtest_vegas_2025-03-01.json"), "rows"),
    }

    print(f"transformer: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in tf_sum.items() if k != 'temps'})}")
    print(f"  temps: {tf_sum.get('temps')}\n")

    verdicts = []
    for name, (ref, ref_sum) in refs.items():
        gate = name != "vegas"
        db, blo, bhi, n = paired_boot(tf, ref, "brier")
        dm, mlo, mhi, _ = paired_boot(tf, ref, "abs_err")
        beats_brier = db < 0 and bhi < 0
        print(f"vs {name} (n={n}):")
        print(f"  Brier diff  {db:+.4f}  CI [{blo:+.4f}, {bhi:+.4f}]"
              f"  -> {'transformer better (CI<0)' if beats_brier else 'NOT significantly better'}")
        print(f"  |margin| diff {dm:+.2f}  CI [{mlo:+.2f}, {mhi:+.2f}]")
        if gate:
            verdicts.append((name, beats_brier, dm))
        print()

    best_mae = min(s["margin_mae"] for name, (_, s) in refs.items()
                   if "margin_mae" in s and name != "vegas")
    mae_ok = tf_sum["margin_mae"] <= best_mae + 0.5
    brier_ok = all(v[1] for v in verdicts)
    print(f"GATE B: brier beats both baselines: {brier_ok}; "
          f"margin MAE {tf_sum['margin_mae']:.2f} <= best baseline "
          f"{best_mae:.2f} + 0.5: {mae_ok}")
    print(f"VERDICT: {'PASS' if brier_ok and mae_ok else 'FAIL'}")


if __name__ == "__main__":
    main()

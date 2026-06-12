#!/usr/bin/env python3
"""Vegas closing-line reference row (Definition of GOOD external anchor).

Closing spreads from the Kaggle dataset 'NBA Betting Data October 2007 to
June 2025' (kagglehub: cviaxmiwnptr/...), which carries spreads for every
2024-25 game but no moneylines after 2023. Win probability therefore comes
from a logistic fit of P(home win) on the closing home spread over the
2008-2024 seasons — strictly before the val window, no leakage. Margin MAE
uses the spread directly. This is the market's number: a ceiling reference,
never a gate.

Run from v2/:  python test_scripts/backtest_vegas.py [--val-cutoff 2025-03-01]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

HERE = os.path.dirname(os.path.abspath(__file__))
ODDS_CSV = os.path.expanduser(
    "~/.cache/kagglehub/datasets/cviaxmiwnptr/"
    "nba-betting-data-october-2007-to-june-2024/versions/2/nba_2008-2025.csv")

ABBREV = {"gs": "GSW", "no": "NOP", "ny": "NYK", "sa": "SAS",
          "utah": "UTA", "wsh": "WAS"}


def load_odds():
    df = pd.read_csv(ODDS_CSV)
    df = df[df.spread.notna()].copy()
    df["home_line"] = np.where(df.whos_favored == "home", df.spread, -df.spread)
    df["home_won"] = df.score_home > df.score_away
    for c in ("away", "home"):
        df[c] = df[c].map(lambda a: ABBREV.get(a, a.upper()))
    return df


def fit_logistic(x, y, iters=200, lr=0.1):
    """P(y=1) = sigmoid(a + b*x) by Newton-free gradient descent (n ~ 20k)."""
    a, b = 0.0, 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        a += lr * (y - p).mean()
        b += lr * ((y - p) * x).mean()
    return a, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cutoff", default="2025-03-01")
    args = p.parse_args()

    df = load_odds()
    fit = df[df.season <= 2024]
    a, b = fit_logistic(fit.home_line, fit.home_won)
    print(f"logistic fit on {len(fit)} games (2008-2024): "
          f"P(home) = sigmoid({a:.4f} + {b:.4f} * line)")

    # The val game list comes from the player baseline so all reference rows
    # share the exact same 343 games.
    val = json.load(open(os.path.join(
        HERE, "..", "results", f"backtest_player_{args.val_cutoff}.json")))["rows"]
    odds = {(r.date, r.away, r.home): r for r in
            df[df.date >= args.val_cutoff].itertuples()}

    rows, missing = [], []
    for r in val:
        away, home = r["matchup"].split("@")
        o = odds.get((r["date"], away, home))
        if o is None:
            missing.append((r["date"], r["matchup"]))
            continue
        prob = float(1 / (1 + np.exp(-(a + b * o.home_line))))
        rows.append({
            "date": r["date"], "game_id": r["game_id"], "matchup": r["matchup"],
            "home_line": float(o.home_line), "home_win_prob": prob,
            "home_won": r["home_won"], "actual_margin": r["actual_margin"],
            "margin_err": float(o.home_line - r["actual_margin"]),
        })
    if missing:
        print(f"WARNING: {len(missing)} val games missing odds: {missing[:5]}")

    probs = np.array([r["home_win_prob"] for r in rows])
    wins = np.array([r["home_won"] for r in rows])
    picks = float(((probs >= 0.5) == wins).mean())
    brier = float(((probs - wins) ** 2).mean())
    clamped = np.clip(probs, 1e-6, 1 - 1e-6)
    log_loss = float(-(wins * np.log(clamped)
                       + (1 - wins) * np.log(1 - clamped)).mean())
    margin_mae = float(np.mean([abs(r["margin_err"]) for r in rows]))

    print(f"\n=== VEGAS CLOSING LINES ({len(rows)} games) ===")
    print(f"  Pick accuracy:  {picks:.1%}")
    print(f"  Brier score:    {brier:.4f}")
    print(f"  Log loss:       {log_loss:.4f}")
    print(f"  Margin MAE:     {margin_mae:.1f} pts")

    out = os.path.join(HERE, "..", "results",
                       f"backtest_vegas_{args.val_cutoff}.json")
    with open(out, "w") as f:
        json.dump({"summary": {
            "n_games": len(rows), "missing": len(missing),
            "source": "kaggle cviaxmiwnptr closing spreads",
            "prob_model": {"a": a, "b": b, "fit_seasons": "2008-2024"},
            "picks": picks, "brier": brier, "log_loss": log_loss,
            "margin_mae": margin_mae,
        }, "rows": rows}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

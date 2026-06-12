#!/usr/bin/env python3
"""Player-form baseline (#15b, Gate B of the Definition of GOOD).

Sees the SAME header information as the transformer backtest — the night's
actual actives per team from tokens.jsonl — and predicts the score by
projecting each active's mpg-weighted season-to-date scoring, scaled to a
240-minute team total and adjusted for opponent defense. The three-way gap
(team-form baseline vs this vs transformer) decomposes the transformer's
edge into roster-information value vs learned-simulation value.

Deterministic model: margin ~ Normal(pred_margin, sigma), where sigma is the
residual sd fit on the DEV window (Jan-Feb 2025) only — the val set is never
used for fitting. Win prob and p10-p90 coverage follow from that normal,
matching protocol v2's smoothed estimator.

Run from v2/:  python test_scripts/backtest_player_baseline.py
               [--val-cutoff 2025-03-01] [--min-prior-games 3]
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sim.form import season_start
from sim.games import load_games

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DEV_START, DEV_END = "2025-01-01", "2025-03-01"
TEAM_MINUTES = 48 * 5  # regulation; OT scoring is part of the residual


def season_tables(games, before_date):
    """Per-player and per-team season-to-date tables from games strictly
    before before_date (any team — trades keep a player's full history)."""
    players = defaultdict(lambda: {"games": 0, "pts": 0, "min": 0.0})
    team_scored = defaultdict(list)
    team_allowed = defaultdict(list)
    home_margins = []
    start = season_start(before_date)
    for g in games:
        if not (start <= g.date < before_date):
            continue
        team_scored[g.home].append(g.home_score)
        team_scored[g.away].append(g.away_score)
        team_allowed[g.home].append(g.away_score)
        team_allowed[g.away].append(g.home_score)
        home_margins.append(g.home_score - g.away_score)
        for name, line in g.players.items():
            t = players[name]
            t["games"] += 1
            t["pts"] += line.pts
            t["min"] += line.minutes
    return players, team_scored, team_allowed, home_margins


def project_team(actives, players, min_prior_games):
    """Sum of mpg-weighted season-to-date scoring, scaled to 240 minutes."""
    pts, mins, used = 0.0, 0.0, 0
    for name in actives:
        t = players.get(name)
        if not t or t["games"] < min_prior_games or t["min"] <= 0:
            continue
        mpg = t["min"] / t["games"]
        pts += (t["pts"] / t["min"]) * mpg
        mins += mpg
        used += 1
    if mins <= 0:
        return None, 0
    return pts * (TEAM_MINUTES / mins), used


def predict(g, games, min_prior_games):
    """Predicted home margin for one tokenized game, or None if not coverable."""
    toks = g["tokens"]
    away, home = toks[1][5:], toks[2][5:]
    ra, rh = toks.index("[ROSTER_A]"), toks.index("[ROSTER_H]")
    end = toks.index("dt:0")
    actives_away = [t[2:] for t in toks[ra + 1:rh]]
    actives_home = [t[2:] for t in toks[rh + 1:end]]

    players, scored, allowed, home_margins = season_tables(games, g["date"])
    if len(scored.get(home, ())) < 15 or len(scored.get(away, ())) < 15:
        return None  # same eligibility rule as sim/evaluate.py

    proj_home, n_home = project_team(actives_home, players, min_prior_games)
    proj_away, n_away = project_team(actives_away, players, min_prior_games)
    if proj_home is None or proj_away is None:
        return None

    league_ppg = np.mean([s for v in scored.values() for s in v])
    def_factor = lambda opp: np.mean(allowed[opp]) / league_ppg
    proj_home *= def_factor(away)
    proj_away *= def_factor(home)
    home_edge = float(np.mean(home_margins))

    return {
        "matchup": f"{away}@{home}",
        "pred_margin": proj_home - proj_away + home_edge,
        "players_used": n_home + n_away,
        "actives": len(actives_home) + len(actives_away),
    }


def run_window(tokens_games, games, min_prior_games):
    rows = []
    for g in tokens_games:
        pred = predict(g, games, min_prior_games)
        if pred is None:
            continue
        pred.update(date=g["date"], game_id=g["game_id"])
        rows.append(pred)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-cutoff", default="2025-03-01")
    p.add_argument("--min-prior-games", type=int, default=3)
    args = p.parse_args()

    games = load_games(os.path.join(CACHE, "games.jsonl"))
    truth = {g.game_id: g for g in games}
    with open(os.path.join(CACHE, "tokens.jsonl")) as f:
        toks = [json.loads(l) for l in f if l.strip()]
    reg = [g for g in toks if g["game_id"].startswith("002")]
    dev = [g for g in reg if DEV_START <= g["date"] < DEV_END]
    val = [g for g in reg if g["date"] >= args.val_cutoff]
    print(f"dev games={len(dev)}  val games={len(val)}")

    # Fit sigma on dev only.
    dev_rows = run_window(dev, games, args.min_prior_games)
    dev_resid = [truth[r["game_id"]].home_score - truth[r["game_id"]].away_score
                 - r["pred_margin"] for r in dev_rows]
    sigma = float(np.std(dev_resid, ddof=1))
    bias = float(np.mean(dev_resid))
    print(f"dev n={len(dev_rows)}  residual sd={sigma:.2f}  bias={bias:+.2f}")

    val_rows = run_window(val, games, args.min_prior_games)
    skipped = len(val) - len(val_rows)
    z90 = 1.281552  # 90th percentile of the standard normal
    for r in val_rows:
        t = truth[r["game_id"]]
        r["actual_margin"] = t.home_score - t.away_score
        r["home_won"] = r["actual_margin"] > 0
        r["home_win_prob"] = 0.5 * (1 + math.erf(
            r["pred_margin"] / (sigma * math.sqrt(2))))
        r["margin_err"] = r["pred_margin"] - r["actual_margin"]
        r["margin_in_p10_p90"] = bool(
            abs(r["actual_margin"] - r["pred_margin"]) <= z90 * sigma)

    probs = np.array([r["home_win_prob"] for r in val_rows])
    wins = np.array([r["home_won"] for r in val_rows])
    picks = float(((probs >= 0.5) == wins).mean())
    brier = float(((probs - wins) ** 2).mean())
    clamped = np.clip(probs, 1e-6, 1 - 1e-6)
    log_loss = float(-(wins * np.log(clamped)
                       + (1 - wins) * np.log(1 - clamped)).mean())
    margin_mae = float(np.mean([abs(r["margin_err"]) for r in val_rows]))
    coverage = float(np.mean([r["margin_in_p10_p90"] for r in val_rows]))
    low_info = sum(1 for r in val_rows if r["players_used"] < r["actives"] - 2)

    print(f"\n=== PLAYER-FORM BASELINE ({len(val_rows)} games"
          f", {skipped} skipped) ===")
    print(f"  Pick accuracy:  {picks:.1%}")
    print(f"  Brier score:    {brier:.4f}")
    print(f"  Log loss:       {log_loss:.4f}")
    print(f"  Margin MAE:     {margin_mae:.1f} pts")
    print(f"  Margin coverage (p10-p90): {coverage:.1%}")
    print(f"  games missing >2 actives' form: {low_info}")

    out = os.path.join(HERE, "..", "results",
                       f"backtest_player_{args.val_cutoff}.json")
    with open(out, "w") as f:
        json.dump({"summary": {
            "n_games": len(val_rows), "skipped": skipped,
            "protocol": "v2-smoothed", "sigma_dev": sigma, "dev_bias": bias,
            "picks": picks, "brier": brier, "log_loss": log_loss,
            "margin_mae": margin_mae, "coverage_p10_p90": coverage,
        }, "rows": val_rows}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

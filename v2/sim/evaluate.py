"""Backtest a simulator against held-out real games.

Philosophy: a simulator is stochastic, so we evaluate it as a probability
model, not a point predictor. For each held-out game we run K simulations and
score the implied win probability (Brier, log loss, calibration) and the
dispersion of outcomes (does the actual margin land inside the simulated
distribution?). Trivial baselines are scored alongside for reference.
"""

import math
from typing import Dict, List

import numpy as np

from .form import team_form
from .games import Game
from .simulate import Simulator


def backtest(sim: Simulator, games: List[Game], start_date: str,
             min_prior_games: int = 15, n_sims: int = 200, seed: int = 0,
             limit: int = 0) -> dict:
    """Evaluate sim on every game on/after start_date where both teams have history."""
    rng = np.random.default_rng(seed)
    rows = []

    eligible = [g for g in games if g.date >= start_date]
    for g in eligible:
        hf = team_form(games, g.home, g.date)
        af = team_form(games, g.away, g.date)
        if hf is None or af is None or hf.games < min_prior_games or af.games < min_prior_games:
            continue

        sims = [sim.simulate(g.away, g.home, g.date, rng) for _ in range(n_sims)]
        home_win_prob = sum(1 for s in sims if s.winner == g.home) / n_sims
        margins = np.array([s.home_score - s.away_score for s in sims])
        totals = np.array([s.home_score + s.away_score for s in sims])

        actual_margin = g.home_score - g.away_score
        actual_total = g.home_score + g.away_score
        home_won = actual_margin > 0

        rows.append({
            "date": g.date,
            "matchup": f"{g.away}@{g.home}",
            "home_win_prob": home_win_prob,
            "home_won": home_won,
            "home_better_record": hf.wins / hf.games >= af.wins / af.games,
            "margin_err": float(margins.mean() - actual_margin),
            "total_err": float(totals.mean() - actual_total),
            "margin_in_p10_p90": bool(
                np.percentile(margins, 10) <= actual_margin <= np.percentile(margins, 90)
            ),
        })
        if limit and len(rows) >= limit:
            break

    return {"summary": _summarize(rows), "games": rows}


def _clamp(p: float, eps: float = 1e-3) -> float:
    return min(max(p, eps), 1 - eps)


def _summarize(rows: List[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_games": 0}

    probs = np.array([r["home_win_prob"] for r in rows])
    outcomes = np.array([r["home_won"] for r in rows], dtype=float)

    picks_correct = ((probs >= 0.5) == outcomes.astype(bool)).mean()
    brier = float(((probs - outcomes) ** 2).mean())
    log_loss = float(-np.mean(
        outcomes * np.log([_clamp(p) for p in probs])
        + (1 - outcomes) * np.log([1 - _clamp(p) for p in probs])
    ))

    home_rate = float(outcomes.mean())
    record_pick_correct = float(np.mean(
        [r["home_better_record"] == r["home_won"] for r in rows]
    ))

    # Calibration: bucket predicted probs, compare to empirical frequency.
    calibration = []
    for lo in np.arange(0.0, 1.0, 0.2):
        mask = (probs >= lo) & (probs < lo + 0.2)
        if mask.sum() > 0:
            calibration.append({
                "bucket": f"{lo:.1f}-{lo + 0.2:.1f}",
                "n": int(mask.sum()),
                "predicted": round(float(probs[mask].mean()), 3),
                "actual": round(float(outcomes[mask].mean()), 3),
            })

    return {
        "n_games": n,
        "pick_accuracy": round(float(picks_correct), 3),
        "brier": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "margin_mae": round(float(np.mean([abs(r["margin_err"]) for r in rows])), 2),
        "total_mae": round(float(np.mean([abs(r["total_err"]) for r in rows])), 2),
        "margin_coverage_p10_p90": round(
            float(np.mean([r["margin_in_p10_p90"] for r in rows])), 3
        ),  # well-calibrated dispersion ≈ 0.80
        "calibration": calibration,
        "baselines": {
            "home_always": {
                "pick_accuracy": round(home_rate, 3),
                "brier": round(float(((0.58 - outcomes) ** 2).mean()), 4),
            },
            "better_record_wins": {"pick_accuracy": round(record_pick_correct, 3)},
        },
    }

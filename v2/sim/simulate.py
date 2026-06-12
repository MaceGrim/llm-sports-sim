"""Simulators: give me two teams, I give you a game.

The Simulator interface is the contract every future engine (statistical,
event-sequence transformer, finetuned LLM) implements, so they all plug into
the same evaluation harness.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np

from .form import TeamForm, league_scoring, team_form
from .games import Game

HOME_EDGE = 3.0  # NBA home advantage in points, split across both teams


@dataclass
class SimPlayerLine:
    pts: int
    reb: int
    ast: int


@dataclass
class SimGame:
    away: str
    home: str
    away_score: int
    home_score: int
    periods: List[List[int]]  # per period [away, home]
    players: Dict[str, SimPlayerLine] = field(default_factory=dict)

    @property
    def winner(self) -> str:
        return self.home if self.home_score > self.away_score else self.away


class Simulator(Protocol):
    def simulate(self, away: str, home: str, as_of: str, rng: np.random.Generator) -> SimGame:
        """Simulate one game between away @ home using only data before as_of."""
        ...


class StatisticalSimulator:
    """Monte Carlo baseline: team strength from season-to-date scoring, normal noise.

    Any learned model must beat this to justify its existence.
    """

    def __init__(self, games: List[Game]):
        self.games = games
        self._form_cache: Dict[tuple, Optional[TeamForm]] = {}
        self._scoring_cache: Dict[str, dict] = {}

    def _form(self, team: str, as_of: str) -> TeamForm:
        key = (team, as_of)
        if key not in self._form_cache:
            self._form_cache[key] = team_form(self.games, team, as_of)
        form = self._form_cache[key]
        if form is None:
            raise ValueError(f"No games for {team} before {as_of}")
        return form

    def expected_scores(self, away: str, home: str, as_of: str) -> tuple:
        af, hf = self._form(away, as_of), self._form(home, as_of)
        mu_away = (af.ppg + hf.opp_ppg) / 2 - HOME_EDGE / 2
        mu_home = (hf.ppg + af.opp_ppg) / 2 + HOME_EDGE / 2
        return mu_away, mu_home

    def simulate(self, away: str, home: str, as_of: str, rng: np.random.Generator) -> SimGame:
        mu_away, mu_home = self.expected_scores(away, home, as_of)
        if as_of not in self._scoring_cache:
            self._scoring_cache[as_of] = league_scoring(self.games, as_of)
        std = self._scoring_cache[as_of]["std"]

        away_score = home_score = 0
        while away_score == home_score:
            away_score = max(70, round(rng.normal(mu_away, std)))
            home_score = max(70, round(rng.normal(mu_home, std)))

        periods = self._split_quarters(away_score, home_score, rng)
        players = self._player_lines(away, home, as_of, away_score, home_score, rng)
        return SimGame(away=away, home=home, away_score=away_score,
                       home_score=home_score, periods=periods, players=players)

    @staticmethod
    def _split_quarters(away_score: int, home_score: int, rng: np.random.Generator) -> List[List[int]]:
        quarters = []
        for total in (away_score, home_score):
            shares = rng.dirichlet([30, 30, 30, 30])  # mild quarter-to-quarter variance
            q = [int(round(total * s)) for s in shares]
            q[3] += total - sum(q)  # rounding remainder into Q4
            quarters.append(q)
        return [[quarters[0][i], quarters[1][i]] for i in range(4)]

    def _player_lines(self, away: str, home: str, as_of: str,
                      away_score: int, home_score: int,
                      rng: np.random.Generator) -> Dict[str, SimPlayerLine]:
        lines: Dict[str, SimPlayerLine] = {}
        for team, score in ((away, away_score), (home, home_score)):
            form = self._form(team, as_of)
            # Points: split the simulated team total by noisy season scoring shares.
            weights = np.array([max(p.ppg, 0.5) for p in form.players])
            noisy = weights * rng.gamma(shape=6.0, scale=1 / 6.0, size=len(weights))
            shares = noisy / noisy.sum()
            pts = np.floor(shares * score).astype(int)
            pts[np.argmax(shares)] += score - pts.sum()  # remainder to top scorer
            for p, player_pts in zip(form.players, pts):
                lines[p.name] = SimPlayerLine(
                    pts=int(player_pts),
                    reb=int(rng.poisson(max(p.rpg, 0.1))),
                    ast=int(rng.poisson(max(p.apg, 0.1))),
                )
        return lines


def simulate_matchup(sim: Simulator, away: str, home: str, as_of: str,
                     n_sims: int = 200, seed: int = 0) -> dict:
    """Run n_sims simulations and summarize: win probability, score distributions."""
    rng = np.random.default_rng(seed)
    results = [sim.simulate(away, home, as_of, rng) for _ in range(n_sims)]

    home_wins = sum(1 for r in results if r.winner == home)
    away_scores = np.array([r.away_score for r in results])
    home_scores = np.array([r.home_score for r in results])
    margins = home_scores - away_scores  # positive = home wins by that much

    return {
        "away": away,
        "home": home,
        "as_of": as_of,
        "n_sims": n_sims,
        "home_win_prob": home_wins / n_sims,
        "mean_score": {"away": round(away_scores.mean(), 1), "home": round(home_scores.mean(), 1)},
        "margin_p10_p50_p90": [int(np.percentile(margins, q)) for q in (10, 50, 90)],
        "sample_game": results[0],
    }

"""Team and player form computed strictly from games BEFORE a cutoff date (no leakage)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .games import Game


@dataclass
class PlayerForm:
    name: str
    games: int = 0
    ppg: float = 0.0
    rpg: float = 0.0
    apg: float = 0.0
    mpg: float = 0.0


@dataclass
class TeamForm:
    team: str
    games: int
    wins: int
    losses: int
    ppg: float  # points scored per game
    opp_ppg: float  # points allowed per game
    last5: str  # e.g. "WWLWL", most recent last
    players: List[PlayerForm]  # sorted by minutes per game, descending


def team_form(games: List[Game], team: str, before_date: str, last_n_players: int = 9) -> Optional[TeamForm]:
    """Compute a team's form from all its games strictly before before_date.

    Returns None if the team has no prior games. `games` must be date-sorted.
    """
    scored, allowed, results = [], [], []
    player_totals: Dict[str, dict] = {}

    for g in games:
        if g.date >= before_date:
            break
        if team == g.home:
            us, them, side = g.home_score, g.away_score, "home"
        elif team == g.away:
            us, them, side = g.away_score, g.home_score, "away"
        else:
            continue
        scored.append(us)
        allowed.append(them)
        results.append("W" if us > them else "L")
        for name, line in g.players.items():
            if line.side != side:
                continue
            t = player_totals.setdefault(name, {"games": 0, "pts": 0, "reb": 0, "ast": 0, "min": 0.0})
            t["games"] += 1
            t["pts"] += line.pts
            t["reb"] += line.reb
            t["ast"] += line.ast
            t["min"] += line.minutes

    n = len(scored)
    if n == 0:
        return None

    players = [
        PlayerForm(
            name=name,
            games=t["games"],
            ppg=round(t["pts"] / t["games"], 1),
            rpg=round(t["reb"] / t["games"], 1),
            apg=round(t["ast"] / t["games"], 1),
            mpg=round(t["min"] / t["games"], 1),
        )
        for name, t in player_totals.items()
    ]
    # Rank by total minutes so one big game from a fringe player doesn't
    # outrank a full-season rotation player.
    players.sort(key=lambda p: p.mpg * p.games, reverse=True)

    return TeamForm(
        team=team,
        games=n,
        wins=results.count("W"),
        losses=results.count("L"),
        ppg=round(sum(scored) / n, 1),
        opp_ppg=round(sum(allowed) / n, 1),
        last5="".join(results[-5:]),
        players=players[:last_n_players],
    )


def league_scoring(games: List[Game], before_date: str) -> dict:
    """League-wide per-team scoring mean and std before a date (for simulation noise)."""
    totals = []
    for g in games:
        if g.date >= before_date:
            break
        totals.extend([g.home_score, g.away_score])
    if not totals:
        return {"mean": 114.0, "std": 12.0}  # 2022-23 league-wide priors
    n = len(totals)
    mean = sum(totals) / n
    var = sum((x - mean) ** 2 for x in totals) / n
    return {"mean": round(mean, 1), "std": round(var ** 0.5, 1)}

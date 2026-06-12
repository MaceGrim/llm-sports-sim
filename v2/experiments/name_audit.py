"""Cross-season player-name audit (TODO #4): find one player spelled two ways.

The "AJ Green lesson": the same player appearing under two spellings splits his
token (and his embedding) in half. Before merging six seasons, scan every
lineup in every game and flag candidate duplicates:

- Tier 1 (mechanical variants): names identical after stripping punctuation,
  spacing, case, and Jr/Sr/II-style suffixes — e.g. "A.J. Green" vs "AJ Green",
  "Marcus Morris" vs "Marcus Morris Sr.". Auto-proposed as NAME_FIXES entries
  when the two spellings never share a game (two real people would co-occur or
  at least overlap rosters).
- Tier 2 (manual review): same last name, same first initial, never co-occur,
  different letters — catches "Nicolas"/"Nic" and "Kenyon Jr."/"KJ" style
  renames that no mechanical rule can prove are the same person. No roster
  overlap required: renames often straddle the missing 2023-24 season.

Usage: python experiments/name_audit.py            # prints both tiers
"""

import os
import re
import sys
import unicodedata
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.games import parse_filename  # noqa: E402
from sim.tokenizer import LINEUP_COLS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "nba_data")

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(name: str) -> str:
    """'A.J. Green' -> 'aj green'; 'Dennis Schröder' -> 'dennis schroder'."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z ]", "", ascii_name.lower()).split()
    while words and words[-1] in SUFFIXES:
        words.pop()
    return " ".join(words)


def season_of(date: str) -> str:
    year, month = int(date[:4]), int(date[5:7])
    start = year if month >= 9 else year - 1
    return f"{start}-{str(start + 1)[2:]}"


def scan(data_dir=DATA_DIR):
    """name -> {(season, team)}, games count, and pairwise same-game co-occurrence."""
    where = defaultdict(set)
    games = defaultdict(int)
    cooccur = defaultdict(set)  # name -> names ever seen in the same game
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    for k, f in enumerate(files):
        date, _, away, home = parse_filename(f)
        season = season_of(date)
        df = pd.read_csv(os.path.join(data_dir, f), usecols=LINEUP_COLS,
                         dtype=str, keep_default_na=False)
        names = set()
        for col in LINEUP_COLS:
            team = away if col.startswith("a") else home
            for n in df[col].unique():
                n = n.strip()
                if n:
                    where[n].add((season, team))
                    names.add(n)
        for n in names:
            games[n] += 1
            cooccur[n] |= names
        if (k + 1) % 500 == 0:
            print(f"  scanned {k + 1}/{len(files)}", file=sys.stderr)
    return where, games, cooccur


def main():
    where, games, cooccur = scan()
    by_key = defaultdict(list)
    for n in where:
        by_key[norm(n)].append(n)

    print("== Tier 1: mechanical variants (proposed NAME_FIXES) ==")
    for key, names in sorted(by_key.items()):
        if len(names) < 2:
            continue
        clash = any(b in cooccur[a] for a in names for b in names if b != a)
        names = sorted(names, key=lambda n: -games[n])
        detail = "; ".join(f"{n!r} {games[n]}g {sorted(where[n])}" for n in names)
        if clash:
            print(f"  CO-OCCUR (two real people?): {detail}")
        else:
            for alias in names[1:]:
                print(f'  "{alias}": "{names[0]}",   # {detail}')

    print("\n== Tier 2: same last name + first initial, never co-occur (review) ==")
    by_last = defaultdict(list)
    for n in where:
        k = norm(n)
        if k:
            by_last[k.split()[-1]].append(n)
    for last, names in sorted(by_last.items()):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if norm(a) == norm(b) or b in cooccur[a]:
                    continue
                if norm(a)[0] == norm(b)[0]:
                    print(f"  {a!r} {games[a]}g {sorted(where[a])}  vs  "
                          f"{b!r} {games[b]}g {sorted(where[b])}")


if __name__ == "__main__":
    main()

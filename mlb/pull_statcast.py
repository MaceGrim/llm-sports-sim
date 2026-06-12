#!/usr/bin/env python3
"""Pull Statcast regular seasons into per-year parquets
(statcast_YYYY.parquet, gitignored).

Windows run opening day (international openers included) through the
regular season's last day; spring/postseason rows inside a window are
dropped at load time by game_type == "R" (sim/data.py). pybaseball caches
day-chunks, so an interrupted pull resumes cheaply.

  python pull_statcast.py              # every missing year
  python pull_statcast.py --year 2021  # one year
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

SEASONS = {
    2020: ("2020-07-23", "2020-09-27"),  # COVID 60-game season
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-30"),
    2025: ("2025-03-18", "2025-09-28"),  # Tokyo opener
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, choices=sorted(SEASONS))
    args = p.parse_args()

    from pybaseball import cache, statcast
    cache.enable()

    years = [args.year] if args.year else sorted(SEASONS)
    for year in years:
        out = os.path.join(HERE, f"statcast_{year}.parquet")
        if os.path.exists(out):
            print(f"{year}: already at {out}, skipping")
            continue
        start, end = SEASONS[year]
        print(f"{year}: pulling {start}..{end}", flush=True)
        df = statcast(start_dt=start, end_dt=end, verbose=False)
        df.to_parquet(out)
        n_reg = (df.game_type == "R").sum()
        print(f"{year}: {len(df):,} rows ({n_reg:,} regular-season) -> {out}",
              flush=True)


if __name__ == "__main__":
    main()

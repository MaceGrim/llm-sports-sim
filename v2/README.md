# v2 — NBA Game Simulator

From-scratch rebuild. Core call: give it two teams, get a simulated game.
See `DESIGN.md` for evaluation philosophy and the modeling roadmap.

## Usage

```bash
cd v2
python run.py build-cache              # one-time: parse ../nba_data into cache/games.jsonl (~30s)
python run.py tokenize                 # encode all games, verify exact round-trip, write vocab
python run.py simulate PHI BOS         # simulate PHI @ BOS with full-season form
python run.py simulate PHI BOS --date 2022-12-25   # using only games before a date
python run.py evaluate                 # backtest on games from 2023-01-15 onward

python train.py --smoke                # 60-step pipeline check (needs torch)
python train.py                        # train EventGPT (auto-detects cuda/mps/cpu)
```

## Layout

```
v2/
├── sim/
│   ├── games.py      # CSV play-by-play → Game ground truth (real minutes, box lines); JSONL cache
│   ├── form.py       # team/player season-to-date form, strictly pre-date (no leakage)
│   ├── simulate.py   # Simulator interface + StatisticalSimulator (Monte Carlo baseline)
│   ├── evaluate.py   # probabilistic backtest: Brier, calibration, coverage, baselines
│   ├── tokenizer.py  # token grammar + Replay state machine (exact round-trip, all 1,320 games)
│   └── model.py      # EventGPT: small transformer + additive state channels
├── train.py          # training loop with per-slot validation (actor/outcome/clock)
├── experiments/
│   ├── embedding_lab.py    # team-leakage vs playstyle test for player embeddings
│   └── token_analysis.py   # full-corpus measurements behind TOKENIZER.md
├── run.py            # CLI: build-cache | simulate | evaluate
├── tests/            # pytest suite verified against a known real game
└── DESIGN.md         # evaluation criteria, modeling assessment, lab findings, next steps
```

## Tests

```bash
cd v2 && python -m pytest tests/
```

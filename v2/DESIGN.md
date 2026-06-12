# v2 Design: Simulator, Evaluation, and the Modeling Question

## The goal

`simulate(team_a, team_b) → a game`. Everything else (UIs, season sims, what-if
lineups) builds on that call. This document records what "good" means for that
function and which modeling path to take.

## Target use cases (and what each demands)

1. **Cross-era fantasy matchups** — MJ's Bulls vs the Splash Bros. Needs
   multi-year data, player-season embeddings (`e_player + δ_season`), and the
   global season channel as an **era dial**: simulate the matchup under 1996
   conditions or 2016 conditions. Caveats: unfalsifiable (no ground truth —
   evaluate plausibility/consistency, not accuracy), and play-by-play data
   thins out before ~1996-2000; pre-1996 teams are behind a data wall.
2. **Mid-season prediction** — fantasy values, over/unders, win probabilities.
   This is exactly what the backtest harness scores (calibration, Brier,
   totals coverage). Additional needs: projected lineups/injury availability
   (we condition on starters; knowing them is the user's input), the
   rest/schedule channel, and honesty about the bar: beating Vegas closing
   lines is far harder than beating Elo — the realistic product is calibrated
   distributions, not guaranteed edge.
3. **Talent evaluation / roster fit** — swap a player (or an archetype token:
   "generic rim protector") into a lineup, simulate the schedule, measure
   Δ(point differential). This is counterfactual simulation, and it is only
   credible because of the embedding-lab result: if player vectors encoded
   team identity, swaps would be meaningless. Defense-aware outcome
   conditioning (risk #1) is load-bearing here too.

Future direction (marked down, not started): **MLB on Statcast**. The
architecture transfers nearly wholesale — pitch/PA event tokens, state
channels (count, bases, outs), factorized player-season identity — and
baseball's discrete structure plus Statcast's measurement depth make it
arguably *more* tractable than basketball. Revisit after the NBA transformer
proves out.

## What exists now

- **`Simulator` interface** (`sim/simulate.py`): `simulate(away, home, as_of, rng) → SimGame`.
  Every future engine — statistical, trained transformer, finetuned LLM — implements
  this, so they all plug into the same backtest.
- **`StatisticalSimulator`**: Monte Carlo from season-to-date scoring rates +
  home edge + league score variance, with player box lines from noisy season
  shares. It is deliberately dumb. **Any learned model must beat it to justify
  its training cost.**
- **Backtest harness** (`sim/evaluate.py`): time-split evaluation on 673 games
  (Jan 15 – Feb 2023), form computed strictly from earlier games.

Baseline results (673 held-out games, 200 sims each):

| Metric | StatisticalSimulator | home-always | better-record |
|---|---|---|---|
| Pick accuracy | 60.9% | 56.0% | 60.2% |
| Brier score | 0.2357 | 0.2468 (p=0.58) | — |
| Margin MAE | 10.7 | — | — |
| Margin coverage (p10–p90) | 90.6% (target ~80%) | — | — |

The coverage number is a caught flaw: the baseline draws two independent team
scores at league std (~12), giving margin std ~17 vs the real ~13.5 — it's
over-dispersed. Public reference points for context: Vegas lines hit roughly
67% / Brier ≈ 0.21 on NBA winners. The gap 61% → 67% is the playing field.

## What a good evaluation is

A simulator is stochastic, so evaluating one sampled score against one real
score is meaningless. Evaluate it as a **probability model over games**:

1. **Win-probability quality** — run K sims per held-out game; score the implied
   win prob with Brier, log loss, and a calibration table (when it says 70%, does
   the home team win 70% of the time?). This is the headline metric.
2. **Dispersion honesty** — the actual margin should land inside the simulated
   p10–p90 about 80% of the time. Too narrow = overconfident, too wide = mushy.
3. **Distributional realism** — simulated team totals, margins, and player lines
   should match historical *distributions* (compare via Wasserstein distance or
   quantile checks), not just means. A simulator that always outputs 112–108 can
   ace MAE while being useless.
4. **Micro-level likelihood (event models only)** — held-out next-event
   perplexity, plus validity checks (possessions alternate, points sum to the
   score, clock decreases). This is the cheap training-time signal that doesn't
   require rollouts.
5. **Leakage discipline** — form features use only games before the target date;
   train/eval split is by time (train Oct–mid-Jan, eval mid-Jan–Feb), never random.

## Scratch vs. finetune: recommendation

**Train a small event-sequence transformer from scratch.** Reasons:

- **Contamination.** Every pretrained LLM has read about the 2022–23 NBA season.
  A finetuned LLM that "predicts" Celtics-beat-Sixers may be reciting memory, and
  no backtest on this season can distinguish that from skill. A from-scratch
  model is provably clean.
- **Player embeddings fall out natively.** Tokenize events with player-ID tokens;
  the embedding table *is* the player-embedding matrix. After training, nearest
  neighbors should recover roles/positions without ever being told them — that's
  both the deliverable you want and a free sanity check of the model.
- **The data is the right shape and size.** ~1,320 games × ~470 events ≈ 620k
  events. Tokenized at 3–5 tokens per event (type, player, outcome, clock/shot
  buckets) that's ~2–3M tokens — small for language, ample for a 5–15M-param
  GPT over a ~1,000-symbol vocabulary (≈600 players + event/outcome/bucket
  tokens). Chess transformers learn strong play from comparable corpora. Trains
  in minutes–hours on CPU/modest GPU.
- **Simulation is just sampling.** Condition on date + lineups, roll the model
  forward autoregressively, and a full play-by-play falls out — scores, box
  lines, quarter splits all *derived* from one generative process rather than
  predicted as separate heads. Every evaluation above applies directly.

Finetuning an open LLM is still worth doing later **as a comparison row in the
backtest table**, with eyes open: it brings basketball priors and a text
interface, but costs more, rolls out slower, has the contamination problem, and
its player representations are smeared across a tokenizer rather than sitting
in a clean embedding table.

Known risks of the scratch path, with mitigations:

- *Thin tails*: bench players have few events → map players below a minutes
  threshold to a shared `<bench>` token per position.
- *Rollout drift*: long generations can lose the clock/score plot → include
  period and clock-bucket tokens so the model is always conditioned on game
  state, and validate rollouts structurally.
- *Small data*: one season only → heavy regularization, small model, and the
  statistical baseline as the floor that tells us if it's working.

## Embedding lab findings (experiments/embedding_lab.py)

Tested locally with count-based embeddings (PPMI + SVD) as a fast proxy for
what a trained model's player vectors would encode. Four context definitions,
368 players (≥500 min), playstyle reference built from a disjoint half of the
games. `teammate@10` = team leakage; `style@10` = playstyle recovery
(split-half ceiling 0.397, chance ≈ 0.03).

| variant | what the player vector predicts | teammate@10 | style@10 (dim 64) |
|---|---|---|---|
| CTX-full | nearby events incl. who acts | **0.894** | 0.027 |
| CTX-anon | nearby events, identities stripped | 0.110 | 0.095 |
| ACT | own actions only | 0.033 | **0.203** |
| CTX-team | CTX-full minus team centroid | 0.256 | 0.028 |

Three conclusions, each answering a standing hypothesis:

1. **Bigger embeddings don't fix team clustering.** Dim 16 → 128 moved
   teammate@10 only 0.94 → 0.87. Team identity is the dominant predictable
   signal whenever teammates appear in the context; extra dimensions just store
   more of it.
2. **Post-hoc team subtraction removes team but does not recover playstyle**
   (0.028 ≈ chance). The style signal was never encoded — it is absent, not
   masked. Subtraction can't restore information the training signal never
   rewarded.
3. **What the player token is asked to predict determines what it encodes.**
   Strip identities from context and leakage collapses (0.89 → 0.11); restrict
   to own actions and style recovery is the best achievable here (half the
   split-half ceiling), with face-valid neighbors: Gobert → Plumlee, Jordan,
   Claxton, Drummond; Klay → Hardaway Jr., Beasley, Isaiah Joe; Harden →
   LaMelo, Haliburton, Dinwiddie; Jokic → KAT, Draymond, Vucevic.

**Confirmed on the trained transformer (EventGPT, 5.7M params).** The probe
(`test_scripts/probe_embeddings.py`, same metrics) ran across four training
configurations:

| configuration | teammate@10 | style@10 |
|---|---|---|
| naive loss (predicts rosters/lineups) | 0.940 | 0.031 |
| + header/Q1 masked from loss | 0.929 | 0.034 |
| + leak-free state channels | 0.938 | 0.033 |
| + **legality-masked player slots** | **0.085** | **0.144** |

The cause was the tied softmax: every prediction pushed ~520 not-in-this-game
players away, and that suppression direction is team identity. Restricting
player-slot softmax to the game's rosters (~23 candidates) removed the force;
team clustering collapsed 11x and playstyle emerged in the same run — Gobert's
neighbors became Gafford/Zubac/Robert Williams/Nurkic (rim protectors, no
teammates), Harden's became Haliburton/CP3/Mitchell (ball-dominant guards),
Jokic's top comp is Sabonis. Masking also exposed a latent encode bug
(P:nan jumpers) as a hard failure and made actor loss interpretable
(~2.5 nats ~ choosing among the legal ~12).

Prescription for the transformer: **factorize identity out of the player
token's job.** Give the model team/lineup context through separate channels
(team tokens, additive team embedding on the input) so the player embedding is
never the cheapest way to encode "who my teammates are", and keep a
player-conditioned own-outcome term in the loss so the embedding is rewarded
for encoding what the player *does*.

## Player-year structure (untestable locally — one season of data)

For the multi-year build, recommend **factorized player-season embeddings**:

    e(player, season) = e_player + δ_player,season  (+ e_season, global)

- `e_player` carries career identity; `δ` (regularized toward zero) carries the
  vintage: 2016 Steph vs 2024 Steph share a core and differ by a small offset.
- A global per-season term absorbs league-wide drift (pace, 3pt era) so player
  deltas don't have to.
- Team creation composes naturally: career-average = `e_player`; a vintage =
  `e_player + δ_y`; an extrapolated vintage is interpolation in δ-space.
- This beats independent player-year tokens (fragments each player's data into
  ~80-game shards, no transfer across seasons) and beats post-hoc averaging of
  separately learned year vectors (nothing aligns those spaces).
- Same principle as the team finding above: build the decomposition into the
  parameterization, don't try to recover it afterward.

## Risk register: what we'd otherwise miss

Domain gaps:

1. **Defense is nearly invisible in play-by-play.** Individual defense surfaces
   only as steals/blocks/fouls/defensive rebounds. The outcome conditional
   P(made | actor, action, dist) must also see the **defensive five** — the
   on-floor lineup channel includes both sides, or Gobert's rim protection
   doesn't exist in the sim. Add counterfactual probes to the eval suite: swap
   a rim protector in, opponent rim FG% should drop; five centers should fail.
2. **Hard rules the token grammar permits violating.** Free-throw sequences
   (2/2 requires 1/2), foul-outs at 6, Δt bounded by the remaining clock,
   forced [END_Q] at 0:00, tie at end of regulation forces OT. These live in
   the rollout state machine as per-slot legality masks, and double as
   validity metrics on generated games.
3. **Rotations are a policy, not noise.** Substitution events are 13% of all
   events. Eval must include minutes-distribution realism (ground truth is in
   the cache), and foul-trouble responses. For user-built rosters the model
   generates rotations from the supplied bench — that's a feature, but only if
   minutes realism is measured.
4. **The dataset includes the postseason** (1,230 regular + 6 play-in + 84
   playoff games, by game_id prefix 002/005/004). Playoff ball differs (pace,
   8-man rotations) and series repeat matchups, violating backtest
   independence. Add a game-type header token/channel; default backtests to
   regular season, playoffs as a separate stress set. NOTE: the current 673-game
   backtest in this doc mixes both — rerun split when the next engine lands.
5. **Identity hygiene.** "A.J. Green" and "AJ Green" are the same player
   (both MIL, never co-occur). The tokenizer needs a canonicalization pass —
   flag same-team near-duplicate names that never co-occur — and multi-year
   data needs stable player IDs, not name strings.
6. **Schedule context.** Rest days / back-to-backs are derivable from the file
   dates and measurably matter; cheap header channel. Home/away is currently
   positional in the header — when teams become player-sets, keep an explicit
   home channel (home edge ≈ 3 pts must come from somewhere).

ML gaps:

7. **Exposure bias.** Training is teacher-forced; simulation free-runs ~2,000
   tokens. Measure drift by rollout depth (does Q4 pace/scoring look worse
   than Q1?); legality masks bound the damage.
8. **Per-slot diagnostics, not one perplexity.** The outcome slot is a binary
   classifier — score its calibration directly (predicted vs actual FG% by
   player and zone). Actor-slot perplexity = usage modeling quality. Aggregate
   perplexity hides which skill is failing.
9. **Per-slot sampling temperature** is the post-hoc dispersion knob — fit on
   validation to hit the ~80% margin-coverage target the statistical baseline
   currently misses (90.6%).
10. **Embedding probes as training callbacks.** The lab metrics (teammate@10,
    style@10) are cheap; run them during training to catch team leakage early,
    and consider warm-starting player embeddings from the ACT PPMI-SVD vectors.
11. **Low-sample players.** 41 of 543 names appear <50 times; star players have
    ~10× the events of role players. Don't hard-replace them with generic
    tokens (their thin data is still real); use **hierarchical shrinkage**:
    `e_player = e_archetype + residual`, residual regularized ∝ 1/sample-size,
    so fringe players collapse to their archetype and stars individuate.
    Archetypes must be *learned* (no position column in the data) — cluster
    the embedding lab's action profiles (~8 clusters). Then make archetype
    tokens first-class via **player-token dropout** in training (randomly swap
    a player token for their archetype token), which teaches the model to
    simulate "a generic 3&D wing" — the interface for rookies, cold-start,
    and roster-fit queries.
12. **Environment**: torch is not installed yet (CPU wheel needed before
    training starts).

## Next steps

The live task list is the repo-root **`TODO.md`** — single source of truth.
Completed milestones for the record: tokenizer with exact round-trip —
1,320/1,320 on 2022-23, then 7,579/7,590 across all six seasons (11 waivers,
each traced to verified source-data corruption; see TOKENIZER.md); cross-season
name canonicalization (13 split identities); training pipeline (state channels
incl. possession, per-slot validation, legality-masked player slots, best-val
checkpointing); 6x multi-season retrain (best val 1.536; FGA/FTA/minutes-corr
all moved sharply toward reality — the old sampler-loop pathologies were
exposure bias); free-throw state machine in the sampler (fouls arm FTs, forced
sequence completion); rollout sampler with preallocated-KV-cache lockstep
batching (0.34s/game at batch 128); embedding probes (playstyle clustering
held at 6x scale: teammate@10 0.098 vs 0.032 chance) and interactive explorer;
64-game distributional smoke vs held-out reality.

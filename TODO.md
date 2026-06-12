# TODO

Single source of truth for project tasks. Architecture rationale lives in
`v2/DESIGN.md`; this file is what's left to do.

## Immediate next cycles (sampler/eval only, no retrain)

1. ~~**Free-throw state machine in the sampler**~~ DONE 2026-06-11 — fouls arm
   FTs, 1/2 -> 2/2 forced, subs/timeouts may intervene, shooter-subbed-out quirk
   handled (tests in v2/tests/test_sample.py). FTA 56.3 -> 47.3 vs real 23;
   the expected bigger win didn't materialize because **the model over-fouls**:
   sims 93 fouls/game vs real 39 (old sampler already at 72), uniform across
   periods, with a foul->FT->foul loop (P(foul after foul) 18% vs 4% real) and
   blocked-FT mass partly diverting into F:unknown. This is exposure bias, not
   grammar — re-measure after the 6x retrain (#6); per-slot calibration (#17)
   is the sampler-side fallback. Post-fix smoke (63 games): totals 119.2±18.0
   (real 115.8±12.4), margin +5.1±30.0, FG% .496/.481, FGA 70.9 vs 88.7,
   winner 37/63, pts-corr 0.563, min-corr 0.322.
2. ~~Minutes-correlation regression~~ RESOLVED by the 6x retrain (0.32 -> 0.48):
   it was exposure bias, not the possession channel.
3. Length-capped games: not observed since the cap rose to 3,300 tokens and
   the 6x model stopped FT-spiraling (0 truncations in recent 64-game runs).
   Keep the skip-with-warning guard; revisit only if it reappears.

## Multi-season integration (in order)

4. ~~**Cross-season name canonicalization**~~ DONE 2026-06-11 —
   experiments/name_audit.py swept all six seasons; 13 split identities now in
   NAME_FIXES: 7 mechanical (punctuation/suffix), 4 nickname switches (Nic
   Claxton, KJ Martin, Bub Carrington, Alexandre Sarr), 2 full renames caught
   by targeted checks (Enes Kanter/Freedom, Didi Louzada/Marcos Louzada Silva).
5. ~~Rebuild cache + round-trip sweep over all 7,590 games~~ DONE 2026-06-11 —
   **7,579/7,590 exact**, 11 documented waivers, all traced to source-data
   corruption (verified against descriptions + historical box scores):
   7 games with a duplicated/blank player in the lineup columns (minutes
   unreconstructable; stats exact), 4 games with a corrupt points column where
   the replay derivation is RIGHT and the cache is wrong (Harden's 40-pt-streak
   game, Beal's 46). Also fixed: duplicated end-of-period row, blank-player
   flagrant FTs (FT_SHOOTER_FIXES), NaN lineup cells on sub/period-start rows
   (the P:nan poison token is gone), per-side sub diffs. Corpus 16.25M tokens,
   vocab 1,385, max game 3,124 tokens (model max_len bumped to 3,328); zero
   body tokens outside their game's header roster (legality-mask requirement).
6. ~~**Retrain at 6x data**~~ DONE 2026-06-11 — 30k steps, --val-cutoff
   2025-03-01 (6,716 train / 343 val), best val **1.536 @ step 19000**
   (plateau ~20k; single-season best was 1.779 on its own window). Smoke vs
   new val: FGA 86.9 vs 89.3 real (was 70.9!), FTA 30.0 vs 21.4 (was 47.3),
   min-corr 0.48 (was 0.32), totals 117.4±14-17 vs 114.8±12.8, FG% .461 vs
   .472 (now slightly under), margin still over-dispersed. Fouls/game 59.5 vs
   real 37.0 (was 93) — exposure bias shrank with data; the residual is the top
   distribution error (levers: #17 calibration, #10). Embedding probes on the
   754-player multi-season universe: teammate@10 0.098 (chance 0.032 — no
   relapse), style@10 0.093 (chance 0.013, count-profile ceiling 0.298 — same
   ~1/3-of-ceiling as the single-season model); face-valid neighbors
   (Gobert->Mobley/Capela/Poeltl, Klay->Hield/Fournier/THJ). Old checkpoint
   kept at v2/cache/model_2223.pt. Sampler now uses a preallocated KV cache
   (sim/model.py KVCache): per-step torch.cat was fragmenting VRAM into
   shared-memory spill — 0.34s/game at batch 128 (was 17.6s at 64).
7. **Season conditioning**: e_player + δ_season factorization + global season
   token/channel (see DESIGN.md) — unlocks cross-era matchups.
8. Playoff/game-type conditioning channel (currently playoffs are just excluded).
9. Scrape the missing **2023-24 season from the official NBA API** (sanctioned
   path per BigDataBall).

## Model quality (after the multi-season baseline lands)

10. Embedding gaps: teammate@10 0.085 (floor 0.034), style@10 0.144 (ceiling
    0.397). Levers: on-floor lineup channel, untied softmax head, more data (#6).
11. **Archetype tokens**: cluster action profiles, hierarchical shrinkage for
    low-sample players, player-token dropout — the interface for rookies and
    "generic 3&D wing" roster queries.
12. Shot-zone tokens from coordinates (deferred until shot-chart realism matters).
13. Run embedding probes (teammate@10 / style@10) as training callbacks.

## Definition of GOOD (ratified with Mason 2026-06-12) — NBA exit criteria

The NBA work is not done until ALL three gates pass. Protocol v2 for every
number: 50 sims/game (200 was too expensive), home-win prob from a normal
fit to the margins (empirical frequency at 50 sims adds ~0.005 Brier noise),
both baselines re-measured under the identical protocol, paired by game.

- **Gate A — calibration**: margin coverage (p10-p90) in [76, 84] (now
  85.1) — widened from [78, 82] on the MLB session's point (2026-06-12)
  that empirical coverage of a PERFECT model has SE ~2.2pp at n=343, so
  the +/-1-SE band fails a flawless simulator a third of the time;
  [76, 84] is +/-2 SE, matching the MLB gates. Margin sd within 10% of
  the real window's (~16.7; now ~26).
  Temperature tuned on a dev window only (Jan-Feb 2025; model-seen — note
  the caveat, fix at next retrain by moving val-cutoff to 2025-01-01),
  one evaluation on the untouched 343-game val set.
- **Gate B — prediction (required, Mason's bar: beat BOTH baselines)**:
  transformer Brier better than the player-form baseline (#15b, same
  roster information) AND the team-form baseline on the 343-game window,
  paired bootstrap; margin MAE within +0.5 of the best baseline. Picks
  reported, never a gate (n=343 cannot distinguish ~5pp).
  **Targets measured 2026-06-12, protocol v2, identical 343 games:**

  | reference                  | picks | Brier  | log loss | margin MAE |
  |----------------------------|-------|--------|----------|------------|
  | Vegas closing lines        | 72.6% | 0.1806 | 0.5376   | 10.5       |
  | player-form baseline #15b  | 67.9% | 0.2117 | 0.6136   | 13.0       |
  | team-form baseline         | 63.6% | 0.2244 | 0.6402   | 12.3       |

  So the gate is: **Brier < 0.2117 and margin MAE <= 12.8.** Files:
  results/backtest_{player,team_paired343,vegas_2025-03-01}.json. The
  team-form number moved from the recorded 0.2205 because (a) protocol v2
  smoothing, and (b) a real bug fixed 2026-06-12: with the multi-season
  cache, sim/form.py team_form had no season boundary, so "season-to-date
  form" silently became career-since-2018 averages (BKN@DET expected
  margin +0.9 vs +6.6 season-scoped). form.py is now season-scoped
  (Aug 1 boundary), matching the original single-season-cache behavior.
- **Gate C — capability**: pre-registered counterfactual suite (#16),
  >= 4 of 5 with correct sign and bootstrap CI excluding zero; embedding
  probes hold (teammate@10 <= 0.11, style@10 no worse than current).
- **External anchor**: historical Vegas closing lines for the val window
  as a reference row in every backtest table (Mason: yes, pull them).

Execution order: protocol v2 in both backtests -> re-run baseline ->
per-slot temperature calibration (#17) on dev -> Gate A check -> build
#15b -> Gate B -> if it fails, #10/#11 levers and re-enter at Gate A.
Counterfactual suite (#16) in parallel. MLB stays on the Mac meanwhile.

**Round 1 verdicts (2026-06-12, 6x checkpoint @19k, identity temps,
results/backtest_transformer_2025-03-01_s50.json + gate_b.py):**
- Transformer protocol-v2 row: 62.4% picks / 0.2371 Brier / 0.6791 log
  loss / 14.0 margin MAE / coverage 80.8% / sim sd 20.1 (real 16.7).
- **Gate A: FAIL** — coverage 80.8 is in [76, 84], but sim margin sd is
  20.1 vs the <= 18.4 the 10%-of-real component requires. Temperature
  can't fix it (#17 negative result); dispersion is structural.
- **Gate B: FAIL** — paired bootstrap: vs player-form +0.0254 Brier, CI
  [+0.0014, +0.0496] (significantly WORSE); vs team-form +0.0127, CI
  [-0.0064, +0.0324] (statistically tied); MAE 14.0 vs <= 12.76 needed.
- Next (pre-committed): retrain with #10 levers (lineup channel + untied
  head, both implemented and smoke-tested 2026-06-12) at --val-cutoff
  2025-01-01 (makes the Jan-Feb dev window clean), then re-enter Gate A.
  Checkpoint backup at cache/model_6x_19k.pt; new run writes
  cache/model_v3.pt.

## Evaluation / product

14. **The proper backtest**: transformer vs StatisticalSimulator, same protocol
    (200 sims/game). Target re-measured 2026-06-11 on the multi-season val
    window (2024-25 regular season, predictions 2025-03-01+, season-to-date
    form): **65.9% picks / 0.2205 Brier / margin MAE 12.0, n=343**
    (results/backtest_baseline_2025-03-01.json). The old 60.9%/0.2357 was the
    2022-23 mid-season window — late-season form is stronger; compare like
    for like. **RESULT 2026-06-12** (343 games x 200 sims,
    results/backtest_transformer_2025-03-01.json): picks 61.8% / Brier
    0.2331 / log loss 0.6654 / margin MAE 14.1 / coverage(p10-p90) 85.1%
    (target 80) / 10 of 68,600 rollouts truncated. **The statistical
    baseline wins every metric** (65.9% / 0.2205 / 12.0) — and the
    conditioning asymmetry favored the transformer (it saw the night's
    actual rosters/starters; the baseline only team form). Diagnosis:
    margin over-dispersion (coverage 85% vs 80% target; sd ~±26 vs real
    ±16.7) flattens win probabilities toward 0.5, hurting picks and Brier
    together; under it, event-level imitation error compounds over ~450
    events/game (residual exposure bias — over-fouling, #1). The simulator
    is trained to imitate play-by-play, not to pick winners. Levers, in
    order: #17 temperature calibration (cheapest — directly attacks
    dispersion), #15b player-form baseline (decomposes roster knowledge vs
    simulation value before any Sloan claim), #10/#11 model quality.
15. Re-run the statistical baseline's backtest split regular-season vs playoffs
    (current 673-game numbers in DESIGN.md mix both).
15b. ~~**Player-form baseline**~~ (Mason, 2026-06-11): the team-form baseline never
    uses player form to predict the score (only to decorate box lines), and it
    doesn't see the night's rosters — but the transformer does, so a win could
    be dismissed as roster knowledge, not modeling. Add a baseline that takes
    the same header (actual actives/starters), projects mpg-weighted player
    season-to-date scoring vs opponent defense, and sums. The three-way gap
    decomposes the transformer's edge into roster-information value vs
    learned-simulation value — the second number is the Sloan headline.
    BUILT 2026-06-12 (v2/test_scripts/backtest_player_baseline.py):
    deterministic margin + Normal(sigma) fit on the Jan-Feb dev window
    (sigma=17.1). Result on the 343-game val set: 67.9% picks / 0.2117
    Brier / 13.0 margin MAE — the strongest baseline and the binding Gate B
    bar. Roster information alone is worth ~4pp picks / 0.013 Brier over
    team form. Vegas reference also pulled (backtest_vegas.py, Kaggle
    closing spreads + spread->prob logistic fit on 2008-2024): 72.6% /
    0.1806 / 10.5 — the market ceiling, never a gate.
16. **Counterfactual sanity suite** (pre-registered in
    test_scripts/counterfactual_suite.py). **ROUND 1: 3/5 — Gate C FAIL**
    (2026-06-12, 6x checkpoint, results/counterfactual_suite.json):
    rest_star PASS (-5.7 [-8.8, -2.7]); five_centers PASS (3PA -8.0);
    venue_flip PASS on sign but implies HCA ~7.4 pts vs ~3 real
    (over-learned home edge — watch after retrain); add_star FAIL
    BACKWARD (Jokic joining a foreign roster *drops* margin -3.4
    [-5.9, -1.2] — player identity entangled with team context, the
    embedding-lab conclusion at the behavioral level; the #10 lineup
    channel attacks exactly this); rim_protector FAIL null (-0.002 rim
    FG%, CI spans 0 — defender identity is invisible in the grammar's
    defensive end; structural, won't fix without defense tokens; candidate
    swap: replace with a falsifiable offense-side counterfactual, or
    accept 4/5 as the realistic ceiling of this suite).
17. ~~Per-slot sampling temperature calibration~~ **NEGATIVE RESULT
    2026-06-12** (results/calibration_grid.json: 9-cell outcome x action
    grid, 48 dev games x 50 sims): margin sd sits at 19.6-20.3 in EVERY
    cell — temperature does not move dispersion, so the over-dispersion is
    structural (residual over-fouling/exposure bias, #1), not a sampler
    knob. Brier spans 0.2208-0.2380 with no coherent structure (the best
    value appears at two unrelated corners; coverage column is pure noise
    at dev n=48, SE ~5.8pp). Decision: NO temperature override — the one
    val evaluation runs at identity temps (any other pick would be fitting
    dev noise; the script's pre-declared closest-to-0.80 rule would have
    chosen the worst-Brier cell, a lesson in selecting on noisy criteria).
    Dispersion lever moves to the #10 retrain.
18. User-facing `simulate(roster_a, roster_b)` CLI (sampler already supports
    arbitrary rosters internally).

## MLB twin (mlb/ — see mlb/DESIGN.md for design + status)

Hardware (settled 2026-06-12): Mac only for MLB tonight; the 3070 Ti maybe
transitions over tomorrow. **GPU handoff protocol**: the line below is the
single source of truth; the NBA session edits it when it releases the
3070 Ti for good, and MLB sessions must not touch the desktop GPU until
it flips (CPU-only inference locally is fine).
GPU status: **NBA-owned** (set 2026-06-12; NBA session — flip this to
"FREE for MLB" + date when your GPU work is done for good). Data scope: 2020-2025 (all pulled 2026-06-12,
~3.83M regular-season pitches; the crashed desktop pull was resumed and
completed — 2022/2023/2025 landed; 2025's parquet keeps non-R rows that
data.py filters at load). 2015-2019 deliberately deferred. **Single-season
2024 model first** as the baseline; multi-season + SEASON:/δ-vintage
conditioning (DESIGN.md 4b) only after the 2024 gates are attempted.

22. ~~Statcast encoder + replay with v2's round-trip contract~~ DONE
    2026-06-12, extended same day to grammar v1.1 (pitch velo/spin, plate
    location, batted-ball physics, O:/B: state transitions, PARK:/MONTH:):
    2,426/2,427 games exact + 1 documented waiver, per-pitch state
    (count/outs/bases) verified against source columns on all 711K
    pitches. 5.92M-token corpus, vocab 1,849.
23. **MLB training**: ~~M1 smoke run~~ DONE 2026-06-12 — 3,000 steps,
    full-size 6.2M-param model, VAL 1.708 (pitch 1.611 / result 1.168 /
    event 0.634 / bases 0.501), every slot still descending at the cap:
    convergence confirmed. Checkpoint copied to desktop as
    mlb/cache/model_m1smoke.pt (Mac keeps model_smoke3k.pt). **Full 2024
    run IN FLIGHT** (Mac, PID 11362, /tmp/mlb_train_full.log): 20k steps,
    batch 4, --val-cutoff 2024-08-01 --val-end 2024-09-01 (1,629 train /
    413 Aug-dev val; September never loaded — train.py gained --val-end
    because best-val checkpoint selection is tuning and must not see the
    test window). ~13h at 0.42 steps/s; restartable on the 3070 Ti when
    the GPU status line flips.
24. ~~**MLB sampler state machine**~~ DONE 2026-06-12 (sim/sample.py):
    deals batters (due batter or fresh bench, slot inherited; E: advances
    the order EXCEPT truncated_pa — batter re-bats, measured), deals
    pitchers (current or fresh pen, removed never return), forces [HALF]
    at exactly three outs, conservation-masks B: (runners + batter =
    runners' + outs + runs) with [MID] +n settling between-row scores,
    forces E: by pitch result (ball4 -> walk family, strike3 -> K family,
    two-strike bunt foul -> K, hit_into_play -> 16 in-play events, HBP),
    handles dropped third strikes (K with zero outs), walk-offs (game
    ends the instant home leads in Bot 9+, no trailing O:/B:), skipped
    Bot 9, extras with the Manfred runner, and per-event minimum-out
    forcing — every rule measured on the corpus first
    (test_scripts/derive_sampler_tables.py). Validation:
    **2,417/2,427 real games drive through fully legal + channel-exact**
    (test_scripts/drive_real_games.py; 10 documented source-data waivers:
    voided pitch, lineup-misfiled injury sub, the suspended/traded-Jansen
    game, 3 count glitches, 4 rain-shortened endings) + 17 handcrafted
    scenario tests (tests/test_sample.py, 31 tests green with tokenizer
    suite). Smoke generation vs the 3k-step checkpoint
    (test_scripts/smoke_generate.py, CPU, ~500 tok/s at batch 6): all
    games complete and Replay() clean, 17-22 halves (extras + walk-offs
    occur), PA 66-78 vs real ~75, K/BB/H plausible for an undertrained
    checkpoint.
25. MLB after the full checkpoint (NEXT): smoke eval vs real 2024 rates
    (R/HR/K/BB, EV/LA distributions), embedding probes (batter
    contact-quality clusters, pitcher arsenal clusters vs team leakage),
    then the gate suite below. Scale to more seasons only after.
26. MLB data niceties: venue lookup for neutral-site PARK: tokens; stand
    token for switch hitters if platoon realism underperforms; bat-speed/
    swing-length tokens when 2025 data (full coverage) lands.

## Definition of GOOD (ratified with Mason 2026-06-12) — MLB exit criteria

The 2024 single-season MLB model is not done until ALL three gates pass.
Protocol for every number: train < 2024-08-01; **dev = August 2024** (all
calibration/temperature tuning happens here and only here); **test =
September 2024** (~380 games, untouched, evaluated ONCE per attempt).
50 sims/game; home-win prob = empirical sim frequency — run margins are
discrete with no ties, so NBA's normal-fit trick is wrong here; if the
~±0.07 noise at 50 sims bites, go to 100 sims, never a parametric fit.
Baselines re-measured under the identical protocol, paired by game.

- **Gate A — calibration**: simmed-vs-real test-window league rates:
  K/game and BB/game within 5% (per-pitch quantities, learned directly);
  R/game and HR/game within 10% (rare, compounded outcomes — the analog
  of NBA's over-fouling lives here). Run-total sd within 10% of real.
  Run-differential p10-p90 coverage in **[76, 84]** — ±2 SE at n≈380; a
  ±1-SE band like NBA's [78, 82] fails a perfectly calibrated model ~1/3
  of the time (flagged to the NBA workstream — same math at n=343).
  Physics tripwires: per-pitch-type median velocity within 0.5 mph of
  real; EV and LA quartiles within one token bin. The physics gates are
  anti-Goodhart canaries, not stretch targets — the data-level numbers
  already satisfy them.
- **Gate B — prediction**: transformer Brier better than the **team-form
  baseline** (season-to-date log5 + home advantage) on the test window,
  paired bootstrap. The **lineup-and-starter baseline** (sees the night's
  lineups + starting pitcher — the baseball #15b, and the starter is the
  single biggest piece of night-of information) is the decomposition row,
  NOT a gate: parity within the bootstrap CI is acceptable, and the
  three-way gap (starter/roster information value vs learned-simulation
  value) is the deliverable. Rationale (Mason's too-strict concern,
  2026-06-12): baseball outcomes are far noisier than NBA — best teams
  win ~60% of games, not ~80% — so a perfectly calibrated simulator can
  legitimately tie a strong baseline on Brier. Margin MAE within +0.2
  runs of the best baseline. Picks reported, never a gate. External
  anchor: closing moneylines as a reference row in every table (source
  still needed). **Targets measured 2026-06-12
  (mlb/test_scripts/backtest_baselines.py; dev-fit bias+sigma on August,
  evaluated on the 385-game September window):**

  | reference                  | picks | Brier  | log loss | margin MAE |
  |----------------------------|-------|--------|----------|------------|
  | lineup-starter baseline    | 55.6% | 0.2468 | 0.6866   | 3.51       |
  | team-form baseline         | 52.5% | 0.2470 | 0.6870   | 3.53       |

  So the gate is: **Brier < 0.2470 and margin MAE <= 3.71.** Both
  baselines hit ~80% p10-p90 coverage (their dev-fit Normal-sigma is
  honest). No-skill (constant-p) Brier is ~0.250 — the bar is thin but
  real, exactly as the gate design anticipated. Starter information is
  worth ~3pp picks over team form at ~equal Brier. Files:
  mlb/results/backtest_{team,lineup_starter}_2024-09.json.
- **Gate C — capability & player-stat recovery** (the end goal — first-
  class here, unlike NBA): (i) per-player simmed test-window rates
  (batters: K%, BB%, wOBA; pitchers: K%, BB%) correlate with actuals at
  **>= 50% of the noise ceiling** — the Spearman-Brown step-up of the
  September split-half correlation, since the sim is judged against
  full-window rates. **Measured 2026-06-12 (noise_ceiling_2024-09.json):
  ceilings/gates — batter K% .635/.317, BB% .437/.219, wOBA~ .192/.096,
  pitcher K% .435/.217, BB% .317/.158** (one September is VERY noisy;
  the wOBA gate is honest but weak — K% is the load-bearing stat,
  matching its known fastest-stabilizing reliability); (ii) pre-registered
  counterfactual suite, >= 4 of 5 correct sign with bootstrap CI
  excluding zero: ace starter -> replacement-level starter (opponent
  runs rise); PARK: swap to Coors / a pitcher's park (totals move
  correctly); same lineup vs a known LHP vs a known RHP (platoon gap
  sign — tests handedness in identity embeddings without a stand token);
  best hitter 9th -> leadoff (PA-volume mechanics); elite bat into a
  weak lineup (team output rises); (iii) embedding probes: teammate@10
  <= 2x chance, pitcher arsenal@10 >= 1/3 of count-profile ceiling (the
  fraction v2 actually achieved).
- **Anti-Goodhart rules** (Mason, 2026-06-12: strict targets breed wild
  solutions): levers pre-registered (per-slot temperature calibration on
  dev first, then model-quality items); the test window is evaluated
  once per attempt; a failed gate sends work back to diagnosis, never to
  test-set tuning. The gates deliberately pull against each other —
  calibration hacks trip the physics tokens, prediction hacks blow out
  coverage. NBA's Gate B miss -> diagnosis -> ordered lever list is the
  model for handling failure.

Execution order: sampler (#24) -> smoke eval vs 2024 rates (#25) -> full
2024 train at the gate split -> dev calibration -> Gate A -> baselines ->
Gate B -> Gate C suite (counterfactuals in parallel once the sampler
takes arbitrary rosters). NBA owns the desktop GPU meanwhile.

## Admin / Sloan

19. **Re-confirm snippet-sharing with BigDataBall before the repo goes public**
    — the question was asked but never answered (see correspondence below).
20. Cite the official NBA API, not BigDataBall, in the submission (unless
    academically affiliated).
21. ~~**Make a first real git commit**~~ DONE 2026-06-11 — three commits: v1
    legacy baseline, v2 simulator, mlb scaffold. nba_data/, caches, and
    checkpoints stay out of git (.gitignore); to run inference elsewhere,
    copy v2/cache/{model.pt, tokens.jsonl, games.jsonl} (~255MB).

Suggested order: 1 -> 4-6 -> 14. Items 7, 11, 16 are where the Sloan paper's
most interesting figures live.

## Data licensing & citation for the Sloan submission

From BigDataBall support (Serhat, ~Dec 2025 correspondence):

- **Unless affiliated with an educational institute, keep BigDataBall usage
  private in the Sloan submission.** Their dataset closely resembles the
  publicly available version, so **cite/reference the official NBA API as the
  data source** if a citation is required.
- Redistributing the full dataset: off-limits (assumed and effectively
  confirmed).
- **Open question — re-confirm before publishing**: sharing a small,
  sample-sized snippet with the repo for reproducibility was asked but never
  explicitly approved in the reply.
- **Future data path**: since BigDataBall mirrors the official NBA API, the
  NBA API is a viable scrape source going forward.
- Contact: BigDataBall support via noreply@bigdataball.com (whitelist it;
  replies may land in spam).

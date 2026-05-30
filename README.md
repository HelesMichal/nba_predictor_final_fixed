# NBA Game Predictor

Pure-Python pipeline that scrapes NBA stats, builds a leakage-safe dataset, trains a margin-regression model with isotonic-calibrated win probabilities, and predicts upcoming games with last-5-days validation. Optional GitHub Actions automation sends a daily Telegram digest.

## What it does

1. **Fetch** — pulls games + team advanced stats from `nba_api` and player advanced stats (BPM/VORP/WS) + current injuries from Basketball-Reference. Regular season and playoffs are tagged separately.
2. **Build features** — see below.
3. **Train** — XGBoost regressor on `HOME_MARGIN` + IsotonicRegression to map margin → P(home win). Trains on the **last 12 seasons** with **exponential per-season sample weights** (last 2 seasons get full weight 1.0, then `0.85^k` per older year). The **2019-20 season is excluded entirely** (COVID bubble anomaly). Greedy correlation pruning drops any feature with `|r| > 0.97`.
4. **Predict** — for each scheduled game: pulls the most recent feature row per team, applies injury adjustments (subtracts star scores of players listed Out/Doubtful), outputs **P(home win)**, **P(away win)**, and **predicted margin**. Also re-predicts the last 5 days of completed games and reports accuracy.

## Feature list (all used in training)

**Rolling team form** (window = **6 games**, leakage-safe via `shift(1)`):
`roll_win_pct`, `roll_pt_diff`, `roll_fg_pct`, `roll_fg3_pct`, `roll_ft_pct`, `roll_tov`, `roll_reb`, `roll_stl`, `roll_blk`

**Why 6 instead of 10**: 6 games reacts faster to lineup changes, trades, and cold/hot streaks while still being long enough to denoise single-game variance. 10 games can lag real form shifts by 2–3 weeks of NBA basketball.

**Pythagorean expectation** (rolling, current-season, exponent 14):
`pyth_win_pct_roll`

**Schedule & travel**:
`rest_days`, `is_b2b`, `travel_miles`, `tz_shift`, `high_altitude_game` (DEN/UTA)

**Home-court advantage** (per team, prior-season):
`home_wp_prior`, `away_wp_prior`, `home_edge = h_home_wp_prior − a_away_wp_prior`

**Season-level advanced** (split by Regular Season vs Playoffs):
`OFF_RATING`, `DEF_RATING`, `NET_RATING`, `PACE`

**Star strength** (per team-season, derived from per-player weighted score over min/g + USG% + BPM + VORP + WS):
`team_star_total`, `team_star_max`, `team_n_stars`

**Elo ratings** (separate books for Regular Season vs Playoffs, with 538-style margin-of-victory multiplier, home bonus +65, season-to-season regression toward 1500):
`elo_pre`, `elo_diff`, plus per-team `elo_home_pre` / `elo_away_pre` home–away splits and `elo_split_diff`

All home/away features are prefixed `h_` / `a_`. The full master list lives in `nba_predictor/features/build.py::FEATURE_COLS`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# One-shot full pipeline (slow first run; ~1–2 hours of scraping)
python -m nba_predictor.cli.main all --start 2013 --end 2024

# Or step-by-step:
python -m nba_predictor.cli.main fetch --start 2013 --end 2024
python -m nba_predictor.cli.main build
python -m nba_predictor.cli.main train
python -m nba_predictor.cli.main predict --days 1

# Daily Telegram digest (after configuring TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
python -m nba_predictor.cli.main notify --days 1
python -m nba_predictor.cli.main notify --days 1 --dry-run   # preview only
```

The model lives at `artifacts/model.joblib` and metadata (tail accuracy, log loss, Brier, margin MAE, seasons used, features kept/dropped) at `artifacts/model_meta.json`.

## Automation

See `DEPLOYMENT.md`. Two GitHub Actions workflows are bundled:

- `.github/workflows/daily-predict.yml` — runs at 15:00 and 16:00 UTC (lands at 17:00 local for CET *and* CEST) and on manual trigger. Sends today's predictions + last-5-days accuracy to your Telegram bot.
- `.github/workflows/weekly-retrain.yml` — Mondays at 06:00 UTC, plus a manual "Run workflow" button with optional season overrides.

Both rely on cached model + data between runs. If the cache is empty, the workflows now fetch the full training window automatically before building. Cost: **$0** on GitHub Actions free tier + Telegram.

## Notes

- **No leakage**: every rolling / cumulative / Elo feature uses pre-game state only (`shift(1)` before rolling; Elo updates *after* the row is emitted).
- **Rate limits**: nba_api ≈ 0.6s/call; basketball-reference ≈ 3.5s/call. Disk cache on BR HTML and incremental parquet fetching for nba_api keep re-runs cheap.

## Project layout

```
nba_predictor/
  config.py
  data/
    nba_api_source.py     # games + team advanced
    br_source.py          # player advanced + injuries (disk-cached)
  features/
    star_score.py         # weighted per-player influence
    travel.py             # travel miles / tz shift / altitude
    elo.py                # per-season-type Elo books w/ MOV + splits
    build.py              # dataset assembly + FEATURE_COLS
  model/
    train.py              # margin XGBRegressor + isotonic + correlation pruning
    predict.py            # upcoming games + last-5-days validation
  notify/
    telegram.py
    format.py
  cli/main.py             # entrypoint
.github/workflows/         # daily-predict + weekly-retrain
artifacts/                 # saved model + metadata
cache/                     # parquet caches of raw + built data
```

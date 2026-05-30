"""Centralized config & paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
ARTIFACT_DIR = ROOT / "artifacts"
CACHE_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

# ---- Files ----
GAMES_PARQUET = CACHE_DIR / "games.parquet"
TEAM_BOX_PARQUET = CACHE_DIR / "team_box.parquet"
PLAYER_SEASON_PARQUET = CACHE_DIR / "player_season.parquet"
PLAYER_GAME_PARQUET = CACHE_DIR / "player_game.parquet"
INJURIES_PARQUET = CACHE_DIR / "injuries.parquet"
DATASET_PARQUET = CACHE_DIR / "dataset.parquet"

MODEL_FILE = ARTIFACT_DIR / "model.joblib"
META_FILE = ARTIFACT_DIR / "model_meta.json"

# ---- Training window ----
# Train on the last 12 seasons. 2019-20 (COVID bubble) is dropped entirely.
SEASONS_BACK = 12
EXCLUDED_SEASONS = {"2019-20"}

# Exponential season weighting:
# the last 2 seasons get full weight (1.0), then each older season is
# discounted by SEASON_DECAY per year. e.g. 0.85 -> 10 yrs old ~ 0.23.
SEASON_FULL_WEIGHT_YEARS = 2
SEASON_DECAY = 0.85

# ---- Feature config ----
ROLL_WINDOW = 6   # rolling team form window (games)

# Elo
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 65.0       # home-court Elo bonus
ELO_SEASON_REGRESS = 0.25 # regress 25% toward 1500 each new season
ELO_MOV_MULT = True       # margin-of-victory multiplier (538-style)

# Star-score weights ("Both, weighted")
STAR_WEIGHTS = {
    "min_per_g": 0.20,
    "usg_pct":   0.15,
    "bpm":       0.25,
    "vorp":      0.20,
    "ws":        0.20,
}

# Polite rate-limiting
NBA_API_SLEEP = 0.6
BR_SLEEP      = 3.5

USER_AGENT = "nba-predictor/0.1 (+research; non-commercial)"

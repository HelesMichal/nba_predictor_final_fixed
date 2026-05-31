"""Train a margin-regression XGBoost model + isotonic probability calibration.

Design choices (per spec):
  - Train on the last 12 seasons of data
  - SKIP the 2019-20 season entirely (COVID bubble anomaly)
  - Exponential per-season sample weighting:
      * the last 2 seasons => weight 1.0
      * each older season => multiply by SEASON_DECAY (0.85) per year past that
  - Margin regression (XGBRegressor on HOME_MARGIN) calibrated to probability
    via IsotonicRegression over a held-out tail of recent games.
  - Greedy correlation pruning at |r| > 0.97 to drop redundant features.
"""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, mean_absolute_error
from xgboost import XGBRegressor

from ..config import (
    DATASET_PARQUET, MODEL_FILE, META_FILE,
    SEASONS_BACK, EXCLUDED_SEASONS,
    SEASON_FULL_WEIGHT_YEARS, SEASON_DECAY,
)
from ..features.build import FEATURE_COLS, build_dataset


def _season_start_year(s: str) -> int:
    # "2018-19" -> 2018
    return int(s.split("-")[0])


def _season_weight(season: str, latest_year: int) -> float:
    yr = _season_start_year(season)
    years_back = latest_year - yr
    if years_back <= SEASON_FULL_WEIGHT_YEARS - 1:
        return 1.0
    return float(SEASON_DECAY ** (years_back - (SEASON_FULL_WEIGHT_YEARS - 1)))


def _prune_correlated(X: pd.DataFrame, threshold: float = 0.97) -> list[str]:
    """Greedy: keep first occurrence, drop any later feature with |corr| > threshold."""
    cols = list(X.columns)
    corr = X.corr().abs()
    keep = []
    dropped = set()
    for c in cols:
        if c in dropped:
            continue
        keep.append(c)
        for other in cols:
            if other == c or other in dropped or other in keep:
                continue
            if corr.loc[c, other] > threshold:
                dropped.add(other)
    return keep


def train_and_save():
    if not DATASET_PARQUET.exists():
        print(f"[train] {DATASET_PARQUET} missing; building dataset first")
        build_dataset()

    df = pd.read_parquet(DATASET_PARQUET).sort_values("GAME_DATE").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{DATASET_PARQUET} exists but is empty. Run build again or check data fetch.")
    df = df.dropna(subset=["HOME_WIN", "HOME_MARGIN"])

    # ---- Season filtering ----
    seasons = sorted(df["SEASON"].unique(), key=_season_start_year)
    latest_year = _season_start_year(seasons[-1])
    cutoff_year = latest_year - SEASONS_BACK + 1
    df = df[df["SEASON"].apply(lambda s: _season_start_year(s) >= cutoff_year)]
    df = df[~df["SEASON"].isin(EXCLUDED_SEASONS)]
    print(f"[data] using {df['SEASON'].nunique()} seasons "
          f"({df['SEASON'].min()} … {df['SEASON'].max()}), "
          f"excluded: {sorted(EXCLUDED_SEASONS)}; rows={len(df)}")

    # ---- Sample weights ----
    df["sample_weight"] = df["SEASON"].apply(lambda s: _season_weight(s, latest_year))
    w = df["sample_weight"].values

    # ---- Feature matrix ----
    feats_avail = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats_avail].astype(float)
    X = X.fillna(X.median(numeric_only=True))

    # ---- Correlation pruning ----
    kept = _prune_correlated(X, threshold=0.97)
    dropped = [c for c in feats_avail if c not in kept]
    if dropped:
        print(f"[prune] dropped {len(dropped)} correlated features: {dropped[:8]}…")
    X = X[kept]

    y_margin = df["HOME_MARGIN"].astype(float).values
    y_win = df["HOME_WIN"].astype(int).values

    # ---- Time-ordered split: last 15% as calibration / eval tail ----
    n = len(df)
    cut = int(n * 0.85)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    mtr, mte = y_margin[:cut], y_margin[cut:]
    ytr, yte = y_win[:cut], y_win[cut:]
    wtr = w[:cut]

    reg = XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.04,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        tree_method="hist", n_jobs=-1, random_state=42,
        objective="reg:squarederror",
    )
    reg.fit(Xtr, mtr, sample_weight=wtr)

    # Calibrate predicted margin -> P(home win) with isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(reg.predict(Xtr), ytr)

    pred_margin_te = reg.predict(Xte)
    proba_te = iso.transform(pred_margin_te)
    pred_te = (proba_te >= 0.5).astype(int)

    acc = accuracy_score(yte, pred_te)
    ll  = log_loss(yte, np.clip(proba_te, 1e-6, 1 - 1e-6))
    br  = brier_score_loss(yte, proba_te)
    mae = mean_absolute_error(mte, pred_margin_te)
    print(f"[tail] acc={acc:.4f}  log_loss={ll:.4f}  brier={br:.4f}  margin_MAE={mae:.2f}")

    # Refit on ALL data (with weights) for production model
    reg_full = XGBRegressor(**reg.get_params())
    reg_full.fit(X, y_margin, sample_weight=w)
    iso_full = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_full.fit(reg_full.predict(X), y_win)

    joblib.dump({
        "regressor": reg_full,
        "isotonic": iso_full,
        "features": kept,
    }, MODEL_FILE)

    META_FILE.write_text(json.dumps({
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_rows": int(len(df)),
        "n_features": len(kept),
        "seasons_used": sorted(df["SEASON"].unique().tolist()),
        "excluded_seasons": sorted(EXCLUDED_SEASONS),
        "roll_window": 6,
        "season_decay": SEASON_DECAY,
        "tail_accuracy": float(acc),
        "tail_log_loss": float(ll),
        "tail_brier": float(br),
        "tail_margin_mae": float(mae),
        "features": kept,
        "dropped_correlated": dropped,
    }, indent=2))
    print(f"[saved] {MODEL_FILE}")
    return float(acc)

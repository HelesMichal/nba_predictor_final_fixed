"""Compute a single 'star score' per player from rate + value stats.

Per spec ("Both, weighted"): combine min/g, USG%, BPM, VORP, WS into one
normalized score per (player, season). Used to:
  (a) quantify how much team strength leans on a few stars
  (b) value the impact of injuries (sum of missing star scores)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..config import STAR_WEIGHTS


def _zscore_per_season(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if not sd or np.isnan(sd):
        return pd.Series(0, index=s.index)
    return (s - mu) / sd


def compute_star_scores(player_adv: pd.DataFrame) -> pd.DataFrame:
    """Returns per-player season rows with `star_score` in [0, ~1] range."""
    df = player_adv.copy()
    feats = list(STAR_WEIGHTS.keys())
    for f in feats:
        if f not in df.columns:
            df[f] = np.nan

    # Per-season z-scores so eras don't dominate
    z = (
        df.groupby("SEASON")[feats]
          .transform(_zscore_per_season)
          .fillna(0)
    )
    weights = np.array([STAR_WEIGHTS[f] for f in feats])
    raw = z.values @ weights
    # squash to [0,1] via logistic so a few superstars don't dominate sums
    df["star_score"] = 1.0 / (1.0 + np.exp(-raw))
    return df[["player", "team", "SEASON", "min_per_g", "star_score"]]


def team_star_strength(star_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (team, season): total star_score and # of high-impact players."""
    g = star_df.groupby(["team", "SEASON"])
    out = g.agg(
        team_star_total=("star_score", "sum"),
        team_star_max=("star_score", "max"),
        team_n_stars=("star_score", lambda s: int((s > 0.75).sum())),
    ).reset_index()
    return out

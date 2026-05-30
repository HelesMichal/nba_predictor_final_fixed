"""Predict today's NBA games + validate against the last 5 days.

Uses the margin-regression + isotonic-calibration bundle written by train.py.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import scoreboardv2

from ..config import MODEL_FILE, DATASET_PARQUET, INJURIES_PARQUET, PLAYER_SEASON_PARQUET
from ..features.build import FEATURE_COLS
from ..data.br_source import fetch_current_injuries
from ..features.star_score import compute_star_scores


def _load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["regressor"], bundle["isotonic"], bundle["features"]


def _predict_proba(reg, iso, X: pd.DataFrame):
    m = reg.predict(X)
    p = iso.transform(m)
    return p, m


def _injury_adjustment(team_abbr: str, side: str) -> dict:
    out = {}
    try:
        inj = pd.read_parquet(INJURIES_PARQUET)
    except FileNotFoundError:
        inj = fetch_current_injuries()
    if inj.empty or "team" not in inj.columns or "player" not in inj.columns:
        return out
    try:
        padv = pd.read_parquet(PLAYER_SEASON_PARQUET)
    except FileNotFoundError:
        return out
    star = compute_star_scores(padv)
    latest = star["SEASON"].max()
    star = star[star["SEASON"] == latest]

    team_inj = inj[inj["team"].str.contains(team_abbr, case=False, na=False)]
    out_status = team_inj["status"].str.lower().str.contains("out|doubt", na=False) \
        if "status" in team_inj.columns else pd.Series(True, index=team_inj.index)
    missing = team_inj[out_status]["player"].tolist()
    impact = star[star["player"].isin(missing)]["star_score"].sum()
    out[f"{side}_team_star_total"] = -float(impact)
    return out


def _build_recent_team_features(team_id: int, on_date: pd.Timestamp,
                                  dataset: pd.DataFrame, side: str) -> dict:
    col_id = f"{side}_TEAM_ID"
    rows = dataset[(dataset[col_id] == team_id) & (dataset["GAME_DATE"] < on_date)]
    if rows.empty:
        return {}
    row = rows.sort_values("GAME_DATE").iloc[-1]
    return {c: row[c] for c in FEATURE_COLS if c.startswith(f"{side}_") and c in row.index}


def predict_upcoming(days_ahead: int = 1):
    reg, iso, feat_list = _load_model()
    dataset = pd.read_parquet(DATASET_PARQUET)
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])

    rows = []
    today = datetime.utcnow().date()
    for d_offset in range(days_ahead):
        d = today + timedelta(days=d_offset)
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=d.strftime("%Y-%m-%d"))
            gh = sb.game_header.get_data_frame()
        except Exception as e:
            print(f"[warn] scoreboard {d}: {e}")
            continue
        if gh.empty:
            continue
        for _, gm in gh.iterrows():
            home_id, away_id = int(gm["HOME_TEAM_ID"]), int(gm["VISITOR_TEAM_ID"])
            ts = pd.Timestamp(d)
            h_feats = _build_recent_team_features(home_id, ts, dataset, "h")
            a_feats = _build_recent_team_features(away_id, ts, dataset, "a")
            if not h_feats or not a_feats:
                continue
            row = {**h_feats, **a_feats}
            if "h_home_wp_prior" in row and "a_away_wp_prior" in row:
                row["home_edge"] = (row.get("h_home_wp_prior") or 0) - (row.get("a_away_wp_prior") or 0)

            h_abbr = gm.get("HOME_TEAM_ABBREVIATION", "")
            a_abbr = gm.get("VISITOR_TEAM_ABBREVIATION", "")
            for k, v in _injury_adjustment(h_abbr, "h").items():
                row[k] = row.get(k, 0) + v
            for k, v in _injury_adjustment(a_abbr, "a").items():
                row[k] = row.get(k, 0) + v

            X = pd.DataFrame([{c: row.get(c, np.nan) for c in feat_list}])
            X = X.fillna(X.median(numeric_only=True)).fillna(0)
            p_arr, m_arr = _predict_proba(reg, iso, X)
            p_home = float(p_arr[0])
            rows.append({
                "date": str(d),
                "home": h_abbr,
                "away": a_abbr,
                "p_home_win": round(p_home, 4),
                "p_away_win": round(1 - p_home, 4),
                "pred_margin": round(float(m_arr[0]), 2),
            })
    return pd.DataFrame(rows)


def validate_last_n_days(n: int = 5) -> dict:
    reg, iso, feat_list = _load_model()
    dataset = pd.read_parquet(DATASET_PARQUET)
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])

    cutoff = dataset["GAME_DATE"].max() - pd.Timedelta(days=n)
    recent = dataset[dataset["GAME_DATE"] > cutoff].copy()
    if recent.empty:
        return {"n_games": 0, "accuracy": None, "log_loss": None, "details": []}

    for c in feat_list:
        if c not in recent.columns:
            recent[c] = np.nan
    X = recent[feat_list].astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    proba, _ = _predict_proba(reg, iso, X)
    pred = (proba >= 0.5).astype(int)
    actual = recent["HOME_WIN"].astype(int).values
    acc = float((pred == actual).mean())
    ll = float(-np.mean(actual * np.log(np.clip(proba, 1e-6, 1)) +
                         (1 - actual) * np.log(np.clip(1 - proba, 1e-6, 1))))
    details = []
    for i, (_, g) in enumerate(recent.iterrows()):
        details.append({
            "date": str(g["GAME_DATE"].date()),
            "home": g.get("h_TEAM_ABBREVIATION", ""),
            "away": g.get("a_TEAM_ABBREVIATION", ""),
            "p_home_win": round(float(proba[i]), 3),
            "actual_home_win": int(actual[i]),
            "correct": bool(pred[i] == actual[i]),
        })
    return {"n_games": int(len(recent)), "accuracy": acc, "log_loss": ll, "details": details}

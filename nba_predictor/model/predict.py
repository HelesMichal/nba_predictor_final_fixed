"""Predict today's NBA games + validate against the last 5 days.

Uses the margin-regression + isotonic-calibration bundle written by train.py.
"""
from __future__ import annotations
from datetime import datetime, date, timedelta
import joblib
import pandas as pd
import numpy as np

from nba_api.stats.endpoints import scoreboardv2, scoreboardv3
import requests

from ..config import MODEL_FILE, DATASET_PARQUET, INJURIES_PARQUET, PLAYER_SEASON_PARQUET
from ..features.build import FEATURE_COLS
from ..data.br_source import fetch_current_injuries
from ..features.star_score import compute_star_scores


def _load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_FILE}. "
            "Run `python -m nba_predictor.cli.main train` first, or use `weekly-retrain` to create the model."
        )
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


def _parse_gamecode_matchup(gamecode: str) -> tuple[str | None, str | None]:
    if not isinstance(gamecode, str) or "/" not in gamecode:
        return None, None
    parts = gamecode.split("/")
    if len(parts) < 2:
        return None, None
    code = parts[-1]
    if len(code) == 6:
        return code[:3], code[3:]
    return None, None


def _build_team_id_map(dataset: pd.DataFrame) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for side in ["h", "a"]:
        abbr_col = f"{side}_TEAM_ABBREVIATION"
        id_col = f"{side}_TEAM_ID"
        if abbr_col in dataset.columns and id_col in dataset.columns:
            valid = dataset[[abbr_col, id_col]].dropna()
            for abbr, team_id in valid.drop_duplicates().values:
                if isinstance(abbr, str) and pd.notna(team_id):
                    mapping.setdefault(abbr, int(team_id))
    return mapping


def _schedule_from_scoreboardv2(d: date, dataset: pd.DataFrame) -> list[dict]:
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=d.strftime("%Y-%m-%d"))
        gh = sb.game_header.get_data_frame()
    except Exception:
        return []
    if gh.empty:
        return []

    team_id_map = _build_team_id_map(dataset)
    rows = []
    for _, gm in gh.iterrows():
        home_id_raw = gm.get("HOME_TEAM_ID")
        away_id_raw = gm.get("VISITOR_TEAM_ID")
        home_abbr = gm.get("HOME_TEAM_ABBREVIATION")
        away_abbr = gm.get("VISITOR_TEAM_ABBREVIATION")
        if pd.isna(home_abbr):
            home_abbr = None
        if pd.isna(away_abbr):
            away_abbr = None

        parsed_home, parsed_away = _parse_gamecode_matchup(gm.get("GAMECODE", ""))
        if not home_abbr and parsed_home:
            home_abbr = parsed_home
        if not away_abbr and parsed_away:
            away_abbr = parsed_away

        if pd.isna(home_id_raw) or pd.isna(away_id_raw):
            if parsed_home and parsed_away:
                home_id_raw = home_id_raw if pd.notna(home_id_raw) else team_id_map.get(home_abbr)
                away_id_raw = away_id_raw if pd.notna(away_id_raw) else team_id_map.get(away_abbr)
        if pd.isna(home_id_raw) or pd.isna(away_id_raw) or home_id_raw is None or away_id_raw is None:
            continue

        rows.append({
            "home_id": int(home_id_raw),
            "away_id": int(away_id_raw),
            "home_abbr": home_abbr or "",
            "away_abbr": away_abbr or "",
            "source": "v2",
        })
    return rows


def _schedule_from_scoreboardv3(d: date) -> list[dict]:
    try:
        sb = scoreboardv3.ScoreboardV3(game_date=d.strftime("%Y-%m-%d"))
        payload = sb.get_json()
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
    except Exception:
        return []
    games = payload.get("scoreboard", {}).get("games", [])
    rows = []
    for game in games:
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        home_id = home.get("teamId")
        away_id = away.get("teamId")
        home_abbr = home.get("teamTricode")
        away_abbr = away.get("teamTricode")
        if home_id is None or away_id is None or not home_abbr or not away_abbr:
            continue
        rows.append({
            "home_id": int(home_id),
            "away_id": int(away_id),
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "source": "v3",
        })
    return rows


def _schedule_from_nba_data_net(d: date) -> list[dict]:
    url = f"https://data.nba.net/10s/prod/v1/{d.strftime('%Y%m%d')}/scoreboard.json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    games = payload.get("games", [])
    rows = []
    for game in games:
        home = game.get("hTeam", {})
        away = game.get("vTeam", {})
        home_id = home.get("teamId")
        away_id = away.get("teamId")
        home_abbr = home.get("triCode")
        away_abbr = away.get("triCode")
        if home_id is None or away_id is None or not home_abbr or not away_abbr:
            continue
        rows.append({
            "home_id": int(home_id),
            "away_id": int(away_id),
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "source": "nba_data_net",
        })
    return rows


def _get_schedule_rows(d: date, dataset: pd.DataFrame) -> tuple[list[dict], bool]:
    rows = _schedule_from_scoreboardv2(d, dataset)
    if rows:
        return rows, True
    rows = _schedule_from_scoreboardv3(d)
    if rows:
        return rows, True
    rows = _schedule_from_nba_data_net(d)
    return rows, bool(rows)


def _predict_for_date(reg, iso, feat_list, dataset: pd.DataFrame, d: date):
    rows = []
    had_placeholder_matchups = False
    schedule_rows, had_schedule = _get_schedule_rows(d, dataset)
    if not schedule_rows:
        return pd.DataFrame(), had_schedule

    for game in schedule_rows:
        home_id = game["home_id"]
        away_id = game["away_id"]
        home_abbr = game["home_abbr"]
        away_abbr = game["away_abbr"]
        ts = pd.Timestamp(d)
        h_feats = _build_recent_team_features(home_id, ts, dataset, "h")
        a_feats = _build_recent_team_features(away_id, ts, dataset, "a")
        if not h_feats or not a_feats:
            continue
        row = {**h_feats, **a_feats}
        if "h_home_wp_prior" in row and "a_away_wp_prior" in row:
            row["home_edge"] = (row.get("h_home_wp_prior") or 0) - (row.get("a_away_wp_prior") or 0)

        for k, v in _injury_adjustment(home_abbr, "h").items():
            row[k] = row.get(k, 0) + v
        for k, v in _injury_adjustment(away_abbr, "a").items():
            row[k] = row.get(k, 0) + v

        X = pd.DataFrame([{c: row.get(c, np.nan) for c in feat_list}])
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        p_arr, m_arr = _predict_proba(reg, iso, X)
        p_home = float(p_arr[0])
        rows.append({
            "date": str(d),
            "home": home_abbr,
            "away": away_abbr,
            "p_home_win": round(p_home, 4),
            "p_away_win": round(1 - p_home, 4),
            "pred_margin": round(float(m_arr[0]), 2),
        })
    return pd.DataFrame(rows), had_schedule


def predict_for_date(d: date):
    reg, iso, feat_list = _load_model()
    dataset = pd.read_parquet(DATASET_PARQUET)
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    df, _ = _predict_for_date(reg, iso, feat_list, dataset, d)
    return df


def predict_for_date_with_status(d: date):
    reg, iso, feat_list = _load_model()
    dataset = pd.read_parquet(DATASET_PARQUET)
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    return _predict_for_date(reg, iso, feat_list, dataset, d)


def predict_upcoming(days_ahead: int = 1):
    reg, iso, feat_list = _load_model()
    dataset = pd.read_parquet(DATASET_PARQUET)
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    rows = []
    today = datetime.utcnow().date()
    for d_offset in range(days_ahead):
        d = today + timedelta(days=d_offset)
        df, _ = _predict_for_date(reg, iso, feat_list, dataset, d)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


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

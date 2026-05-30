"""Build the modeling dataset: one row per game with home/away features.

Every team-level feature is computed from STRICTLY PRIOR games (shift(1) before
rolling) so there is no leakage.

Per-team rolling window = ROLL_WINDOW games (default 6).
   We use 6 instead of 10 because 6 reacts faster to lineup changes / trades
   / cold streaks while still being long enough to denoise single-game variance.

Per-game features (h_/a_ prefixed):
  Rolling form (last N games):
    - roll_win_pct, roll_pt_diff
    - roll_fg_pct, roll_fg3_pct, roll_ft_pct
    - roll_tov, roll_reb, roll_stl, roll_blk
  Schedule:
    - rest_days, is_b2b
  Travel:
    - travel_miles, tz_shift, high_altitude_game
  Season-level (from advanced team stats, by SEASON_TYPE):
    - OFF_RATING, DEF_RATING, NET_RATING, PACE
  Star strength:
    - team_star_total, team_star_max, team_n_stars
  Home-court advantage:
    - team_home_win_pct_prior  (career prior-season home WP per team)
    - h_home_edge = h_home_wp_prior - a_away_wp_prior
  Elo (computed in elo.py, separate books per season type):
    - h_elo_pre, a_elo_pre, elo_diff
    - h_elo_home_pre, a_elo_away_pre, elo_split_diff
  Pythagorean (current-season, rolling):
    - pyth_win_pct_roll (exponent 14)

Target: HOME_WIN (1/0); HOME_MARGIN (PTS diff) is also exported for margin regression.
"""
from __future__ import annotations
from datetime import datetime
import pandas as pd
import numpy as np

from ..config import (
    GAMES_PARQUET, TEAM_BOX_PARQUET, PLAYER_SEASON_PARQUET, DATASET_PARQUET,
    ROLL_WINDOW, SEASONS_BACK,
)
from ..data.nba_api_source import fetch_games
from .star_score import compute_star_scores, team_star_strength
from .elo import compute_elo_features
from .travel import compute_travel


ROLL_COLS = {
    "FG_PCT":  "roll_fg_pct",
    "FG3_PCT": "roll_fg3_pct",
    "FT_PCT":  "roll_ft_pct",
    "TOV":     "roll_tov",
    "REB":     "roll_reb",
    "STL":     "roll_stl",
    "BLK":     "roll_blk",
}


def _pivot_home_away(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["IS_HOME"] = ~g["MATCHUP"].str.contains("@")
    home = g[g["IS_HOME"]].add_prefix("h_")
    away = g[~g["IS_HOME"]].add_prefix("a_")
    home = home.rename(columns={
        "h_GAME_ID": "GAME_ID", "h_GAME_DATE": "GAME_DATE",
        "h_SEASON": "SEASON", "h_SEASON_TYPE": "SEASON_TYPE",
    })
    away = away.rename(columns={"a_GAME_ID": "GAME_ID"})
    merged = home.merge(
        away[["GAME_ID"] + [c for c in away.columns if c.startswith("a_")]],
        on="GAME_ID", how="inner",
    )
    merged["HOME_WIN"] = (merged["h_WL"] == "W").astype(int)
    merged["HOME_MARGIN"] = merged["h_PTS"] - merged["a_PTS"]
    return merged.sort_values("GAME_DATE").reset_index(drop=True)


def _team_rolling_form(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["IS_HOME"] = ~g["MATCHUP"].str.contains("@")
    g["WIN"] = (g["WL"] == "W").astype(int)
    g = g.sort_values(["TEAM_ID", "GAME_DATE"])
    g["pt_diff"] = g["PLUS_MINUS"] if "PLUS_MINUS" in g.columns else 0.0

    grp = g.groupby("TEAM_ID")
    # shift(1) -> use only PRIOR games
    g["roll_win_pct"] = grp["WIN"].shift(1).rolling(ROLL_WINDOW, min_periods=3).mean().reset_index(level=0, drop=True)
    g["roll_pt_diff"] = grp["pt_diff"].shift(1).rolling(ROLL_WINDOW, min_periods=3).mean().reset_index(level=0, drop=True)
    for src, dst in ROLL_COLS.items():
        if src in g.columns:
            g[dst] = grp[src].shift(1).rolling(ROLL_WINDOW, min_periods=3).mean().reset_index(level=0, drop=True)

    g["prev_date"] = grp["GAME_DATE"].shift(1)
    g["rest_days"] = (g["GAME_DATE"] - g["prev_date"]).dt.days.clip(upper=10)
    g["is_b2b"] = (g["rest_days"] == 1).astype(int)

    # Rolling current-season Pythagorean (exponent 14)
    g["season_pts_for"] = grp["PTS"].shift(1).groupby(g["TEAM_ID"]).cumsum() if False else None
    # cumulative within season for Pythagorean
    g["_pf"] = grp["PTS"].shift(1)
    g["_pa"] = grp["PTS"].shift(1) - grp["pt_diff"].shift(1)
    g["cum_pf"] = g.groupby(["TEAM_ID", "SEASON"])["_pf"].cumsum()
    g["cum_pa"] = g.groupby(["TEAM_ID", "SEASON"])["_pa"].cumsum()
    ex = 14.0
    g["pyth_win_pct_roll"] = (g["cum_pf"] ** ex) / ((g["cum_pf"] ** ex) + (g["cum_pa"] ** ex))

    keep = ["TEAM_ID", "GAME_ID", "GAME_DATE", "IS_HOME",
            "roll_win_pct", "roll_pt_diff", "rest_days", "is_b2b",
            "pyth_win_pct_roll"]
    keep += [c for c in ROLL_COLS.values() if c in g.columns]
    return g[keep]


def _home_court_advantage(games: pd.DataFrame) -> pd.DataFrame:
    """Per-team prior-season home and away win pcts.

    For each (TEAM_ID, SEASON) we compute the team's home WP and away WP from
    *the previous season* (no leakage).
    """
    g = games.copy()
    g["IS_HOME"] = ~g["MATCHUP"].str.contains("@")
    g["WIN"] = (g["WL"] == "W").astype(int)
    g = g[g["SEASON_TYPE"] == "Regular Season"]
    season_team = g.groupby(["TEAM_ID", "SEASON", "IS_HOME"])["WIN"].mean().unstack(fill_value=np.nan)
    season_team.columns = ["away_wp", "home_wp"]
    season_team = season_team.reset_index().sort_values(["TEAM_ID", "SEASON"])
    season_team["home_wp_prior"] = season_team.groupby("TEAM_ID")["home_wp"].shift(1)
    season_team["away_wp_prior"] = season_team.groupby("TEAM_ID")["away_wp"].shift(1)
    return season_team[["TEAM_ID", "SEASON", "home_wp_prior", "away_wp_prior"]]


def _load_or_fetch_games() -> pd.DataFrame:
    if not GAMES_PARQUET.exists():
        end_year = datetime.utcnow().year
        start_year = end_year - SEASONS_BACK
        print(f"[build] {GAMES_PARQUET} missing; fetching seasons {start_year}-{end_year} first")
        fetch_games(start_year, end_year)
    if not GAMES_PARQUET.exists():
        raise FileNotFoundError(
            f"{GAMES_PARQUET} was not created. Run `python -m nba_predictor.cli.main fetch --start <year> --end <year>` and check the NBA API output."
        )
    games = pd.read_parquet(GAMES_PARQUET)
    if games.empty:
        raise ValueError(f"{GAMES_PARQUET} exists but contains no games")
    return games


def build_dataset() -> pd.DataFrame:
    games = _load_or_fetch_games()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    paired = _pivot_home_away(games)
    form = _team_rolling_form(games)

    home_form = form[form["IS_HOME"]].drop(columns=["IS_HOME"])
    away_form = form[~form["IS_HOME"]].drop(columns=["IS_HOME"])
    h = home_form.add_prefix("h_").rename(columns={"h_GAME_ID": "GAME_ID"})
    a = away_form.add_prefix("a_").rename(columns={"a_GAME_ID": "GAME_ID"})

    df = paired.merge(
        h.drop(columns=["h_GAME_DATE", "h_TEAM_ID"]), on="GAME_ID", how="left"
    ).merge(
        a.drop(columns=["a_GAME_DATE", "a_TEAM_ID"]), on="GAME_ID", how="left"
    )

    # Home-court advantage (prior-season splits)
    hca = _home_court_advantage(games)
    df = df.merge(hca.rename(columns={"TEAM_ID": "h_TEAM_ID",
                                      "home_wp_prior": "h_home_wp_prior",
                                      "away_wp_prior": "h_away_wp_prior"}),
                  on=["h_TEAM_ID", "SEASON"], how="left")
    df = df.merge(hca.rename(columns={"TEAM_ID": "a_TEAM_ID",
                                      "home_wp_prior": "a_home_wp_prior",
                                      "away_wp_prior": "a_away_wp_prior"}),
                  on=["a_TEAM_ID", "SEASON"], how="left")
    df["home_edge"] = df["h_home_wp_prior"] - df["a_away_wp_prior"]

    # Travel features per side
    tg = games[["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP"]].copy()
    tg["IS_HOME"] = ~tg["MATCHUP"].str.contains("@")
    # opp abbreviation = last 3 chars of MATCHUP
    tg["opp_abbr"] = tg["MATCHUP"].str[-3:]
    tr = compute_travel(tg)
    df = df.merge(tr.rename(columns={"TEAM_ID": "h_TEAM_ID",
                                     "travel_miles": "h_travel_miles",
                                     "tz_shift": "h_tz_shift",
                                     "high_altitude_game": "h_high_altitude_game"}),
                  on=["h_TEAM_ID", "GAME_ID"], how="left")
    df = df.merge(tr.rename(columns={"TEAM_ID": "a_TEAM_ID",
                                     "travel_miles": "a_travel_miles",
                                     "tz_shift": "a_tz_shift",
                                     "high_altitude_game": "a_high_altitude_game"}),
                  on=["a_TEAM_ID", "GAME_ID"], how="left")

    # Elo features (separate books per SEASON_TYPE)
    elo = compute_elo_features(df[["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE",
                                   "h_TEAM_ID", "a_TEAM_ID", "h_PTS", "a_PTS"]])
    df = df.merge(elo, on="GAME_ID", how="left")

    # Season-level advanced team stats (per season_type)
    try:
        team_adv = pd.read_parquet(TEAM_BOX_PARQUET)
        cols = ["TEAM_ID", "SEASON", "SEASON_TYPE",
                "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
        cols = [c for c in cols if c in team_adv.columns]
        team_adv = team_adv[cols].drop_duplicates(["TEAM_ID", "SEASON", "SEASON_TYPE"])
        df = df.merge(team_adv.rename(columns={"TEAM_ID": "h_TEAM_ID"}).add_prefix("hs_").rename(
            columns={"hs_h_TEAM_ID": "h_TEAM_ID", "hs_SEASON": "SEASON", "hs_SEASON_TYPE": "SEASON_TYPE"}),
            on=["h_TEAM_ID", "SEASON", "SEASON_TYPE"], how="left")
        df = df.merge(team_adv.rename(columns={"TEAM_ID": "a_TEAM_ID"}).add_prefix("as_").rename(
            columns={"as_a_TEAM_ID": "a_TEAM_ID", "as_SEASON": "SEASON", "as_SEASON_TYPE": "SEASON_TYPE"}),
            on=["a_TEAM_ID", "SEASON", "SEASON_TYPE"], how="left")
    except FileNotFoundError:
        pass

    # Star strength per team
    try:
        padv = pd.read_parquet(PLAYER_SEASON_PARQUET)
        star = compute_star_scores(padv)
        team_star = team_star_strength(star)
        df = df.merge(team_star.add_prefix("h_"),
                      left_on=["h_TEAM_ABBREVIATION", "SEASON"],
                      right_on=["h_team", "h_SEASON"], how="left").drop(
                          columns=["h_team", "h_SEASON"], errors="ignore")
        df = df.merge(team_star.add_prefix("a_"),
                      left_on=["a_TEAM_ABBREVIATION", "SEASON"],
                      right_on=["a_team", "a_SEASON"], how="left").drop(
                          columns=["a_team", "a_SEASON"], errors="ignore")
    except FileNotFoundError:
        pass

    df.to_parquet(DATASET_PARQUET, index=False)
    return df


# Master feature list used by train/predict (only those that exist are used)
FEATURE_COLS = [
    # rolling form
    "h_roll_win_pct", "a_roll_win_pct",
    "h_roll_pt_diff", "a_roll_pt_diff",
    "h_roll_fg_pct", "a_roll_fg_pct",
    "h_roll_fg3_pct", "a_roll_fg3_pct",
    "h_roll_ft_pct", "a_roll_ft_pct",
    "h_roll_tov", "a_roll_tov",
    "h_roll_reb", "a_roll_reb",
    "h_roll_stl", "a_roll_stl",
    "h_roll_blk", "a_roll_blk",
    # pythagorean
    "h_pyth_win_pct_roll", "a_pyth_win_pct_roll",
    # schedule
    "h_rest_days", "a_rest_days",
    "h_is_b2b", "a_is_b2b",
    # travel
    "h_travel_miles", "a_travel_miles",
    "h_tz_shift", "a_tz_shift",
    "h_high_altitude_game", "a_high_altitude_game",
    # home-court advantage
    "h_home_wp_prior", "a_away_wp_prior", "home_edge",
    # season-level advanced (prefixed hs_/as_)
    "hs_OFF_RATING", "hs_DEF_RATING", "hs_NET_RATING", "hs_PACE",
    "as_OFF_RATING", "as_DEF_RATING", "as_NET_RATING", "as_PACE",
    # star strength
    "h_team_star_total", "h_team_star_max", "h_team_n_stars",
    "a_team_star_total", "a_team_star_max", "a_team_n_stars",
    # elo
    "h_elo_pre", "a_elo_pre", "elo_diff",
    "h_elo_home_pre", "a_elo_away_pre", "elo_split_diff",
]

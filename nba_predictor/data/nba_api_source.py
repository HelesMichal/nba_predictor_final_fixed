"""Pull schedule + team box scores from nba_api.

Covers regular season AND playoffs (separate queries, tagged with `season_type`).
"""
from __future__ import annotations
import time
import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import leaguegamefinder, leaguedashteamstats
from nba_api.stats.static import teams as static_teams

from ..config import NBA_API_SLEEP, GAMES_PARQUET, TEAM_BOX_PARQUET


def _merge_with_existing(new_rows: list[pd.DataFrame], path, unique_cols: list[str]) -> pd.DataFrame:
    frames = []
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if not existing.empty:
                frames.append(existing)
        except Exception as e:
            print(f"[warn] could not read existing cache {path}: {e}")
    frames.extend([df for df in new_rows if df is not None and not df.empty])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    present_unique = [c for c in unique_cols if c in out.columns]
    if present_unique:
        out = out.drop_duplicates(present_unique, keep="last")
    return out


def _season_str(start_year: int) -> str:
    """2018 -> '2018-19'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_games(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch every team-game row for the seasons [start_year, end_year] inclusive.

    Each NBA game produces 2 rows (one per team). We pivot into one row per game
    with home/away columns later in the dataset builder.
    """
    rows = []
    for yr in tqdm(range(start_year, end_year + 1), desc="seasons"):
        season = _season_str(yr)
        for stype in ("Regular Season", "Playoffs"):
            try:
                gf = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable=stype,
                    league_id_nullable="00",
                )
                df = gf.get_data_frames()[0]
                if df.empty:
                    continue
                df["SEASON"] = season
                df["SEASON_TYPE"] = stype
                rows.append(df)
            except Exception as e:
                print(f"[warn] {season} {stype}: {e}")
            time.sleep(NBA_API_SLEEP)

    out = _merge_with_existing(rows, GAMES_PARQUET, ["GAME_ID", "TEAM_ID"])
    if out.empty:
        print("[warn] no game rows fetched and no existing games cache is available")
        return out
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])
    out = out.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"], kind="stable")
    out.to_parquet(GAMES_PARQUET, index=False)
    return out


def fetch_team_season_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """Season-level advanced team stats (Off/Def rating, pace, NetRtg).

    Stored per (season, season_type) so playoff-vs-regular splits are honored.
    """
    rows = []
    for yr in tqdm(range(start_year, end_year + 1), desc="team-stats"):
        season = _season_str(yr)
        for stype in ("Regular Season", "Playoffs"):
            try:
                d = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    season_type_all_star=stype,
                    measure_type_detailed_defense="Advanced",
                ).get_data_frames()[0]
                d["SEASON"] = season
                d["SEASON_TYPE"] = stype
                rows.append(d)
            except Exception as e:
                print(f"[warn] team-stats {season} {stype}: {e}")
            time.sleep(NBA_API_SLEEP)

    out = _merge_with_existing(rows, TEAM_BOX_PARQUET, ["TEAM_ID", "SEASON", "SEASON_TYPE"])
    if out.empty:
        print("[warn] no team-stat rows fetched and no existing team stats cache is available")
        return out
    out.to_parquet(TEAM_BOX_PARQUET, index=False)
    return out


def team_id_to_abbrev() -> dict:
    return {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

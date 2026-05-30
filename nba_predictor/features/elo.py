"""Elo ratings, computed game-by-game with leakage-safe pre-game snapshots.

Per the spec:
  - separate Elo for Regular Season vs Playoffs
  - home-court Elo bonus
  - per-team home Elo / away Elo splits
  - margin-of-victory multiplier (538-style)
  - mild season-to-season regression toward the mean
"""
from __future__ import annotations
from collections import defaultdict
import math
import pandas as pd

from ..config import (
    ELO_START, ELO_K, ELO_HOME_ADV, ELO_SEASON_REGRESS, ELO_MOV_MULT,
)


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _mov_mult(margin: int, elo_diff: float) -> float:
    if not ELO_MOV_MULT:
        return 1.0
    # 538 formula: log(|MOV|+1) * 2.2 / (elo_diff_winner * 0.001 + 2.2)
    return math.log(abs(margin) + 1.0) * 2.2 / (elo_diff * 0.001 + 2.2)


def compute_elo_features(games_paired: pd.DataFrame) -> pd.DataFrame:
    """Input: one row per game with h_/a_ TEAM_ID, PTS, SEASON, SEASON_TYPE.
    Output: same index with pre-game Elo features merged in.
    """
    df = games_paired.sort_values("GAME_DATE").reset_index(drop=True)

    # Separate Elo books per season_type
    reg_overall = defaultdict(lambda: ELO_START)
    reg_home    = defaultdict(lambda: ELO_START)
    reg_away    = defaultdict(lambda: ELO_START)
    po_overall  = defaultdict(lambda: ELO_START)

    last_season = {"Regular Season": None, "Playoffs": None}

    rows = []
    for r in df.itertuples(index=False):
        st = r.SEASON_TYPE
        season = r.SEASON
        h, a = r.h_TEAM_ID, r.a_TEAM_ID

        book_overall = reg_overall if st == "Regular Season" else po_overall

        # season turnover: regress all teams toward 1500
        if last_season[st] != season:
            if last_season[st] is not None:
                for t in list(book_overall.keys()):
                    book_overall[t] = (
                        ELO_START * ELO_SEASON_REGRESS
                        + book_overall[t] * (1 - ELO_SEASON_REGRESS)
                    )
                if st == "Regular Season":
                    for t in list(reg_home.keys()):
                        reg_home[t] = ELO_START * ELO_SEASON_REGRESS + reg_home[t] * (1 - ELO_SEASON_REGRESS)
                    for t in list(reg_away.keys()):
                        reg_away[t] = ELO_START * ELO_SEASON_REGRESS + reg_away[t] * (1 - ELO_SEASON_REGRESS)
            last_season[st] = season

        h_elo = book_overall[h]
        a_elo = book_overall[a]
        h_elo_home = reg_home[h] if st == "Regular Season" else h_elo
        a_elo_away = reg_away[a] if st == "Regular Season" else a_elo

        rows.append({
            "GAME_ID": r.GAME_ID,
            "h_elo_pre": h_elo,
            "a_elo_pre": a_elo,
            "elo_diff": (h_elo + ELO_HOME_ADV) - a_elo,
            "h_elo_home_pre": h_elo_home,
            "a_elo_away_pre": a_elo_away,
            "elo_split_diff": (h_elo_home + ELO_HOME_ADV) - a_elo_away,
        })

        # update with actual result (post-game)
        margin = int(r.h_PTS - r.a_PTS) if pd.notna(r.h_PTS) and pd.notna(r.a_PTS) else 0
        home_win = 1.0 if margin > 0 else 0.0
        exp_h = _expected(h_elo + ELO_HOME_ADV, a_elo)
        winner_diff = (h_elo + ELO_HOME_ADV - a_elo) if home_win else (a_elo - h_elo - ELO_HOME_ADV)
        mult = _mov_mult(margin if margin != 0 else 1, winner_diff)
        delta = ELO_K * mult * (home_win - exp_h)
        book_overall[h] = h_elo + delta
        book_overall[a] = a_elo - delta

        if st == "Regular Season":
            # home/away split books update on their respective sides only
            exp_h_split = _expected(h_elo_home + ELO_HOME_ADV, a_elo_away)
            delta_split = ELO_K * mult * (home_win - exp_h_split)
            reg_home[h] = h_elo_home + delta_split
            reg_away[a] = a_elo_away - delta_split

    return pd.DataFrame(rows)

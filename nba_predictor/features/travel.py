"""Travel & altitude features.

Pre-game per-team: miles since last game, time-zone shift, high-altitude flag.
Strictly past-only (no leakage).
"""
from __future__ import annotations
import math
import pandas as pd

# Approximate arena coords (lat, lon, tz_offset_hours_from_UTC, elevation_ft)
ARENAS = {
    "ATL": (33.7573, -84.3963, -5, 1050),
    "BOS": (42.3662, -71.0621, -5, 141),
    "BKN": (40.6826, -73.9754, -5, 30),
    "CHA": (35.2251, -80.8392, -5, 751),
    "CHI": (41.8807, -87.6742, -6, 594),
    "CLE": (41.4965, -81.6882, -5, 653),
    "DAL": (32.7905, -96.8104, -6, 430),
    "DEN": (39.7487, -105.0077, -7, 5280),
    "DET": (42.6960, -83.2454, -5, 745),
    "GSW": (37.7680, -122.3877, -8, 13),
    "HOU": (29.7508, -95.3621, -6, 49),
    "IND": (39.7639, -86.1555, -5, 717),
    "LAC": (33.9430, -118.3411, -8, 89),
    "LAL": (34.0430, -118.2673, -8, 233),
    "MEM": (35.1382, -90.0506, -6, 337),
    "MIA": (25.7814, -80.1870, -5, 6),
    "MIL": (43.0451, -87.9173, -6, 617),
    "MIN": (44.9795, -93.2761, -6, 830),
    "NOP": (29.9490, -90.0821, -6, 7),
    "NYK": (40.7505, -73.9934, -5, 30),
    "OKC": (35.4634, -97.5151, -6, 1201),
    "ORL": (28.5392, -81.3839, -5, 82),
    "PHI": (39.9012, -75.1720, -5, 39),
    "PHX": (33.4457, -112.0712, -7, 1086),
    "POR": (45.5316, -122.6668, -8, 50),
    "SAC": (38.5802, -121.4997, -8, 30),
    "SAS": (29.4271, -98.4375, -6, 650),
    "TOR": (43.6435, -79.3791, -5, 250),
    "UTA": (40.7683, -111.9011, -7, 4226),
    "WAS": (38.8981, -77.0209, -5, 25),
}

HIGH_ALT = {"DEN", "UTA"}


def _haversine(a, b) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 3958.7613  # miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))


def compute_travel(team_games: pd.DataFrame) -> pd.DataFrame:
    """team_games: cols [TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, IS_HOME, opp_abbr].

    Returns travel_miles, tz_shift, high_altitude_game keyed by (TEAM_ID, GAME_ID).
    """
    g = team_games.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    out = []
    for tid, grp in g.groupby("TEAM_ID"):
        prev_loc = None
        prev_tz = None
        for r in grp.itertuples(index=False):
            cur_abbr = r.TEAM_ABBREVIATION if r.IS_HOME else r.opp_abbr
            cur = ARENAS.get(cur_abbr)
            if cur is None:
                out.append((tid, r.GAME_ID, 0.0, 0, 0))
                continue
            miles = _haversine(prev_loc[:2], cur[:2]) if prev_loc else 0.0
            tz_sh = abs(cur[2] - prev_tz) if prev_tz is not None else 0
            alt_flag = int(cur_abbr in HIGH_ALT)
            out.append((tid, r.GAME_ID, miles, tz_sh, alt_flag))
            prev_loc = cur
            prev_tz = cur[2]
    return pd.DataFrame(out, columns=["TEAM_ID", "GAME_ID", "travel_miles", "tz_shift", "high_altitude_game"])

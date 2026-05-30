"""Scrape Basketball-Reference for player advanced stats (BPM/VORP/WS) and injuries.

We use BR for value metrics (cleaner than nba_api's exposure) and for the daily
injury report.
"""
from __future__ import annotations
import time
import io
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm

from ..config import BR_SLEEP, USER_AGENT, PLAYER_SEASON_PARQUET, INJURIES_PARQUET

BR_BASE = "https://www.basketball-reference.com"
HEADERS = {"User-Agent": USER_AGENT}


def _get(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(BR_SLEEP)
    return r.text


def _read_table(html: str, table_id: str) -> pd.DataFrame:
    """BR hides many tables in HTML comments; parse them out."""
    soup = BeautifulSoup(html, "lxml")
    node = soup.find("table", id=table_id)
    if node is None:
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            if table_id in c:
                node = BeautifulSoup(c, "lxml").find("table", id=table_id)
                if node is not None:
                    break
    if node is None:
        return pd.DataFrame()
    return pd.read_html(io.StringIO(str(node)))[0]


def fetch_player_advanced(start_year: int, end_year: int) -> pd.DataFrame:
    """Per-player season advanced stats (BPM, VORP, WS, USG%, MP)."""
    frames = []
    for yr in tqdm(range(start_year, end_year + 1), desc="adv-seasons"):
        season_end = yr + 1  # BR uses end-year (e.g. 2019 for 2018-19)
        url = f"{BR_BASE}/leagues/NBA_{season_end}_advanced.html"
        try:
            html = _get(url)
            df = _read_table(html, "advanced_stats")
            if df.empty:
                df = _read_table(html, "advanced")
        except Exception as e:
            print(f"[warn] advanced {season_end}: {e}")
            continue
        if df.empty:
            continue
        df = df[df["Player"] != "Player"].copy()
        df["SEASON"] = f"{yr}-{str(season_end)[-2:]}"
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # normalize columns
    rename = {
        "Player": "player",
        "Tm": "team",
        "Team": "team",
        "MP": "mp",
        "USG%": "usg_pct",
        "BPM": "bpm",
        "VORP": "vorp",
        "WS": "ws",
        "G": "g",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    for col in ["mp", "usg_pct", "bpm", "vorp", "ws", "g"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["min_per_g"] = (out["mp"] / out["g"]).where(out.get("g", 0) > 0)
    out.to_parquet(PLAYER_SEASON_PARQUET, index=False)
    return out


def fetch_current_injuries() -> pd.DataFrame:
    """BR's current injury report. Used at predict time."""
    url = f"{BR_BASE}/friv/injuries.fcgi"
    try:
        html = _get(url)
        df = _read_table(html, "injuries")
    except Exception as e:
        print(f"[warn] injuries: {e}")
        return pd.DataFrame()
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    df.to_parquet(INJURIES_PARQUET, index=False)
    return df

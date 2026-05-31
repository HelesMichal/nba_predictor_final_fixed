"""Generate predictions and (optionally) push them to Telegram.

Usage:
    python -m nba_predictor.predict --mode today
    python -m nba_predictor.predict --mode tomorrow --chat-id 123456789
    python -m nba_predictor.predict --mode week
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

from .config import MODEL_PATH
from .features.build import build_feature_frame, ensure_cache
from .notify.telegram import send_message

CET = dt.timezone(dt.timedelta(hours=1))  # display only; DST handled by GH cron


def _load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m nba_predictor.model.train` first."
        )
    return joblib.load(MODEL_PATH)


def _date_range(mode: str) -> List[dt.date]:
    today = dt.datetime.now(CET).date()
    if mode == "today":
        return [today]
    if mode == "tomorrow":
        return [today + dt.timedelta(days=1)]
    if mode == "week":
        return [today + dt.timedelta(days=i) for i in range(7)]
    raise ValueError(f"unknown mode: {mode}")


def _accuracy_last_n(days: int = 5) -> Optional[float]:
    """Return accuracy over the last ``days`` of completed games, if available."""
    try:
        from .config import CACHE_DIR
        games = pd.read_parquet(Path(CACHE_DIR) / "games.parquet")
        preds = pd.read_parquet(Path(CACHE_DIR) / "predictions_log.parquet")
    except Exception:
        return None
    cutoff = dt.date.today() - dt.timedelta(days=days)
    games = games[games["GAME_DATE"].dt.date >= cutoff]
    merged = preds.merge(games[["GAME_ID", "HOME_WIN"]], on="GAME_ID")
    if merged.empty:
        return None
    merged["correct"] = (merged["PRED_HOME_WIN_PROB"] >= 0.5) == (merged["HOME_WIN"] == 1)
    return float(merged["correct"].mean())


def _format(games: pd.DataFrame, mode: str) -> str:
    if games.empty:
        return f"<b>NBA Predictions ({mode})</b>\nNo games scheduled."
    lines = [f"<b>NBA Predictions — {mode.capitalize()}</b>"]
    current_date = None
    for _, g in games.sort_values(["GAME_DATE", "GAME_TIME"]).iterrows():
        d = g["GAME_DATE"].strftime("%a %d %b")
        if d != current_date:
            lines.append(f"\n<b>{d}</b>")
            current_date = d
        p = g["PRED_HOME_WIN_PROB"]
        winner = g["HOME_TEAM"] if p >= 0.5 else g["AWAY_TEAM"]
        conf = p if p >= 0.5 else 1 - p
        lines.append(
            f"  {g['AWAY_TEAM']} @ {g['HOME_TEAM']} → <b>{winner}</b> ({conf:.0%})"
        )
    acc = _accuracy_last_n(5)
    if acc is not None:
        lines.append(f"\n<i>Last 5-day accuracy: {acc:.1%}</i>")
    return "\n".join(lines)


def predict(mode: str = "today", chat_id: Optional[str] = None, *, send: bool = True) -> str:
    ensure_cache()
    model = _load_model()
    dates = _date_range(mode)
    frame = build_feature_frame(dates=dates, for_prediction=True)
    if frame.empty:
        text = f"<b>NBA Predictions — {mode.capitalize()}</b>\nNo games scheduled."
    else:
        frame["PRED_HOME_WIN_PROB"] = model.predict_proba(frame)
        text = _format(frame, mode)
    if send:
        send_message(text, chat_id=chat_id)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["today", "tomorrow", "week"], default="today")
    parser.add_argument("--chat-id", default=None, help="Send only to this chat (default: all)")
    parser.add_argument("--no-send", action="store_true")
    args = parser.parse_args()
    text = predict(args.mode, chat_id=args.chat_id, send=not args.no_send)
    print(text)


if __name__ == "__main__":
    main()

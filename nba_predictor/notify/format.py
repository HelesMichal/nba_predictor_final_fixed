"""Formatting helpers for daily Telegram digests."""
from __future__ import annotations

from datetime import datetime, timezone
import html


def _esc(s) -> str:
    return html.escape(str(s))


def format_daily_digest(predictions_df, validation: dict) -> str:
    """Build the daily Telegram message.

    predictions_df: DataFrame with columns date, home, away, p_home_win, p_away_win
                    (optional: pred_margin)
    validation: dict returned by validate_last_n_days(5)
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"<b>🏀 NBA predictions — {today}</b>", ""]

    # --- Today's games ---
    if predictions_df is None or predictions_df.empty:
        lines.append("<i>No games scheduled today.</i>")
    else:
        lines.append("<b>Today's matchups</b>")
        for _, r in predictions_df.iterrows():
            ph = float(r["p_home_win"]) * 100
            pa = float(r["p_away_win"]) * 100
            fav = "🏠" if ph >= pa else "✈️"
            margin = ""
            if "pred_margin" in r and r["pred_margin"] is not None:
                try:
                    m = float(r["pred_margin"])
                    sign = "+" if m >= 0 else ""
                    margin = f"  <i>(H {sign}{m:.1f})</i>"
                except Exception:
                    pass
            lines.append(
                f"{fav} <b>{_esc(r['away'])}</b> @ <b>{_esc(r['home'])}</b> — "
                f"H {ph:.0f}% / A {pa:.0f}%{margin}"
            )

    lines.append("")
    # --- Last 5 days accuracy ---
    lines.append("<b>Last 5 days accuracy</b>")
    if not validation or validation.get("accuracy") is None:
        lines.append("<i>No recent completed games.</i>")
    else:
        acc = validation["accuracy"] * 100
        n = validation["n_games"]
        ll = validation.get("log_loss")
        ll_s = f"  log-loss {ll:.3f}" if ll is not None else ""
        lines.append(f"✅ {acc:.1f}% over {n} games{ll_s}")
        # show last few
        for d in (validation.get("details") or [])[-5:]:
            mark = "✓" if d["correct"] else "✗"
            lines.append(
                f"  {mark} {_esc(d['date'])} {_esc(d['away'])} @ {_esc(d['home'])} "
                f"— p(H)={float(d['p_home_win']):.2f}"
            )

    return "\n".join(lines)

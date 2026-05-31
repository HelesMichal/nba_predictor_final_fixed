"""Telegram notification helpers.

Supports broadcasting to multiple chats (multiple devices / users) via the
``TELEGRAM_CHAT_IDS`` env var (comma separated). ``TELEGRAM_CHAT_ID``
(singular) is kept as a fallback for backward compatibility.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional

import requests

TELEGRAM_API = "https://api.telegram.org"
_MAX_LEN = 4000  # Telegram hard limit is 4096; leave headroom


def _token() -> Optional[str]:
    return os.environ.get("TELEGRAM_BOT_TOKEN")


def get_chat_ids() -> List[str]:
    """Return the configured chat IDs (de-duplicated, order preserved)."""
    raw = os.environ.get("TELEGRAM_CHAT_IDS") or os.environ.get("TELEGRAM_CHAT_ID") or ""
    ids: List[str] = []
    for piece in raw.replace(";", ",").split(","):
        piece = piece.strip()
        if piece and piece not in ids:
            ids.append(piece)
    return ids


def _chunks(text: str, size: int = _MAX_LEN) -> Iterable[str]:
    if len(text) <= size:
        yield text
        return
    buf: List[str] = []
    length = 0
    for line in text.splitlines(keepends=True):
        if length + len(line) > size and buf:
            yield "".join(buf)
            buf, length = [], 0
        buf.append(line)
        length += len(line)
    if buf:
        yield "".join(buf)


def _post(method: str, payload: dict, *, attempts: int = 4) -> dict:
    token = _token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    url = f"{TELEGRAM_API}/bot{token}/{method}"
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 429:
                retry = int(r.json().get("parameters", {}).get("retry_after", 2))
                time.sleep(retry + 1)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** i)
    raise RuntimeError(f"Telegram {method} failed after {attempts} tries: {last_err}")


def send_message(text: str, chat_id: Optional[str] = None) -> None:
    """Send ``text`` to ``chat_id`` (default: every configured chat)."""
    if not _token():
        print("[telegram] TELEGRAM_BOT_TOKEN not set – skipping notification.")
        return

    targets = [chat_id] if chat_id else get_chat_ids()
    if not targets:
        print("[telegram] No TELEGRAM_CHAT_IDS configured – skipping notification.")
        return

    for cid in targets:
        for chunk in _chunks(text):
            try:
                _post(
                    "sendMessage",
                    {
                        "chat_id": cid,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
            except Exception as e:  # noqa: BLE001
                print(f"[telegram] failed for chat {cid}: {e}")


__all__ = ["send_message", "get_chat_ids"]

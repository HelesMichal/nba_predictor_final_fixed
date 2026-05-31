"""Telegram notification helper.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment
(set as GitHub Actions secrets in CI, or exported locally).
"""
from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
import json


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def send_message(text: str, *, parse_mode: str = "HTML", disable_preview: bool = True) -> dict:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Telegram hard-limits messages to 4096 chars. Split if needed.
    chunks = _split(text, 3900)
    last = {}
    for chunk in chunks:
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": parse_mode,
            "disable_web_page_preview": "true" if disable_preview else "false",
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            last = json.loads(body)
            if not last.get("ok"):
                raise RuntimeError(f"Telegram error: {body}")
    return last


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > limit:
            out.append(buf); buf = ""
        buf += line
    if buf:
        out.append(buf)
    return out


if __name__ == "__main__":
    send_message(" ".join(sys.argv[1:]) or "test ping from nba_predictor")

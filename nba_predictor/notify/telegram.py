"""Telegram notification helper.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment
(set as GitHub Actions secrets in CI, or exported locally).

If either var is missing we print a warning and exit 0 so a CI
notification step never masks the real upstream error.
"""
from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
import urllib.error
import json
import time


def _env_optional(name: str) -> str:
    return os.environ.get(name, "").strip()


def send_message(text: str, *, parse_mode: str = "HTML", disable_preview: bool = True) -> dict:
    token = _env_optional("TELEGRAM_BOT_TOKEN")
    chat_id = _env_optional("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping notification.")
        print("[telegram] Add both as repository secrets (Settings -> Secrets and variables -> Actions).")
        return {"ok": False, "skipped": True}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = _split(text, 3900)
    last: dict = {}
    for chunk in chunks:
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": parse_mode,
            "disable_web_page_preview": "true" if disable_preview else "false",
        }).encode()
        last = _post_with_retry(url, data)
    return last


def _post_with_retry(url: str, data: bytes, attempts: int = 3) -> dict:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
                parsed = json.loads(body)
                if not parsed.get("ok"):
                    raise RuntimeError(f"Telegram error: {body}")
                return parsed
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            if i == attempts - 1:
                print(f"[telegram] send failed after {attempts} attempts: {e}")
                return {"ok": False, "error": str(e)}
            time.sleep(2 ** i)
    return {"ok": False}


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

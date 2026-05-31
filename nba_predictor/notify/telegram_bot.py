"""Telegram bot polling for command-based predictions."""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..model.predict import predict_for_date, predict_for_date_with_status
from .format import format_predictions, format_week_predictions

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
POLL_OFFSET_FILE = Path(__file__).resolve().parent.parent / "cache" / "telegram_update_id.txt"


def _api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def _load_last_update_id() -> int | None:
    try:
        return int(POLL_OFFSET_FILE.read_text().strip())
    except FileNotFoundError:
        return None
    except ValueError:
        return None


def _save_last_update_id(update_id: int) -> None:
    POLL_OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    POLL_OFFSET_FILE.write_text(str(update_id))


def _http_post(url: str, payload: dict) -> dict:
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_updates(offset: int | None = None, timeout: int = 30) -> dict:
    url = _api_url("getUpdates")
    params = {"timeout": timeout, "allowed_updates": json.dumps(["message"])}
    if offset is not None:
        params["offset"] = offset
    return _http_post(url, params)


def send_reply(chat_id: str, text: str) -> dict:
    url = _api_url("sendMessage")
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }
    return _http_post(url, payload)


def _build_response_text(command: str, target_date: date) -> str:
    df, had_placeholder = predict_for_date_with_status(target_date)
    text = format_predictions(df, target_date.strftime("%Y-%m-%d"))
    if df.empty and had_placeholder:
        text += (
            "\n\n<i>Games are scheduled for this date, but matchup details are not yet available "
            "from the NBA API.</i>"
        )
    return text


def _build_week_response_text() -> str:
    today = date.today()
    dfs = []
    had_placeholder = False
    for offset in range(7):
        target = today + timedelta(days=offset)
        df, placeholder = predict_for_date_with_status(target)
        dfs.append(df)
        had_placeholder = had_placeholder or placeholder
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    text = format_week_predictions(df, today.strftime("%Y-%m-%d"), (today + timedelta(days=6)).strftime("%Y-%m-%d"))
    if df.empty and had_placeholder:
        text += (
            "\n\n<i>Games are scheduled for this week, but matchup details are not yet available "
            "from the NBA API.</i>"
        )
    return text


def _normalize_command(text: str) -> str:
    return text.strip().lower().split()[0] if text else ""


def handle_message(message: dict) -> None:
    chat = message.get("chat", {})
    chat_id = str(chat.get("id", ""))
    text = message.get("text", "")
    if not chat_id or not text:
        return

    command = _normalize_command(text)
    try:
        if command in {"/today", "today"}:
            response = _build_response_text(command, date.today())
        elif command in {"/tomorrow", "tomorrow"}:
            response = _build_response_text(command, date.today() + timedelta(days=1))
        elif command == "/week" or command == "week":
            response = _build_week_response_text()
        elif command in {"/help", "help", "/start", "start"}:
            response = (
                "Send /today for today\'s NBA predictions, "
                "/tomorrow for tomorrow\'s predictions, "
                "/week for the next 7 days, or "
                "/help for this message."
            )
        else:
            response = (
                "I can send NBA predictions for you.\n"
                "Use /today, /tomorrow, or /week.\n"
                "If you want the bot to answer, run the bot process with the correct Telegram token."
            )
    except FileNotFoundError as exc:
        response = (
            "Model or data not available yet. "
            "Please train the model first by running `python -m nba_predictor.cli.main train` "
            "or a full retrain workflow, then try again.\n"
            f"Details: {exc}"
        )
    except Exception as exc:
        response = (
            "Sorry, I could not compute predictions right now. "
            "Please check the model and data files.\n"
            f"Error: {exc}"
        )

    send_reply(chat_id, response)


def run_bot(poll_interval: int = 5) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    print("[telegram-bot] starting polling loop...")
    offset = _load_last_update_id()
    while True:
        try:
            data = get_updates(offset=offset, timeout=30)
            if not data.get("ok"):
                print(f"[telegram-bot] getUpdates failed: {data}")
                time.sleep(poll_interval)
                continue
            updates = data.get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                _save_last_update_id(offset)
                if "message" in update:
                    handle_message(update["message"])
        except urllib.error.URLError as e:
            print(f"[telegram-bot] network error: {e}")
            time.sleep(poll_interval)
        except Exception as e:
            print(f"[telegram-bot] unexpected error: {e}")
            time.sleep(poll_interval)
        time.sleep(poll_interval)


if __name__ == "__main__":
    run_bot()

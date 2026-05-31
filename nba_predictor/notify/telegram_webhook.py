"""Telegram webhook server for command-based predictions."""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from .telegram_bot import handle_message, TELEGRAM_BOT_TOKEN

WEBHOOK_PATH = os.environ.get("TELEGRAM_WEBHOOK_PATH", "/telegram-webhook").strip()
if not WEBHOOK_PATH.startswith("/"):
    WEBHOOK_PATH = "/" + WEBHOOK_PATH
WEBHOOK_HOST = os.environ.get("TELEGRAM_WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8000"))


def _api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def _http_post(url: str, payload: dict) -> dict:
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def set_webhook(webhook_url: Optional[str] = None) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if webhook_url is None:
        webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
    if not webhook_url:
        raise RuntimeError("TELEGRAM_WEBHOOK_URL must be set or passed to set_webhook()")
    return _http_post(_api_url("setWebhook"), {"url": webhook_url})


class _TelegramWebhookHandler(BaseHTTPRequestHandler):
    def _send_text(self, code: int, text: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def do_GET(self) -> None:
        self._send_text(200, "NBA Predictor Telegram webhook is running")

    def do_POST(self) -> None:
        if self.path != WEBHOOK_PATH:
            self._send_text(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            update = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self._send_text(400, f"Invalid JSON: {exc}")
            return
        message = update.get("message")
        if message:
            try:
                handle_message(message)
            except Exception as exc:
                print(f"[telegram-webhook] message handling failed: {exc}")
        self._send_text(200, "OK")


def run_webhook_server() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    server = HTTPServer((WEBHOOK_HOST, WEBHOOK_PORT), _TelegramWebhookHandler)
    print(f"[telegram-webhook] running on http://{WEBHOOK_HOST}:{WEBHOOK_PORT}{WEBHOOK_PATH}")
    print("[telegram-webhook] set Telegram webhook to this URL or proxy to it.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[telegram-webhook] stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_webhook_server()

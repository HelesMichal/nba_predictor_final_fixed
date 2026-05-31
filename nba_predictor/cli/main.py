"""Single CLI entrypoint.

Usage:
  python -m nba_predictor.cli.main fetch            --start 2013 --end 2024
  python -m nba_predictor.cli.main build
  python -m nba_predictor.cli.main train
  python -m nba_predictor.cli.main predict
  python -m nba_predictor.cli.main notify           --days 1   # predict + send to Telegram
  python -m nba_predictor.cli.main telegram-bot     # start Telegram polling bot
  python -m nba_predictor.cli.main telegram-webhook  # start HTTP webhook server
  python -m nba_predictor.cli.main set-webhook      --url https://example.com/telegram-webhook
  python -m nba_predictor.cli.main all              --start 2013 --end 2024
"""
from __future__ import annotations
import argparse
import sys

from ..data.nba_api_source import fetch_games, fetch_team_season_stats
from ..data.br_source import fetch_player_advanced, fetch_current_injuries
from ..features.build import build_dataset
from ..model.train import train_and_save
from ..model.predict import predict_upcoming, predict_for_date, validate_last_n_days
from ..notify.telegram_bot import run_bot as run_telegram_bot
from ..notify.telegram_webhook import run_webhook_server, set_webhook


def cmd_fetch(args):
    print(f"[fetch] games {args.start}-{args.end}")
    fetch_games(args.start, args.end)
    print("[fetch] team advanced stats")
    fetch_team_season_stats(args.start, args.end)
    print("[fetch] player advanced stats from BR")
    fetch_player_advanced(args.start, args.end)
    print("[fetch] current injuries from BR")
    fetch_current_injuries()


def cmd_build(_):
    df = build_dataset()
    print(f"[build] dataset rows={len(df)} cols={len(df.columns)}")


def cmd_train(_):
    train_and_save()


def cmd_telegram_bot(args):
    print("[telegram-bot] starting...")
    run_telegram_bot()


def cmd_telegram_webhook(args):
    print("[telegram-webhook] starting server...")
    run_webhook_server()


def cmd_set_webhook(args):
    result = set_webhook(args.url or None)
    print("[telegram-webhook] setWebhook response:")
    print(result)


def cmd_predict(args):
    print("\n=== Validation: last 5 days ===")
    v = validate_last_n_days(5)
    if v["accuracy"] is None:
        print("No recent games in dataset.")
    else:
        print(f"games={v['n_games']}  accuracy={v['accuracy']:.3f}  log_loss={v['log_loss']:.3f}")
        for d in v["details"][-15:]:
            mark = "✓" if d["correct"] else "✗"
            print(f"  {mark} {d['date']} {d['away']} @ {d['home']}  "
                  f"p(home)={d['p_home_win']:.2f}  actual_home_win={d['actual_home_win']}")

    print("\n=== Upcoming games ===")
    p = predict_upcoming(days_ahead=args.days)
    if p.empty:
        print("No upcoming games found for the requested window.")
    else:
        for _, r in p.iterrows():
            print(f"  {r['date']} {r['away']} @ {r['home']}  "
                  f"p(home win)={r['p_home_win']:.2f}  p(away win)={r['p_away_win']:.2f}")


def cmd_notify(args):
    """Run predictions + validation, then send a Telegram message."""
    from ..notify.format import format_daily_digest
    from ..notify.telegram import send_message

    print("[notify] computing validation (last 5 days)...")
    v = validate_last_n_days(5)
    print("[notify] computing today's predictions...")
    p = predict_upcoming(days_ahead=args.days)

    msg = format_daily_digest(p, v)
    print("[notify] message preview:\n" + msg + "\n")

    if args.dry_run:
        print("[notify] --dry-run set, not sending.")
        return
    send_message(msg)
    print("[notify] sent ✓")


def cmd_all(args):
    cmd_fetch(args); cmd_build(args); cmd_train(args); cmd_predict(args)


def main():
    p = argparse.ArgumentParser(prog="nba_predictor")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch"); f.add_argument("--start", type=int, default=2013); f.add_argument("--end", type=int, default=2024); f.set_defaults(func=cmd_fetch)
    sub.add_parser("build").set_defaults(func=cmd_build)
    sub.add_parser("train").set_defaults(func=cmd_train)
    pr = sub.add_parser("predict"); pr.add_argument("--days", type=int, default=1); pr.set_defaults(func=cmd_predict)
    n = sub.add_parser("notify"); n.add_argument("--days", type=int, default=1); n.add_argument("--dry-run", action="store_true"); n.set_defaults(func=cmd_notify)
    tb = sub.add_parser("telegram-bot"); tb.set_defaults(func=cmd_telegram_bot)
    tw = sub.add_parser("telegram-webhook"); tw.set_defaults(func=cmd_telegram_webhook)
    sw = sub.add_parser("set-webhook"); sw.add_argument("--url", type=str, default="", help="Webhook URL to register with Telegram"); sw.set_defaults(func=cmd_set_webhook)
    a = sub.add_parser("all"); a.add_argument("--start", type=int, default=2013); a.add_argument("--end", type=int, default=2024); a.add_argument("--days", type=int, default=1); a.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

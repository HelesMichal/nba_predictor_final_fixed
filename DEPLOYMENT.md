# Running NBA Predictor on GitHub Actions (free) with Telegram alerts

This project can run autonomously in the cloud — no laptop needed — using
**GitHub Actions** as the scheduler and **Telegram** as the notification channel.
Both are free for personal use.

## What you get

- **Daily at ~17:00 Central European time** → a Telegram message with today's
  games, home/away win probabilities, and the model's accuracy over the last 5 days.
- **Weekly retrain (Mondays 07:00 CET)** that refreshes the model on the latest data.
- **Manual retrain on demand** from the GitHub Actions UI.

## One-time setup (≈10 minutes)

### 1. Create a Telegram bot

1. Open Telegram, search for **@BotFather**, send `/newbot`.
2. Pick a name and username — BotFather replies with a **bot token** like
   `1234567890:AAH...`. Save it.
3. Open a chat with your new bot and send any message (`hi`).
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser.
   Find `"chat":{"id": 123456789, ...}` — that number is your **chat ID**.

### 2. Push this project to GitHub

Create a new GitHub repo (public is fine; the model files don't contain anything
secret). Upload the project contents, including the `.github/workflows/` folder.

### 3. Add secrets to the repo

In the repo, go to **Settings → Secrets and variables → Actions → New repository secret**
and add two secrets:

| Name                  | Value                          |
|-----------------------|--------------------------------|
| `TELEGRAM_BOT_TOKEN`  | The token from BotFather       |
| `TELEGRAM_CHAT_ID`    | Your numeric chat ID           |

### 4. Trigger the first retrain

In the repo, go to **Actions → Weekly retrain → Run workflow**.
This trains the model and stores it in the workflow cache so the daily job can use it.
Takes ~30–90 minutes the first time (full data fetch); future weekly runs are faster because data and model files are cached.

That's it. The daily workflow will start firing automatically at the next 17:00 CET/CEST. It is also safe to run before the cache exists; it will fetch the full training window instead of failing on a missing `cache/games.parquet` file.

## Manual controls

- **Run predictions right now** → Actions → *Daily NBA predictions* → Run workflow
- **Retrain the model** → Actions → *Weekly retrain* → Run workflow
- **Test the Telegram integration locally**:
  ```bash
  export TELEGRAM_BOT_TOKEN=...
  export TELEGRAM_CHAT_ID=...
  python -m nba_predictor.notify.telegram "hello from my laptop"
  ```
- **Dry-run the notifier** (no message sent):
  ```bash
  python -m nba_predictor.cli.main notify --dry-run
  ```

## Schedule notes

GitHub Actions cron runs in **UTC** and Central Europe switches between CET (UTC+1) and
CEST (UTC+2). The workflow defines two cron lines (15:00 and 16:00 UTC) so that one
of them always lands at 17:00 local. If you'd rather get exactly one notification per
day, delete whichever cron line you don't want from `.github/workflows/daily-predict.yml`.

GitHub Actions cron is best-effort and can be delayed by a few minutes during high load —
don't rely on second-level precision.

## Cost

- **GitHub Actions**: 2,000 free minutes/month on private repos (unlimited on public).
  Daily job ≈ 5 min, weekly retrain ≈ 45 min → ~330 min/month. Well within free tier.
- **Telegram**: free.

## Alternatives if you'd rather not use GitHub

- **Render.com cron jobs** (free tier) — same idea, run a container on a schedule.
- **Fly.io machines** — start/stop a tiny VM on a schedule.
- **A spare Raspberry Pi at home + cron + systemd-timer** — zero cloud cost.

The notifier itself (`nba_predictor.notify.telegram`) doesn't care where it runs;
swap the scheduler however you like.

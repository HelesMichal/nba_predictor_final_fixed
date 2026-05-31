# NBA Predictor – Deployment guide

Everything in this app runs on **free infrastructure that stays on 24/7**, so
your laptop can be off. Two pieces:

| Piece | Where it runs | What it does |
|------|----------------|--------------|
| Training + scheduled daily push | **GitHub Actions** (free) | Retrains weekly, sends a 17:00 CET digest to Telegram |
| `/today` `/tomorrow` `/week` commands | **Cloudflare Worker** (free) | Listens for your Telegram messages and triggers an on-demand GitHub Action that replies with predictions |

You only set this up once. After that it runs by itself.

---

## 1 · Create the Telegram bot (2 min)

1. Open Telegram, talk to [`@BotFather`](https://t.me/BotFather) → `/newbot` → pick a name.
2. Copy the **bot token** (looks like `1234:ABC…`). Keep it open.
3. Talk to your new bot and send `/start`, then visit
   `https://api.telegram.org/bot<TOKEN>/getUpdates` to copy your **chat id**
   (the integer inside `"chat":{"id":…}`).
4. To receive on multiple devices/users: every Telegram account that should
   get the digest must `/start` the bot. Grab each `chat id` the same way.
   You can have as many as you want (3, 10, …) — they're just comma-separated.

---

## 2 · Push the repo to GitHub (1 min)

```bash
gh repo create nba_predictor --private --source . --push
```

Or just create an empty GitHub repo in the UI and `git push`.

---

## 3 · Add GitHub Secrets (1 min)

GitHub repo → **Settings → Secrets and variables → Actions → New secret**:

| Name | Value |
|------|-------|
| `TELEGRAM_BOT_TOKEN` | the bot token from step 1 |
| `TELEGRAM_CHAT_IDS` | one or more chat ids, comma-separated, e.g. `12345,67890,11111` |

That's enough for the **daily 17:00 CET digest** and **weekly Monday retrain**
to start working. Trigger them once manually from the **Actions** tab to seed
the cache.

---

## 4 · Deploy the Cloudflare Worker for on-demand commands (5 min)

This is what makes `/today`, `/tomorrow`, `/week` work from any device, at any
time, without your computer being on.

### 4.1 Install wrangler and log in

```bash
npm i -g wrangler
wrangler login
```

### 4.2 Create a GitHub Personal Access Token

GitHub → **Settings → Developer settings → Personal access tokens →
Fine-grained tokens → Generate new token**:

- Repository access: **Only select repositories** → your `nba_predictor` repo
- Permissions:
  - **Contents**: Read and write
  - **Actions**: Read and write
  - **Metadata**: Read

Copy the token (starts with `github_pat_…`).

### 4.3 Deploy the worker

```bash
cd cloudflare-worker
wrangler deploy
```

Wrangler prints the worker URL, e.g.
`https://nba-predictor-bot.<your-subdomain>.workers.dev`. **Copy it.**

### 4.4 Add the worker secrets

```bash
wrangler secret put TELEGRAM_BOT_TOKEN          # paste BotFather token
wrangler secret put TELEGRAM_WEBHOOK_SECRET     # any random string, e.g. `openssl rand -hex 24`
wrangler secret put GITHUB_TOKEN                # the PAT from 4.2
wrangler secret put GITHUB_REPO                 # e.g. yourname/nba_predictor
wrangler secret put ALLOWED_CHAT_IDS            # same comma list as TELEGRAM_CHAT_IDS
```

### 4.5 Point Telegram at the worker

Replace the two placeholders and run:

```bash
TOKEN="<bot token>"
URL="https://nba-predictor-bot.<subdomain>.workers.dev"
SECRET="<the same TELEGRAM_WEBHOOK_SECRET you set above>"

curl -s "https://api.telegram.org/bot$TOKEN/setWebhook" \
  -d "url=$URL" \
  -d "secret_token=$SECRET" \
  -d "allowed_updates=[\"message\"]"
```

You should see `{"ok":true,…}`.

---

## 5 · Try it

In Telegram, message the bot:

- `/help` – list commands
- `/today` – today's games + win probabilities
- `/tomorrow` – tomorrow's slate
- `/week` – next 7 days

The worker queues a GitHub Action which replies in ~1–2 min directly to the
chat that asked. The same bot works simultaneously on every device logged
into Telegram – that's a Telegram-side feature, no extra config needed.

---

## How "works when my computer is off" actually works

- **GitHub Actions** runs on GitHub's servers on a cron — your machine is
  irrelevant.
- **Cloudflare Workers** runs at the edge 24/7 — also nothing to do with your
  machine.
- The only thing your laptop is used for is the one-time `wrangler deploy`.

---

## Manual operations

- **Retrain now**: GitHub → Actions → *Weekly Retrain* → Run workflow.
- **Send a prediction now without Telegram**: GitHub → Actions → *On-Demand
  Predict* → Run workflow → choose mode.
- **Add a new device/user**: have them `/start` the bot, grab their chat id,
  append it to both `TELEGRAM_CHAT_IDS` (GitHub secret) and
  `ALLOWED_CHAT_IDS` (worker secret).

GitHub repo → **Settings → Secrets and variables → Actions → New secret**:
| `TELEGRAM_CHAT_IDS` | e.g. `12345,67890,11111` |      - 8951792524, ... 

in Terminal (powershell - cloudflare-wroker):
wrangler secret put ALLOWED_CHAT_IDS                    - 8951792524, ... 

---

## Files you can safely delete

These were leftovers from earlier iterations and aren't used anywhere:

- any `*.bak`, `*.old`, `__pycache__/` directories
- `notebooks/` (exploration only)
- `scripts/local_run.sh` / `Dockerfile.dev` if present
- `tests/fixtures/large_*.parquet` (regenerated from the API)

Keeping the project to: `nba_predictor/`, `.github/workflows/`,
`cloudflare-worker/`, `requirements.txt`, `README.md`, `DEPLOYMENT.md`.

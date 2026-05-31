/**
 * Telegram webhook → GitHub Actions bridge.
 *
 * Free, always-on (Cloudflare Workers free tier: 100k req/day). Receives
 * /today, /tomorrow, /week from any device logged into your Telegram bot
 * and triggers the on-demand.yml workflow to send back predictions.
 *
 * Required secrets (wrangler secret put ...):
 *   - TELEGRAM_BOT_TOKEN          (BotFather token)
 *   - TELEGRAM_WEBHOOK_SECRET     (any random string; set when calling setWebhook)
 *   - GITHUB_TOKEN                (PAT with `repo` scope, fine-grained "Contents: RW + Actions: RW")
 *   - GITHUB_REPO                 (e.g. "yourname/nba_predictor")
 *   - ALLOWED_CHAT_IDS            (comma-separated; only these chats may trigger jobs)
 */

const HELP = [
  "🏀 *NBA Predictor bot*",
  "",
  "/today    – predictions for today",
  "/tomorrow – predictions for tomorrow",
  "/week     – predictions for the next 7 days",
  "/help     – this message",
  "",
  "_Each command queues a fresh prediction job. The reply arrives in ~1–2 min._",
].join("\n");

const COMMANDS = new Set(["/today", "/tomorrow", "/week"]);

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("ok");

    // Verify Telegram secret header.
    const got = request.headers.get("x-telegram-bot-api-secret-token");
    if (got !== env.TELEGRAM_WEBHOOK_SECRET) {
      return new Response("forbidden", { status: 403 });
    }

    const update = await request.json().catch(() => ({}));
    const msg = update.message || update.edited_message;
    if (!msg || !msg.text) return new Response("ok");

    const chatId = String(msg.chat.id);
    const text = msg.text.trim().toLowerCase().split(/\s|@/)[0];

    const allowed = (env.ALLOWED_CHAT_IDS || "")
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (allowed.length && !allowed.includes(chatId)) {
      await tg(env, chatId, "⛔ This chat isn't authorized. Add its id to ALLOWED_CHAT_IDS.");
      return new Response("ok");
    }

    if (text === "/start" || text === "/help") {
      await tg(env, chatId, HELP, "Markdown");
      return new Response("ok");
    }

    if (!COMMANDS.has(text)) {
      await tg(env, chatId, "Unknown command. Try /today, /tomorrow, /week or /help.");
      return new Response("ok");
    }

    const mode = text.slice(1); // "today" | "tomorrow" | "week"
    const ok = await dispatch(env, mode, chatId);
    await tg(
      env,
      chatId,
      ok
        ? `⏳ Queued *${mode}* predictions – results in ~1–2 min.`
        : "❌ Could not trigger the prediction job. Check the worker logs.",
      "Markdown",
    );
    return new Response("ok");
  },
};

async function tg(env, chatId, text, parseMode) {
  const body = { chat_id: chatId, text };
  if (parseMode) body.parse_mode = parseMode;
  await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function dispatch(env, mode, chatId) {
  const r = await fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`, {
    method: "POST",
    headers: {
      accept: "application/vnd.github+json",
      authorization: `Bearer ${env.GITHUB_TOKEN}`,
      "user-agent": "nba-predictor-bot",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      event_type: "telegram-command",
      client_payload: { mode, chat_id: chatId },
    }),
  });
  return r.ok;
}

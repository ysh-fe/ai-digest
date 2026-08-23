# ai-digest

A small cron-friendly script that turns the day's AI newsletters into a single
Russian-language digest and posts it to Telegram.

It reads the newsletters from their **public web archives / RSS feeds** — not
from email — so there is no mail API, no OAuth and no message-size limit to
work around. An LLM (Gemma via the Gemini API) merges overlapping stories,
strips sponsor blocks, keeps the source links, and formats the result with
Telegram HTML markup.

## Sources

| Source | How it is fetched |
| --- | --- |
| [TLDR AI](https://tldr.tech/ai) | `https://tldr.tech/api/latest/ai` — always redirects to the latest issue |
| [The Neuron](https://www.theneurondaily.com/) | beehiiv RSS feed (the site itself 403s bots), falls back to scraping the archive page |
| [What's Up in AI](https://whatsupinai.beehiiv.com/) | archive page — first post link is the latest issue |

Adding a source means adding one entry to the `SOURCES` dict in
[`ai_digest.py`](ai_digest.py): either a `latest_url`, an `rss_url`, or an
`archive_url` + `link_prefix` pair.

## How it works

1. **Fetch** — for each source, get the latest issue (RSS `content:encoded`
   when available, otherwise the article HTML).
2. **Extract** — strip scripts/nav/footer, inline links as `[text](url)` so the
   model can preserve them, take the text of `<body>`.
3. **Skip repeats** — `ai_digest_state.json` stores the last-sent URL per
   source, so a newsletter that didn't publish today is not sent twice.
4. **Compile** — one prompt with all fresh issues → one digest, structured by
   section (news, tools, prompt of the day, jokes — whatever was actually in
   the issues), ads excluded.
5. **Send** — split on `<b>` section boundaries to stay under Telegram's
   4096-character cap, then post each part.

State is only updated after every message is delivered, so a failed run simply
retries on the next one.

## Setup

```bash
git clone https://github.com/<you>/ai-digest.git
cd ai-digest
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY
```

Run it:

```bash
./venv/bin/python ai_digest.py
```

Daily at 09:00 via cron:

```cron
0 9 * * * cd /path/to/ai-digest && ./venv/bin/python ai_digest.py >> ai_digest.log 2>&1
```

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | yes | Bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | yes | Chat or channel id to post into |
| `GEMINI_API_KEY` | yes | Key from [Google AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | no | Model used to compile the digest (default `gemma-4-31b-it`) |

The digest prompt (`DIGEST_PROMPT_TEMPLATE` in `ai_digest.py`) is written in
Russian because it asks for a Russian-language digest — rewrite it in your own
language to change the output language.

## Notes

- No database: the only state is `ai_digest_state.json` (gitignored).
- Logs go to stdout and `ai_digest.log`.
- Per-source input is capped at 15 000 characters before it reaches the model.

## License

MIT

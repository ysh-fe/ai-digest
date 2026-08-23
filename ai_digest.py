"""
ai_digest.py — Daily AI newsletter digest -> Telegram.

Fetches the latest issue of three AI newsletters from their public web
archives (NOT email — avoids mail API size limits), asks an LLM (Gemma via
the Gemini API) to compile a Russian-language digest, and posts it to
Telegram.

Sources:
  - TLDR AI          — https://tldr.tech/api/latest/ai (always redirects to
                        today's/latest issue)
  - The Neuron       — https://www.theneurondaily.com/ (beehiiv RSS, falls
                        back to the archive page)
  - What's Up in AI  — https://whatsupinai.beehiiv.com/ (archive page, first
                        post link = latest issue)

State (ai_digest_state.json) tracks the last-sent article URL per source so
the same issue isn't sent twice on days a newsletter didn't publish.

Usage:
  python ai_digest.py

Cron (daily at 11:00 — server time):
  0 11 * * * cd /path/to/ai-digest && ./venv/bin/python ai_digest.py >> ai_digest.log 2>&1

Configuration — see .env.example:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GEMINI_API_KEY
"""

import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "ai_digest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ai_digest")

STATE_PATH = Path(__file__).parent / "ai_digest_state.json"
# Some newsletter sites (e.g. theneurondaily.com) 403 on non-browser UAs.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
FETCH_TIMEOUT = 20
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-31b-it")
TELEGRAM_MSG_LIMIT = 3500  # stay comfortably under Telegram's 4096-char cap
MAX_SOURCE_CHARS = 15000  # per-source cap before feeding to the model

SOURCES = {
    "tldr": {
        "name": "TLDR AI",
        "latest_url": "https://tldr.tech/api/latest/ai",
    },
    "the_neuron": {
        "name": "The Neuron",
        # theneurondaily.com 403s plain requests (bot protection) even with a
        # browser UA — the beehiiv-hosted RSS feed is not behind the same
        # protection and includes the full post content, so try it first.
        "rss_url": "https://rss.beehiiv.com/feeds/N4eCstxvgX.xml",
        "archive_url": "https://www.theneurondaily.com/",
        "link_prefix": "https://www.theneurondaily.com/p/",
    },
    "whats_up_in_ai": {
        "name": "What's Up in AI",
        "archive_url": "https://whatsupinai.beehiiv.com/",
        "link_prefix": "https://whatsupinai.beehiiv.com/p/",
    },
}


# ─── State ───────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read state file, starting fresh: {e}")
    return {}


def _save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Fetching ────────────────────────────────────────────────────────────

FETCH_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch(url: str) -> tuple[str, str]:
    """GET a URL, following redirects. Returns (final_url, html)."""
    resp = requests.get(url, headers=FETCH_HEADERS, timeout=FETCH_TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    return resp.url, resp.text


def _latest_archive_url(archive_html: str, archive_url: str, link_prefix: str) -> str | None:
    """Finds the first (most recent) post link on a beehiiv-style archive page.

    Links on these pages are sometimes relative (e.g. "/p/slug"), so resolve
    against the archive page's own URL before matching the prefix.
    """
    soup = BeautifulSoup(archive_html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = urljoin(archive_url, a["href"]).split("?")[0]
        if href.startswith(link_prefix) and href != link_prefix:
            return href
    return None


def _extract_article_text(html: str) -> str:
    """Extracts readable article text, preserving links as [text](url).

    Deliberately does NOT narrow to <article>/<main> — on some newsletter
    platforms those tags wrap very little of the actual content, which
    silently produces a too-short extraction. Using <body> (minus obvious
    chrome) is more inclusive and safer.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()

    container = soup.body or soup

    # Inline-ify links before extracting text so the model can keep them.
    for a in container.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if text and href.startswith("http"):
            a.replace_with(f"[{text}]({href})")
        else:
            a.unwrap()

    return container.get_text("\n", strip=True)


RSS_CONTENT_ENCODED = "{http://purl.org/rss/1.0/modules/content/}encoded"


def _fetch_via_rss(rss_url: str) -> tuple[str, str] | None:
    """Fetches the first <item> of an RSS/Atom feed. Beehiiv feeds embed the
    full post HTML in <content:encoded>, so this avoids hitting the
    (sometimes bot-protected) publication website at all."""
    try:
        resp = requests.get(rss_url, headers=FETCH_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        item = root.find(".//item")
        if item is None:
            return None
        link = (item.findtext("link") or "").strip()
        content_el = item.find(RSS_CONTENT_ENCODED)
        html = (content_el.text if content_el is not None else None) or item.findtext("description") or ""
        if not link or not html.strip():
            return None
        return link, html
    except Exception as e:
        logger.warning(f"RSS fetch failed for {rss_url}: {e}")
        return None


def get_latest_issue(key: str, cfg: dict) -> tuple[str, str] | None:
    """Returns (article_url, article_text) for a source's latest issue, or None on failure."""
    try:
        final_url: str | None = None
        html: str | None = None

        if cfg.get("rss_url"):
            rss_result = _fetch_via_rss(cfg["rss_url"])
            if rss_result:
                final_url, html = rss_result
            else:
                logger.warning(f"[{key}] RSS unavailable, falling back to page scrape")

        if html is None:
            if "latest_url" in cfg:
                final_url, html = _fetch(cfg["latest_url"])
            elif "archive_url" in cfg:
                archive_final_url, archive_html = _fetch(cfg["archive_url"])
                article_url = _latest_archive_url(archive_html, archive_final_url, cfg["link_prefix"])
                if not article_url:
                    logger.warning(f"[{key}] Could not find latest article link on archive page")
                    return None
                final_url, html = _fetch(article_url)
            else:
                logger.error(f"[{key}] No rss_url/latest_url/archive_url configured")
                return None

        text = _extract_article_text(html)
        logger.info(f"[{key}] extracted {len(text)} chars from {final_url}")
        if not text.strip():
            logger.warning(f"[{key}] Extracted empty text, skipping")
            return None
        return final_url, text
    except Exception as e:
        logger.warning(f"[{key}] fetch failed: {e}")
        return None


# ─── Digest compilation (Gemma via Gemini API) ──────────────────────────

DIGEST_PROMPT_TEMPLATE = """Ты собираешь ежедневную сводку AI-новостей на русском языке для Telegram-канала.

Ниже — полный текст свежих выпусков AI-рассылок (ссылки внутри текста в формате [текст](url)).

{combined}

Составь одно сообщение-сводку на русском языке для Telegram. Используй Telegram HTML-разметку:
<b>жирный</b> для заголовков разделов, "• " в начале строки для списков, <a href="URL">текст</a> для ссылок на источники, <pre>...</pre> для промпт-шаблонов или кода. НЕ используй Markdown-заголовки (#) — Telegram их не поддерживает. Не используй вложенные теги.

Требования к содержанию:
- Объединяй похожие/пересекающиеся новости из разных источников в один общий пункт.
- Сохраняй ВСЕ типы контента, которые реально есть в выпусках, а не только новости: инструменты/находки ("что попробовать"), промпт/скилл дня (шаблон приводи целиком внутри <pre>), шутки — если они есть.
- У каждого пункта, где в исходном тексте была ссылка на источник, статью или продукт, сохрани эту ссылку через <a href="URL">текст</a>.
- Полностью исключи рекламные блоки: "Sponsor", "From our partners", "Advertise", промокоды, партнёрские предложения, купоны вида "Get X% off".
- Структурируй по смысловым разделам (например: новости, инструменты, промпт дня, шутки — конкретный набор зависит от того, что реально было в выпусках).
- В начале сообщения — заголовок <b>AI-сводка на {date}</b> и список источников, вошедших в сегодняшний выпуск: {source_names}.
- Не добавляй пояснений о том, что ты делаешь — выведи только готовый текст сообщения.
"""


def compile_digest(new_items: dict[str, tuple[str, str]]) -> str | None:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai not installed. Run: pip install google-genai")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set in .env")
        return None

    blocks = []
    names = []
    for key, (url, text) in new_items.items():
        name = SOURCES[key]["name"]
        names.append(name)
        blocks.append(f"=== {name} ({url}) ===\n{text[:MAX_SOURCE_CHARS]}")
    combined = "\n\n".join(blocks)

    prompt = DIGEST_PROMPT_TEMPLATE.format(
        combined=combined,
        date=datetime.now().strftime("%d.%m.%Y"),
        source_names=", ".join(names),
    )

    client = genai.Client(api_key=api_key)
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    )

    try:
        chunks = []
        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL, contents=contents, config=config,
        ):
            if chunk.text:
                chunks.append(chunk.text)
        result = "".join(chunks).strip()
        return result or None
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return None


# ─── Telegram ────────────────────────────────────────────────────────────

def _split_for_telegram(text: str, limit: int = TELEGRAM_MSG_LIMIT) -> list[str]:
    """Splits on section boundaries (<b>...) so HTML tags never get cut mid-way."""
    if len(text) <= limit:
        return [text]

    parts = re.split(r"(?=<b>)", text)
    messages: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > limit:
            messages.append(current.strip())
            current = part
        else:
            current += part
    if current.strip():
        messages.append(current.strip())
    return messages


def send_telegram_html(bot_token: str, chat_id: str, text: str) -> bool:
    """Sends one HTML-formatted message via the Telegram Bot API."""
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": False,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            logger.error(f"[Telegram] API error: {data}")
            return False
        logger.info("[Telegram] Message sent successfully")
        return True
    except Exception as e:
        logger.error(f"[Telegram] Request failed: {e}")
        return False


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    state = _load_state()
    new_items: dict[str, tuple[str, str]] = {}

    for key, cfg in SOURCES.items():
        result = get_latest_issue(key, cfg)
        if result is None:
            continue
        url, text = result
        if state.get(key, {}).get("url") == url:
            logger.info(f"[{key}] No new issue since last run ({url})")
            continue
        logger.info(f"[{key}] New issue: {url}")
        new_items[key] = (url, text)

    if not new_items:
        logger.info("No new issues from any source — nothing to send.")
        return

    digest = compile_digest(new_items)
    if not digest:
        logger.error("Digest compilation failed — not sending, state not updated (will retry next run).")
        return

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env")
        return

    all_ok = True
    for msg in _split_for_telegram(digest):
        if not send_telegram_html(bot_token, chat_id, msg):
            all_ok = False

    if all_ok:
        now_iso = datetime.now().isoformat()
        for key, (url, _) in new_items.items():
            state[key] = {"url": url, "date": now_iso}
        _save_state(state)
        logger.info(f"Digest sent ({len(new_items)} source(s)) and state updated.")
    else:
        logger.error("Telegram send failed for at least one message — state not updated.")


if __name__ == "__main__":
    main()

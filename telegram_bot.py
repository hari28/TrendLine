"""Sends watchlist alerts via the Telegram Bot API.

Reads credentials from environment variables, loaded via python-dotenv from a
.env file in the project root (gitignored) -- same pattern as whatsapp.py, so
this works identically whether imported from the Streamlit app process or the
standalone check_watchlist.py script run by launchd.
"""
import os
import logging

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("trendline.telegram")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram_message(text: str, parse_mode: str | None = None) -> bool:
    """Send a message to the configured Telegram chat/group via the Bot API.
    Never raises -- returns False on any failure (missing config, network
    error, non-2xx response) so a transient Telegram outage never breaks a
    scan or watchlist check. Pass parse_mode="HTML" for messages using HTML
    tags (e.g. <pre> for monospace tables) -- plain alerts should leave it
    unset."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        logger.warning("Telegram not configured (missing env vars) -- skipping alert: %s", text[:80])
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text[:4096]}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code >= 400:
            logger.error("Telegram send failed (%s): %s", resp.status_code, resp.text[:300])
            return False
        return True
    except Exception as e:
        logger.error("Telegram send raised: %s", e)
        return False

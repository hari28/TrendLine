"""Sends watchlist alerts via the Meta WhatsApp Cloud API (official Graph API).

Reads credentials from environment variables, loaded via python-dotenv from a
.env file in the project root (gitignored). This works identically whether
imported from the Streamlit app process or from the standalone check_watchlist.py
script run by launchd -- st.secrets alone wouldn't be readable from that script.

Setup and known limitations (24h token expiry, 24h messaging window) are
documented in README.md under "Watchlist & WhatsApp Alerts" -- not solved here.
"""
import os
import logging

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("trendline.whatsapp")

GRAPH_API_VERSION = os.environ.get("WHATSAPP_API_VERSION", "v21.0")
META_WHATSAPP_TOKEN = os.environ.get("META_WHATSAPP_TOKEN")
META_PHONE_NUMBER_ID = os.environ.get("META_PHONE_NUMBER_ID")
META_RECIPIENT_NUMBER = os.environ.get("META_RECIPIENT_NUMBER")


def send_whatsapp_message(body: str) -> bool:
    """Send a WhatsApp text message via the Meta Graph API. Never raises --
    returns False on any failure (missing config, network error, non-2xx
    response) so a transient WhatsApp outage never breaks a scan or check."""
    if not (META_WHATSAPP_TOKEN and META_PHONE_NUMBER_ID and META_RECIPIENT_NUMBER):
        logger.warning("WhatsApp not configured (missing env vars) -- skipping alert: %s", body[:80])
        return False

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {META_WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": META_RECIPIENT_NUMBER,
        "type": "text",
        "text": {"body": body[:4096]},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code >= 400:
            logger.error("WhatsApp send failed (%s): %s", resp.status_code, resp.text[:300])
            return False
        return True
    except Exception as e:
        logger.error("WhatsApp send raised: %s", e)
        return False

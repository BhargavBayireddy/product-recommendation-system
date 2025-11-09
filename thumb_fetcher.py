# thumb_fetcher.py
import io
import re
import base64
import random
import requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

USER_AGENT = "ReccoVerse/1.0 (https://streamlit.app)"

def _http_get(url, timeout=7):
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def _wikipedia_thumb(title: str):
    q = requests.utils.quote(title)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=6)
        if r.status_code == 200:
            data = r.json()
            img = (data.get("thumbnail") or {}).get("source")
            if img:
                b = _http_get(img, timeout=6)
                return b
    except Exception:
        pass
    return None

def _unsplash_source(title: str):
    # Free endpoint without key; result is a redirect to an image
    q = requests.utils.quote(title)
    url = f"https://source.unsplash.com/featured/800x1200/?{q}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=7)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def _svg_placeholder_bytes(text: str, seed: int = 7):
    palette = [
        "#1f1c2c", "#3a1c71", "#0f2027", "#23074d", "#16384c",
        "#1b2735", "#1b1b2f", "#0a0a0a", "#2A0A29", "#201F3A"
    ]
    color = palette[seed % len(palette)]
    title = (text[:28] + "…") if len(text) > 28 else text
    svg = f"""
    <svg width="800" height="1200" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="{color}" />
          <stop offset="100%" stop-color="#0d0d0d" />
        </linearGradient>
      </defs>
      <rect width="800" height="1200" fill="url(#g)" />
      <text x="50" y="1070" font-size="44" fill="#f2f2f2" font-family="Arial,sans-serif">{title}</text>
    </svg>
    """
    return svg.encode("utf-8")

@st.cache_data(show_spinner=False)
def get_or_create_thumb(item_id: str, title: str, domain: str, tags: list) -> bytes:
    """
    Hybrid: try Wikipedia -> Unsplash -> SVG placeholder.
    Returns image bytes (PNG/JPEG/SVG).
    """
    # Wikipedia for named entities (movies, persons, albums)
    b = _wikipedia_thumb(title)
    if b:
        return b

    # Unsplash generic if Wikipedia misses
    b = _unsplash_source(title)
    if b:
        return b

    # Fallback SVG with gradient + title
    return _svg_placeholder_bytes(title, seed=abs(hash(item_id)) % 97)

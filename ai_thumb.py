# ai_thumb.py — Stable Diffusion thumbnail generator via HuggingFace
# Uses Streamlit secret HF_TOKEN. Caches to artifacts/thumbs and (optional) Firestore.

import os, base64, json, time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import streamlit as st

BASE = Path(__file__).parent
ART  = BASE / "artifacts"
THUMBS = ART / "thumbs"
THUMBS.mkdir(parents=True, exist_ok=True)

HF_MODEL = "stabilityai/stable-diffusion-2-1"

def _hf_token() -> Optional[str]:
    # Prefer secrets; fallback to env
    if "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]
    return os.environ.get("HF_TOKEN")

def _thumb_path(item_id: str) -> Path:
    safe = "".join([c for c in item_id if c.isalnum() or c in ("-","_")])
    return THUMBS / f"{safe}.png"

def _pillow_fallback(title: str) -> bytes:
    W, H = 700, 1000
    bg = Image.new("RGB", (W, H), (11, 14, 28))
    draw = ImageDraw.Draw(bg)
    for y in range(H):
        r = int(10 + 60*(y/H))
        g = int(18 + 40*(y/H))
        b = int(60 + 120*(y/H))
        draw.line([(0,y),(W,y)], fill=(r,g,b))
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse((-160, -120, W+160, H*0.9), fill=190)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    bg = Image.composite(Image.new("RGB",(W,H),(20,26,58)), bg, glow)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
    except Exception:
        font = ImageFont.load_default()
    t = (title or "Recommended").upper()[:28]
    tw = draw.textlength(t, font=font)
    draw.text(((W-tw)/2, int(H*0.7)), t, font=font, fill=(238,242,248))
    out = BytesIO(); bg.save(out, format="PNG"); return out.getvalue()

def _firestore_get_b64(item_id: str, firestore) -> Optional[bytes]:
    if not firestore: return None
    try:
        doc = firestore.collection("posters").document(item_id).get()
        if doc.exists:
            b64 = doc.to_dict().get("b64")
            if b64:
                return base64.b64decode(b64)
    except Exception:
        return None
    return None

def _firestore_put_b64(item_id: str, img_bytes: bytes, firestore) -> None:
    if not firestore: return
    try:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        firestore.collection("posters").document(item_id).set({"b64": b64, "ts": time.time()})
    except Exception:
        pass

def _call_hf(prompt: str, negative: str = "text, watermark, logo, words, lowres, bad anatomy") -> Optional[bytes]:
    token = _hf_token()
    if not token:
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"negative_prompt": negative, "num_inference_steps": 30}}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        # HF returns raw image bytes when model is loaded; else JSON while loading
        if r.headers.get("content-type","").startswith("image/"):
            return r.content
        # if json (loading) — retry once
        time.sleep(3)
        r2 = requests.post(url, headers=headers, json=payload, timeout=120)
        if r2.headers.get("content-type","").startswith("image/"):
            return r2.content
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def get_or_create_thumb(item_id: str, title: str, domain: str = "", tags: Optional[List[str]] = None, firestore=None) -> bytes:
    """
    Returns PNG bytes. Order:
      1) Local cache (artifacts/thumbs/<id>.png)
      2) Firestore base64 cache
      3) HuggingFace SD 2.1 generation (if token available)
      4) Pillow fallback poster
    """
    p = _thumb_path(item_id)
    if p.exists():
        return p.read_bytes()

    # Firestore cached?
    img = _firestore_get_b64(item_id, firestore)
    if img:
        p.write_bytes(img)  # persist locally too
        return img

    # Try HF
    prompt = (
        f"Netflix-style photo-real poster with no text. "
        f"Subject: {title}. Domain: {domain}. Tags: {(tags or [])}. "
        f"Cinematic lighting, dramatic contrast, center composition, glossy, UHD."
    )
    img = _call_hf(prompt)
    if img is None:
        img = _pillow_fallback(title)
    # persist
    try:
        p.write_bytes(img)
    except Exception:
        pass
    _firestore_put_b64(item_id, img, firestore)
    return img

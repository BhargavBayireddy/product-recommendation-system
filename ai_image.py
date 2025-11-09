# ai_image.py
# Auto-generates a poster-style image for any item missing an image.
# Uses OpenAI Images API when OPENAI key is present; otherwise falls back to a styled Pillow poster.

import io
import os
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- Optional OpenAI backend (uses env/Streamlit secrets) ---
_USE_OPENAI = False
try:
    from openai import OpenAI  # pip install openai
    _USE_OPENAI = True
except Exception:
    _USE_OPENAI = False


def _prompt_from_meta(
    title: str,
    domain: str = "",
    category: str = "",
    tags: Optional[List[str]] = None,
    style_profile: Optional[str] = None,
) -> str:
    tags = tags or []
    vibe = style_profile or "premium minimal, cinematic lighting, deep contrast"
    base = f"""Create a Netflix-style poster thumbnail with no text.
Subject: {title}.
Domain: {domain}. Category: {category}. Tags: {", ".join(tags)}.
Style: {vibe}. 4k photo-real poster art, dramatic rim light, shallow depth of field, glossy finish, center composition."""
    return base


def _png_bytes_from_pillow(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _fallback_pillow_poster(title: str) -> bytes:
    """
    If no OpenAI key is available, render a clean poster from scratch:
    gradient + vignette + glow bar + subtle texture + title (trimmed).
    """
    W, H = 700, 1000  # poster-ish
    bg = Image.new("RGB", (W, H), (8, 10, 22))
    draw = ImageDraw.Draw(bg)

    # radial glow
    glow = Image.new("L", (W, H), 0)
    gdraw = ImageDraw.Draw(glow)
    gdraw.ellipse((-150, -100, W + 150, H * 0.9), fill=190)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    bg = Image.composite(Image.new("RGB", (W, H), (18, 25, 55)), bg, glow)

    # diagonal color wash
    grad = Image.new("RGB", (W, H), (0, 0, 0))
    for y in range(H):
        r = int(12 + 110 * (y / H))
        g = int(20 + 40 * (y / H))
        b = int(55 + 160 * (y / H))
        ImageDraw.Draw(grad).line([(0, y), (W, y)], fill=(r, g, b))
    grad = grad.filter(ImageFilter.GaussianBlur(12))
    bg = Image.blend(bg, grad, 0.35)

    # center glossy bar
    bar = Image.new("RGBA", (W, int(H * 0.55)), (255, 255, 255, 16))
    bar = bar.filter(ImageFilter.GaussianBlur(6))
    bg.paste(bar, (0, int(H * 0.22)), bar)

    # Title (trim & uppercase)
    t = (title or "Recommended").strip()
    if len(t) > 28:
        t = t[:25] + "…"
    t = t.upper()

    # load a safe default font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
        sub = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
        sub = ImageFont.load_default()

    # text shadow
    tw, th = draw.textlength(t, font=font), 52
    x = (W - tw) / 2
    y = int(H * 0.70)
    for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
        draw.text((x + dx, y + dy), t, font=font, fill=(0, 0, 0))
    draw.text((x, y), t, font=font, fill=(240, 242, 248))

    draw.text((W // 2 - 110, y + 58), "AI-generated visual", font=sub, fill=(180, 188, 210))
    return _png_bytes_from_pillow(bg)


def generate_tile(
    title: str,
    domain: str = "",
    category: str = "",
    tags: Optional[List[str]] = None,
    style_profile: Optional[str] = None,
) -> bytes:
    """
    Returns PNG bytes for a poster tile.
    1) If OPENAI_API_KEY present -> generate with OpenAI Images
    2) Else -> return Pillow fallback poster
    """
    # OpenAI path (only if key present + lib available)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower()) or os.getenv("openai_api_key")
    if _USE_OPENAI and api_key:
        try:
            client = OpenAI(api_key=api_key)
            prompt = _prompt_from_meta(title, domain, category, tags, style_profile)
            img = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1536",
                quality="high",
                n=1,
            )
            b64 = img.data[0].b64_json
            import base64
            return base64.b64decode(b64)
        except Exception:
            # Fall through to Pillow
            pass

    # Fallback
    return _fallback_pillow_poster(title)

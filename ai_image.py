# ai_image.py  ✅ FINAL STABLE VERSION (works on all Pillow versions)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random


def generate_poster(item_id: str, name: str, domain: str, size=(280, 395)):
    """
    Generates a fake poster-like image with gradient + labels.
    No transparency, no alpha_composite, fully compatible with Stream-lit & Pillow 2024.
    """
    W, H = size
    bg = Image.new("RGB", (W, H), color=(15, 15, 15))

    # gradient background
    arr = np.linspace(0, 255, H).astype(np.float32)
    gradient = np.tile(arr, (W, 1)).T
    tint = random.choice([(160, 0, 220), (0, 140, 255), (255, 40, 140)])
    grad_rgb = np.stack([gradient * (t / 255) for t in tint], axis=2).astype(np.uint8)
    grad_img = Image.fromarray(grad_rgb, mode="RGB")
    bg = Image.blend(bg, grad_img, 0.55)

    draw = ImageDraw.Draw(bg)

    # fonts
    try:
        font_small = ImageFont.truetype("arial.ttf", 18)
        font_big = ImageFont.truetype("arial.ttf", 24)
    except:
        font_small = ImageFont.load_default()
        font_big = ImageFont.load_default()

    # ✅ replacement for draw.textsize()
    def text_size(txt, font):
        try:
            return font.getsize(txt)   # Pillow <10
        except:
            return draw.textbbox((0, 0), txt, font=font)[2:]  # Pillow >=10

    # --- domain pill ---
    pill = f" {domain.title()} "
    tw, th = text_size(pill, font_small)
    draw.rounded_rectangle((8, 8, 8 + tw + 12, 8 + th + 6),
                           radius=10, fill=(255, 255, 255))
    draw.text((14, 10), pill, fill=(0, 0, 0), font=font_small)

    # --- item title (bottom center) ---
    title = name[:22]
    tw2, th2 = text_size(title, font_big)
    draw.text(((W - tw2) / 2, H - th2 - 22),
              title, fill=(255, 255, 255), font=font_big)

    return bg

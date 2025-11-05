# ui_texts.py
import random

CHEEKY_CAPTIONS = [
    "Hot pick. Zero regrets.",
    "Chef’s kiss material.",
    "Trust the vibes.",
    "Tiny click. Big vibe.",
    "Your next favorite, probably.",
    "Spicy pick. Handle with care.",
    "Low effort, high serotonin.",
]

def cheesy() -> str:
    return random.choice(CHEEKY_CAPTIONS)

def mood_line(hour: int) -> str:
    if 0 <= hour < 5:  return "Night owl mode detected. We brought the cozy stuff."
    if 5 <= hour < 9:  return "Fresh start energy. Let’s line up feel-good picks."
    if 9 <= hour < 12: return "Peak focus hours. Clean, crisp recommendations incoming."
    if 12 <= hour < 16:return "Post-lunch lull? We’ve got dopamine on tap."
    if 16 <= hour < 20:return "Golden hour glow. Time for crowd-pleasers."
    return "Late night cravings unlocked. S-tier comfort picks only."

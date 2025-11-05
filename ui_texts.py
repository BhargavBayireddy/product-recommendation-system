# ui_texts.py
from __future__ import annotations
import random
from datetime import datetime

CHEESY = [
    "Chef's kiss material 😘",
    "Hot pick. Zero regrets 🔥",
    "Trust the vibes. You’ll love it ✨",
    "You had me at ‘Add to Bag’ 😏",
    "Flirting with perfection 😉",
    "Recommended by your future self 💌",
    "This belongs in your life. And cart. 🛒",
    "Viral for a reason. Try it 😮‍💨",
]

COLLAB_TITLES = [
    "🔥 Vibe-twins also loved…",
    "People with your taste grabbed these too 🫶",
    "Taste-match picks just for you 🎯",
]

EXPLORE_TITLES = [
    "Explore something different",
    "Break your pattern. Try these 💫",
    "Plot twist recs 🎬",
]

def pick_line() -> str:
    return random.choice(CHEESY)

def collab_header() -> str:
    return random.choice(COLLAB_TITLES)

def explore_header() -> str:
    return random.choice(EXPLORE_TITLES)

def mood_label(now: datetime, keystrokes: int) -> str:
    hr = now.hour
    if keystrokes >= 30:
        return "curious"
    if 0 <= hr < 6:
        return "night-owl"
    if 6 <= hr < 12:
        return "fresh"
    if 12 <= hr < 18:
        return "buzzy"
    return "chill"

def mood_copy(tag: str) -> str:
    return {
        "curious": "You’re on a roll. Here’s deeper stuff to binge 🤓",
        "night-owl": "Late scrolls deserve late treats 🌙",
        "fresh": "Morning magic. Smart picks to start strong ☀️",
        "buzzy": "Mid-day mojo. Keep the streak alive ⚡",
        "chill": "Easy mode engaged. Cozy selections for you 🧸",
    }[tag]

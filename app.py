import os, time, random, math, hashlib, json
from typing import Dict, List, Tuple
from functools import lru_cache

import requests
import pandas as pd
import streamlit as st

# -------------------------
# App constants & helpers
# -------------------------
APP_TITLE = "ReccoVerse"
PRIMARY_GRADIENT = "linear-gradient(90deg, #19C3FB 0%, #7B2FF7 100%)"
DARK_BG = "#0b1220"
CARD_BG = "#0f172a"
TEXT = "#E6EDF6"
SUBTEXT = "#9FB0C1"
ACCENT = "#7B2FF7"

def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def uid_of(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

# -------------------------
# Styling
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
st.markdown(f"""
    <style>
        html, body, [class*="block-container"] {{
            background: radial-gradient(1200px 600px at 20% 0%, #0d1730 0%, {DARK_BG} 35%, {DARK_BG} 100%) !important;
            color: {TEXT};
        }}
        .title-hero {{
            font-weight: 800; letter-spacing: .5px;
            font-size: 40px; margin: 8px 0 24px 0;
        }}
        .row-title {{
            font-weight: 700; font-size: 22px; margin: 16px 0 8px 0;
        }}
        .pill {{
            padding: 6px 12px; border-radius: 999px; font-size: 12px; 
            background: rgba(123,47,247,.12); color: #c8baff; border: 1px solid rgba(123,47,247,.35);
            display: inline-flex; gap: 8px; align-items: center;
        }}
        .btn-grad {{
            background: {PRIMARY_GRADIENT};
            color: white; font-weight: 700; border: none; padding: 10px 16px; 
            border-radius: 999px; box-shadow: 0 6px 22px rgba(123,47,247,.35);
        }}
        .card {{
            background: {CARD_BG}; border: 1px solid #111a2e; border-radius: 16px; overflow: hidden;
            transition: all .25s ease; position: relative;
        }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 26px rgba(0,0,0,.4); }}
        .meta {{ color: {SUBTEXT}; font-size: 12px; }}
        .nov-wrap {{ height: 8px; background: #111a2e; border-radius: 8px; overflow: hidden; }}
        .nov-fill {{ height: 8px; background: {PRIMARY_GRADIENT}; }}
        .searchbox input {{
            background: #0f1528 !important; color: {TEXT} !important; border: 1px solid #223254 !important;
        }}
        .sidebar .stButton>button {{
            background: {PRIMARY_GRADIENT}; color: white; border: 0; border-radius: 999px; font-weight: 700;
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Secrets & tokens
# -------------------------
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET", "")
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

# -------------------------
# Session State
# -------------------------
_ss("liked_ids", set())
_ss("bag_ids", set())
_ss("surprise", False)
_ss("search", "")
_ss("user_profile", {"uid": "guest"})

# -------------------------
# Spotify Client Credentials
# -------------------------
@st.cache_data(ttl=3200, show_spinner=False)
def spotify_token() -> str:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return ""
    url = "https://accounts.spotify.com/api/token"
    resp = requests.post(
        url,
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json().get("access_token", "")
    return ""

# -------------------------
# TMDB helpers
# -------------------------
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

@st.cache_data(ttl=1800, show_spinner=False)
def tmdb_popular() -> List[dict]:
    if not TMDB_API_KEY:
        return []
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []

@st.cache_data(ttl=1200, show_spinner=False)
def tmdb_search(q: str) -> List[dict]:
    if not TMDB_API_KEY or not q.strip():
        return []
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(q)}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []

# -------------------------
# Spotify helpers
# -------------------------
@st.cache_data(ttl=1200, show_spinner=False)
def spotify_search_tracks(q: str) -> List[dict]:
    tok = spotify_token()
    if not tok or not q.strip():
        return []
    url = f"https://api.spotify.com/v1/search?type=track&limit=20&q={requests.utils.quote(q)}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=15)
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
        return items
    except Exception:
        return []

@st.cache_data(ttl=1200, show_spinner=False)
def spotify_popular_seed() -> List[dict]:
    """Use a fixed popular-ish seed query to populate a first row."""
    tok = spotify_token()
    if not tok:
        return []
    url = "https://api.spotify.com/v1/search?type=track&limit=20&q=top%20hits"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, timeout=15)
        r.raise_for_status()
        return r.json().get("tracks", {}).get("items", [])
    except Exception:
        return []

# -------------------------
# Product mock + future hooks
# -------------------------
MOCK_PRODUCTS = [
    {
        "id": f"prod-{i}",
        "title": t,
        "brand": b,
        "img": img,
        "category": c,
        "price": p,
        "url": u
    }
    for i, (t, b, img, c, p, u) in enumerate([
        ("Wireless Over-Ear Headphones", "Novaclear", "https://images.unsplash.com/photo-1518449180961-6d4a67759e1a?q=80&w=800", "Electronics", 69.0, "https://example.com/p1"),
        ("Lightweight Running Shoes", "RoadRun", "https://images.unsplash.com/photo-1542291026-7eec264c27ff?q=80&w=800", "Footwear", 49.0, "https://example.com/p2"),
        ("Stainless Water Bottle 1L", "HydroPro", "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?q=80&w=800", "Outdoors", 19.0, "https://example.com/p3"),
        ("Cotton Graphic Tee", "UrbanInk", "https://images.unsplash.com/photo-1520975682031-137cc8f6f0a5?q=80&w=800", "Apparel", 15.0, "https://example.com/p4"),
        ("Smart Fitness Band", "FitLoop", "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?q=80&w=800", "Wearables", 39.0, "https://example.com/p5"),
    ])
]

@st.cache_data(ttl=900, show_spinner=False)
def product_search(q: str) -> List[dict]:
    """Future: swap this with Walmart/Amazon search.
       For now: fuzzy contains on mock catalog."""
    if not q.strip():
        return MOCK_PRODUCTS
    ql = q.lower()
    out = []
    for p in MOCK_PRODUCTS:
        blob = " ".join([p["title"], p["brand"], p["category"]]).lower()
        if ql in blob:
            out.append(p)
    return out

# -------------------------
# Unify into a single catalog
# -------------------------
def normalize_movie(m: dict) -> dict:
    mid = f"movie-{m.get('id')}"
    img = m.get("poster_path")
    return {
        "id": mid,
        "kind": "movie",
        "title": m.get("title") or m.get("name") or "Untitled",
        "subtitle": "Movies • Film",
        "img": TMDB_IMG + img if img else "https://images.unsplash.com/photo-1497032628192-86f99bcd76bc?w=800",
        "tags": [g for g in [m.get("original_language"), "movie"] if g],
        "score": m.get("vote_average", 0),
        "meta": f"{m.get('release_date', '')[:4] or ''}",
        "url": f"https://www.themoviedb.org/movie/{m.get('id')}",
    }

def normalize_track(t: dict) -> dict:
    tid_raw = t.get("id", "")
    tid = f"track-{tid_raw}"
    img = ""
    try:
        album_imgs = t.get("album", {}).get("images", [])
        if album_imgs:
            img = album_imgs[0]["url"]
    except Exception:
        pass
    artists = ", ".join([a["name"] for a in t.get("artists", [])]) or "Artist"
    return {
        "id": tid,
        "kind": "music",
        "title": t.get("name") or "Untitled",
        "subtitle": artists,
        "img": img or "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=800",
        "tags": [a["name"] for a in t.get("artists", [])][:3] + ["track"],
        "score": t.get("popularity", 0),
        "meta": "Music • Track",
        "url": t.get("external_urls", {}).get("spotify", "https://open.spotify.com/"),
    }

def normalize_product(p: dict) -> dict:
    return {
        "id": p["id"],
        "kind": "product",
        "title": p["title"],
        "subtitle": p["brand"],
        "img": p["img"],
        "tags": [p["category"], "product"],
        "score": p["price"],
        "meta": f"₹{int(round(float(p['price'])))}",
        "url": p["url"],
    }

def novelty_score(item_tags: List[str], liked_tag_pool: List[str]) -> float:
    """Quanta-style lightweight novelty = 1 - (overlap / union) based on tags."""
    a = set(t.lower() for t in item_tags)
    b = set(t.lower() for t in liked_tag_pool)
    if not a:
        return 0.5
    union = len(a.union(b)) or 1
    overlap = len(a.intersection(b))
    return max(0.0, min(1.0, 1.0 - (overlap / union)))

def gather_liked_tag_pool(catalog: List[dict]) -> List[str]:
    liked = [x for x in catalog if x["id"] in st.session_state.liked_ids]
    pool = []
    for x in liked:
        pool.extend(x.get("tags", []))
    return pool[:100]

# -------------------------
# Build the homepage rows
# -------------------------
def fetch_catalog(search_q: str = "") -> Dict[str, List[dict]]:
    # Movies
    movies = tmdb_search(search_q) if search_q else tmdb_popular()
    movies_norm = [normalize_movie(m) for m in (movies or [])][:20]

    # Music
    tracks = spotify_search_tracks(search_q) if search_q else spotify_popular_seed()
    tracks_norm = [normalize_track(t) for t in (tracks or [])][:20]

    # Products (mock or future API)
    prods = product_search(search_q)
    prods_norm = [normalize_product(p) for p in prods][:20]

    return {"movies": movies_norm, "music": tracks_norm, "products": prods_norm}

# -------------------------
# UI pieces
# -------------------------
def like_bag_row(item: dict):
    left, right = st.columns([1, 1])
    liked = item["id"] in st.session_state.liked_ids
    in_bag = item["id"] in st.session_state.bag_ids

    if left.button(("❤️" if liked else "♡  Like"), key=f"like_{item['id']}"):
        if liked:
            st.session_state.liked_ids.remove(item["id"])
        else:
            st.session_state.liked_ids.add(item["id"])
        st.rerun()

    if right.button(("🛍️ In Bag" if in_bag else "＋  Bag"), key=f"bag_{item['id']}"):
        if in_bag:
            st.session_state.bag_ids.remove(item["id"])
        else:
            st.session_state.bag_ids.add(item["id"])
        st.rerun()

def render_card(item: dict, liked_pool: List[str]):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(item["img"], use_container_width=True)
        st.markdown(f"<div style='padding:10px 12px'>"
                    f"<div style='font-weight:700'>{item['title']}</div>"
                    f"<div class='meta'>{item['subtitle']}</div>", unsafe_allow_html=True)
        # Novelty
        nov = novelty_score(item.get("tags", []), liked_pool)
        st.markdown(f"""
            <div style='padding:0 12px 8px 12px'>
              <div class='pill'>🧭 Novelty {int(nov*100)}%</div>
              <div style='height:10px'></div>
              <div class='nov-wrap'><div class='nov-fill' style='width:{int(nov*100)}%'></div></div>
            </div>
        """, unsafe_allow_html=True)
        with st.container():
            like_bag_row(item)
        st.markdown("</div>", unsafe_allow_html=True)

def render_row(title: str, items: List[dict], liked_pool: List[str]):
    if not items:
        return
    st.markdown(f"<div class='row-title'>{title}</div>", unsafe_allow_html=True)
    # 5 cards per row
    chunks = [items[i:i+5] for i in range(0, len(items), 5)]
    for row in chunks:
        cols = st.columns(5, gap="large")
        for c, it in zip(cols, row):
            with c:
                render_card(it, liked_pool)

def sidebar():
    with st.sidebar:
        st.markdown("## ReccoVerse")
        st.button("Sign Out", key="signout")
        st.markdown("#### ")
        st.text_input("Search anything…", key="search", placeholder="movies, artists, shoes…", label_visibility="visible")
        st.checkbox("📈 Surprise Mode (shuffle)", key="surprise")

        # Likes & Bag quick peek
        if st.session_state.liked_ids:
            st.markdown("### ❤️ Liked")
            st.caption(f"{len(st.session_state.liked_ids)} items")
        if st.session_state.bag_ids:
            st.markdown("### 🛍️ Bag")
            st.caption(f"{len(st.session_state.bag_ids)} items")

def top_header():
    st.markdown(f"<div class='title-hero'>🍿 Top Picks For You</div>", unsafe_allow_html=True)

def page_likes_bag(catalog_all: List[dict]):
    st.markdown("## ❤️ Your Liked")
    liked = [x for x in catalog_all if x["id"] in st.session_state.liked_ids]
    pool = gather_liked_tag_pool(catalog_all)
    render_row("Liked Items", liked, pool)

    st.markdown("## 🛍️ Your Bag")
    bag = [x for x in catalog_all if x["id"] in st.session_state.bag_ids]
    render_row("Bag Items", bag, pool)

# -------------------------
# MAIN
# -------------------------
def main():
    sidebar()
    st.markdown(f"<h1 style='margin-top:-10px'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    top_header()

    q = st.session_state.search.strip()
    cat = fetch_catalog(q)

    # Surprise mode randomize
    if st.session_state.surprise:
        random.seed(42)
        for k in cat:
            random.shuffle(cat[k])

    # Build a single list for liked-pool computation
    all_items = cat["movies"] + cat["music"] + cat["products"]
    liked_pool = gather_liked_tag_pool(all_items)

    # Rows
    render_row("🎬 Popular Movies", cat["movies"], liked_pool)
    render_row("🎵 Top Music Tracks", cat["music"], liked_pool)
    render_row("🛍️ Trending Products", cat["products"], liked_pool)

    # Because you liked
    if st.session_state.liked_ids:
        # simple filter: different kind than what you liked + higher novelty
        liked_kinds = {x["kind"] for x in all_items if x["id"] in st.session_state.liked_ids}
        recs = [x for x in all_items if x["kind"] not in liked_kinds]
        recs.sort(key=lambda r: novelty_score(r.get("tags", []), liked_pool), reverse=True)
        render_row("💖 Because You Liked These", recs[:10], liked_pool)

    st.markdown("---")
    if st.button("Open Liked & Bag", key="open_lb"):
        st.session_state["__lb"] = True
        st.rerun()

    if st.session_state.get("__lb"):
        page_likes_bag(all_items)
        if st.button("Close", key="close_lb"):
            st.session_state["__lb"] = False
            st.rerun()

if __name__ == "__main__":
    main()

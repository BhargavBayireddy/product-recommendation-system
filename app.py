import os, time, random, math, textwrap, functools
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import requests

# ------------------------------
# Config & Theme
# ------------------------------
st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")

PRIMARY_GRADIENT = "linear-gradient(90deg, #26C6DA, #7C4DFF)"  # Fresh, not Jio/Hotstar colors
BG_DARK = "#0B1220"
CARD_BG = "#121A2B"

APP_TITLE = """
<style>
body, .stApp { background: """ + BG_DARK + """; }
h1,h2,h3,h4 { color: #EAF2FF; }
.small { color:#9FB3D9; font-size:0.9rem; }
.badge { background:#1F2A44; padding:4px 10px; border-radius:12px; color:#BFD1FF; }
.gradbtn {
  background: """ + PRIMARY_GRADIENT + """;
  color: white; padding: 8px 16px; border-radius: 999px; border: none;
}
.card {
  background: """ + CARD_BG + """;
  border: 1px solid #1B2540; border-radius: 18px; padding: 14px;
}
.thumb { border-radius: 12px; object-fit: cover; }
.metricbar {
  height:10px; border-radius:8px; background:#12203C; overflow:hidden;
}
.metricbar > div { height:100%; background: """ + PRIMARY_GRADIENT + """; }
.rowtitle { display:flex; align-items:center; gap:10px; }
.rowtitle .emoji { font-size:1.3rem; }
.searchbox input { color:#EAF2FF !important; }
</style>
"""

st.markdown(APP_TITLE, unsafe_allow_html=True)

# ------------------------------
# Secrets / Keys
# ------------------------------
TMDB_KEY = st.secrets.get("TMDB_API_KEY")
SPOTIFY_ID = st.secrets.get("SPOTIFY_CLIENT_ID")
SPOTIFY_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET")

# ------------------------------
# Session Init
# ------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("liked", set())
    ss.setdefault("bag", set())
    ss.setdefault("surprise", False)
    ss.setdefault("search", "")
    ss.setdefault("pool", [])         # unified item pool
    ss.setdefault("byid", {})         # id-> item
    ss.setdefault("last_rebuilt_at", 0.0)

_init_state()

# ------------------------------
# Utilities
# ------------------------------
def uid(domain: str, raw_id: Any) -> str:
    return f"{domain}:{raw_id}"

def human(s: str, limit=70) -> str:
    return (s[:limit] + "…") if len(s) > limit else s

def novelty_score(item_tags: List[str], liked_tags: List[str]) -> float:
    """1.0 = very novel, 0.0 = same as likes (simple Jaccard complement)."""
    if not item_tags:
        return 0.6
    A, B = set(item_tags), set(liked_tags)
    if not A and not B:
        return 0.6
    inter = len(A & B)
    union = len(A | B) or 1
    sim = inter / union
    return float(max(0.0, 1.0 - sim))

def ensure_rerun():
    # Aggressive updates like Netflix UI:
    st.rerun()

# ------------------------------
# Data Sources
# ------------------------------
def fetch_tmdb_movies() -> List[Dict[str, Any]]:
    # If key missing, return small curated demo
    if not TMDB_KEY:
        demo = [
            dict(id=101, domain="movie", title="Black Phone 2", subtitle="Movies • Film",
                 image="https://image.tmdb.org/t/p/w500/8fLC4n7LxQ5Zp7p8DDRD9YfL6bE.jpg",
                 url="https://www.themoviedb.org/",
                 tags=["horror","thriller","kidnap"]),
            dict(id=102, domain="movie", title="Frankenstein", subtitle="Movies • Film",
                 image="https://image.tmdb.org/t/p/w500/1SgK3Zf8dSV2mVqf0K0ePpI6LwQ.jpg",
                 url="https://www.themoviedb.org/",
                 tags=["classic","monster","gothic"]),
        ]
        return demo

    try:
        url = f"https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"
        r = requests.get(url, headers={"Authorization": f"Bearer {TMDB_KEY}"}, timeout=12)
        r.raise_for_status()
        results = r.json().get("results", [])[:20]
        out = []
        for m in results:
            out.append(dict(
                id=m["id"],
                domain="movie",
                title=m.get("title") or m.get("name") or "Untitled",
                subtitle="Movies • Film",
                image=(f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}"
                       if m.get("poster_path") else ""),
                url=f"https://www.themoviedb.org/movie/{m.get('id')}",
                tags=[g.lower() for g in ["movie","popular"]]
            ))
        return out
    except Exception:
        return []

def _spotify_token() -> str:
    if not (SPOTIFY_ID and SPOTIFY_SECRET):
        return ""
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type":"client_credentials"},
            auth=(SPOTIFY_ID, SPOTIFY_SECRET),
            timeout=12
        )
        r.raise_for_status()
        return r.json().get("access_token","")
    except Exception:
        return ""

def fetch_spotify_tracks() -> List[Dict[str, Any]]:
    token = _spotify_token()
    if not token:
        # fallback demo
        return [
            dict(id="sp1", domain="music", title="Blue Eyes", subtitle="Music • Track",
                 image="https://i.scdn.co/image/ab67616d0000b273d3c2f0e0ad57f9a7f5e3fabc",
                 url="https://open.spotify.com/",
                 tags=["pop","romance"]),
            dict(id="sp2", domain="music", title="Life is Beautiful", subtitle="Music • Track",
                 image="https://i.scdn.co/image/ab67616d0000b2731b1b4b6a83f3f9d9db1c02e1",
                 url="https://open.spotify.com/",
                 tags=["feelgood","indie"]),
        ]
    try:
        # Simple: get a popular editorial playlist's first tracks (Global Top 50)
        r = requests.get(
            "https://api.spotify.com/v1/playlists/37i9dQZEVXbMDoHDwVN2tF/tracks?limit=20",
            headers={"Authorization": f"Bearer {token}"},
            timeout=12
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        out = []
        for it in items:
            tr = it.get("track") or {}
            out.append(dict(
                id=tr.get("id") or f"t{random.randint(1,1e9)}",
                domain="music",
                title=tr.get("name","Track"),
                subtitle="Music • Track",
                image=(tr.get("album",{}).get("images",[{"url":""}])[0]["url"] or ""),
                url=tr.get("external_urls",{}).get("spotify","https://open.spotify.com"),
                tags=[a.get("name","") for a in tr.get("artists",[])]
            ))
        return out
    except Exception:
        return []

def fetch_products_mock() -> List[Dict[str, Any]]:
    # Replace later with Walmart/Flipkart once credentials are approved.
    # Keep small catalog for demo.
    cats = [
        dict(id=f"p{n}", domain="product",
             title=random.choice(["NoiseFit Nova","Boat Airdopes 141","Redmi Note Case",
                                  "Puma Running Shoes","Dell Backpack 15"]),
             subtitle="Products • E-commerce",
             image=random.choice([
                 "https://images.unsplash.com/photo-1519741497674-611481863552?q=80&w=600&auto=format&fit=crop",
                 "https://images.unsplash.com/photo-1516726817505-f5ed825624d8?q=80&w=600&auto=format&fit=crop",
                 "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?q=80&w=600&auto=format&fit=crop",
             ]),
             url="https://www.flipkart.com/",
             tags=random.choice([["electronics","audio"],["mobile","case"],["shoes","running"],["bag","laptop"]]))
        for n in range(1, 18)
    ]
    return cats

# ------------------------------
# Build Unified Pool
# ------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def build_pool() -> List[Dict[str, Any]]:
    movies  = fetch_tmdb_movies()
    music   = fetch_spotify_tracks()
    prods   = fetch_products_mock()
    pool = movies + music + prods

    # Give every item a tags list and novelty default
    for it in pool:
        it.setdefault("tags", [])
        it["novelty"] = 0.7  # baseline; recomputed per-user below
    return pool

def index_pool(pool: List[Dict[str,Any]]) -> Dict[str, Dict[str,Any]]:
    return { uid(it["domain"], it["id"]) : it for it in pool }

# ------------------------------
# Recommender (lightweight, fast)
# ------------------------------
def liked_tags(byid: Dict[str,Any], liked_ids: List[str]) -> List[str]:
    tags = []
    for k in liked_ids:
        it = byid.get(k)
        if it: tags += it.get("tags", [])
    return [t for t in pd.Series(tags).value_counts().index.tolist()]

def score_item(it: Dict[str,Any], liked_tag_list: List[str], surprise: bool) -> float:
    n = novelty_score(it.get("tags", []), liked_tag_list)
    base = 0.5
    # If user liked similar tags, boost (unless surprise mode)
    overlap = len(set(it.get("tags", [])) & set(liked_tag_list))
    sim_boost = 0.15 * overlap
    if surprise:
        return base + n * 0.7 - sim_boost * 0.3
    return base + sim_boost * 0.5 + n * 0.25

def recommend(pool, byid, liked_ids, k=20, surprise=False):
    lt = liked_tags(byid, list(liked_ids))
    candidates = []
    for it in pool:
        iid = uid(it["domain"], it["id"])
        if iid in liked_ids:
            continue
        candidates.append((score_item(it, lt, surprise), it))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in candidates[:k]]

# ------------------------------
# UI: Controls
# ------------------------------
def sidebar_controls():
    with st.sidebar:
        st.markdown("## ReccoVerse")
        st.button("Sign Out", use_container_width=True, type="secondary", key="signout_btn")

        st.markdown("#### Search anything…")
        q = st.text_input("", key="search", placeholder="movie, artist, shoes…", label_visibility="collapsed")
        st.checkbox("Surprise Mode (shuffle)", key="surprise")
        st.markdown('<div class="small">Tip: Try queries like <b>romance</b>, <b>phone</b>, <b>Taylor</b>.</div>', unsafe_allow_html=True)

# ------------------------------
# UI Helpers
# ------------------------------
def novelty_bar(n: float):
    n = max(0.0, min(1.0, n))
    st.markdown(
        f"""
        <div class="metricbar"><div style="width:{int(n*100)}%"></div></div>
        <div class="small">Novelty {int(n*100)}%</div>
        """, unsafe_allow_html=True
    )

def like_bag_row(item: Dict[str,Any]):
    iid = uid(item["domain"], item["id"])
    left, right = st.columns([1,1], vertical_alignment="center")

    liked = iid in st.session_state.liked
    bagged = iid in st.session_state.bag

    if left.button(("❤️" if liked else "♡  Like"), key=f"like_{iid}"):
        if liked:
            st.session_state.liked.remove(iid)
        else:
            st.session_state.liked.add(iid)
        ensure_rerun()

    if right.button(("👜  In Bag" if bagged else "＋  Bag"), key=f"bag_{iid}"):
        if bagged:
            st.session_state.bag.remove(iid)
        else:
            st.session_state.bag.add(iid)
        ensure_rerun()

def card(item: Dict[str,Any]):
    # Image + text
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        img = item.get("image") or "https://via.placeholder.com/600x800?text=No+Image"
        st.image(img, use_container_width=True, clamp=True)
        st.markdown(f"**{human(item['title'], 60)}**")
        st.markdown(f"<div class='small'>{item.get('subtitle','')}</div>", unsafe_allow_html=True)
        novelty_bar(item.get("novelty", 0.6))
        like_bag_row(item)
        st.markdown("</div>", unsafe_allow_html=True)

def render_row(title: str, items: List[Dict[str,Any]], anchor: str):
    st.markdown(f"""<div class="rowtitle"><span class="emoji">🍿</span><h3>{title}</h3></div>""", unsafe_allow_html=True)
    if not items:
        st.info("No items found for this section.")
        return

    # Grid 5 per row
    per_row = 5
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        chunk = items[i:i+per_row]
        for c, it in zip(cols, chunk):
            with c:
                card(it)

# ------------------------------
# Search / Filter
# ------------------------------
def filter_items(pool: List[Dict[str,Any]], q:str) -> List[Dict[str,Any]]:
    q = (q or "").strip().lower()
    if not q:
        return pool
    out = []
    for it in pool:
        hay = " ".join([
            it.get("title",""), it.get("subtitle",""),
            " ".join(it.get("tags",[]))
        ]).lower()
        if q in hay:
            out.append(it)
    return out

# ------------------------------
# Pages
# ------------------------------
def page_home(all_items: List[Dict[str,Any]], byid: Dict[str,Any]):
    # Compute novelty per item for display relative to user's current likes
    lt = liked_tags(byid, list(st.session_state.liked))
    for it in all_items:
        it["novelty"] = novelty_score(it.get("tags",[]), lt)

    # Search filter
    items = filter_items(all_items, st.session_state.search)

    # Popular rows by domain
    movies  = [it for it in items if it["domain"]=="movie"][:15]
    music   = [it for it in items if it["domain"]=="music"][:15]
    prods   = [it for it in items if it["domain"]=="product"][:15]

    recs = recommend(items, byid, st.session_state.liked, k=20, surprise=st.session_state.surprise)

    st.markdown("## 🍿 Top Picks For You")
    if movies: render_row("Popular Movies", movies, "m")
    if music:  render_row("Top Music Tracks", music, "mu")
    if prods:  render_row("Trending Products", prods, "p")

    # Personalized
    if st.session_state.liked:
        render_row("Because You Liked These", recs, "rec")

def page_likes_bag(all_items: List[Dict[str,Any]], byid: Dict[str,Any]):
    liked   = [byid[i] for i in st.session_state.liked if i in byid]
    bagged  = [byid[i] for i in st.session_state.bag if i in byid]

    st.markdown("## ❤️ Liked & 👜 Bag")
    render_row("Liked Items", liked, "liked")
    render_row("In Your Bag", bagged, "bag")

# ------------------------------
# App
# ------------------------------
def main():
    sidebar_controls()

    # (Re)build pool when empty or every 15 minutes
    now = time.time()
    if (now - st.session_state.last_rebuilt_at) > 900 or not st.session_state.pool:
        pool = build_pool()
        st.session_state.pool = pool
        st.session_state.byid = index_pool(pool)
        st.session_state.last_rebuilt_at = now

    pool = st.session_state.pool
    byid = st.session_state.byid

    # Nav (Netflix keeps it single page; we show two tabs)
    tabs = st.tabs(["🏠 Home", "❤️ Liked & 👜 Bag"])
    with tabs[0]:
        page_home(pool, byid)
    with tabs[1]:
        page_likes_bag(pool, byid)

if __name__ == "__main__":
    main()

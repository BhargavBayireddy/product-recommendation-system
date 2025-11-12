# ReccoVerse — AI-powered multi-domain recommendation dashboard (Diagnostic version)
# Adds debug panels for API key & data validation

import os, json, time, random, base64
from typing import List, Dict, Any
import requests
import streamlit as st

# ========== PAGE SETUP ==========
st.set_page_config(page_title="ReccoVerse", page_icon="🍿", layout="wide")

# ---- Netflix Dark Theme ----
st.markdown("""
<style>
:root {
  --bg: #0f1321;
  --bg2: #0b1020;
  --border: #1e293b;
  --text: #e2e8f0;
  --subtle: #94a3b8;
}
html, body, .block-container { background: var(--bg) !important; color: var(--text) !important; }
.card { background: var(--bg2); border-radius:18px; padding:12px; border:1px solid var(--border); }
img.poster { width:100%; height:240px; object-fit:cover; border-radius:12px; background:#0a0f1a; }
.grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:24px; }
.progress-wrap { background:#0b1020; border-radius:9999px; height:10px; border:1px solid #1e293b; }
.progress { background:linear-gradient(90deg,#38bdf8,#a78bfa); height:100%; border-radius:9999px; }
.rowhead { display:flex; align-items:center; gap:.5rem; margin:.25rem 0 .75rem 0 }
.rowhead h2 { margin:0; }
</style>
""", unsafe_allow_html=True)

# ========== SECRETS ==========
TMDB_API = st.secrets.get("APIS", {}).get("TMDB_API_KEY", "")
SPOTIFY_ID = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_SECRET = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_SECRET", "")
FIREBASE_WEB = st.secrets.get("FIREBASE_WEB_CONFIG", {})
FIREBASE_SA = st.secrets.get("FIREBASE_SERVICE_ACCOUNT", {})

# ========== FIREBASE INIT (Optional) ==========
FIREBASE_READY = False
try:
    import pyrebase
    import firebase_admin
    from firebase_admin import credentials

    if FIREBASE_WEB and FIREBASE_SA:
        firebase = pyrebase.initialize_app({
            "apiKey": FIREBASE_WEB["apiKey"],
            "authDomain": FIREBASE_WEB["authDomain"],
            "projectId": FIREBASE_WEB["projectId"],
            "storageBucket": FIREBASE_WEB.get("storageBucket", ""),
            "messagingSenderId": FIREBASE_WEB.get("messagingSenderId", ""),
            "appId": FIREBASE_WEB["appId"],
            "databaseURL": f"https://{FIREBASE_WEB['projectId']}.firebaseio.com"
        })
        _auth = firebase.auth()
        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_SA)
            firebase_admin.initialize_app(cred)
        FIREBASE_READY = True
except Exception as e:
    FIREBASE_READY = False

# ========== SESSION DEFAULTS ==========
def _ensure_state():
    ss = st.session_state
    ss.setdefault("authed", True)  # enable mock mode for now
    ss.setdefault("uid", "debug-user")
    ss.setdefault("liked", set())
    ss.setdefault("bag", set())
    ss.setdefault("query", "")
    ss.setdefault("surprise", False)
_ensure_state()

# ========== API FETCHERS ==========
@st.cache_data(ttl=3600)
def tmdb_popular():
    if not TMDB_API:
        return []
    try:
        url = "https://api.themoviedb.org/3/movie/popular"
        r = requests.get(url, params={"api_key": TMDB_API, "language": "en-US", "page": 1}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json().get("results", [])[:20]
        out = []
        for m in data:
            out.append({
                "id": f"mv_{m['id']}",
                "title": m.get("title", ""),
                "type": "movie",
                "subtitle": "Movies • Film",
                "img": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get("poster_path") else "",
                "novelty": random.uniform(0.2, 1.0)
            })
        return out
    except Exception as e:
        st.error(f"TMDB fetch failed: {e}")
        return []

@st.cache_data(ttl=3400)
def spotify_token():
    if not (SPOTIFY_ID and SPOTIFY_SECRET):
        return ""
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_ID, SPOTIFY_SECRET),
        timeout=15
    )
    if r.status_code != 200:
        st.error("Spotify token failed: " + r.text)
        return ""
    return r.json().get("access_token", "")

@st.cache_data(ttl=1200)
def spotify_top_tracks():
    token = spotify_token()
    if not token:
        return []
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get("https://api.spotify.com/v1/playlists/37i9dQZF1DXcBWIGoYBM5M", headers=headers, timeout=15)
    if r.status_code != 200:
        st.error("Spotify fetch failed: " + r.text)
        return []
    tracks = []
    for it in r.json().get("tracks", {}).get("items", [])[:20]:
        t = it.get("track", {})
        img = t.get("album", {}).get("images", [{}])[0].get("url", "")
        tracks.append({
            "id": f"sp_{t.get('id','')}",
            "title": t.get("name", ""),
            "type": "music",
            "subtitle": "Music • Track",
            "img": img,
            "novelty": random.uniform(0.3, 1.0)
        })
    return tracks

@st.cache_data(ttl=3600)
def trending_products():
    return [
        {"id": "pd_1", "title": "Nike Air Winflo 10", "type": "product", "subtitle": "Products • Shoes", "img": "https://static.nike.com/a/images/t_PDP_864_v1/f_auto,q_auto:eco/e3f09c1b-63f1-4a9f-8f21-7a7f0a60f6f3/air-winflo-10-road-running-shoes.png", "novelty": 0.7},
        {"id": "pd_2", "title": "boAt Airdopes 141", "type": "product", "subtitle": "Products • Audio", "img": "https://cdn.shopify.com/s/files/1/0057/8938/4802/files/Airdopes-141-black.png?v=171", "novelty": 0.6},
        {"id": "pd_3", "title": "Redmi Note 13 Pro", "type": "product", "subtitle": "Products • Phone", "img": "https://i02.appmifile.com/mi-com-product/fly-birds/redmi-note-13-pro/pc/7399a7.png", "novelty": 0.9},
    ]

# ========== COMBINE ==========
def all_items():
    items = tmdb_popular() + spotify_top_tracks() + trending_products()
    return items

# ========== RENDER ==========
def render_card(it):
    st.markdown(f"<div class='card'><img class='poster' src='{it['img']}'/><div><b>{it['title']}</b><br><span style='color:#94a3b8'>{it['subtitle']}</span></div></div>", unsafe_allow_html=True)

def render_row(title, items):
    if items:
        st.markdown(f"<div class='rowhead'><h2>{title}</h2></div>", unsafe_allow_html=True)
        st.markdown("<div class='grid'>", unsafe_allow_html=True)
        for it in items:
            render_card(it)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"No data fetched for {title}")

# ========== MAIN ==========
def main():
    st.title("🍿 ReccoVerse (Diagnostic Mode)")

    # --- DEBUG PANEL ---
    with st.expander("🔍 Debug: Secret Configs", expanded=False):
        st.write({
            "TMDB_API_KEY": TMDB_API,
            "SPOTIFY_CLIENT_ID": SPOTIFY_ID,
            "SPOTIFY_CLIENT_SECRET": SPOTIFY_SECRET,
            "Firebase Web Config": bool(FIREBASE_WEB),
            "Firebase Service Account": bool(FIREBASE_SA),
            "Firebase Ready": FIREBASE_READY,
        })

    movies = tmdb_popular()
    music = spotify_top_tracks()
    prods = trending_products()

    with st.expander("📦 Debug: API Fetch Counts", expanded=True):
        st.write({
            "Movies fetched": len(movies),
            "Music fetched": len(music),
            "Products fetched": len(prods),
        })

    st.markdown("---")

    render_row("🎬 Popular Movies", movies)
    render_row("🎵 Top Music Tracks", music)
    render_row("🛍 Trending Products", prods)

if __name__ == "__main__":
    main()

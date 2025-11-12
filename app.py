# ReccoVerse — AI-powered multi-domain recommendation dashboard
# Streamlit Cloud ready

import os, json, time, random, base64
from typing import List, Dict, Any, Tuple
from functools import lru_cache

import requests
import streamlit as st

# ========== PAGE SETUP ==========
st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")

# ---- Dark Netflix-like Theme & UI ----
st.markdown("""
<style>
:root {
  --bg: #0f1321;
  --bg2: #0b1020;
  --border: #1e293b;
  --text: #e2e8f0;
  --subtle: #94a3b8;
  --muted: #64748b;
}
html, body, .block-container { background: var(--bg) !important; color: var(--text) !important; }
.block-container { padding-top: 1rem; }
h1,h2,h3 { letter-spacing: .2px; margin: 0 0 .25rem 0 }
small.muted { color: var(--muted) }
.input-dark input, .input-dark textarea { color: var(--text) !important; }
.topbar { display:flex; align-items:center; gap:.5rem; }
.badge {
  display:inline-flex; align-items:center; gap:.35rem; font-weight:600;
  background: var(--bg2); color:#cbd5e1; padding:6px 10px; border-radius:999px;
  border:1px solid var(--border); font-size:.85rem; margin-left:.35rem;
}
.card {
  background: var(--bg);
  border-radius: 18px; padding: 14px; border: 1px solid var(--border);
  transition: transform .1s ease;
}
.card:hover { transform: translateY(-2px); }
.title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-top:.4rem; line-height:1.2 }
.subtle { color: var(--subtle); font-size:.9rem }
.grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 24px; }
img.poster { width: 100%; height: 240px; object-fit: cover; border-radius: 12px; background: #0a0f1a }
.rowhead { display:flex; align-items:center; gap:.6rem; margin:.25rem 0 .75rem 0 }
.rowhead h2 { margin:0; }
.progress-wrap { background:var(--bg2); border-radius:9999px; height:10px; border:1px solid var(--border) }
.progress { background:linear-gradient(90deg,#38bdf8,#a78bfa); height:100%; border-radius:9999px }
hr.hr-soft { border:0; border-top:1px solid var(--border); margin: 1.2rem 0; }
a.card-link { text-decoration:none; color:inherit; }
</style>
""", unsafe_allow_html=True)

# ========== SECRETS & KEYS ==========
# Expect these in Streamlit Secrets (see bottom of this file for template)
TMDB_API = st.secrets.get("APIS", {}).get("TMDB_API_KEY", "")
SPOTIFY_ID = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_SECRET = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_SECRET", "")

# ========== FIREBASE AUTH ==========
FIREBASE_READY = False
_auth = None

def _init_firebase():
    """Initialize pyrebase (client) + firebase_admin (server)."""
    global FIREBASE_READY, _auth
    try:
        import pyrebase
        import firebase_admin
        from firebase_admin import credentials

        WEB = dict(st.secrets.get("FIREBASE_WEB_CONFIG", {}))
        SA = dict(st.secrets.get("FIREBASE_SERVICE_ACCOUNT", {}))
        if not (WEB and SA):
            FIREBASE_READY = False
            return

        # pyrebase client (email/password)
        firebase = pyrebase.initialize_app({
            "apiKey": WEB["apiKey"],
            "authDomain": WEB["authDomain"],
            "projectId": WEB["projectId"],
            "storageBucket": WEB.get("storageBucket", ""),
            "messagingSenderId": WEB.get("messagingSenderId",""),
            "appId": WEB["appId"],
            "databaseURL": f"https://{WEB['projectId']}.firebaseio.com"
        })
        _auth = firebase.auth()

        # firebase_admin
        if not firebase_admin._apps:
            cred = credentials.Certificate(SA)
            firebase_admin.initialize_app(cred)

        FIREBASE_READY = True
    except Exception:
        FIREBASE_READY = False

_init_firebase()

# ========== SESSION DEFAULTS ==========
def _ensure_state():
    ss = st.session_state
    ss.setdefault("authed", False)
    ss.setdefault("uid", None)
    ss.setdefault("liked", set())
    ss.setdefault("bag", set())
    ss.setdefault("surprise", False)
    ss.setdefault("query", "")
_ensure_state()

# ========== IMAGE FALLBACK HANDLING ==========
# 24x24 dark placeholder PNG (base64) — ensures images "never break"
_PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAVUlEQVR4nO3UsQkAIBAEQXv/"
    "l1wzq1o0sXcQmV1kqkC2wqk9oYzJkJQ4Nw8k8t9m0H3yYgB+4Qy8nWQzvX9K4G8l4TtT1fE0w1"
    "m8H0p2m3G0g2G8F4K7I2k0JXQb7vQ8mC8f3c9b3wF7sCw2z2a4gAAAABJRU5ErkJggg=="
)

def _placeholder_data_uri() -> str:
    return "data:image/png;base64," + _PLACEHOLDER_PNG_B64

@lru_cache(maxsize=256)
def _image_is_ok(url: str) -> bool:
    """HEAD check; tolerate some CDNs that block HEAD by falling back to GET with stream."""
    if not url:
        return False
    try:
        r = requests.head(url, timeout=6)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", "").lower():
            return True
        # Fallback: some CDNs disallow HEAD
        r = requests.get(url, stream=True, timeout=8)
        return r.status_code == 200 and "image" in r.headers.get("Content-Type","").lower()
    except Exception:
        return False

def safe_img(url: str) -> str:
    return url if _image_is_ok(url) else _placeholder_data_uri()

# ========== APIS (CACHED) ==========
@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_popular() -> List[Dict[str, Any]]:
    if not TMDB_API:
        return []
    url = "https://api.themoviedb.org/3/movie/popular"
    params = {"api_key": TMDB_API, "language": "en-US", "page": 1}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    out = []
    for m in r.json().get("results", [])[:20]:
        poster = f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get("poster_path") else ""
        out.append({
            "id": f"mv_{m['id']}",
            "type": "movie",
            "title": m.get("title") or m.get("name",""),
            "subtitle": "Movies • Film",
            "img": poster,
            "href": f"https://www.themoviedb.org/movie/{m['id']}",
            "novelty": random.uniform(0.15, 1.0)
        })
    # Validate images / apply fallback
    for it in out:
        it["img"] = safe_img(it["img"])
    return out

@st.cache_data(ttl=3400, show_spinner=False)
def spotify_token() -> str:
    if not (SPOTIFY_ID and SPOTIFY_SECRET):
        return ""
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_ID, SPOTIFY_SECRET),
        timeout=15
    )
    r.raise_for_status()
    return r.json().get("access_token","")

@st.cache_data(ttl=1200, show_spinner=False)
def spotify_top_tracks() -> List[Dict[str, Any]]:
    tok = spotify_token()
    if not tok:
        return []
    headers = {"Authorization": f"Bearer {tok}"}
    tracks: List[Dict[str, Any]] = []

    # Editorial playlists: Today's Top Hits + Pop Rising
    for playlist in ("37i9dQZF1DXcBWIGoYBM5M", "37i9dQZF1DX0XUsuxWHRQd"):
        r = requests.get(f"https://api.spotify.com/v1/playlists/{playlist}", headers=headers, timeout=15)
        if r.status_code != 200:
            continue
        items = r.json().get("tracks", {}).get("items", [])
        for it in items[:20]:
            t = it.get("track") or {}
            # Album art (official from Spotify CDN)
            img = ""
            try:
                imgs = t.get("album", {}).get("images", [])
                if imgs:
                    img = imgs[0]["url"]
            except Exception:
                img = ""
            entry = {
                "id": f"sp_{t.get('id','x')}",
                "type": "music",
                "title": t.get("name", ""),
                "subtitle": "Music • Track",
                "img": img,
                "href": t.get("external_urls", {}).get("spotify", ""),
                "novelty": random.uniform(0.6, 1.0)
            }
            entry["img"] = safe_img(entry["img"])
            if entry["title"]:
                tracks.append(entry)

    # Deduplicate by id, keep first 20
    seen = set()
    deduped = []
    for x in tracks:
        if x["id"] not in seen:
            deduped.append(x)
            seen.add(x["id"])
        if len(deduped) >= 20:
            break
    return deduped

@st.cache_data(ttl=3600, show_spinner=False)
def trending_products() -> List[Dict[str, Any]]:
    """
    Product thumbnails from official CDNs (validated) with official site links.
    If any URL fails validation, a data-URI placeholder is used.
    """
    catalog: List[Tuple[str, str, str, str]] = [
        # (Title, Subtitle, ImageURL (official CDN), Link (official))
        ("Nike Air Winflo 10", "Products • Shoes",
         "https://static.nike.com/a/images/t_PDP_864_v1/f_auto,q_auto:eco/e3f09c1b-63f1-4a9f-8f21-7a7f0a60f6f3/air-winflo-10-road-running-shoes.png",
         "https://www.nike.com/"),
        ("boAt Airdopes 141", "Products • Audio",
         "https://cdn.shopify.com/s/files/1/0057/8938/4802/files/Airdopes-141-black.png?v=171",
         "https://www.boat-lifestyle.com/"),
        ("Redmi Note 13 Pro", "Products • Phone",
         "https://i02.appmifile.com/mi-com-product/fly-birds/redmi-note-13-pro/pc/7399a7.png",
         "https://www.mi.com/"),
        ("Wildcraft 45L Backpack", "Products • Bags",
         "https://cdn.wildcraft.com/product/images/large/12269_1.jpg",
         "https://wildcraft.com/"),
        ("H&M Oversized Tee", "Products • Fashion",
         "https://www2.hm.com/content/dam/men_s06/july_2024/8031/8031-3x2-1.jpg",
         "https://www2.hm.com/")
    ]
    out = []
    for i, (t, sub, img, href) in enumerate(catalog):
        out.append({
            "id": f"pd_{i}",
            "type": "product",
            "title": t,
            "subtitle": sub,
            "img": safe_img(img),
            "href": href,
            "novelty": random.uniform(0.3, 0.9)
        })
    return out

# ========== COMBINE POOL ==========
def all_items() -> List[Dict[str, Any]]:
    items = []
    items += tmdb_popular()
    items += spotify_top_tracks()
    items += trending_products()
    # Ensure absolutely no empty images
    for it in items:
        if not it.get("img"):
            it["img"] = _placeholder_data_uri()
    return items

# ========== FUTURE GNN/AI HOOK ==========
@st.cache_data(ttl=1200, show_spinner=False)
def gnn_score_recommendations(items: List[Dict[str, Any]], liked_ids: List[str]) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Placeholder for the future QUNTA Recommendation Engine.
    Return list of (score, item). For now, we blend category affinity + novelty.
    """
    liked_ids_set = set(liked_ids)
    liked_types = {it["type"] for it in items if it["id"] in liked_ids_set}
    scored = []
    for x in items:
        base = 1.0
        if x["type"] in liked_types and liked_types:
            base += 1.2  # category affinity
        base += float(x.get("novelty", 0.5)) * 0.6
        scored.append((base + random.random() * 0.15, x))
    scored.sort(key=lambda z: z[0], reverse=True)
    return scored

def because_you_liked(items: List[Dict[str, Any]], liked_ids: set) -> List[Dict[str, Any]]:
    if not liked_ids:
        # cold start: pick a diversified shuffle of top novelty
        tmp = sorted(items, key=lambda x: x.get("novelty", 0), reverse=True)
        random.shuffle(tmp)
        return tmp[:15]
    scored = gnn_score_recommendations(items, list(liked_ids))
    return [x for s, x in scored if x["id"] not in liked_ids][:15]

# ========== AUTH UI ==========
def _signed(uid: str):
    st.session_state.authed = True
    st.session_state.uid = uid
    st.rerun()

def login_screen():
    st.markdown("<h1>🍿 ReccoVerse</h1><small class='muted'>AI-curated picks across Movies • Music • Products</small>", unsafe_allow_html=True)
    st.write("")
    with st.container():
        col1, col2 = st.columns([1.2, 1])
        with col2:
            st.caption(f"Backend: {'Firebase' if FIREBASE_READY else 'Local mock'}")
            email = st.text_input("Email", key="auth_email", placeholder="you@example.com")
            pwd = st.text_input("Password", type="password", key="auth_pwd", placeholder="min 6 chars")

            c1, c2 = st.columns(2)
            if FIREBASE_READY:
                if c1.button("Login", use_container_width=True, key="btn_login"):
                    try:
                        user = _auth.sign_in_with_email_and_password(email, pwd)
                        _signed(user["localId"])
                    except Exception:
                        st.error("Login failed. Check email/password.")
                if c2.button("Create Account", use_container_width=True, key="btn_signup"):
                    try:
                        _auth.create_user_with_email_and_password(email, pwd)
                        user = _auth.sign_in_with_email_and_password(email, pwd)
                        _signed(user["localId"])
                    except Exception:
                        st.error("Sign-up failed. Use a valid email & 6+ char password.")
            else:
                # Local mock (for development if Firebase not configured)
                if c1.button("Login (mock)", use_container_width=True, key="btn_login_mock"):
                    if email:
                        _signed(email)
                    else:
                        st.error("Enter any email to continue in mock mode.")
                if c2.button("Create (mock)", use_container_width=True, key="btn_signup_mock"):
                    if email:
                        _signed(email)
                    else:
                        st.error("Enter any email to continue in mock mode.")

# ========== UI HELPERS ==========
def novelty_bar(v: float, key_suffix: str):
    pct = int(min(max(float(v), 0.0), 1.0) * 100)
    st.markdown(
        f"<div class='subtle' style='margin:.35rem 0 .3rem 0'>Novelty {pct}%</div>"
        f"<div class='progress-wrap'><div class='progress' style='width:{pct}%'></div></div>",
        unsafe_allow_html=True
    )

def like_bag_row(item: Dict[str, Any], key_prefix: str):
    lid = item["id"]
    liked = lid in st.session_state.liked
    bagged = lid in st.session_state.bag
    c1, c2 = st.columns(2)
    if c1.button(("❤️ Liked" if liked else "♡  Like"), key=f"{key_prefix}-like-{lid}"):
        if liked:
            st.session_state.liked.remove(lid)
        else:
            st.session_state.liked.add(lid)
        st.rerun()
    if c2.button(("✔ In Bag" if bagged else "👜  Add to Bag"), key=f"{key_prefix}-bag-{lid}"):
        if bagged:
            st.session_state.bag.remove(lid)
        else:
            st.session_state.bag.add(lid)
        st.rerun()

def render_card(item: Dict[str, Any], key_prefix: str):
    # Whole card — no duplicate keys: use id-suffixed keys for controls only
    img_html = f"<img class='poster' src='{item['img']}'/>"
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if item.get("href"):
            st.markdown(f"<a class='card-link' href='{item['href']}' target='_blank'>{img_html}</a>", unsafe_allow_html=True)
        else:
            st.markdown(img_html, unsafe_allow_html=True)
        st.markdown(f"<div class='title'>{item.get('title','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtle'>{item.get('subtitle','')}</div>", unsafe_allow_html=True)
        novelty_bar(item.get("novelty", 0.5), key_suffix=item["id"])
        like_bag_row(item, key_prefix=key_prefix)
        st.markdown("</div>", unsafe_allow_html=True)

def render_row(title: str, items: List[Dict[str, Any]], key_prefix: str):
    if title:
        st.markdown(f"<div class='rowhead'><h2>{title}</h2>"
                    f"<span class='badge'>grid ×5</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='grid'>", unsafe_allow_html=True)
    for it in items:
        render_card(it, key_prefix=key_prefix)
    st.markdown("</div>", unsafe_allow_html=True)

# ========== FILTERING / SURPRISE ==========
def apply_filters(pool: List[Dict[str, Any]], query: str, surprise: bool) -> List[Dict[str, Any]]:
    items = pool[:]
    q = (query or "").strip().lower()
    if q:
        def _match(x):
            hay = " ".join([
                x.get("title", ""),
                x.get("subtitle", ""),
                x.get("type", "")
            ]).lower()
            return q in hay
        items = [x for x in items if _match(x)]
    if surprise:
        random.shuffle(items)
    return items

# ========== PAGES ==========
def page_home(pool: List[Dict[str, Any]]):
    # Rows per spec (Netflix style) — Popular Movies, Top Music Tracks, Trending Products, Because You Liked These
    movies = [x for x in pool if x["type"] == "movie"][:10]
    music  = [x for x in pool if x["type"] == "music"][:10]
    prods  = [x for x in pool if x["type"] == "product"][:10]
    liked_recs = because_you_liked(pool, st.session_state.liked)

    if movies: render_row("🎬 Popular Movies", movies, key_prefix="row-movie")
    if music:
        st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
        render_row("🎵 Top Music Tracks", music, key_prefix="row-music")
    if prods:
        st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
        render_row("🛍 Trending Products", prods, key_prefix="row-product")
    if liked_recs:
        st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
        render_row("💡 Because You Liked These", liked_recs[:15], key_prefix="row-because")

def page_liked(pool: List[Dict[str, Any]]):
    liked_items = [x for x in pool if x["id"] in st.session_state.liked]
    if not liked_items:
        st.info("No likes yet. Go to Home and tap ♡ Like on items you enjoy!")
        return
    render_row("❤️ Liked", liked_items, key_prefix="row-liked")

def page_bag(pool: List[Dict[str, Any]]):
    bag_items = [x for x in pool if x["id"] in st.session_state.bag]
    if not bag_items:
        st.info("Your bag is empty. Add items with 👜 Add to Bag.")
        return
    render_row("🛍 Bag", bag_items, key_prefix="row-bag")

# ========== SIDEBAR ==========
def sidebar_controls():
    with st.sidebar:
        st.text_input("Search anything", key="query", placeholder="movies, tracks, phones, tees…", help="Filters by title, subtitle, or type")
        st.checkbox("Surprise Mode (shuffle)", key="surprise")
        st.write("")
        st.button("Sign Out", key="btn_signout", type="secondary",
                  on_click=lambda: (st.session_state.clear(), _ensure_state(), st.experimental_rerun()))

# ========== MAIN ==========
def main():
    if not st.session_state.authed:
        login_screen()
        return

    # Top heading
    st.markdown("<div class='topbar'><h1>🍿 ReccoVerse</h1>"
                "<span class='badge'>AI picks</span>"
                "<span class='badge'>Movies</span>"
                "<span class='badge'>Music</span>"
                "<span class='badge'>Products</span></div>", unsafe_allow_html=True)
    st.markdown("<small class='muted'>Log in session is stored safely in session_state (uid, likes, bag).</small>", unsafe_allow_html=True)

    sidebar_controls()

    # Load + filter
    pool = all_items()
    filtered = apply_filters(pool, st.session_state.query, st.session_state.surprise)

    # Top tabs: Home | Liked | Bag
    t1, t2, t3 = st.tabs(["🏠 Home", "❤️ Liked", "🛍 Bag"])
    with t1:
        page_home(filtered)
    with t2:
        page_liked(filtered)
    with t3:
        page_bag(filtered)

if __name__ == "__main__":
    main()

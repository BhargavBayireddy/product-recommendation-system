# ReccoVerse — AI-powered multi-domain recommendation dashboard
# Final production version (Streamlit Cloud ready)

import random
import base64
from functools import lru_cache
from typing import List, Dict, Any, Tuple

import requests
import streamlit as st

# =================== PAGE & THEME ===================
st.set_page_config(page_title="ReccoVerse", page_icon="🍿", layout="wide")

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
.rowhead { display:flex; align-items:center; gap:.6rem; margin:.25rem 0 .75rem 0 }
.rowhead h2 { margin:0; }
.grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 24px; }
.card {
  background: var(--bg);
  border-radius: 18px; padding: 14px; border: 1px solid var(--border);
  transition: transform .1s ease;
}
.card:hover { transform: translateY(-2px); }
img.poster { width: 100%; height: 240px; object-fit: cover; border-radius: 12px; background: #0a0f1a }
.title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-top:.4rem; line-height:1.2 }
.subtle { color: var(--subtle); font-size:.9rem }
.progress-wrap { background:var(--bg2); border-radius:9999px; height:10px; border:1px solid var(--border) }
.progress { background:linear-gradient(90deg,#38bdf8,#a78bfa); height:100%; border-radius:9999px }
hr.hr-soft { border:0; border-top:1px solid var(--border); margin: 1.2rem 0; }
a.card-link { text-decoration:none; color:inherit; }
.btn-like, .btn-bag { width:100%; padding:6px 10px; border-radius:10px; border:1px solid var(--border); background:var(--bg2); }
.btn-like:hover, .btn-bag:hover { filter:brightness(1.15); }
</style>
""", unsafe_allow_html=True)

# =================== SECRETS ===================
TMDB_API = st.secrets.get("APIS", {}).get("TMDB_API_KEY", "")
SPOTIFY_ID = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_SECRET = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_SECRET", "")

# =================== FIREBASE (EMAIL/PASSWORD) ===================
FIREBASE_READY = False
_auth = None
def _init_firebase():
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
        firebase = pyrebase.initialize_app({
            "apiKey": WEB["apiKey"],
            "authDomain": WEB["authDomain"],
            "projectId": WEB["projectId"],
            "storageBucket": WEB.get("storageBucket", ""),
            "messagingSenderId": WEB.get("messagingSenderId",""),
            "appId": WEB["appId"],
            "databaseURL": WEB.get("databaseURL") or f"https://{WEB['projectId']}.firebaseio.com"
        })
        _auth = firebase.auth()
        import firebase_admin
        if not firebase_admin._apps:
            cred = credentials.Certificate(SA)
            firebase_admin.initialize_app(cred)
        FIREBASE_READY = True
    except Exception:
        FIREBASE_READY = False
_init_firebase()

# =================== SESSION DEFAULTS ===================
def _ensure_state():
    ss = st.session_state
    ss.setdefault("authed", False)
    ss.setdefault("uid", None)
    ss.setdefault("liked", set())
    ss.setdefault("bag", set())
    ss.setdefault("surprise", False)
    ss.setdefault("query", "")
_ensure_state()

# =================== IMAGE FALLBACK ===================
_PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAAVUlEQVR4nO3UsQkAIBAEQXv/"
    "l1wzq1o0sXcQmV1kqkC2wqk9oYzJkJQ4Nw8k8t9m0H3yYgB+4Qy8nWQzvX9K4G8l4TtT1fE0w1"
    "m8H0p2m3G0g2G8F4K7I2k0JXQb7vQ8mC8f3c9b3wF7sCw2z2a4gAAAABJRU5ErkJggg=="
)
def _placeholder() -> str:
    return "data:image/png;base64," + _PLACEHOLDER_PNG_B64

@lru_cache(maxsize=1024)
def _img_ok(url: str) -> bool:
    if not url:
        return False
    try:
        r = requests.head(url, timeout=6)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type","").lower():
            return True
        r = requests.get(url, stream=True, timeout=8)
        return r.status_code == 200 and "image" in r.headers.get("Content-Type","").lower()
    except Exception:
        return False

def safe_img(url: str) -> str:
    return url if _img_ok(url) else _placeholder()

# =================== APIS (CACHED) ===================
# -------- TMDB: pull many pages (20 per page) --------
@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_popular_many(target_count: int = 1000) -> List[Dict[str, Any]]:
    if not TMDB_API:
        return []
    out: List[Dict[str, Any]] = []
    page = 1
    # TMDB allows up to 500 pages for many lists; keep a sane upper bound
    while len(out) < target_count and page <= 60:
        r = requests.get(
            "https://api.themoviedb.org/3/movie/popular",
            params={"api_key": TMDB_API, "language": "en-US", "page": page},
            timeout=15
        )
        if r.status_code != 200:
            break
        for m in r.json().get("results", []):
            poster = f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get("poster_path") else ""
            out.append({
                "id": f"mv_{m['id']}",
                "type": "movie",
                "title": m.get("title") or m.get("name",""),
                "subtitle": "Movies • Film",
                "img": safe_img(poster),
                "href": f"https://www.themoviedb.org/movie/{m['id']}",
                "novelty": random.uniform(0.15, 1.0)
            })
        page += 1
    # Dedup by id just in case
    seen = set()
    ded = []
    for x in out:
        if x["id"] not in seen:
            seen.add(x["id"])
            ded.append(x)
        if len(ded) >= target_count:
            break
    return ded

# -------- Spotify (aggregate many editorial playlists) --------
SPOTIFY_PLAYLISTS = [
    # Global hits + regional; adding many to reach ~1000 unique tracks
    "37i9dQZF1DXcBWIGoYBM5M",  # Today's Top Hits
    "37i9dQZEVXbNG2KDcFcKOF",  # Global Top 50
    "37i9dQZEVXbMDoHDwVN2tF",  # Viral 50 Global
    "37i9dQZF1DX0XUsuxWHRQd",  # Pop Rising
    "37i9dQZF1DX1ngEVM0lKrb",  # All Out 2010s
    "37i9dQZF1DX4dyzvuaRJ0n",  # Rock Classics
    "37i9dQZF1DWWxPM4nWdhy3",  # Chill Hits
    "37i9dQZF1DWXRqgorJj26U",  # Lofi Beats
    "37i9dQZF1DX3JRswwCJn24",  # Bollywood Butter
    "37i9dQZF1DX0XUfTFmNBRM",  # Punjabi 101
    "37i9dQZF1DXaE9T3ob22rZ",  # Hot Country
    "37i9dQZF1DX4JAvHpjipBk",  # RapCaviar
    "37i9dQZF1DX2RxBh64BHjQ",  # Mint
    "37i9dQZF1DX4Wpq3hjjFPA",  # Dance Hits
    "37i9dQZF1DX4sWSpwq3LiO",  # Mega Hit Mix
    "37i9dQZF1DWXmlLSKkfdAk",  # New Music Friday
    "37i9dQZF1DX2pSTOxoPbx9",  # Peaceful Piano
    "37i9dQZF1DX8FwnYE6PRvL",  # Deep Focus
    "37i9dQZF1DWTJ7xPn4vNaz",  # Beast Mode
    "37i9dQZF1DXaKIA8E7WcJj",  # Songs to Sing in the Car
]

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
    if r.status_code != 200:
        return ""
    return r.json().get("access_token","")

@st.cache_data(ttl=1800, show_spinner=False)
def spotify_many_tracks(target_count: int = 1000) -> List[Dict[str, Any]]:
    tok = spotify_token()
    if not tok:
        return []
    headers = {"Authorization": f"Bearer {tok}"}
    out: List[Dict[str, Any]] = []
    seen = set()
    for pid in SPOTIFY_PLAYLISTS:
        r = requests.get(f"https://api.spotify.com/v1/playlists/{pid}", headers=headers, timeout=20)
        if r.status_code != 200:
            continue
        items = r.json().get("tracks", {}).get("items", [])
        for it in items:
            t = it.get("track") or {}
            tid = t.get("id") or ""
            if not tid or tid in seen:
                continue
            seen.add(tid)
            img = ""
            try:
                imgs = t.get("album", {}).get("images", [])
                if imgs:
                    img = imgs[0]["url"]
            except Exception:
                img = ""
            out.append({
                "id": f"sp_{tid}",
                "type": "music",
                "title": t.get("name",""),
                "subtitle": "Music • Track",
                "img": safe_img(img),
                "href": t.get("external_urls",{}).get("spotify",""),
                "novelty": random.uniform(0.4, 1.0)
            })
            if len(out) >= target_count:
                return out
    return out

# -------- Products (official CDNs) --------
@st.cache_data(ttl=3600, show_spinner=False)
def trending_products() -> List[Dict[str, Any]]:
    # Curated stable images from official brand CDNs / established stores
    catalog = [
        ("Nike Air Winflo 10", "Products • Shoes",
         "https://static.nike.com/a/images/t_PDP_864_v1/f_auto,q_auto:eco/e3f09c1b-63f1-4a9f-8f21-7a7f0a60f6f3/air-winflo-10-road-running-shoes.png", "https://www.nike.com/"),
        ("boAt Airdopes 141", "Products • Audio",
         "https://cdn.shopify.com/s/files/1/0057/8938/4802/files/Airdopes-141-black.png?v=171", "https://www.boat-lifestyle.com/"),
        ("Redmi Note 13 Pro", "Products • Phone",
         "https://i02.appmifile.com/mi-com-product/fly-birds/redmi-note-13-pro/pc/7399a7.png", "https://www.mi.com/"),
        ("Wildcraft 45L Backpack", "Products • Bags",
         "https://m.media-amazon.com/images/I/61A6zK02jXL._AC_UL600_FMwebp_QL65_.jpg", "https://wildcraft.com/"),
        ("H&M Oversized Tee", "Products • Fashion",
         "https://www2.hm.com/content/dam/men_s06/july_2024/8031/8031-3x2-1.jpg", "https://www2.hm.com/"),
        # add more items (can extend as needed)
        ("Nike Revolution 7", "Products • Shoes",
         "https://static.nike.com/a/images/t_PDP_864_v1/f_auto,q_auto:eco/2b9a9d20-9b7d-44ef-a5d6-a9c09d4b01f3/revolution-7-road-running-shoes.png", "https://www.nike.com/"),
        ("boAt Rockerz 255 Pro+", "Products • Audio",
         "https://cdn.shopify.com/s/files/1/0057/8938/4802/files/Rockerz-255-Pro-Plus-black.png?v=169", "https://www.boat-lifestyle.com/"),
        ("Redmi Buds 5", "Products • Audio",
         "https://i02.appmifile.com/mi-com-product/images/redmi-buds-5-white.png", "https://www.mi.com/"),
        ("H&M Cargo Joggers", "Products • Fashion",
         "https://www2.hm.com/content/dam/men_s06/2024_b/8079e/8079e-3x2-1.jpg", "https://www2.hm.com/")
    ]
    out=[]
    for i,(t,sub,img,href) in enumerate(catalog):
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

# =================== DATA COMBINERS ===================
def gather_pool() -> List[Dict[str, Any]]:
    items = []
    items += tmdb_popular_many(1000)
    items += spotify_many_tracks(1000)
    items += trending_products()
    for it in items:
        if not it.get("img"):
            it["img"] = _placeholder()
    return items

# =================== GNN / RANKING ===================
@st.cache_data(ttl=1200, show_spinner=False)
def gnn_score_recommendations(items: List[Dict[str, Any]], liked_ids: List[str]) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Placeholder for QUNTA Recommendation Engine:
    score = 0.6*novelty + 0.4*category_affinity + small_noise
    """
    liked_ids_set = set(liked_ids)
    liked_types = {it["type"] for it in items if it["id"] in liked_ids_set}
    scored = []
    for x in items:
        novelty = float(x.get("novelty", 0.5))
        aff = 1.0 + (0.4 if x["type"] in liked_types and liked_types else 0.0)
        score = novelty * 0.6 + aff * 0.4 + random.random() * 0.05
        scored.append((score, x))
    scored.sort(key=lambda z: z[0], reverse=True)
    return scored

def because_you_liked(items: List[Dict[str, Any]], liked_ids: set) -> List[Dict[str, Any]]:
    if not liked_ids:
        return sorted(items, key=lambda x: x.get("novelty", 0), reverse=True)[:20]
    ranked = gnn_score_recommendations(items, list(liked_ids))
    return [x for s,x in ranked if x["id"] not in liked_ids][:20]

# =================== UI HELPERS ===================
def novelty_bar(v: float):
    pct = int(min(max(float(v), 0.0), 1.0) * 100)
    st.markdown(
        f"<div class='subtle' style='margin:.35rem 0 .3rem 0'>Novelty {pct}%</div>"
        f"<div class='progress-wrap'><div class='progress' style='width:{pct}%'></div></div>",
        unsafe_allow_html=True
    )

def like_bag_buttons(item: Dict[str, Any], key_prefix: str):
    lid = item["id"]
    liked = lid in st.session_state.liked
    bagged = lid in st.session_state.bag
    c1, c2 = st.columns(2)
    if c1.button(("❤️ Liked" if liked else "♡  Like"), key=f"{key_prefix}-like-{lid}", use_container_width=True):
        if liked: st.session_state.liked.remove(lid)
        else: st.session_state.liked.add(lid)
        st.rerun()
    if c2.button(("✔ In Bag" if bagged else "👜  Add to Bag"), key=f"{key_prefix}-bag-{lid}", use_container_width=True):
        if bagged: st.session_state.bag.remove(lid)
        else: st.session_state.bag.add(lid)
        st.rerun()

def render_card(item: Dict[str, Any], ctx: str):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if item.get("href"):
            st.markdown(f"<a class='card-link' href='{item['href']}' target='_blank'><img class='poster' src='{item['img']}'/></a>", unsafe_allow_html=True)
        else:
            st.markdown(f"<img class='poster' src='{item['img']}'/>", unsafe_allow_html=True)
        st.markdown(f"<div class='title'>{item.get('title','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtle'>{item.get('subtitle','')}</div>", unsafe_allow_html=True)
        novelty_bar(item.get("novelty", 0.5))
        like_bag_buttons(item, key_prefix=ctx)
        st.markdown("</div>", unsafe_allow_html=True)

def render_row(title: str, items: List[Dict[str, Any]], ctx: str):
    st.markdown(f"<div class='rowhead'><h2>{title}</h2><span class='badge'>grid ×5</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='grid'>", unsafe_allow_html=True)
    for it in items:
        render_card(it, ctx=ctx)
    st.markdown("</div>", unsafe_allow_html=True)

def sidebar_controls():
    with st.sidebar:
        st.text_input("Search anything", key="query", placeholder="movies, tracks, phones, tees…",
                      help="Filters by title, subtitle, or type")
        st.checkbox("Surprise Mode (shuffle)", key="surprise")
        st.write("")
        st.button("Sign Out", key="btn_signout", type="secondary",
                  on_click=lambda: (st.session_state.clear(), _ensure_state(), st.experimental_rerun()))

def filter_pool(pool: List[Dict[str, Any]], query: str, surprise: bool) -> List[Dict[str, Any]]:
    items = pool[:]
    q = (query or "").strip().lower()
    if q:
        def _match(x):
            return q in (x.get("title","") + " " + x.get("subtitle","") + " " + x.get("type","")).lower()
        items = [x for x in items if _match(x)]
    if surprise:
        random.shuffle(items)
    return items

# =================== AUTH UI ===================
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
                if c1.button("Login (mock)", use_container_width=True, key="btn_login_mock"):
                    if email: _signed(email)
                    else: st.error("Enter any email to continue in mock mode.")
                if c2.button("Create (mock)", use_container_width=True, key="btn_signup_mock"):
                    if email: _signed(email)
                    else: st.error("Enter any email to continue in mock mode.")

# =================== PAGES ===================
def page_home(pool: List[Dict[str, Any]]):
    movies = [x for x in pool if x["type"] == "movie"][:200]
    music  = [x for x in pool if x["type"] == "music"][:200]
    prods  = [x for x in pool if x["type"] == "product"][:30]
    liked_recs = because_you_liked(pool, st.session_state.liked)

    if movies: render_row("🎬 Popular Movies", movies, ctx="row-movie")
    if music:
        st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
        render_row("🎵 Top Music Tracks", music, ctx="row-music")
    if prods:
        st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
        render_row("🛍 Trending Products", prods, ctx="row-product")
    if liked_recs:
        st.markdown("<hr class='hr-soft'/>", unsafe_allow_html=True)
        render_row("💡 Because You Liked These", liked_recs[:20], ctx="row-because")

def page_liked(pool: List[Dict[str, Any]]):
    liked_items = [x for x in pool if x["id"] in st.session_state.liked]
    if not liked_items:
        st.info("No likes yet. Go to Home and tap ♡ Like on items you enjoy!")
        return
    render_row("❤️ Liked", liked_items, ctx="row-liked")

def page_bag(pool: List[Dict[str, Any]]):
    bag_items = [x for x in pool if x["id"] in st.session_state.bag]
    if not bag_items:
        st.info("Your bag is empty. Add items with 👜 Add to Bag.")
        return
    render_row("🛍 Bag", bag_items, ctx="row-bag")

# =================== MAIN ===================
def main():
    if not st.session_state.authed:
        login_screen()
        return

    st.markdown("<div class='topbar'><h1>🍿 ReccoVerse</h1>"
                "<span class='badge'>AI picks</span>"
                "<span class='badge'>Movies</span>"
                "<span class='badge'>Music</span>"
                "<span class='badge'>Products</span></div>", unsafe_allow_html=True)
    st.markdown("<small class='muted'>Your session is stored safely in memory (uid, likes, bag).</small>", unsafe_allow_html=True)

    sidebar_controls()

    # Fetch + Filter (cached)
    pool = gather_pool()
    filtered = filter_pool(pool, st.session_state.query, st.session_state.surprise)

    # Tabs
    t1, t2, t3 = st.tabs(["🏠 Home", "❤️ Liked", "🛍 Bag"])
    with t1: page_home(filtered)
    with t2: page_liked(filtered)
    with t3: page_bag(filtered)

if __name__ == "__main__":
    main()

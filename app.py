import os, json, time, random, textwrap, base64
from typing import List, Dict, Any
import requests
import streamlit as st

# ---------- UI THEME TWEAK ----------
st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")
st.markdown("""
<style>
:root { --card-bg: #0f1321; --chip: linear-gradient(135deg,#38bdf8,#a78bfa); }
.block-container { padding-top: 1rem; }
h1,h2,h3 { letter-spacing: 0.2px; }
.btn-chip {
  background: var(--chip); color: white; padding: 8px 16px; border-radius: 9999px;
  border: 0; font-weight: 600;
}
.badge {
  display:inline-flex; align-items:center; gap:.35rem; font-weight:600;
  background:#0b1020; color:#cbd5e1; padding:6px 10px; border-radius:999px;
  border:1px solid #1e293b; font-size:.85rem
}
.card {
  background: var(--card-bg);
  border-radius: 18px; padding: 14px; border: 1px solid #1e293b;
}
.title { font-size: 1.05rem; font-weight: 700; color:#e2e8f0; margin-top:.4rem }
.subtle { color:#94a3b8; font-size:.9rem }
.grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 24px; }
img.poster { width: 100%; height: 240px; object-fit: cover; border-radius: 12px; }
.skel { width:100%; height: 14px; background: #0b1020; border-radius: 9999px; }
.progress-wrap { background:#0b1020; border-radius:9999px; height:10px; border:1px solid #1e293b }
.progress { background:linear-gradient(90deg,#38bdf8,#a78bfa); height:100%; border-radius:9999px }
.rowhead { display:flex; align-items:center; gap:.6rem }
.rowhead h2 { margin:0; }
.input-dark input { color:#e2e8f0 !important; }
small.muted { color:#64748b }
</style>
""", unsafe_allow_html=True)

# ============== SECRETS & KEYS ==============
TMDB_API = st.secrets.get("APIS", {}).get("TMDB_API_KEY", "")
SPOTIFY_ID = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_SECRET = st.secrets.get("APIS", {}).get("SPOTIFY_CLIENT_SECRET", "")

# ============== FIREBASE INIT (email/password) ==============
FIREBASE_READY = False
_auth = _db = None
try:
    import pyrebase
    import firebase_admin
    from firebase_admin import credentials, auth as admin_auth

    WEB = dict(st.secrets.get("FIREBASE_WEB_CONFIG", {}))
    SA = dict(st.secrets.get("FIREBASE_SERVICE_ACCOUNT", {}))
    if WEB and SA:
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
        if not firebase_admin._apps:
            cred = credentials.Certificate(SA)
            firebase_admin.initialize_app(cred)
        FIREBASE_READY = True
except Exception as e:
    FIREBASE_READY = False  # Fall back to local-mock below


# ============== SESSION DEFAULTS ==============
def _ensure_state():
    ss = st.session_state
    ss.setdefault("authed", False)
    ss.setdefault("uid", None)
    ss.setdefault("liked", set())
    ss.setdefault("bag", set())
    ss.setdefault("surprise", False)
    ss.setdefault("query", "")
    ss.setdefault("view", "Home")  # Home | Liked | Bag
_ensure_state()

# ============== DATA FETCHERS ==============
@st.cache_data(ttl=3600)
def tmdb_popular() -> List[Dict[str,Any]]:
    if not TMDB_API:
        return []
    url = "https://api.themoviedb.org/3/movie/popular"
    r = requests.get(url, params={"api_key": TMDB_API, "language":"en-US", "page":1}, timeout=15)
    r.raise_for_status()
    out=[]
    for m in r.json().get("results", [])[:20]:
        out.append({
            "id": f"mv_{m['id']}",
            "type": "movie",
            "title": m.get("title") or m.get("name",""),
            "subtitle": "Movies • Film",
            "img": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get("poster_path") else "",
            "href": f"https://www.themoviedb.org/movie/{m['id']}",
            "novelty": random.uniform(0.15, 1.0)
        })
    return out

@st.cache_data(ttl=3400)
def spotify_token() -> str:
    if not (SPOTIFY_ID and SPOTIFY_SECRET):
        return ""
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type":"client_credentials"},
        auth=(SPOTIFY_ID, SPOTIFY_SECRET), timeout=15
    )
    r.raise_for_status()
    return r.json()["access_token"]

@st.cache_data(ttl=1200)
def spotify_top_tracks() -> List[Dict[str,Any]]:
    tok = spotify_token()
    if not tok: return []
    # Pull some editorial playlists then pick items
    headers = {"Authorization": f"Bearer {tok}"}
    tracks=[]
    for playlist in ("37i9dQZF1DXcBWIGoYBM5M","37i9dQZF1DX0XUsuxWHRQd"):  # Today's Top Hits, Pop Rising
        r = requests.get(f"https://api.spotify.com/v1/playlists/{playlist}", headers=headers, timeout=15)
        if r.status_code!=200: continue
        for it in r.json().get("tracks", {}).get("items", [])[:20]:
            t = it.get("track") or {}
            img = ""
            if t.get("album",{}).get("images"):
                img = t["album"]["images"][0]["url"]
            tracks.append({
                "id": f"sp_{t.get('id','x')}",
                "type": "music",
                "title": t.get("name",""),
                "subtitle": "Music • Track",
                "img": img,
                "href": t.get("external_urls",{}).get("spotify",""),
                "novelty": random.uniform(0.6, 1.0)
            })
    # Dedup, keep first 20
    ded={}
    for x in tracks:
        if x["id"] not in ded and x["title"]:
            ded[x["id"]] = x
    return list(ded.values())[:20]

@st.cache_data(ttl=3600)
def trending_products() -> List[Dict[str,Any]]:
    # Mock catalog (safe + image-guaranteed). Replace with Walmart/Flipkart later.
    catalog = [
        ("Nike Air Winflo 10","Products • Shoes","https://i.imgur.com/5QFQG6x.jpeg","https://www.nike.com/"),
        ("Boat Airdopes 141","Products • Audio","https://i.imgur.com/Zi5X9rI.jpeg","https://www.boat-lifestyle.com/"),
        ("Redmi Note 13 Pro","Products • Phone","https://i.imgur.com/3z7r3ah.jpeg","https://www.mi.com/"),
        ("Wildcraft Backpack","Products • Bags","https://i.imgur.com/kbcl2jE.jpeg","https://wildcraft.com/"),
        ("H&M Oversized Tee","Products • Fashion","https://i.imgur.com/7oTzB0z.jpeg","https://www2.hm.com/")
    ]
    out=[]
    for i,(t,sub,img,href) in enumerate(catalog):
        out.append({
            "id": f"pd_{i}",
            "type":"product",
            "title":t, "subtitle":sub,
            "img": img, "href": href,
            "novelty": random.uniform(0.3, 0.9)
        })
    return out

# Combine pool
def all_items() -> List[Dict[str,Any]]:
    items = []
    items += spotify_top_tracks()
    items += trending_products()
    items += tmdb_popular()
    # ensure image fallback
    for it in items:
        if not it["img"]:
            it["img"] = "https://i.imgur.com/0Zf7rQe.jpeg"
    return items

# ============== SIMPLE RE-RANK: “Because You Liked These” ==============
def because_you_liked(items: List[Dict[str,Any]], liked_ids: set) -> List[Dict[str,Any]]:
    if not liked_ids:
        return random.sample(items, k=min(10, len(items)))
    liked = [x for x in items if x["id"] in liked_ids]
    buckets = { "movie": set(), "music": set(), "product": set() }
    for x in liked:
        buckets[x["type"]].add(x["id"])
    # bias: recommend from categories you interacted with
    cand = []
    for x in items:
        score = 1.0
        if x["type"] in buckets and buckets[x["type"]]:
            score += 1.2
        score += x["novelty"] * 0.5
        cand.append((score + random.random()*0.2, x))
    cand.sort(reverse=True, key=lambda z: z[0])
    return [x for _,x in cand if x["id"] not in liked_ids][:15]

# ============== AUTH UI (EMAIL/PASSWORD) ==============
def login_screen():
    st.markdown("<h1 style='margin:0 0 .5rem 0'>🍿 ReccoVerse</h1><small class='muted'>AI-curated picks across Movies • Music • Products</small>", unsafe_allow_html=True)
    st.write("")
    colA, colB = st.columns([1.2, 1])
    with colB:
        st.caption("Backend: " + ("Firebase" if FIREBASE_READY else "Local mock"))
        email = st.text_input("Email", key="auth_email", placeholder="you@example.com")
        pwd = st.text_input("Password", type="password", key="auth_pwd", placeholder="min 6 chars")
        c1, c2 = st.columns(2)
        def _signed(uid):
            st.session_state.authed = True
            st.session_state.uid = uid
            st.rerun()

        if FIREBASE_READY:
            if c1.button("Login", use_container_width=True):
                try:
                    user = _auth.sign_in_with_email_and_password(email, pwd)
                    _signed(user["localId"])
                except Exception as e:
                    st.error("Login failed. Check email/password.")
            if c2.button("Create Account", use_container_width=True):
                try:
                    _auth.create_user_with_email_and_password(email, pwd)
                    user = _auth.sign_in_with_email_and_password(email, pwd)
                    _signed(user["localId"])
                except Exception as e:
                    st.error("Sign-up failed. Use a valid email & 6+ char password.")
        else:
            # Local mock
            if c1.button("Login (mock)", use_container_width=True):
                if email:
                    _signed(email)
                else:
                    st.error("Enter any email to continue in mock mode.")
            if c2.button("Create (mock)", use_container_width=True):
                if email:
                    _signed(email)
                else:
                    st.error("Enter any email to continue in mock mode.")

# ============== UI HELPERS ==============
def novelty_bar(v: float):
    pct = int(min(max(v,0),1)*100)
    st.markdown(f"""
    <div class='subtle' style='margin:.35rem 0 .3rem 0'>Novelty {pct}%</div>
    <div class='progress-wrap'><div class='progress' style='width:{pct}%'></div></div>
    """, unsafe_allow_html=True)

def like_bag_row(item: Dict[str,Any], ctx: str):
    lid = item["id"]
    liked = lid in st.session_state.liked
    bagged = lid in st.session_state.bag
    left, right = st.columns(2)
    if left.button(("❤️" if liked else "♡  Like"), key=f"{ctx}_like_{lid}"):
        if liked: st.session_state.liked.remove(lid)
        else: st.session_state.liked.add(lid)
        st.rerun()
    if right.button(("✔ In Bag" if bagged else "+  Bag"), key=f"{ctx}_bag_{lid}"):
        if bagged: st.session_state.bag.remove(lid)
        else: st.session_state.bag.add(lid)
        st.rerun()

def render_card(item: Dict[str,Any], ctx: str):
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<img class='poster' src='{item['img']}'/>", unsafe_allow_html=True)
        st.markdown(f"<div class='title'>{item['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtle'>{item['subtitle']}</div>", unsafe_allow_html=True)
        novelty_bar(item["novelty"])
        like_bag_row(item, ctx=ctx)
        st.markdown("</div>", unsafe_allow_html=True)

def render_row(title: str, items: List[Dict[str,Any]], ctx: str):
    st.markdown(f"<div class='rowhead'><h2> {title}</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='grid'>", unsafe_allow_html=True)
    for it in items:
        with st.container():
            render_card(it, ctx=ctx)
    st.markdown("</div>", unsafe_allow_html=True)

def top_nav():
    tabs = st.tabs(["🏠 Home", "❤️ Liked", "🛍 Bag"])
    st.session_state.view = ["Home","Liked","Bag"][ [0,1,2][tabs.index(tabs[0]) ] ]  # (visual only)

# ============== PAGES ==============
def page_home(pool: List[Dict[str,Any]]):
    # Sidebar
    with st.sidebar:
        st.button("Sign Out", key="signout", type="secondary",
                  on_click=lambda: (st.session_state.clear(), _ensure_state(), None))
        st.write("")
        st.text_input("Search anything…", key="query", placeholder="love, phone, Taylor", help="Title keyword match")
        st.checkbox("Surprise Mode (shuffle)", key="surprise")

    # Filter/shuffle
    items = pool[:]
    q = (st.session_state.query or "").strip().lower()
    if q:
        items = [x for x in items if q in (x["title"] or "").lower()]
    if st.session_state.surprise:
        random.shuffle(items)

    st.markdown("<h1>🍿 Top Picks For You</h1>", unsafe_allow_html=True)
    # group rows
    movies = [x for x in items if x["type"]=="movie"][:10]
    music  = [x for x in items if x["type"]=="music"][:10]
    prods  = [x for x in items if x["type"]=="product"][:10]
    liked_recs = because_you_liked(items, st.session_state.liked)

    if music:
        render_row("Top Music Tracks", music, ctx="music")
        st.write("")
    if prods:
        render_row("Trending Products", prods, ctx="product")
        st.write("")
    if movies:
        render_row("Popular Movies", movies, ctx="movie")
        st.write("")
    if liked_recs:
        render_row("Because You Liked These", liked_recs, ctx="recs")

def page_liked_bag(pool: List[Dict[str,Any]]):
    liked_items = [x for x in pool if x["id"] in st.session_state.liked]
    bag_items   = [x for x in pool if x["id"] in st.session_state.bag]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h1>❤️ Liked Items</h1>", unsafe_allow_html=True)
        if not liked_items: st.info("No likes yet."); 
        else: render_row("", liked_items, ctx="likedlist")
    with col2:
        st.markdown("<h1>🛍 Bag</h1>", unsafe_allow_html=True)
        if not bag_items: st.info("Your bag is empty.")
        else: render_row("", bag_items, ctx="baglist")

# ============== MAIN ==============
def main():
    if not st.session_state.authed:
        login_screen()
        return

    # header tabs look
    top_nav()
    pool = all_items()

    # Use radio-like top bar
    colh = st.columns([0.12,0.12,0.12,0.64])
    with colh[0]:
        if st.button("🏠 Home", key="go_home"): st.session_state.view="Home"; st.rerun()
    with colh[1]:
        if st.button("❤️ Liked", key="go_liked"): st.session_state.view="Liked"; st.rerun()
    with colh[2]:
        if st.button("🛍 Bag", key="go_bag"): st.session_state.view="Bag"; st.rerun()

    if st.session_state.view=="Home":
        page_home(pool)
    else:
        page_liked_bag(pool)

if __name__ == "__main__":
    main()

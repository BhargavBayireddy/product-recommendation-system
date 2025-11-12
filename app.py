# ==========================================================
# ReccoVerse — Netflix-Style Multidomain Recommender (v5.3)
# ==========================================================
import streamlit as st
import pandas as pd, requests, uuid, random

st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")

# -------------------- THEME (cinematic, not Hotstar) --------------------
st.markdown("""
<style>
:root{
  --bg:#05060c;          /* deep cinema black */
  --panel:#0b0f1a;       /* card/controls */
  --ink:#f5f7ff;         /* main text */
  --muted:#a8b0c3;       /* muted text */
  --accent1:#d946ef;     /* magenta */
  --accent2:#22d3ee;     /* cyan */
}
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 700px at 20% 0%, #0a0d16 0%, var(--bg) 55%) !important;
  color: var(--ink);
  font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
h1,h2,h3,h4,strong { color: var(--ink); }
small, .markdown-text-container p { color: var(--muted); }

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #070a13 0%, #070a13 100%) !important;
  border-right: 1px solid rgba(255,255,255,.06);
}

/* Buttons */
.stButton>button{
  background: linear-gradient(90deg, var(--accent2), var(--accent1));
  border: none; border-radius: 28px; color: white; font-weight: 600;
  padding: .45rem 1rem; transition: .2s transform ease;
}
.stButton>button:hover{ transform: translateY(-1px) scale(1.03); }

/* Inputs (including sidebar search) */
input, textarea{
  background: var(--panel) !important;
  color: var(--ink) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  border-radius: 12px !important;
}
input::placeholder{ color: #c7cce0 !important; opacity: .7; }

/* Row titles */
.rowtitle{ font-size:1.15rem; margin:.75rem 0 .25rem; font-weight:700; }

/* Card hover */
.card { transition: transform .18s ease; }
.card:hover { transform: translateY(-4px); }

/* Chip/badge */
.badge{
  background: rgba(255,255,255,.1); padding:.18rem .55rem; border-radius:999px;
  display:inline-block; font-size:.75rem; color:var(--ink);
}

/* Progress (novelty) */
.progress{height:8px;border-radius:10px;overflow:hidden;background:#1b2030;margin:.25rem 0 .35rem;}
.bar{height:8px;background: linear-gradient(90deg, var(--accent2), var(--accent1));}
</style>
""", unsafe_allow_html=True)

# -------------------- API KEYS --------------------
TMDB_KEY = st.secrets.get("TMDB_API_KEY", "57b87af46cd78b943c23b3b94c68cfef")

# -------------------- SESSION INIT ----------------
if "authed" not in st.session_state: st.session_state["authed"] = False
if "liked"  not in st.session_state: st.session_state["liked"]  = set()
if "bag"    not in st.session_state: st.session_state["bag"]    = set()

# -------------------- DATA FETCHERS ----------------
@st.cache_data(show_spinner=False)
def fetch_movies(n=40):
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_KEY}&language=en-US&page=1",
            timeout=8).json()
        rows = [{
            "id": f"mv_{m['id']}",
            "title": m["title"],
            "category": "Movies",
            "genre": "Film",
            "image": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else ""
        } for m in r.get("results", [])[:n]]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_music(n=30):
    try:
        r = requests.get(f"https://api.deezer.com/chart/0/tracks?limit={n}", timeout=8).json()
        rows = [{
            "id": f"mu_{t['id']}",
            "title": t["title_short"],
            "category": "Music",
            "genre": t["artist"]["name"],
            "image": t["album"]["cover_medium"]
        } for t in r.get("data", [])]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_products(n=40):
    try:
        r = requests.get(f"https://dummyjson.com/products?limit={n}", timeout=8).json()
        rows = []
        for p in r.get("products", []):
            cat = p["category"].title()
            domain = "Beauty" if "beauty" in cat.lower() else \
                     "Fashion" if any(k in cat.lower() for k in ["shirt","shoe","bag","dress","watch"]) else \
                     "Tech"
            rows.append({
                "id": f"pr_{p['id']}",
                "title": p["title"],
                "category": domain,
                "genre": cat,
                "image": p["thumbnail"]
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner="Loading catalogs…")
def load_catalog():
    df = pd.concat([fetch_movies(), fetch_music(), fetch_products()], ignore_index=True)
    df = df.drop_duplicates("title").reset_index(drop=True)
    return df

# -------------------- NOVELTY (simple overlap) ---------------
def novelty_score(title:str, liked_titles:list[str])->float:
    if not liked_titles: return random.uniform(.7, 1.0)
    overlap = sum(1 for t in liked_titles if any(w in title.lower() for w in t.lower().split()))
    return max(0.12, 1 - overlap / max(1,len(liked_titles)))

# -------------------- UI HELPERS -----------------------------
def render_card(row: pd.Series):
    """One item card with Like/Bag + novelty meter."""
    liked  = row.id in st.session_state.liked
    bagged = row.id in st.session_state.bag

    st.container().markdown("", unsafe_allow_html=True)  # spacer
    st.image(row.image or "https://placehold.co/500x750/0b0f1a/ffffff?text=ReccoVerse",
             use_column_width=True, output_format="auto")
    st.markdown(f"**{row.title}**  \n<small>{row.category} • {row.genre}</small>", unsafe_allow_html=True)

    liked_titles = [t for t in st.session_state.get("_liked_titles_cache", [])]
    n = round(novelty_score(row.title, liked_titles)*100)
    st.markdown(f"<span class='badge'>🧬 Novelty {n}%</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='progress'><div class='bar' style='width:{n}%;'></div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    nonce = str(uuid.uuid4())[:8]  # avoid duplicate keys
    with c1:
        if st.button(("❤️" if liked else "♡ Like"), key=f"l_{row.id}_{nonce}"):
            st.session_state.liked.symmetric_difference_update([row.id])
            _refresh_liked_titles_cache()
            st.rerun()
    with c2:
        if st.button(("👜" if bagged else "➕ Bag"), key=f"b_{row.id}_{nonce}"):
            st.session_state.bag.symmetric_difference_update([row.id])
            st.rerun()

def render_row(df: pd.DataFrame, title: str):
    st.markdown(f"<div class='rowtitle'>🎞️ {title}</div>", unsafe_allow_html=True)
    if df.empty:
        st.caption("No items available.")
        return
    cols = st.columns(5)
    for i, (_, r) in enumerate(df.head(5).iterrows()):
        with cols[i]:
            with st.container(border=False):
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                render_card(r)
                st.markdown("</div>", unsafe_allow_html=True)

def _refresh_liked_titles_cache():
    """Cache titles for novelty overlap without refetching."""
    df = st.session_state.get("_catalog_df", pd.DataFrame())
    liked_ids = list(st.session_state.liked)
    titles = []
    if not df.empty and liked_ids:
        titles = df[df.id.isin(liked_ids)]["title"].tolist()
    st.session_state["_liked_titles_cache"] = titles

# -------------------- LOGIN ----------------------------
def login_screen():
    st.markdown("<h1 style='text-align:center;'>🎬 ReccoVerse</h1>", unsafe_allow_html=True)
    st.video("https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4")
    email = st.text_input("Email", placeholder="you@example.com")
    pwd   = st.text_input("Password", type="password", placeholder="••••••••")
    if st.button("Login / Sign-Up", use_container_width=True):
        if email and pwd:
            st.session_state.authed = True
            st.session_state.user = email
            st.success(f"Welcome, {email.split('@')[0].capitalize()}!")
            st.rerun()
        else:
            st.error("Please enter both email and password.")
    st.stop()

# -------------------- MAIN APP -------------------------
def main_app():
    # Sidebar
    st.sidebar.header("ReccoVerse")
    if st.sidebar.button("Sign Out"):
        st.session_state.clear()
        st.rerun()

    query = st.sidebar.text_input("🔎 Search anything...", placeholder="movie, track, product…")
    surprise = st.sidebar.checkbox("🎢 Surprise Mode (shuffle)")

    # Data
    df = load_catalog()
    st.session_state["_catalog_df"] = df  # for novelty cache usage

    # Search (case-insensitive across ALL domains)
    if query:
        q = query.strip().lower()
        df = df[df["title"].str.lower().str.contains(q)]

    # Surprise shuffle (after filtering)
    if surprise and not df.empty:
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)

    # Keep novelty cache updated
    _refresh_liked_titles_cache()

    # Results / empty state
    st.markdown("## 🍿 Top Picks For You")
    if df.empty:
        st.info("No matches found. Try a different search, e.g., **'love'**, **'phone'**, **'Taylor'**.")
        return

    # Rows by domain
    render_row(df[df.category=="Movies"], "Popular Movies")
    render_row(df[df.category=="Music"],  "Top Music Tracks")
    render_row(df[df.category.isin(["Beauty","Fashion","Tech"])], "Trending Products")

    if st.session_state.liked:
        liked_df = df[df.id.isin(st.session_state.liked)]
        render_row(liked_df, "Because You Liked These")

# -------------------- ENTRYPOINT ----------------------
if not st.session_state.authed:
    login_screen()
else:
    main_app()

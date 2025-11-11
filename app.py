# ==========================================================
#  ReccoVerse v4 – Netflix-Style AI Multidomain Recommender
#  Author: Pavan Kumar Reddy & GPT-5
# ==========================================================
import streamlit as st
import pandas as pd, numpy as np, requests, random, uuid

# ----------------------------------------------------------
# PAGE CONFIG & STYLE
st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #000015 0%, #000010 100%) !important;
  color: #fff;
  font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4 { color: white; }
.stButton>button {
  border: none; border-radius: 25px; font-weight: 600;
  background: linear-gradient(90deg,#00c6ff,#ff0080);
  color: white; padding: .4rem 1rem;
  transition: 0.3s;
}
.stButton>button:hover { transform: scale(1.05); background: linear-gradient(90deg,#00f5a0,#00d9ff); }
.card { transition: transform .2s ease; }
.card:hover { transform: scale(1.05); }
input, textarea { background: rgba(255,255,255,0.07)!important; color:#fff!important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# API KEYS
TMDB_KEY = st.secrets.get("TMDB_API_KEY", "57b87af46cd78b943c23b3b94c68cfef")

# ----------------------------------------------------------
# DATA FETCHERS
@st.cache_data
def get_movies(limit=20):
    try:
        res = requests.get(f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_KEY}&language=en-US&page=1").json()
        movies = []
        for m in res["results"][:limit]:
            movies.append({
                "id": f"mv_{m['id']}",
                "title": m["title"],
                "category": "Movies",
                "genre": "Film",
                "image": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else "",
                "source": "TMDB"
            })
        return pd.DataFrame(movies)
    except Exception as e:
        st.error(f"TMDB error: {e}")
        return pd.DataFrame()

@st.cache_data
def get_music(limit=20):
    try:
        res = requests.get("https://api.deezer.com/chart/0/tracks?limit="+str(limit)).json()
        data = []
        for t in res["data"]:
            data.append({
                "id": f"mu_{t['id']}",
                "title": t["title_short"],
                "category": "Music",
                "genre": t["artist"]["name"],
                "image": t["album"]["cover_medium"],
                "source": "Deezer"
            })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

@st.cache_data
def get_products(limit=20):
    try:
        res = requests.get("https://dummyjson.com/products?limit="+str(limit)).json()
        data = []
        for p in res["products"]:
            cat = p["category"].title()
            domain = "Beauty" if "beauty" in cat.lower() else "Tech" if "tech" in cat.lower() else "Fashion"
            data.append({
                "id": f"pr_{p['id']}",
                "title": p["title"],
                "category": domain,
                "genre": cat,
                "image": p["thumbnail"],
                "source": "DummyJSON"
            })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

@st.cache_data
def load_data():
    df = pd.concat([get_movies(), get_music(), get_products()], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------------------------------------
# LOGIN PAGE
def login_screen():
    st.markdown("<h1 style='text-align:center;'>🍿 ReccoVerse</h1>", unsafe_allow_html=True)
    st.video("https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4")

    st.write("")
    email = st.text_input("Email", key="email")
    pwd = st.text_input("Password", type="password", key="pwd")
    if st.button("Login / Sign-Up", use_container_width=True):
        if email and pwd:
            st.session_state["authed"] = True
            st.session_state["user"] = email
            st.success(f"Welcome, {email.split('@')[0]}!")
            st.rerun()
        else:
            st.error("Please enter both fields.")

# ----------------------------------------------------------
# CARD UI
def render_row(df, title):
    st.markdown(f"### {title}")
    cols = st.columns(5)
    for i, (_, r) in enumerate(df.head(5).iterrows()):
        with cols[i]:
            st.image(r["image"], use_column_width=True)
            st.markdown(f"**{r['title']}**  \n_{r['category']} • {r['genre']}_")
            liked = r["id"] in st.session_state["liked"]
            bagged = r["id"] in st.session_state["bag"]
            c1, c2 = st.columns(2)
            with c1:
                if st.button(("❤️" if liked else "♡ Like"), key=f"l_{r['id']}"):
                    if liked: st.session_state["liked"].remove(r["id"])
                    else: st.session_state["liked"].add(r["id"])
                    st.rerun()
            with c2:
                if st.button(("👜" if bagged else "➕ Bag"), key=f"b_{r['id']}"):
                    if bagged: st.session_state["bag"].remove(r["id"])
                    else: st.session_state["bag"].add(r["id"])
                    st.rerun()

# ----------------------------------------------------------
# MAIN DASHBOARD
def main_app():
    st.sidebar.header("ReccoVerse")
    if st.sidebar.button("Sign Out"):
        st.session_state.clear()
        st.rerun()

    search = st.sidebar.text_input("Search", placeholder="Search all domains...")
    df = load_data()
    if search:
        df = df[df["title"].str.lower().str.contains(search.lower())]

    # user prefs
    if "liked" not in st.session_state: st.session_state["liked"] = set()
    if "bag" not in st.session_state: st.session_state["bag"] = set()

    st.markdown("## 🎬 Top Picks For You")
    render_row(df[df["category"]=="Movies"], "Popular Movies")
    render_row(df[df["category"]=="Music"], "Top Music Tracks")
    render_row(df[df["category"].isin(["Beauty","Fashion","Tech"])], "Trending Products")

    if st.session_state["liked"]:
        liked_df = df[df["id"].isin(st.session_state["liked"])]
        render_row(liked_df, "Because You Liked These")

# ----------------------------------------------------------
# ENTRYPOINT
if "authed" not in st.session_state:
    st.session_state["authed"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

if not st.session_state["authed"]:
    login_screen()
else:
    main_app()

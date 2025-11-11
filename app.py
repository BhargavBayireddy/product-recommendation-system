# app.py — ReccoVerse v2 (Cinematic Multi-Domain AI Recommender)
import os, io, json, zipfile, gzip, base64, requests, uuid, time
from pathlib import Path
import pandas as pd
import streamlit as st

from firebase_init import (
    signup_email_password, login_email_password,
    add_interaction, fetch_user_interactions, ensure_user,
    email_exists, send_phone_otp, verify_phone_otp, FIREBASE_READY
)
from gnn_infer import (
    load_item_embeddings, make_user_vector, recommend_items, cold_start_mmr
)

# -------------------------------------------------------
# CONFIG
APP_NAME = "ReccoVerse"
ART = Path("artifacts")
st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

# -------------------------------------------------------
# STYLE (Netflix + AI Glow)
st.markdown("""
<style>
body,[data-testid="stAppViewContainer"]{
  background:radial-gradient(circle at 30% 20%,#0a0f25,#000) !important;
  color:#f5f5f5;font-family:'Poppins',sans-serif;}
h1,h2,h3{color:#fff !important;}
.stButton>button{background:linear-gradient(90deg,#00ffff,#ff00c3);
  border:none;border-radius:50px;color:white;font-weight:600;
  padding:.6rem 1.3rem;transition:.3s;}
.stButton>button:hover{transform:scale(1.05);}
.searchbar input{border-radius:30px;border:1px solid #00ffff;
  background:rgba(255,255,255,0.05);color:#fff;padding:.5rem 1rem;width:100%;}
.card{transition:transform .2s ease;}
.card:hover{transform:scale(1.03);}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HERO SECTION
def hero():
    video_url = "https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4"
    st.markdown(f"""
    <div style='position:relative;height:50vh;overflow:hidden;border-radius:18px;'>
      <video autoplay muted loop playsinline style='width:100%;height:100%;object-fit:cover;filter:brightness(0.7)'>
        <source src="{video_url}" type="video/mp4">
      </video>
      <div style='position:absolute;inset:0;display:flex;flex-direction:column;
        justify-content:center;align-items:center;background:linear-gradient(180deg,rgba(0,0,0,0.3),rgba(0,0,0,0.85));'>
        <h1 style='font-size:3.5rem;background:linear-gradient(90deg,#00ffff,#ff00c3);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>ReccoVerse</h1>
        <p style='font-size:1.2rem;color:#ddd;'>AI-curated picks across Movies · Music · Fashion · Tech · Beauty</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# IMAGE FETCH HELPERS
@st.cache_data(show_spinner=False)
def get_poster(title, domain="movie"):
    key = st.secrets.get("TMDB_API_KEY", "57b87af46cd78b943c23b3b94c68cfef")
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={key}&query={requests.utils.quote(title)}"
        r = requests.get(url, timeout=5).json()
        if r.get("results") and r["results"][0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + r["results"][0]["poster_path"]
    except:
        pass
    fallback = {
        "movie": "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80",
        "music": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80",
        "beauty": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?q=80",
        "fashion": "https://images.unsplash.com/photo-1521335629791-ce4aec67dd47?q=80",
        "tech": "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80"
    }
    return fallback.get(domain, fallback["movie"])

# -------------------------------------------------------
# MULTI-DOMAIN DATASET
@st.cache_data(show_spinner="Loading AI-curated datasets…")
def load_multidomain():
    frames = []

    # Movies
    z = requests.get("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", timeout=15)
    with zipfile.ZipFile(io.BytesIO(z.content)) as f:
        with f.open("ml-latest-small/movies.csv") as c:
            mv = pd.read_csv(c).sample(120)
    mv["image"] = [get_poster(t, "movie") for t in mv["title"]]
    frames.append(pd.DataFrame({
        "item_id": "mv_" + mv["movieId"].astype(str),
        "title": mv["title"],
        "category": "Movies",
        "source": "Netflix",
        "genre": mv["genres"].str.split("|").str[0],
        "image": mv["image"]
    }))

    # Music
    music = pd.read_csv("https://raw.githubusercontent.com/yg397/music-recommender-dataset/master/data.csv").dropna().sample(80)
    frames.append(pd.DataFrame({
        "item_id": "mu_" + music["artist"].astype(str) + "_" + music["track"].astype(str),
        "title": music["track"],
        "category": "Music",
        "source": "Spotify",
        "genre": "Pop",
        "image": get_poster("music album", "music")
    }))

    # Beauty
    beauty = pd.DataFrame([
        {"title": "Maybelline Lipstick", "genre": "Beauty", "source": "Amazon",
         "image": get_poster("lipstick", "beauty")},
        {"title": "Lakme Kajal", "genre": "Beauty", "source": "Amazon",
         "image": get_poster("kajal", "beauty")},
        {"title": "Nivea Cream", "genre": "Beauty", "source": "Amazon",
         "image": get_poster("nivea", "beauty")}
    ])

    beauty["item_id"] = ["pr_" + str(i) for i in range(len(beauty))]
    beauty["category"] = "Beauty"
    frames.append(beauty)

    # Tech Gadgets
    tech = pd.DataFrame([
        {"title": "Apple iPhone 15 Pro", "genre": "Tech", "source": "Apple",
         "image": get_poster("iphone", "tech")},
        {"title": "Sony WH-1000XM5 Headphones", "genre": "Tech", "source": "Sony",
         "image": get_poster("headphones", "tech")},
        {"title": "Samsung Galaxy Watch", "genre": "Tech", "source": "Samsung",
         "image": get_poster("watch", "tech")}
    ])
    tech["item_id"] = ["te_" + str(i) for i in range(len(tech))]
    tech["category"] = "Tech"
    frames.append(tech)

    # Fashion
    fashion = pd.DataFrame([
        {"title": "Zara Denim Jacket", "genre": "Fashion", "source": "Zara",
         "image": get_poster("jacket", "fashion")},
        {"title": "Nike Air Max 270", "genre": "Fashion", "source": "Nike",
         "image": get_poster("shoe", "fashion")},
        {"title": "Adidas Hoodie", "genre": "Fashion", "source": "Adidas",
         "image": get_poster("hoodie", "fashion")}
    ])
    fashion["item_id"] = ["fa_" + str(i) for i in range(len(fashion))]
    fashion["category"] = "Fashion"
    frames.append(fashion)

    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1).reset_index(drop=True)

# -------------------------------------------------------
# INITIALIZE STATE
def init_state():
    for k, v in {
        "authed": False, "uid": None, "email": None,
        "liked": set(), "bag": set(), "auth_mode": "email"
    }.items():
        st.session_state.setdefault(k, v)
init_state()

# -------------------------------------------------------
# AUTH
def auth_page():
    hero()
    email = st.text_input("Email")
    pwd = st.text_input("Password", type="password")
    c1, c2 = st.columns(2)
    if c1.button("Login", use_container_width=True):
        if email_exists(email):
            ok, uid = login_email_password(email, pwd)
            if ok:
                st.session_state.update(authed=True, uid=uid, email=email)
                ensure_user(uid, email=email)
                st.success("✅ Logged in!"); st.rerun()
            else:
                st.error("Invalid credentials.")
        else:
            st.warning("User not found.")
    if c2.button("Sign Up", use_container_width=True):
        ok, uid = signup_email_password(email, pwd)
        if ok:
            st.session_state.update(authed=True, uid=uid, email=email)
            ensure_user(uid, email=email)
            st.success("✅ Account created!"); st.rerun()
        else:
            st.error(uid)

# -------------------------------------------------------
# CARD DISPLAY (Fixed buttons)
def render_card(row):
    c = st.container()
    with c:
        st.image(row.image, use_column_width=True)
        st.markdown(f"**{row.title}**  \n_{row.source} • {row.genre}_")
        c1, c2 = st.columns(2)
        like_key = f"like_{row.item_id}"
        bag_key = f"bag_{row.item_id}"
        liked = row.item_id in st.session_state.liked
        bagged = row.item_id in st.session_state.bag
        if c1.button("❤️" if liked else "♡ Like", key=like_key):
            if liked:
                st.session_state.liked.remove(row.item_id)
            else:
                st.session_state.liked.add(row.item_id)
            st.rerun()
        if c2.button("👜" if bagged else "➕ Bag", key=bag_key):
            if bagged:
                st.session_state.bag.remove(row.item_id)
            else:
                st.session_state.bag.add(row.item_id)
            st.rerun()

# -------------------------------------------------------
# MAIN APP
def main():
    if not st.session_state.authed:
        auth_page()
        return

    df = load_multidomain()

    # Sidebar
    st.sidebar.title("ReccoVerse")
    choice = st.sidebar.radio("Navigate", ["Home", "Liked", "Bag"])
    if st.sidebar.button("Sign Out"):
        st.session_state.authed = False
        st.session_state.liked.clear()
        st.session_state.bag.clear()
        st.rerun()

    # Search bar
    query = st.text_input("🔍 Search", placeholder="Search for movies, music, products, etc...", key="search", label_visibility="collapsed")
    if query:
        df = df[df["title"].str.contains(query, case=False)]

    # Home feed
    if choice == "Home":
        hero()
        st.markdown("## ✨ Explore AI-curated Picks")
        for _, row in df.iterrows():
            with st.container():
                render_card(row)
                st.markdown("---")

    elif choice == "Liked":
        st.markdown("## ❤️ Your Liked Items")
        liked_df = df[df["item_id"].isin(st.session_state.liked)]
        if liked_df.empty:
            st.info("No liked items yet.")
        for _, row in liked_df.iterrows():
            render_card(row)

    elif choice == "Bag":
        st.markdown("## 👜 Your Bag")
        bag_df = df[df["item_id"].isin(st.session_state.bag)]
        if bag_df.empty:
            st.info("Your bag is empty.")
        for _, row in bag_df.iterrows():
            render_card(row)

# -------------------------------------------------------
if __name__ == "__main__":
    main()

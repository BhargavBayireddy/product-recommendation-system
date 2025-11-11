# app.py — ReccoVerse Cinematic Multi-Domain Recommender
import os, io, json, zipfile, gzip, base64, requests
from pathlib import Path
from typing import List
import pandas as pd
import streamlit as st
from firebase_init import (
    signup_email_password, login_email_password,
    add_interaction, fetch_user_interactions, fetch_global_interactions,
    ensure_user, email_exists, send_phone_otp, verify_phone_otp, FIREBASE_READY
)
from gnn_infer import (
    load_item_embeddings, make_user_vector, recommend_items,
    diversity_personalization_novelty, cold_start_mmr
)

APP_NAME = "ReccoVerse"
ART = Path("artifacts")
REFRESH_MS = 5000
CARD_COLS = 5

st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

# ================================================
#  CSS (Neon Glowing + Cinematic Motion Background)
# ================================================
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 20% 10%, #050c1f, #000000) !important;
  color: #f2f2f2;
  font-family: 'Poppins', sans-serif;
}

/* Hero section cinematic */
.hero-wrap {
  position: relative;
  height: 52vh;
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.1);
  box-shadow: 0 0 80px rgba(73,187,255,0.15);
}
.hero-video {
  position: absolute;top: 0;left: 0;width: 100%;height: 100%;
  object-fit: cover;filter: brightness(.75) contrast(1.2) saturate(1.3);
}
.hero-overlay {
  position: absolute;inset: 0;
  background: linear-gradient(180deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.9) 90%);
  display: flex;flex-direction: column;justify-content: center;align-items: center;
}
.brand {
  font-size: 3.5rem;font-weight: 900;letter-spacing: .05em;
  background: linear-gradient(90deg,#00f2ff,#ff00c3);
  -webkit-background-clip: text;-webkit-text-fill-color: transparent;
  text-shadow: 0 0 25px rgba(0,255,255,0.2);
}
.tagline {
  font-size: 1.2rem;opacity: 0.85;margin-top: 0.6rem;color: #d4d4d4;
}

/* Glowing input boxes */
input[type="text"], input[type="password"], textarea {
  background: rgba(255,255,255,0.12) !important;
  color: #ffffff !important;
  border: 1px solid rgba(0,255,255,0.4) !important;
  border-radius: 12px !important;
  padding: 0.7rem 0.9rem !important;
  font-size: 1rem !important;
  box-shadow: 0 0 8px rgba(0,255,255,0.2);
}
input:focus {
  outline: none !important;
  border-color: #00ffff !important;
  box-shadow: 0 0 20px rgba(0,255,255,0.5);
}
input::placeholder {
  color: rgba(255,255,255,0.5);
}

/* Gradient buttons */
.stButton>button {
  background: linear-gradient(90deg,#00ffff,#ff00c3);
  border: none !important;
  border-radius: 999px !important;
  color: white !important;
  font-weight: 700 !important;
  padding: .6rem 1.2rem !important;
  transition: 0.3s;
}
.stButton>button:hover {
  transform: scale(1.05);
  box-shadow: 0 0 18px rgba(0,255,255,0.5);
}

/* Card hover */
.card {
  border-radius:18px; overflow:hidden; border:1px solid rgba(255,255,255,0.1);
  transition:transform .18s ease, box-shadow .18s ease;
}
.card:hover {
  transform: translateY(-4px) scale(1.02);
  box-shadow:0 18px 44px rgba(0,0,0,.45), 0 0 60px rgba(73,187,255,.07);
}
</style>
""", unsafe_allow_html=True)

# ================================================
#  Hero Section (with Cinematic Motion Background)
# ================================================
def hero_with_video():
    local = ART / "hero.mp4"
    if local.exists():
        b64 = base64.b64encode(local.read_bytes()).decode("utf-8")
        src = f"data:video/mp4;base64,{b64}"
    else:
        # fallback futuristic AI motion
        src = "https://cdn.pixabay.com/vimeo/927530021/ai-neural-17839.mp4?width=1920&hash=ebc8a5e6422"
    st.markdown(f"""
    <div class="hero-wrap">
      <video class="hero-video" autoplay muted loop playsinline>
        <source src="{src}" type="video/mp4">
      </video>
      <div class="hero-overlay">
        <div class="brand">ReccoVerse</div>
        <div class="tagline">AI-curated picks across Movies • Music • Products</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================
#  Session State Initialization
# ================================================
def init_state():
    s = st.session_state
    defaults = {
        "authed": False, "uid": None, "email": None,
        "items_df": None, "embeddings": None, "id_to_idx": None,
        "A": None, "liked": set(), "bag": set(),
        "auth_mode": "email", "otp_sent": False, "otp_phone": ""
    }
    for k, v in defaults.items():
        s.setdefault(k, v)
init_state()

# ================================================
#  Load Multi-Domain Data (Movies + Music + Beauty)
# ================================================
@st.cache_data(show_spinner="Fetching Movies • Music • Beauty datasets…")
def load_multidomain_online():
    frames = []

    # Movies
    try:
        z = requests.get("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", timeout=10)
        with zipfile.ZipFile(io.BytesIO(z.content)) as f:
            with f.open("ml-latest-small/movies.csv") as c:
                movies = pd.read_csv(c)
        frames.append(pd.DataFrame({
            "item_id": "mv_" + movies["movieId"].astype(str),
            "title": movies["title"],
            "provider": "Netflix",
            "genre": movies["genres"].str.split("|").str[0],
            "image": "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80&w=1200",
            "text": movies["title"] + " " + movies["genres"]
        }).sample(200))
    except Exception as e:
        st.warning(f"Movies dataset failed: {e}")

    # Music
    try:
        music = pd.read_csv("https://raw.githubusercontent.com/yg397/music-recommender-dataset/master/data.csv").dropna().sample(200)
        frames.append(pd.DataFrame({
            "item_id": "mu_" + music["artist"].astype(str) + "_" + music["track"].astype(str),
            "title": music["track"],
            "provider": "Spotify",
            "genre": "Music",
            "image": "https://images.unsplash.com/photo-1511379938547-c1f69419868d?q=80&w=1200",
            "text": music["artist"] + " " + music["track"]
        }))
    except Exception as e:
        st.warning(f"Music dataset failed: {e}")

    # Beauty Products
    try:
        r = requests.get("https://datarepo.s3.amazonaws.com/beauty_5.json.gz", timeout=10)
        lines = gzip.decompress(r.content).splitlines()[:1500]
        beauty = pd.DataFrame([json.loads(l) for l in lines])
        frames.append(pd.DataFrame({
            "item_id": "pr_" + beauty["asin"].astype(str),
            "title": beauty["title"],
            "provider": "Amazon",
            "genre": "Beauty",
            "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?q=80&w=1200",
            "text": beauty["title"] + " " + beauty.get("description", "").astype(str)
        }).dropna().sample(200))
    except Exception as e:
        st.warning(f"Beauty dataset failed: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates("title", inplace=True)
    return df

# ================================================
#  Authentication Page (Email + Mobile OTP)
# ================================================
def auth_page():
    hero_with_video()
    st.write("")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Join with")
        c1, c2 = st.columns(2)
        if c1.button("📧 Email", use_container_width=True):
            st.session_state.auth_mode = "email"
        if c2.button("📱 Mobile (OTP)", use_container_width=True):
            st.session_state.auth_mode = "phone"
        st.caption(f"Backend: {'Firebase' if FIREBASE_READY else 'Local mock'}")

    with right:
        if st.session_state.auth_mode == "email":
            st.markdown("##### Continue with Email")
            email = st.text_input("Email")
            pwd = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            if col1.button("Continue", use_container_width=True):
                if email_exists(email):
                    ok, uid = login_email_password(email, pwd)
                    if ok:
                        st.session_state.update(authed=True, uid=uid, email=email)
                        ensure_user(uid, email=email)
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                else:
                    st.warning("Account not found.")
            if col2.button("Create Account", use_container_width=True):
                ok, uid = signup_email_password(email, pwd)
                if ok:
                    st.session_state.update(authed=True, uid=uid, email=email)
                    ensure_user(uid, email=email)
                    st.success("Account created and logged in.")
                    st.rerun()
                else:
                    st.error(uid)
        else:
            st.markdown("##### Continue with Mobile (OTP)")
            phone = st.text_input("Mobile number (+91XXXXXXXXXX)")
            otp = st.text_input("Enter OTP")
            c1, c2 = st.columns(2)
            if c1.button("Send OTP", use_container_width=True):
                ok, msg = send_phone_otp(phone)
                st.session_state.otp_sent = ok
                st.session_state.otp_phone = phone
                st.info(f"OTP: {msg}" if msg.isdigit() else msg)
            if c2.button("Verify & Continue", use_container_width=True):
                ok, uid = verify_phone_otp(st.session_state.otp_phone, otp)
                if ok:
                    st.session_state.update(authed=True, uid=uid, email=f"{phone}@phone.local")
                    ensure_user(uid, phone=phone)
                    st.success("Logged in with mobile.")
                    st.rerun()
                else:
                    st.error(uid)

# ================================================
#  Main Pages
# ================================================
def render_card(row, liked, bagged, uid):
    st.image(row.image, use_column_width=True)
    st.markdown(f"**{row.title}**  \n_{row.provider} • {row.genre}_")
    c1, c2 = st.columns(2)
    if c1.button("❤️" if liked else "♡ Like", key=f"l{row.item_id}"):
        add_interaction(uid, row.item_id, "like" if not liked else "unlike")
        (st.session_state.liked.add if not liked else st.session_state.liked.discard)(row.item_id)
        st.rerun()
    if c2.button("👜" if bagged else "➕ Bag", key=f"b{row.item_id}"):
        add_interaction(uid, row.item_id, "bag" if not bagged else "remove_bag")
        (st.session_state.bag.add if not bagged else st.session_state.bag.discard)(row.item_id)
        st.rerun()

def section(title, df, ids, uid):
    st.markdown(f"### {title}")
    cols = st.columns(CARD_COLS)
    for i, iid in enumerate(ids[:CARD_COLS*2+5]):
        row = df[df.item_id == iid]
        if not row.empty:
            with cols[i % CARD_COLS]:
                render_card(row.iloc[0], iid in st.session_state.liked, iid in st.session_state.bag, uid)

def home(uid):
    st.markdown(f"## Welcome to {APP_NAME}")
    ensure_embs()
    df, embs, idmap, A = st.session_state.items_df, st.session_state.embeddings, st.session_state.id_to_idx, st.session_state.A
    inter = fetch_user_interactions(uid)
    st.session_state.liked = {x["item_id"] for x in inter if x["action"] == "like"}
    st.session_state.bag = {x["item_id"] for x in inter if x["action"] == "bag"}
    userv = make_user_vector(st.session_state.liked, st.session_state.bag, idmap, embs)
    top = recommend_items(userv, embs, df, exclude=set(st.session_state.liked), topk=15, A=A)
    crowd = fetch_global_interactions()
    ppl = recommend_items(userv, embs, df, topk=12, A=A, crowd=crowd)
    sim = recommend_items(userv, embs, df, topk=12, A=A, force_content=True)
    cold = cold_start_mmr(df, embs, 0.65, 12)
    section("Top Picks For You", df, top, uid)
    section("People Like You Also Liked", df, ppl, uid)
    section("Because You Liked Similar Items", df, sim, uid)
    section("Explore Something Different", df, cold, uid)
    st.autorefresh(interval=REFRESH_MS)

def ensure_embs():
    if st.session_state.items_df is None:
        df = load_multidomain_online()
        items, embs, idmap, A = load_item_embeddings(df, ART)
        st.session_state.items_df, st.session_state.embeddings, st.session_state.id_to_idx, st.session_state.A = items, embs, idmap, A

def navbar():
    st.sidebar.markdown("### ReccoVerse")
    page = st.sidebar.radio("Navigate", ["Home", "Liked", "Bag"])
    if st.sidebar.button("Sign Out"):
        for k in ["authed", "uid", "email"]:
            st.session_state[k] = None
        st.session_state.liked.clear()
        st.session_state.bag.clear()
        st.session_state.authed = False
        st.rerun()
    return page

# ================================================
#  App Runner
# ================================================
def main():
    if not st.session_state.authed:
        auth_page()
    else:
        p = navbar()
        uid = st.session_state.uid
        {"Home": home, "Liked": lambda u: section("Liked", st.session_state.items_df, list(st.session_state.liked), u),
         "Bag": lambda u: section("Bag", st.session_state.items_df, list(st.session_state.bag), u)}[p](uid)

if __name__ == "__main__":
    main()

# app.py — ReccoVerse (Stable Interactive Edition)
import os, io, json, zipfile, gzip, base64, requests, uuid
from pathlib import Path
import pandas as pd
import streamlit as st
from firebase_init import (
    signup_email_password, login_email_password,
    add_interaction, fetch_user_interactions, fetch_global_interactions,
    ensure_user, email_exists, send_phone_otp, verify_phone_otp, FIREBASE_READY
)
from gnn_infer import (
    load_item_embeddings, make_user_vector, recommend_items, cold_start_mmr
)

# --- Config
APP_NAME = "ReccoVerse"
ART = Path("artifacts")
CARD_COLS = 5
st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

# --- Styles
st.markdown("""
<style>
body,[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 20% 10%, #050c1f, #000) !important;
  color:#f2f2f2;font-family:'Poppins',sans-serif;}
.stButton>button{
  background:linear-gradient(135deg,#00ffff,#ff00c3);
  border:none;border-radius:999px;color:#fff;font-weight:600;
  padding:.6rem 1.2rem;transition:.3s;}
.stButton>button:hover{transform:scale(1.05);}
.card:hover{transform:translateY(-3px)scale(1.01);
  box-shadow:0 0 40px rgba(0,255,255,.2);}
.brand{font-size:3.6rem;font-weight:900;
  background:linear-gradient(90deg,#00f2ff,#ff00c3);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.tagline{font-size:1.1rem;opacity:.8;color:#ddd;}
</style>
""", unsafe_allow_html=True)

# --- Hero
def hero():
    vid = "https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4"
    st.markdown(f"""
    <div style='position:relative;height:52vh;overflow:hidden;border-radius:20px;'>
      <video autoplay muted loop playsinline style='width:100%;height:100%;object-fit:cover;filter:brightness(0.7)'>
        <source src="{vid}" type="video/mp4">
      </video>
      <div style='position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;align-items:center;background:linear-gradient(180deg,rgba(0,0,0,0.2),rgba(0,0,0,0.8));'>
        <div class='brand'>ReccoVerse</div>
        <div class='tagline'>AI-curated picks across Movies · Music · Products</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- TMDB Poster Cache
@st.cache_data(show_spinner=False)
def get_poster(title: str, domain="movie"):
    key = st.secrets.get("TMDB_API_KEY", "57b87af46cd78b943c23b3b94c68cfef")
    try:
        url = f"https://api.themoviedb.org/3/search/{domain}?api_key={key}&query={requests.utils.quote(title)}"
        r = requests.get(url, timeout=6).json()
        if r.get("results") and r["results"][0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + r["results"][0]["poster_path"]
    except Exception:
        pass
    return {
        "movie": "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80&w=1200",
        "music": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=1200",
        "product": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?q=80&w=1200"
    }.get(domain, "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80&w=1200")

# --- Load multi-domain dataset
@st.cache_data(show_spinner="Loading AI datasets…")
def load_data():
    frames = []
    # Movies
    z = requests.get("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", timeout=15)
    with zipfile.ZipFile(io.BytesIO(z.content)) as f:
        with f.open("ml-latest-small/movies.csv") as c:
            mv = pd.read_csv(c).sample(150)
    mv["image"] = [get_poster(t, "movie") for t in mv["title"]]
    frames.append(pd.DataFrame({
        "item_id": "mv_" + mv["movieId"].astype(str),
        "title": mv["title"],
        "provider": "Netflix",
        "genre": mv["genres"].str.split("|").str[0],
        "image": mv["image"],
        "text": mv["title"] + " " + mv["genres"]
    }))
    # Music
    try:
        music = pd.read_csv("https://raw.githubusercontent.com/yg397/music-recommender-dataset/master/data.csv").dropna().sample(100)
        frames.append(pd.DataFrame({
            "item_id": "mu_" + music["artist"].astype(str) + "_" + music["track"].astype(str),
            "title": music["track"],
            "provider": "Spotify",
            "genre": "Music",
            "image": get_poster("music album", "music"),
            "text": music["artist"] + " " + music["track"]
        }))
    except Exception:
        pass
    # Beauty
    try:
        r = requests.get("https://datarepo.s3.amazonaws.com/beauty_5.json.gz", timeout=10)
        lines = gzip.decompress(r.content).splitlines()[:600]
        beauty = pd.DataFrame([json.loads(l) for l in lines])
        frames.append(pd.DataFrame({
            "item_id": "pr_" + beauty["asin"].astype(str),
            "title": beauty["title"],
            "provider": "Amazon",
            "genre": "Beauty",
            "image": get_poster("cosmetics", "product"),
            "text": beauty["title"]
        }).dropna().sample(80))
    except Exception:
        pass
    df = pd.concat(frames, ignore_index=True).drop_duplicates("title")
    return df

# --- Auth
def auth_page():
    hero(); st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Join with")
        c1, c2 = st.columns(2)
        if c1.button("📧 Email", use_container_width=True): st.session_state.auth_mode = "email"
        if c2.button("📱 Mobile (OTP)", use_container_width=True): st.session_state.auth_mode = "phone"
        st.caption(f"Backend: {'Firebase' if FIREBASE_READY else 'Local mock'}")
    with col2:
        if st.session_state.auth_mode == "email":
            email = st.text_input("Email")
            pwd = st.text_input("Password", type="password")
            c1, c2 = st.columns(2)
            if c1.button("Continue", use_container_width=True):
                if email_exists(email):
                    ok, uid = login_email_password(email, pwd)
                    if ok:
                        st.session_state.update(authed=True, uid=uid, email=email)
                        ensure_user(uid, email=email); st.rerun()
                    else: st.error("Invalid credentials")
                else: st.warning("No account found.")
            if c2.button("Create Account", use_container_width=True):
                ok, uid = signup_email_password(email, pwd)
                if ok:
                    st.session_state.update(authed=True, uid=uid, email=email)
                    ensure_user(uid, email=email); st.rerun()
                else: st.error(uid)
        else:
            phone = st.text_input("Mobile (+91XXXXXXXXXX)")
            otp = st.text_input("Enter OTP")
            c1, c2 = st.columns(2)
            if c1.button("Send OTP"): ok, msg = send_phone_otp(phone); st.info(msg)
            if c2.button("Verify & Continue"):
                ok, uid = verify_phone_otp(phone, otp)
                if ok: st.session_state.update(authed=True, uid=uid, email=f"{phone}@local"); st.rerun()
                else: st.error(uid)

# --- Cards
def render_card(row, liked, bagged, uid):
    st.image(row.image, use_column_width=True)
    st.markdown(f"**{row.title}**  \n_{row.provider} • {row.genre}_")
    c1, c2 = st.columns(2)
    lk_key = f"lk_{row.item_id}_{uuid.uuid4().hex[:6]}"
    bg_key = f"bg_{row.item_id}_{uuid.uuid4().hex[:6]}"
    if c1.button("❤️" if liked else "♡ Like", key=lk_key):
        if liked: st.session_state.liked.discard(row.item_id)
        else: st.session_state.liked.add(row.item_id)
        add_interaction(uid, row.item_id, "like" if not liked else "unlike")
        st.experimental_rerun()
    if c2.button("👜" if bagged else "➕ Bag", key=bg_key):
        if bagged: st.session_state.bag.discard(row.item_id)
        else: st.session_state.bag.add(row.item_id)
        add_interaction(uid, row.item_id, "bag" if not bagged else "remove_bag")
        st.experimental_rerun()

def section(title, df, ids, uid):
    if not ids: st.write("Nothing here yet."); return
    st.markdown(f"### {title}")
    cols = st.columns(CARD_COLS)
    for i, iid in enumerate(ids[:CARD_COLS*2+5]):
        r = df[df.item_id == iid]
        if not r.empty:
            with cols[i % CARD_COLS]:
                render_card(r.iloc[0], iid in st.session_state.liked, iid in st.session_state.bag, uid)

# --- Core
def main():
    if "authed" not in st.session_state: st.session_state.authed = False
    if not st.session_state.authed: return auth_page()
    uid = st.session_state.uid
    page = st.sidebar.radio("Navigate", ["Home", "Liked", "Bag"])
    if st.sidebar.button("Sign Out"):
        for k in ["authed", "uid", "email"]: st.session_state[k] = None
        st.session_state.liked, st.session_state.bag = set(), set()
        st.rerun()

    df = load_data()
    st.session_state.setdefault("items_df", df)
    st.session_state.setdefault("liked", set())
    st.session_state.setdefault("bag", set())

    if page == "Home":
        hero()
        st.markdown("## Welcome to ReccoVerse")
        section("Top Picks For You", df, df["item_id"].tolist()[:20], uid)
    elif page == "Liked":
        section("Liked Items", df, list(st.session_state.liked), uid)
    else:
        section("Your Bag", df, list(st.session_state.bag), uid)

if __name__ == "__main__":
    main()

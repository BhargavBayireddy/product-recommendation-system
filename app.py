# app.py — ReccoVerse Cinematic Multi-Domain Recommender (Final)
import os, io, json, zipfile, gzip, base64, requests, uuid, time
from pathlib import Path
import pandas as pd
import streamlit as st
from firebase_init import (
    signup_email_password, login_email_password,
    add_interaction, fetch_user_interactions, fetch_global_interactions,
    ensure_user, email_exists, send_phone_otp, verify_phone_otp, FIREBASE_READY
)
from gnn_infer import (
    load_item_embeddings, make_user_vector, recommend_items,
    cold_start_mmr
)

APP_NAME = "ReccoVerse"
ART = Path("artifacts")
CARD_COLS = 5
st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

# --------------------------------------------------------------
# UI Style (Dark Netflix Look)
# --------------------------------------------------------------
st.markdown("""
<style>
body,[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 20% 10%, #050c1f, #000) !important;
  color:#f2f2f2;font-family:'Poppins',sans-serif;
}
.hero-wrap{position:relative;height:52vh;border-radius:20px;overflow:hidden;
  border:1px solid rgba(255,255,255,.1);box-shadow:0 0 80px rgba(73,187,255,.15);}
.hero-video{position:absolute;top:0;left:0;width:100%;height:100%;
  object-fit:cover;filter:brightness(.7) contrast(1.2);}
.hero-overlay{position:absolute;inset:0;
  background:linear-gradient(180deg,rgba(0,0,0,.3)0%,rgba(0,0,0,.9)90%);
  display:flex;flex-direction:column;justify-content:center;align-items:center;}
.brand{font-size:3.8rem;font-weight:900;
  background:linear-gradient(90deg,#00f2ff,#ff00c3);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  text-shadow:0 0 25px rgba(0,255,255,.2);}
.tagline{font-size:1.2rem;opacity:.85;margin-top:.6rem;color:#d4d4d4;}
.stButton>button{
  background:linear-gradient(135deg,#00ffff,#ff00c3);
  border:none;border-radius:999px;color:#fff;font-weight:600;
  padding:.6rem 1.2rem;transition:.3s;}
.stButton>button:hover{transform:scale(1.06);
  box-shadow:0 0 18px rgba(0,255,255,.5);}
.card{border-radius:18px;overflow:hidden;
  border:1px solid rgba(255,255,255,.1);
  transition:transform .18s ease,box-shadow .18s ease;}
.card:hover{transform:translateY(-4px)scale(1.02);
  box-shadow:0 18px 44px rgba(0,0,0,.45),0 0 60px rgba(73,187,255,.07);}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
def hero_with_video():
    local = ART / "hero.mp4"
    if local.exists():
        b64 = base64.b64encode(local.read_bytes()).decode("utf-8")
        src = f"data:video/mp4;base64,{b64}"
    else:
        src = "https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4"
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

# --------------------------------------------------------------
def get_tmdb_poster(title: str):
    key = st.secrets.get("TMDB_API_KEY", "57b87af46cd78b943c23b3b94c68cfef")
    if not key:
        return "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80&w=1200"
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={key}&query={requests.utils.quote(title)}"
        r = requests.get(url, timeout=5).json()
        if r.get("results"):
            path = r["results"][0].get("poster_path")
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception:
        pass
    return "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80&w=1200"

# --------------------------------------------------------------
def init_state():
    for k, v in {
        "authed": False, "uid": None, "email": None,
        "items_df": None, "embeddings": None, "id_to_idx": None, "A": None,
        "liked": set(), "bag": set(),
        "auth_mode": "email", "otp_sent": False, "otp_phone": ""
    }.items():
        st.session_state.setdefault(k, v)
init_state()

# --------------------------------------------------------------
@st.cache_data(show_spinner="Loading datasets…")
def load_multidomain_online():
    frames = []
    # --- Movies
    try:
        z = requests.get("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", timeout=15)
        with zipfile.ZipFile(io.BytesIO(z.content)) as f:
            with f.open("ml-latest-small/movies.csv") as c:
                movies = pd.read_csv(c)
        movies = movies.sample(min(200, len(movies)))
        posters = [get_tmdb_poster(t) for t in movies["title"].tolist()]
        frames.append(pd.DataFrame({
            "item_id": "mv_" + movies["movieId"].astype(str),
            "title": movies["title"],
            "provider": "Netflix",
            "genre": movies["genres"].str.split("|").str[0],
            "image": posters,
            "text": movies["title"] + " " + movies["genres"]
        }))
    except Exception as e:
        st.warning(f"Movies load failed: {e}")

    # --- Music
    try:
        music = pd.read_csv("https://raw.githubusercontent.com/yg397/music-recommender-dataset/master/data.csv").dropna().sample(150)
        frames.append(pd.DataFrame({
            "item_id": "mu_" + music["artist"].astype(str) + "_" + music["track"].astype(str),
            "title": music["track"],
            "provider": "Spotify",
            "genre": "Music",
            "image": "https://images.unsplash.com/photo-1507874457470-272b3c8d8ee2?q=80&w=1200",
            "text": music["artist"] + " " + music["track"]
        }))
    except Exception:
        pass

    # --- Beauty Products
    try:
        r = requests.get("https://datarepo.s3.amazonaws.com/beauty_5.json.gz", timeout=10)
        lines = gzip.decompress(r.content).splitlines()[:800]
        beauty = pd.DataFrame([json.loads(l) for l in lines])
        frames.append(pd.DataFrame({
            "item_id": "pr_" + beauty["asin"].astype(str),
            "title": beauty["title"],
            "provider": "Amazon",
            "genre": "Beauty",
            "image": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?q=80&w=1200",
            "text": beauty["title"] + " " + beauty.get("description", "").astype(str)
        }).dropna().sample(150))
    except Exception:
        pass

    df = pd.concat(frames, ignore_index=True).drop_duplicates("title")
    return df

# --------------------------------------------------------------
def auth_page():
    hero_with_video(); st.write("")
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
            email = st.text_input("Email")
            pwd = st.text_input("Password", type="password")
            c1, c2 = st.columns(2)
            if c1.button("Continue", use_container_width=True):
                if email_exists(email):
                    ok, uid = login_email_password(email, pwd)
                    if ok:
                        st.session_state.update(authed=True, uid=uid, email=email)
                        ensure_user(uid, email=email)
                        st.success("Logged in!"); st.rerun()
                    else: st.error("Invalid credentials.")
                else: st.warning("Account not found.")
            if c2.button("Create Account", use_container_width=True):
                ok, uid = signup_email_password(email, pwd)
                if ok:
                    st.session_state.update(authed=True, uid=uid, email=email)
                    ensure_user(uid, email=email)
                    st.success("Account created!"); st.rerun()
                else: st.error(uid)
        else:
            phone = st.text_input("Mobile (+91XXXXXXXXXX)")
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
                    st.success("Logged in with mobile."); st.rerun()
                else: st.error(uid)

# --------------------------------------------------------------
def render_card(row, liked, bagged, uid, prefix=""):
    st.image(row.image, use_column_width=True)
    st.markdown(f"**{row.title}**  \n_{row.provider} • {row.genre}_")
    c1, c2 = st.columns(2)
    like_key = f"{prefix}_like_{row.item_id}_{uuid.uuid4().hex[:5]}"
    bag_key = f"{prefix}_bag_{row.item_id}_{uuid.uuid4().hex[:5]}"
    if c1.button("❤️" if liked else "♡ Like", key=like_key):
        if liked:
            st.session_state.liked.discard(row.item_id)
            add_interaction(uid, row.item_id, "unlike")
        else:
            st.session_state.liked.add(row.item_id)
            add_interaction(uid, row.item_id, "like")
        st.toast("❤️ Updated!", icon="❤️")
    if c2.button("👜" if bagged else "➕ Bag", key=bag_key):
        if bagged:
            st.session_state.bag.discard(row.item_id)
            add_interaction(uid, row.item_id, "remove_bag")
        else:
            st.session_state.bag.add(row.item_id)
            add_interaction(uid, row.item_id, "bag")
        st.toast("👜 Bag updated!", icon="👜")

def section(title, df, ids, uid):
    st.markdown(f"### {title}")
    cols = st.columns(CARD_COLS)
    for i, iid in enumerate(ids[:CARD_COLS*2+5]):
        row = df[df.item_id == iid]
        if not row.empty:
            with cols[i % CARD_COLS]:
                render_card(row.iloc[0], iid in st.session_state.liked, iid in st.session_state.bag, uid, prefix=title.replace(" ", "_"))

def ensure_embs():
    if st.session_state.items_df is None:
        df = load_multidomain_online()
        items, embs, idmap, A = load_item_embeddings(df, ART)
        st.session_state.items_df, st.session_state.embeddings, st.session_state.id_to_idx, st.session_state.A = items, embs, idmap, A

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

# --------------------------------------------------------------
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

def main():
    if not st.session_state.authed:
        auth_page()
    else:
        p = navbar(); uid = st.session_state.uid
        if p == "Home": home(uid)
        elif p == "Liked": section("Liked Items", st.session_state.items_df, list(st.session_state.liked), uid)
        else: section("Your Bag", st.session_state.items_df, list(st.session_state.bag), uid)

if __name__ == "__main__":
    main()

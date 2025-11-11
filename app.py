# app.py — ReccoVerse (cinematic, single login page + email/OTP + motion video)
import os, io, json, zipfile, gzip, base64, requests
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# ---------- CSS (login visibility improved) ----------
st.markdown("""
<style>
.hero-wrap{position:relative;height:48vh;min-height:380px;border-radius:28px;overflow:hidden;border:1px solid #1e2738;}
.hero-video{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;filter:contrast(1.05) brightness(.82);}
.hero-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;
  background: radial-gradient(900px 420px at 20% 10%, rgba(73,187,255,.10), transparent 60%),
              radial-gradient(900px 420px at 80% 10%, rgba(255,73,146,.08), transparent 60%);}
.brand{font-size:3.1rem;font-weight:900;letter-spacing:.02em;text-shadow:0 0 18px rgba(73,187,255,.28);}
.tagline{opacity:.9;margin-top:.35rem;}
/* Inputs visible on dark */
input[type="text"], input[type="password"]{
  background: rgba(255,255,255,0.08) !important;
  color: #e7e7ea !important;
  border: 1px solid rgba(255,255,255,0.28) !important;
  border-radius: 10px !important;
  padding: .65rem .85rem !important;
  font-size: 1rem !important;
}
input::placeholder{color: rgba(255,255,255,0.55)}
/* Buttons */
.stButton>button{
  background: linear-gradient(90deg, #2979ff, #00bfa5) !important;
  color: #fff !important; border: none !important; border-radius: 999px !important;
  font-weight: 700 !important; padding: .6rem 1rem !important;
}
.card{border-radius:18px; overflow:hidden; border:1px solid #1f2a3a; transition:transform .18s ease, box-shadow .18s ease;}
.card:hover{ transform: translateY(-4px) scale(1.02); box-shadow:0 18px 44px rgba(0,0,0,.45), 0 0 60px rgba(73,187,255,.07); }
</style>
""", unsafe_allow_html=True)

# ---------- Session ----------
def init_state():
    s = st.session_state
    s.setdefault("authed", False)
    s.setdefault("uid", None)
    s.setdefault("email", None)
    s.setdefault("items_df", None)
    s.setdefault("embeddings", None)
    s.setdefault("id_to_idx", None)
    s.setdefault("A", None)
    s.setdefault("liked", set())
    s.setdefault("bag", set())
    s.setdefault("auth_mode", "email")
    s.setdefault("otp_sent", False)
    s.setdefault("otp_phone", "")
init_state()

# ---------- Motion video hero ----------
def hero_with_video():
    local = ART / "hero.mp4"
    if local.exists():
        b64 = base64.b64encode(local.read_bytes()).decode("utf-8")
        src = f"data:video/mp4;base64,{b64}"
    else:
        # default motion video (no need to host yourself)
        src = "https://assets.mixkit.co/videos/preview/mixkit-artificial-intelligence-visualization-984-large.mp4"
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

# ---------- Data loader (movies + music + beauty) ----------
@st.cache_data(show_spinner="Fetching Movies + Music + Beauty…")
def load_multidomain_online():
    frames = []

    # MovieLens (movies)
    try:
        z = requests.get("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", timeout=12)
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
        st.warning(f"Movies load failed: {e}")

    # Music (Last.fm small)
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
        st.warning(f"Music load failed: {e}")

    # Amazon Beauty subset (mirror)
    try:
        r = requests.get("https://datarepo.s3.amazonaws.com/beauty_5.json.gz", timeout=12)
        lines = gzip.decompress(r.content).splitlines()[:2000]
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
        st.warning(f"Beauty load failed: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.drop_duplicates(subset=["title"], inplace=True)
    return df

# ---------- Auth (single page with Email or Mobile) ----------
def auth_page():
    hero_with_video()
    st.write("")
    left, right = st.columns([1,1])

    with left:
        st.markdown("#### Join with")
        c1, c2 = st.columns(2)
        if c1.button("📧 Email", use_container_width=True): st.session_state.auth_mode = "email"
        if c2.button("📱 Mobile (OTP)", use_container_width=True): st.session_state.auth_mode = "phone"
        st.caption(f"Backend: {'Firebase' if FIREBASE_READY else 'Local mock'}")

    with right:
        if st.session_state.auth_mode == "email":
            st.markdown("##### Continue with Email")
            email = st.text_input("Email")
            pwd = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            if col1.button("Continue", use_container_width=True):
                if email_exists(email):
                    ok, uid_or_msg = login_email_password(email, pwd)
                    if ok:
                        st.session_state.update(authed=True, uid=uid_or_msg, email=email)
                        ensure_user(uid_or_msg, email=email)
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")
                else:
                    st.warning("Account does not exist. Click 'Create Account'.")
            if col2.button("Create Account", use_container_width=True):
                if not email or not pwd:
                    st.error("Enter email and password.")
                else:
                    ok, uid_or_msg = signup_email_password(email, pwd)
                    if ok:
                        st.session_state.update(authed=True, uid=uid_or_msg, email=email)
                        ensure_user(uid_or_msg, email=email)
                        st.success("Account created and logged in.")
                        st.rerun()
                    else:
                        st.error(uid_or_msg)
        else:
            st.markdown("##### Continue with Mobile (OTP)")
            phone = st.text_input("Mobile number (E.164, e.g., +91XXXXXXXXXX)")
            otp = st.text_input("Enter OTP")
            c1, c2 = st.columns(2)
            if c1.button("Send OTP", use_container_width=True):
                if not phone:
                    st.error("Enter mobile number.")
                else:
                    ok, msg = send_phone_otp(phone)
                    st.session_state.otp_sent = ok
                    st.session_state.otp_phone = phone
                    if ok:
                        if msg.isdigit(): st.info(f"(Dev) OTP: {msg}")
                        else: st.success(msg)
                    else:
                        st.error(msg)
            if c2.button("Verify & Continue", use_container_width=True):
                if not st.session_state.otp_sent:
                    st.error("Send OTP first.")
                elif not otp:
                    st.error("Enter the OTP.")
                else:
                    ok, uid_or_msg = verify_phone_otp(st.session_state.otp_phone, otp)
                    if ok:
                        st.session_state.update(authed=True, uid=uid_or_msg, email=f"{phone}@phone.local")
                        ensure_user(uid_or_msg, phone=phone)
                        st.success("Logged in with mobile.")
                        st.rerun()
                    else:
                        st.error(uid_or_msg)

# ---------- Grid UI ----------
def render_item_card(row, liked, bagged, uid):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(row["image"], use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"**{row['title']}**")
    st.caption(f"{row['provider']} • {row['genre']}")
    c1, c2 = st.columns(2)
    if c1.button(("❤️ Liked" if liked else "♡ Like"), key=f"like_{row.item_id}"):
        add_interaction(uid, row.item_id, "unlike" if liked else "like")
        (st.session_state.liked.discard if liked else st.session_state.liked.add)(row.item_id)
        st.rerun()
    if c2.button(("👜 In Bag" if bagged else "➕ Add to Bag"), key=f"bag_{row.item_id}"):
        add_interaction(uid, row.item_id, "remove_bag" if bagged else "bag")
        (st.session_state.bag.discard if bagged else st.session_state.bag.add)(row.item_id)
        st.rerun()

def section_grid(title, items_df, ids, uid):
    st.markdown(f"### {title}")
    if not ids: return
    cols = st.columns(CARD_COLS)
    for i, iid in enumerate(ids[:CARD_COLS*2+5]):
        row = items_df[items_df.item_id == iid]
        if row.empty: continue
        with cols[i % CARD_COLS]:
            r = row.iloc[0]
            render_item_card(r, iid in st.session_state.liked, iid in st.session_state.bag, uid)

# ---------- Embeddings ----------
def ensure_embeddings_loaded():
    if st.session_state.items_df is None:
        src = load_multidomain_online()
        items_df, embs, id2idx, A = load_item_embeddings(items=src, artifacts_dir=ART)
        st.session_state.items_df = items_df
        st.session_state.embeddings = embs
        st.session_state.id_to_idx = id2idx
        st.session_state.A = A

# ---------- Pages ----------
def page_home(uid):
    st.markdown(f"## {APP_NAME}")
    st.caption(f"Signed in as **{st.session_state.email}** • Backend: {'Firebase' if FIREBASE_READY else 'Mock'}")
    ensure_embeddings_loaded()
    items_df = st.session_state.items_df
    embs = st.session_state.embeddings
    id2idx = st.session_state.id_to_idx
    A = st.session_state.A

    interactions = fetch_user_interactions(uid)
    st.session_state.liked = set([x["item_id"] for x in interactions if x["action"] == "like"])
    st.session_state.bag = set([x["item_id"] for x in interactions if x["action"] == "bag"])

    user_vec = make_user_vector(st.session_state.liked, st.session_state.bag, id2idx, embs)

    top_picks = recommend_items(user_vec, embs, items_df, exclude=set(st.session_state.liked), topk=15, A=A)
    crowd = fetch_global_interactions(limit=300)
    ppl_like_you = recommend_items(user_vec, embs, items_df, topk=12, A=A, crowd=crowd)
    because_similar = recommend_items(user_vec, embs, items_df, topk=12, A=A, force_content=True)
    cold = cold_start_mmr(items_df, embs, lambda_=0.65, k=12)

    section_grid("Top Picks For You", items_df, top_picks, uid)
    section_grid("People Like You Also Liked", items_df, ppl_like_you, uid)
    section_grid("Because You Liked Similar Items", items_df, because_similar, uid)
    section_grid("Explore Something Different", items_df, cold, uid)

    st.autorefresh(interval=REFRESH_MS, key="home_refresh")

def page_liked(uid):
    st.markdown("## Liked Items")
    ensure_embeddings_loaded()
    section_grid("Your ❤️ Likes", st.session_state.items_df, list(st.session_state.liked), uid)

def page_bag(uid):
    st.markdown("## Your Bag")
    ensure_embeddings_loaded()
    section_grid("Saved for Later", st.session_state.items_df, list(st.session_state.bag), uid)

# ---------- Nav ----------
def navbar():
    st.sidebar.markdown("### ReccoVerse")
    page = st.sidebar.radio("Navigation", ["Home", "Liked Items", "Bag"])
    if st.sidebar.button("Sign Out"):
        st.session_state.authed = False
        st.session_state.uid = None
        st.session_state.email = None
        st.session_state.liked.clear()
        st.session_state.bag.clear()
        st.rerun()
    return page

# ---------- Entrypoint ----------
def main():
    if not st.session_state.authed:
        auth_page(); return
    page = navbar()
    uid = st.session_state.uid
    {"Home": page_home, "Liked Items": page_liked, "Bag": page_bag}[page](uid)

if __name__ == "__main__":
    main()

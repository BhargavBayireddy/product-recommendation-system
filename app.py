import os, io, json, time, zipfile, gzip, hashlib, random, base64, requests
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from firebase_init import (
    signup_email_password,
    login_email_password,
    add_interaction,
    fetch_user_interactions,
    fetch_global_interactions,
    ensure_user,
    email_exists,
    send_phone_otp,
    verify_phone_otp,
    FIREBASE_READY,
)

from gnn_infer import (
    load_item_embeddings,
    make_user_vector,
    recommend_items,
    diversity_personalization_novelty,
    cold_start_mmr,
)

APP_NAME = "ReccoVerse"
ART = Path("artifacts")
REFRESH_MS = 5000
CARD_COLS = 5

st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

# ------------------ CSS ------------------
def inject_css():
    css_path = Path("ui.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    # hero/video styles
    st.markdown("""
    <style>
    .hero-wrap { position:relative; height:48vh; min-height:380px; border-radius:28px; overflow:hidden; border:1px solid #1e2738; }
    .hero-video { position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; filter:contrast(1.08) saturate(1.05) brightness(.82); }
    .hero-overlay { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; flex-direction:column;
        background: radial-gradient(900px 420px at 20% 10%, rgba(73,187,255,.10), transparent 60%),
                    radial-gradient(900px 420px at 80% 10%, rgba(255,73,146,.08), transparent 60%); }
    .brand { font-size:3.1rem; font-weight:900; letter-spacing:.02em; text-shadow:0 0 18px rgba(73,187,255,.28); }
    .tagline { opacity:.88; margin-top:.35rem; }
    .login-card { background:rgba(12,18,33,.55); border:1px solid #1f2a3a; border-radius:18px; padding:18px; backdrop-filter: blur(6px); }
    .option-btn { width:100%; padding:.8rem 1rem; border-radius:999px; border:1px solid #24324a; background:#0f1626; color:#e7e7ea; font-weight:700; }
    .option-btn:hover { background:#111b2e; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ------------------ Session ------------------
def init_state():
    ss = st.session_state
    ss.setdefault("authed", False)
    ss.setdefault("uid", None)
    ss.setdefault("email", None)
    ss.setdefault("items_df", None)
    ss.setdefault("embeddings", None)
    ss.setdefault("id_to_idx", None)
    ss.setdefault("liked", set())
    ss.setdefault("bag", set())
    ss.setdefault("auth_mode", "email")    # 'email' or 'phone'
    ss.setdefault("otp_sent", False)
    ss.setdefault("otp_phone", "")
    ss.setdefault("otp_hint", "")          # dev mode: show OTP when Twilio not set

init_state()

# ------------------ Online Multi-domain Loader ------------------
@st.cache_data(show_spinner="Fetching Movies + Music + Beauty…")
def load_multidomain_online():
    frames = []

    # Movies (MovieLens)
    try:
        mov_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        with zipfile.ZipFile(io.BytesIO(requests.get(mov_url, timeout=12).content)) as z:
            with z.open("ml-latest-small/movies.csv") as f:
                movies = pd.read_csv(f)
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

    # Music (Last.fm tiny sample)
    try:
        mus_url = "https://raw.githubusercontent.com/yg397/music-recommender-dataset/master/data.csv"
        music = pd.read_csv(mus_url).dropna().sample(200)
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

    # Beauty (Amazon subset mirror)
    try:
        beauty_url = "https://datarepo.s3.amazonaws.com/beauty_5.json.gz"
        with gzip.open(io.BytesIO(requests.get(beauty_url, timeout=12).content)) as f:
            lines = [json.loads(l) for l in f.readlines()[:2000]]
        beauty = pd.DataFrame(lines)
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

    all_df = pd.concat(frames, ignore_index=True)
    all_df.drop_duplicates(subset=["title"], inplace=True)
    return all_df

# ------------------ Hero (AI video background) ------------------
def hero_with_video():
    # Use a local video if available at artifacts/hero.mp4 else remote fallback.
    local_video = ART / "hero.mp4"
    if local_video.exists():
        b64 = base64.b64encode(local_video.read_bytes()).decode("utf-8")
        src = f"data:video/mp4;base64,{b64}"
    else:
        # Fallback public AI/tech motion video
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

# ------------------ Auth UI (single page) ------------------
def auth_page():
    hero_with_video()
    st.write("")
    left, right = st.columns([1,1])

    # Mode switcher
    with left:
        st.markdown("#### Join with")
        colA, colB = st.columns(2)
        with colA:
            if st.button("📧 Email", use_container_width=True):
                st.session_state.auth_mode = "email"
        with colB:
            if st.button("📱 Mobile (OTP)", use_container_width=True):
                st.session_state.auth_mode = "phone"
        st.caption(f"Backend: {'Firebase' if FIREBASE_READY else 'Local mock'}")

    with right:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        if st.session_state.auth_mode == "email":
            st.markdown("##### Continue with Email")
            email = st.text_input("Email", key="email_login")
            pwd = st.text_input("Password", type="password", key="pwd_login")

            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Continue", use_container_width=True):
                    if not email or not pwd:
                        st.error("Enter email and password.")
                    else:
                        if email_exists(email):
                            ok, uid_or_msg = login_email_password(email, pwd)
                            if ok:
                                st.session_state.authed = True
                                st.session_state.uid = uid_or_msg
                                st.session_state.email = email
                                ensure_user(uid_or_msg, email=email)
                                st.success("Logged in!")
                                st.experimental_rerun()
                            else:
                                # Distinguish wrong password vs other
                                if "INVALID_PASSWORD" in uid_or_msg or "password" in uid_or_msg.lower():
                                    st.error("Incorrect password. Try again.")
                                else:
                                    st.error("Login failed. Check credentials.")
                        else:
                            st.warning("Account does not exist. Click 'Create Account' to sign up.")
            with c2:
                if st.button("Create Account", use_container_width=True):
                    if not email or not pwd:
                        st.error("Enter email and password to sign up.")
                    else:
                        if email_exists(email):
                            st.error("Email already registered. Use Continue to sign in.")
                        else:
                            ok, uid_or_msg = signup_email_password(email, pwd)
                            if ok:
                                st.session_state.authed = True
                                st.session_state.uid = uid_or_msg
                                st.session_state.email = email
                                ensure_user(uid_or_msg, email=email)
                                st.success("Account created and logged in.")
                                st.experimental_rerun()
                            else:
                                st.error(uid_or_msg)

        else:
            st.markdown("##### Continue with Mobile (OTP)")
            phone = st.text_input("Mobile number (E.164, e.g., +91xxxxxxxxxx)", key="phone_input")
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("Send OTP", use_container_width=True):
                    if not phone:
                        st.error("Enter your mobile number.")
                    else:
                        ok, msg = send_phone_otp(phone)
                        st.session_state.otp_sent = ok
                        st.session_state.otp_phone = phone
                        st.session_state.otp_hint = msg if msg.isdigit() else ""
                        if ok:
                            if msg.isdigit():
                                st.info(f"(Dev) OTP: {msg}")
                            else:
                                st.success(msg)
                        else:
                            st.error(msg)
            with col2:
                otp = st.text_input("Enter OTP", key="otp_code")
                if st.button("Verify & Continue", use_container_width=True):
                    if not st.session_state.otp_sent:
                        st.error("Send OTP first.")
                    elif not otp:
                        st.error("Enter the OTP.")
                    else:
                        ok, uid_or_msg = verify_phone_otp(st.session_state.otp_phone, otp)
                        if ok:
                            st.session_state.authed = True
                            st.session_state.uid = uid_or_msg
                            st.session_state.email = f"{st.session_state.otp_phone}@phone.local"
                            ensure_user(uid_or_msg, phone=st.session_state.otp_phone)
                            st.success("Logged in with mobile.")
                            st.experimental_rerun()
                        else:
                            st.error(uid_or_msg)

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Item Grid/UI ------------------
def render_item_card(row: pd.Series, liked: bool, bagged: bool, uid: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(row["image"], use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"**{row['title']}**")
        st.caption(f"{row['provider']} • {row['genre']}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(("❤️ Liked" if liked else "♡ Like"), key=f"like_{row.item_id}"):
                if liked:
                    add_interaction(uid, row.item_id, "unlike")
                    st.session_state.liked.discard(row.item_id)
                else:
                    add_interaction(uid, row.item_id, "like")
                    st.session_state.liked.add(row.item_id)
                st.experimental_rerun()
        with c2:
            if st.button(("👜 In Bag" if bagged else "➕ Add to Bag"), key=f"bag_{row.item_id}"):
                if bagged:
                    add_interaction(uid, row.item_id, "remove_bag")
                    st.session_state.bag.discard(row.item_id)
                else:
                    add_interaction(uid, row.item_id, "bag")
                    st.session_state.bag.add(row.item_id)
                st.experimental_rerun()

def section_grid(title: str, items_df: pd.DataFrame, ids: List[str], uid: str):
    st.markdown(f"### {title}")
    if not ids:
        st.info("No items to show yet.")
        return
    cols = st.columns(CARD_COLS)
    for i, iid in enumerate(ids[:CARD_COLS * 2 + 5]):
        row = items_df[items_df.item_id == iid]
        if row.empty: 
            continue
        row = row.iloc[0]
        with cols[i % CARD_COLS]:
            render_item_card(row, iid in st.session_state.liked, iid in st.session_state.bag, uid)

# ------------------ Data/Embeddings ------------------
def ensure_embeddings_loaded():
    if st.session_state.items_df is None or st.session_state.embeddings is None:
        items_online = load_multidomain_online()
        items_df, embs, id_to_idx, A = load_item_embeddings(items=items_online, artifacts_dir=ART)
        st.session_state.items_df = items_df
        st.session_state.embeddings = embs
        st.session_state.id_to_idx = id_to_idx
        st.session_state.A = A

# ------------------ Pages ------------------
def page_home(uid: str):
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

def page_liked(uid: str):
    st.markdown("## Liked Items")
    ensure_embeddings_loaded()
    section_grid("Your ❤️ Likes", st.session_state.items_df, list(st.session_state.liked), uid)

def page_bag(uid: str):
    st.markdown("## Your Bag")
    ensure_embeddings_loaded()
    section_grid("Saved for Later", st.session_state.items_df, list(st.session_state.bag), uid)

def page_compare(uid: str):
    st.markdown("## Compare Engines")
    ensure_embeddings_loaded()

    items_df = st.session_state.items_df
    embs = st.session_state.embeddings
    id2idx = st.session_state.id_to_idx
    A = st.session_state.A
    user_vec = make_user_vector(st.session_state.liked, st.session_state.bag, id2idx, embs)

    rec_personal = recommend_items(user_vec, embs, items_df, topk=12, A=A)
    rec_pop = [r for r in items_df.sample(frac=1, random_state=3).item_id.tolist()[:12]]
    rec_rand = [r for r in items_df.sample(frac=1, random_state=7).item_id.tolist()[:12]]

    def bundle(ids): return items_df[items_df.item_id.isin(ids)]
    p_div, p_per, p_nov = diversity_personalization_novelty(bundle(rec_personal), user_vec, embs, id2idx)
    o_div, o_per, o_nov = diversity_personalization_novelty(bundle(rec_pop), user_vec, embs, id2idx)
    r_div, r_per, r_nov = diversity_personalization_novelty(bundle(rec_rand), user_vec, embs, id2idx)

    labels = ["Diversity", "Personalization", "Novelty"]
    fig = go.Figure()
    fig.add_bar(name="Quanta-GNN", x=labels, y=[p_div,p_per,p_nov])
    fig.add_bar(name="Popularity", x=labels, y=[o_div,o_per,o_nov])
    fig.add_bar(name="Random", x=labels, y=[r_div,r_per,r_nov])
    fig.update_layout(barmode="group", height=420, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Navbar & Router ------------------
def navbar():
    st.sidebar.markdown("### ReccoVerse")
    page = st.sidebar.radio("Navigation", ["Home", "Liked Items", "Bag", "Compare"])
    if st.sidebar.button("Sign Out"):
        for k in ["authed","uid","email","liked","bag"]:
            st.session_state[k] = None if k in ["uid","email"] else False if k=="authed" else set()
        st.experimental_rerun()
    return page

def main():
    if not st.session_state.authed:
        auth_page()
        return
    page = navbar()
    uid = st.session_state.uid
    if page == "Home":
        page_home(uid)
    elif page == "Liked Items":
        page_liked(uid)
    elif page == "Bag":
        page_bag(uid)
    elif page == "Compare":
        page_compare(uid)

if __name__ == "__main__":
    main()

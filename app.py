# app.py — ReccoVerse (cinematic AI-powered hybrid recommender)
# Version: Multi-domain online dataset (Movies + Music + Beauty)
# Run locally: streamlit run app.py

import os, io, json, time, zipfile, gzip, hashlib, random, requests
from pathlib import Path
from datetime import datetime
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
    FIREBASE_READY,
)

from gnn_infer import (
    load_item_embeddings,
    make_user_vector,
    recommend_items,
    diversity_personalization_novelty,
    cold_start_mmr,
)

# ---------- App Config ----------
APP_NAME = "ReccoVerse"
ART = Path("artifacts")
REFRESH_MS = 5000
CARD_COLS = 5

st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

# ---------- Inject CSS ----------
def inject_css():
    css_path = Path("ui.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            body{background:#0b0e14;color:#e7e7ea;}
            .recco-title{font-size:2.4rem;font-weight:900;letter-spacing:.02em;margin:1rem 0;}
            .card{border-radius:14px;overflow:hidden;border:1px solid #1f2633;margin-bottom:10px;transition:all .2s ease;}
            .card:hover{transform:scale(1.02);box-shadow:0 0 25px rgba(73,187,255,.1);}
            .section-title{font-size:1.3rem;font-weight:800;margin-top:1rem;}
            </style>
            """,
            unsafe_allow_html=True,
        )

inject_css()

# ---------- Session State ----------
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

init_state()

# ---------- Multi-Domain Online Loader ----------
@st.cache_data(show_spinner="Fetching multi-domain datasets (Movies, Music, Beauty)...")
def load_multidomain_online():
    frames = []

    # 🎬 MOVIES — MovieLens
    try:
        mov_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        with zipfile.ZipFile(io.BytesIO(requests.get(mov_url, timeout=10).content)) as z:
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
        st.warning(f"Movies dataset load failed: {e}")

    # 🎧 MUSIC — Last.fm open sample
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
        st.warning(f"Music dataset load failed: {e}")

    # 💄 BEAUTY PRODUCTS — Amazon subset
    try:
        beauty_url = "https://datarepo.s3.amazonaws.com/beauty_5.json.gz"
        with gzip.open(io.BytesIO(requests.get(beauty_url, timeout=10).content)) as f:
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
        st.warning(f"Beauty dataset load failed: {e}")

    all_df = pd.concat(frames, ignore_index=True)
    all_df.drop_duplicates(subset=["title"], inplace=True)
    st.success(f"Fetched {len(all_df)} items across {len(frames)} domains.")
    return all_df

# ---------- Auth UI ----------
def splash_login():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-inner">
                <div class="brand-glow">ReccoVerse</div>
                <div class="tagline">AI-curated picks across Movies • Music • Products</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Sign In / Sign Up")

    col1, col2 = st.columns(2)
    with col1:
        with st.form("login"):
            email = st.text_input("Email", key="login_email")
            pwd = st.text_input("Password", type="password", key="login_pwd")
            submitted = st.form_submit_button("Sign In")
        if submitted:
            ok, uid_or_msg = login_email_password(email, pwd)
            if ok:
                st.session_state.authed = True
                st.session_state.uid = uid_or_msg
                st.session_state.email = email
                ensure_user(uid_or_msg, email)
                st.success("Welcome back!")
                st.experimental_rerun()
            else:
                st.error(uid_or_msg)
    with col2:
        with st.form("signup"):
            email_up = st.text_input("New Email", key="signup_email")
            pwd_up = st.text_input("New Password", type="password", key="signup_pwd")
            submitted_up = st.form_submit_button("Create Account")
        if submitted_up:
            ok, uid_or_msg = signup_email_password(email_up, pwd_up)
            if ok:
                st.success("Account created! Please sign in.")
            else:
                st.error(uid_or_msg)

    st.caption(("Using " + ("Firebase" if FIREBASE_READY else "local mock")) + " backend.")

# ---------- Helper Buttons ----------
def toggle_like(uid, item_id, action):
    if action == "like":
        st.session_state.liked.add(item_id)
        add_interaction(uid, item_id, "like")
    else:
        st.session_state.liked.discard(item_id)
        add_interaction(uid, item_id, "unlike")

def toggle_bag(uid, item_id, action):
    if action == "bag":
        st.session_state.bag.add(item_id)
        add_interaction(uid, item_id, "bag")
    else:
        st.session_state.bag.discard(item_id)
        add_interaction(uid, item_id, "remove_bag")

def render_item_card(row, liked, bagged, uid):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(row["image"], use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"**{row['title']}**")
        st.caption(f"{row['provider']} • {row['genre']}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(("❤️ Liked" if liked else "♡ Like"), key=f"like_{row.item_id}"):
                toggle_like(uid, row.item_id, "unlike" if liked else "like")
                st.experimental_rerun()
        with c2:
            if st.button(("👜 In Bag" if bagged else "➕ Add"), key=f"bag_{row.item_id}"):
                toggle_bag(uid, row.item_id, "remove_bag" if bagged else "bag")
                st.experimental_rerun()

def section_grid(title, items_df, ids, uid):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if not ids:
        st.info("No items yet.")
        return
    cols = st.columns(CARD_COLS)
    for i, iid in enumerate(ids[:CARD_COLS * 2]):
        row = items_df[items_df.item_id == iid].iloc[0]
        with cols[i % CARD_COLS]:
            render_item_card(row, iid in st.session_state.liked, iid in st.session_state.bag, uid)

# ---------- Main Pages ----------
def ensure_embeddings_loaded():
    if st.session_state.items_df is None or st.session_state.embeddings is None:
        items_online = load_multidomain_online()
        items_df, embs, id_to_idx, A = load_item_embeddings(items=items_online, artifacts_dir=ART)
        st.session_state.items_df = items_df
        st.session_state.embeddings = embs
        st.session_state.id_to_idx = id_to_idx
        st.session_state.A = A

def page_home(uid):
    st.markdown(f"<div class='recco-title'>{APP_NAME}</div>", unsafe_allow_html=True)
    st.caption(f"Logged in as **{st.session_state.email}**")

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

    st.autorefresh(interval=REFRESH_MS, key="auto_refresh_home")

def page_liked(uid):
    st.markdown("<div class='recco-title'>Liked Items</div>", unsafe_allow_html=True)
    ensure_embeddings_loaded()
    section_grid("Your ❤️ Likes", st.session_state.items_df, list(st.session_state.liked), uid)

def page_bag(uid):
    st.markdown("<div class='recco-title'>Your Bag</div>", unsafe_allow_html=True)
    ensure_embeddings_loaded()
    section_grid("Saved for Later", st.session_state.items_df, list(st.session_state.bag), uid)

def page_compare(uid):
    st.markdown("<div class='recco-title'>Compare Engines</div>", unsafe_allow_html=True)
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

# ---------- Navbar ----------
def navbar():
    st.sidebar.markdown("### ReccoVerse")
    page = st.sidebar.radio("Navigation", ["Home", "Liked Items", "Bag", "Compare"])
    if st.sidebar.button("Sign Out"):
        for k in ["authed","uid","email","liked","bag"]:
            st.session_state[k] = None if k in ["uid","email"] else False if k=="authed" else set()
        st.experimental_rerun()
    return page

# ---------- Entrypoint ----------
def main():
    if not st.session_state.authed:
        splash_login()
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

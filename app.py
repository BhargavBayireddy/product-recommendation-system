# app.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --- Local modules (must exist) ---
# firebase_init provides: signup_email_password, login_email_password,
# ensure_user, add_interaction, remove_interaction,
# fetch_user_interactions, fetch_global_interactions
from firebase_init import (
    signup_email_password,
    login_email_password,
    ensure_user,
    add_interaction,
    remove_interaction,
    fetch_user_interactions,
    fetch_global_interactions,
)

# GNN embeddings / inference (provided in your project)
from gnn_infer import load_item_embeddings, make_user_vector

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_items_catalog() -> pd.DataFrame:
    """
    Replace this with your own loader (e.g., data_real.load_items()).
    Must return a DataFrame with at least:
    ['item_id','title','domain','platform'] and optional ['image','year','blurb','category','mood']
    """
    # ---- TRY your real loader first ----
    try:
        # from data_real import load_items  # Uncomment if you have it
        # return load_items()
        raise ImportError()  # force demo if you have no loader line above
    except Exception:
        # ---- Minimal demo fallback so app still runs ----
        demo = [
            # item_id, title, domain, platform, year, blurb
            ("m1", "Afternoon Acoustic", "Music", "Spotify", 2018, "Tiny click. Big vibe."),
            ("f1", "Four Rooms (1995)", "Entertainment", "Netflix", 1995, "Chef’s kiss material."),
            ("f2", "Sabrina (1995)", "Entertainment", "Netflix", 1995, "Hot pick. Zero regrets."),
            ("f3", "Restoration (1995)", "Entertainment", "Netflix", 1995, "Tiny click. Big vibe."),
            ("f4", "Evita (1996)", "Entertainment", "Netflix", 1996, "Your next favorite."),
            ("f5", "Evil Dead II (1987)", "Entertainment", "Netflix", 1987, "Trust the vibes."),
            ("f6", "Men in Black (1997)", "Entertainment", "Netflix", 1997, "Iconic, obviously."),
            ("p1", "Bass Therapy", "Music", "Spotify", 2020, "Feel the floor shake."),
        ]
        df = pd.DataFrame(demo, columns=["item_id","title","domain","platform","year","blurb"])
        df["image"] = ""  # optional column
        df["category"] = df["domain"]
        df["mood"] = np.where(df["domain"].eq("Music"), "chill", "fun")
        return df

ITEMS: pd.DataFrame = load_items_catalog()
ITEMS["item_id"] = ITEMS["item_id"].astype(str)
ITEM_ID_SET = set(ITEMS["item_id"].tolist())

ARTIFACTS_DIR = Path("artifacts")
ITEM_EMBS, IID2IDX, BACKEND_NAME = load_item_embeddings(ITEMS, ARTIFACTS_DIR)

# ------------------------------------------------------------
# Utility: fast lookups
# ------------------------------------------------------------
def df_by_ids(ids: List[str]) -> pd.DataFrame:
    if not ids: 
        return ITEMS.iloc[0:0]
    return ITEMS.set_index("item_id").loc[[i for i in ids if i in ITEM_ID_SET]].reset_index()

def tagline_for(row: pd.Series) -> str:
    dom = str(row.get("domain",""))
    plat = str(row.get("platform",""))
    lines = {
        "Music": [
            "Hot pick. Zero regrets.",
            "Chef’s playlist approves.",
            "Eargasm alert.",
            "Vibes? Delivered.",
        ],
        "Entertainment": [
            "Chef’s kiss material.",
            "You won’t text back.",
            "Popcorn’s new bestie.",
            "Binge. Regret nothing.",
        ],
        "Shopping": [
            "Cart it like you mean it.",
            "Your wallet is shaking.",
            "Steal of the day. Literally (not).",
            "Fits so good it flirts back.",
        ]
    }
    bank = lines.get(dom, ["Certified fresh pick."])
    msg = np.random.choice(bank)
    if plat:
        msg = f"{msg}"
    return msg

# ------------------------------------------------------------
# Recommendation engines
# ------------------------------------------------------------
def user_recent_vectors(uid: str) -> np.ndarray:
    try:
        inter = fetch_user_interactions(uid, limit=500)
    except Exception:
        inter = []
    return make_user_vector(inter, IID2IDX, ITEM_EMBS)

def score_items_uservec(user_vec: np.ndarray) -> pd.DataFrame:
    # cosine similarity
    em = ITEM_EMBS / (np.linalg.norm(ITEM_EMBS, axis=1, keepdims=True) + 1e-8)
    uv = user_vec / (np.linalg.norm(user_vec, axis=1, keepdims=True) + 1e-8)
    sims = (em @ uv.T).ravel()
    out = ITEMS.copy()
    out["score"] = sims
    return out.sort_values("score", ascending=False)

def popular_from_global(k: int = 50) -> List[str]:
    try:
        g = fetch_global_interactions(limit=5000)
    except Exception:
        g = []
    # simple popularity by likes + bag
    weights = {"like": 2.0, "bag": 3.0}
    pop: Dict[str, float] = {}
    for e in g:
        it = str(e.get("item_id",""))
        act = str(e.get("action",""))
        if it in ITEM_ID_SET:
            pop[it] = pop.get(it, 0.0) + weights.get(act, 0.5)
    if not pop:
        return ITEMS.sample(min(k, len(ITEMS)), random_state=7)["item_id"].tolist()
    ids = sorted(pop, key=lambda x: pop[x], reverse=True)
    return ids[:k]

def recommend(uid: str, k: int = 48) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (top_for_you, collab_because, explore_all)
    - top_for_you: GNN/embedding ranked list
    - collab_because: popularity/collab across users (not your own)
    - explore_all: quick shuffle for discovery
    """
    # per-user vector
    u = user_recent_vectors(uid)
    top = score_items_uservec(u).head(k)

    # collab via global interactions popularity
    collab_ids = popular_from_global(k=k)
    collab = df_by_ids(collab_ids)

    # explore (domain-balanced shuffle)
    explore = ITEMS.sample(min(k, len(ITEMS)), random_state=42)
    return top, collab, explore

# ------------------------------------------------------------
# UI Helpers
# ------------------------------------------------------------
def pill(text: str) -> str:
    return f'<span style="padding:2px 8px;border-radius:999px;background:#f1f3f5;font-size:12px;color:#333;border:1px solid #e6e6e6;">{text}</span>'

def render_card(row: pd.Series, uid: str, section: str):
    """
    One item card with Like / Bag / Remove (with unique keys).
    section is a short string: 'top' | 'collab' | 'explore' | 'liked' | 'bag'
    """
    item_id = str(row["item_id"])
    title = str(row.get("title",""))
    dom = str(row.get("domain",""))
    plat = str(row.get("platform",""))
    year = row.get("year", "")
    blurb = str(row.get("blurb","").strip() or tagline_for(row))
    img = str(row.get("image",""))

    st.markdown(
        f"""
        <div style="border-radius:16px;border:1px solid #eee;padding:14px;margin-bottom:10px;background:#fff;">
          <div style="display:flex;gap:16px;align-items:center;">
            <div style="width:80px;height:54px;background:#ff8a8a;border-radius:8px;"></div>
            <div style="flex:1;">
              <div style="font-weight:600">{title}{f" ({year})" if year else ""}</div>
              <div style="opacity:.8;margin:4px 0 6px 0;">
                {dom} · {plat} · {blurb}
              </div>
              <div style="display:flex;gap:6px;flex-wrap:wrap;">
                {pill(dom)} {pill(plat)}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    if cols[0].button("❤️ Like", key=f"{item_id}-{section}-like"):
        try:
            add_interaction(uid, item_id, "like")
            st.success("Added to Likes")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

    if cols[1].button("👜 Bag", key=f"{item_id}-{section}-bag"):
        try:
            add_interaction(uid, item_id, "bag")
            st.success("Added to Bag")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

    if cols[2].button("🗑️ Remove", key=f"{item_id}-{section}-remove"):
        # remove both like & bag entries for this item
        try:
            remove_interaction(uid, item_id, "like")
            remove_interaction(uid, item_id, "bag")
            st.info("Removed")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

def render_grid(df: pd.DataFrame, uid: str, header: str, section: str):
    st.subheader(header)
    # 5-column responsive grid with unique keys per item
    for _, row in df.iterrows():
        render_card(row, uid, section)

def list_user_bucket(uid: str, action: str) -> pd.DataFrame:
    try:
        hist = fetch_user_interactions(uid, limit=1000)
    except Exception:
        hist = []
    ids = [h["item_id"] for h in hist if h.get("action")==action]
    # ensure uniqueness, keep order by most recent
    ids = list(dict.fromkeys(ids))
    return df_by_ids(ids)

# ------------------------------------------------------------
# Auth UI
# ------------------------------------------------------------
def auth_gate() -> str | None:
    st.title("Sign in to continue")
    email = st.text_input("Email", value=st.session_state.get("last_email",""))
    pw = st.text_input("Password", type="password")
    c1, c2 = st.columns(2)

    # (Login)
    if c1.button("Login", use_container_width=True):
        try:
            u = login_email_password(email, pw)
            st.session_state["uid"] = u["localId"]
            st.session_state["last_email"] = email
            ensure_user(u["localId"], email=email)
            st.success("Logged in")
            st.experimental_rerun()
        except Exception as e:
            st.error("Invalid email or password. Try again or create an account.")

    # (Signup)
    if c2.button("Create account", use_container_width=True):
        if len(pw) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            try:
                u = signup_email_password(email, pw)
                st.session_state["uid"] = u["localId"]
                st.session_state["last_email"] = email
                ensure_user(u["localId"], email=email)
                st.success("Account created. Welcome!")
                st.experimental_rerun()
            except Exception:
                st.error("Could not create account. Try a different email or stronger password.")
    return None

# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_home(uid: str):
    st.caption(f"🧠 Backend: {BACKEND_NAME} · domain-colored tiles (no images) for speed")

    # Netflix-style instant search (shows ALL items while typing)
    q = st.text_input("🔎 Search anything (name, domain, category, mood)...", placeholder="Type to search...")
    qn = q.strip().lower()

    if qn:
        # contains match across several fields
        mask = (
            ITEMS["title"].str.lower().str.contains(qn, na=False) |
            ITEMS["domain"].str.lower().str.contains(qn, na=False) |
            ITEMS["platform"].str.lower().str.contains(qn, na=False) |
            ITEMS.get("category", pd.Series([""]*len(ITEMS))).astype(str).str.lower().str.contains(qn, na=False) |
            ITEMS.get("mood", pd.Series([""]*len(ITEMS))).astype(str).str.lower().str.contains(qn, na=False)
        )
        results = ITEMS[mask].copy()
        st.markdown("### 🍿 All matches")
        if results.empty:
            st.info("Nothing yet. Try a different word.")
        else:
            for _, r in results.iterrows():
                render_card(r, uid, "search")
        return

    # Otherwise: recommendations first (aggressive + cheesy copy)
    top, collab, explore = recommend(uid, k=48)

    st.markdown("## 🔥 Top picks for you")
    st.caption("If taste had a leaderboard, these would be S-tier 🥇")
    for _, r in top.iterrows():
        render_card(r, uid, "top")

    st.markdown("## 💞 Vibe-twins also loved…")
    st.caption("People who liked your faves are going feral for these 😏")
    for _, r in collab.iterrows():
        render_card(r, uid, "collab")

    st.markdown("## 🧪 Explore something different")
    st.caption("Swipe right on a new mood — we won’t tell.")
    for _, r in explore.iterrows():
        render_card(r, uid, "explore")

def page_liked(uid: str):
    st.header("❤️ Your Likes")
    liked = list_user_bucket(uid, "like")
    if liked.empty:
        st.info("You haven’t liked anything yet.")
        return
    for _, r in liked.iterrows():
        render_card(r, uid, "liked")

def page_bag(uid: str):
    st.header("👜 Your Bag")
    bag = list_user_bucket(uid, "bag")
    if bag.empty:
        st.info("Your bag is empty.")
        return
    for _, r in bag.iterrows():
        render_card(r, uid, "bag")

def page_compare(uid: str, k: int = 20):
    """
    Side-by-side model output:
    - Left: GNN (or RandomFallback) personalized ranking
    - Right: Collaborative/Popular (global)
    """
    st.header("⚔️ Model vs Model — Who recommends better?")
    st.caption("Left: **GNN** personalized ranking. Right: **Popular/Collaborative** across users.")

    # left: GNN
    u = user_recent_vectors(uid)
    left = score_items_uservec(u).head(k)

    # right: popularity/collab
    right_ids = popular_from_global(k=k)
    right = df_by_ids(right_ids)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"🔮 {BACKEND_NAME} top {k}")
        for _, r in left.iterrows():
            render_card(r, uid, "cmp-left")
    with c2:
        st.subheader(f"⭐ Collab/Popular top {k}")
        for _, r in right.iterrows():
            render_card(r, uid, "cmp-right")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Multi-Domain Recommender (GNN)", page_icon="🍿", layout="wide")

    # Guard: authenticated?
    uid = st.session_state.get("uid")
    if not uid:
        return auth_gate()

    # Sidebar
    with st.sidebar:
        st.success(f"Logged in: {st.session_state.get('last_email','')}")
        if st.button("Logout"):
            for k in ["uid"]:
                st.session_state.pop(k, None)
            st.experimental_rerun()

        page = st.radio("Go to", ["Home","Liked","Bag","Compare"], index=0)

    # Router
    if page == "Home":
        page_home(uid)
    elif page == "Liked":
        page_liked(uid)
    elif page == "Bag":
        page_bag(uid)
    else:
        page_compare(uid)

if __name__ == "__main__":
    main()

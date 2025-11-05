from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Local modules (present in your repo)
from firebase_init import (
    signup_email_password,
    login_email_password,
    add_interaction,
    remove_interaction,
    fetch_user_interactions,
    fetch_global_interactions,
    ensure_user,
)

# Optional GNN helper
try:
    from gnn_infer import load_item_embeddings, make_user_vector
    HAS_GNN = True
except Exception:
    HAS_GNN = False


# -------------------- Data --------------------
@st.cache_data(show_spinner=False)
def load_items() -> pd.DataFrame:
    """
    Expect your items CSV/loader to provide columns at least:
        item_id, name, domain, provider, category, year, summary
    If something is missing, we create safe defaults.
    """
    # Try your existing module first
    try:
        from data_real import load_items as _loader  # your existing function
        df = _loader()
    except Exception:
        # Fallback tiny demo set (still works end-to-end)
        df = pd.DataFrame(
            [
                # item_id, name, domain, provider, category, year, summary
                ["m1", "Afternoon Acoustic", "Music", "Spotify", "Playlist", 2018, "Chill acoustic vibes."],
                ["m2", "Bass Therapy", "Music", "Spotify", "Playlist", 2020, "Feel the floor shake."],
                ["f1", "Evil Dead II", "Entertainment", "Netflix", "Movie", 1987, "Trust the vibes."],
                ["f2", "Men in Black", "Entertainment", "Netflix", "Movie", 1997, "Tiny click. Big vibe."],
                ["a1", "Echo Dot (5th Gen)", "Shopping", "Amazon", "Smart Speaker", 2023, "Small size, big sound."],
                ["a2", "Kindle Paperwhite", "Shopping", "Amazon", "E-Reader", 2022, "Crisp display for books."],
            ],
            columns=["item_id", "name", "domain", "provider", "category", "year", "summary"],
        )

    # Safety: required columns + search blob
    for col, default in [
        ("item_id", ""),
        ("name", ""),
        ("domain", ""),
        ("provider", ""),
        ("category", ""),
        ("year", ""),
        ("summary", ""),
    ]:
        if col not in df.columns:
            df[col] = default
    # Search blob (lowercased)
    df["search_blob"] = (
        df["name"].fillna("").astype(str)
        + " " + df["domain"].fillna("").astype(str)
        + " " + df["provider"].fillna("").astype(str)
        + " " + df["category"].fillna("").astype(str)
        + " " + df["summary"].fillna("").astype(str)
    ).str.lower()
    return df


ITEMS = load_items()
ARTIFACTS_DIR = Path("artifacts")


# -------------------- Brand colors & badges --------------------
BRAND_BADGES = {
    "Netflix": {"emoji": "🍿", "bg": "#FF3B30", "fg": "#FFFFFF"},   # red
    "Amazon": {"emoji": "🛒", "bg": "#FF9900", "fg": "#000000"},   # orange
    "Spotify": {"emoji": "🎧", "bg": "#1DB954", "fg": "#000000"},  # green
}

def badge_capsule(label: str) -> str:
    """Return HTML capsule for provider."""
    p = BRAND_BADGES.get(label, None)
    if not p:
        # neutral capsule
        return f"""<span style="
            padding: 2px 10px; border-radius: 999px;
            background:#E6E6E6; color:#111; font-size:12px;">{label}</span>"""
    return f"""<span style="
        padding: 2px 10px; border-radius: 999px;
        background:{p['bg']}; color:{p['fg']}; font-size:12px;">{p['emoji']} {label}</span>"""


# -------------------- Auth UI --------------------
def ui_auth() -> str | None:
    st.title("Sign in to continue")

    email = st.text_input("Email", value=st.session_state.get("email", ""), key="email_input")
    pw = st.text_input("Password", type="password", key="pw_input")

    c1, c2 = st.columns([1, 1])
    uid = st.session_state.get("uid")

    def _finish_login(user):
        st.session_state["uid"] = user["localId"]
        st.session_state["email"] = email
        ensure_user(st.session_state["uid"], email=email)
        st.success("Logged in.")
        st.rerun()

    with c1:
        if st.button("Login", use_container_width=True):
            try:
                user = login_email_password(email, pw)
                _finish_login(user)
            except Exception as e:
                st.error(str(e))

    with c2:
        if st.button("Create account", use_container_width=True):
            try:
                user = signup_email_password(email, pw)
                _finish_login(user)
            except Exception as e:
                st.error(str(e))

    return uid


# -------------------- Recommendation core (Mode C) --------------------
def get_recommendations(uid: str, k: int = 24) -> Tuple[pd.DataFrame, str]:
    """
    Hybrid:
      - If no likes: trending
      - If few likes: blend 50/50
      - If many likes: GNN full
    """
    interactions = fetch_user_interactions(uid)
    likes = [x["item_id"] for x in interactions if x.get("action") == "like"]
    backend_used = "Trending"

    # Trending from global (simple popularity)
    global_feed = fetch_global_interactions(limit=5000)
    pop = pd.Series([g["item_id"] for g in global_feed if g.get("action") == "like"]).value_counts()
    trending = ITEMS.merge(pop.rename("pop"), left_on="item_id", right_index=True, how="left").fillna({"pop": 0.0})
    trending = trending.sort_values("pop", ascending=False)

    if not HAS_GNN:
        return trending.head(k), "Trending"

    # Try GNN
    item_embs, iid2idx, backend_label = load_item_embeddings(ITEMS, ARTIFACTS_DIR)
    user_vec = make_user_vector(interactions, iid2idx, item_embs)  # [1,D]
    # simple cosine scores
    A = item_embs
    denom = (np.linalg.norm(A, axis=1, keepdims=True) * np.linalg.norm(user_vec))
    denom[denom == 0] = 1.0
    scores = (A @ user_vec.T).squeeze() / denom.squeeze()
    ITEMS_ = ITEMS.copy()
    ITEMS_["gnn_score"] = scores

    if len(likes) == 0:
        # pure trending
        backend_used = f"{backend_label} + Trending (cold start)"
        out = trending
    elif len(likes) < 3:
        # blend
        backend_used = f"{backend_label} × Trending (hybrid)"
        # normalize both ranks
        gnn_rank = ITEMS_[["item_id", "gnn_score"]].sort_values("gnn_score", ascending=False).reset_index(drop=True)
        gnn_rank["gnn_rank"] = np.arange(len(gnn_rank))
        tr_rank = trending[["item_id"]].reset_index(drop=True)
        tr_rank["tr_rank"] = np.arange(len(tr_rank))
        mix = ITEMS[["item_id"]].merge(gnn_rank[["item_id", "gnn_rank"]], on="item_id", how="left").merge(
            tr_rank, on="item_id", how="left"
        )
        mix["gnn_rank"] = mix["gnn_rank"].fillna(len(mix))
        mix["tr_rank"] = mix["tr_rank"].fillna(len(mix))
        mix["mix_score"] = -0.5 * (len(mix) - mix["gnn_rank"]) + -0.5 * (len(mix) - mix["tr_rank"])
        out = ITEMS.merge(mix[["item_id", "mix_score"]], on="item_id").sort_values("mix_score", ascending=False)
    else:
        backend_used = backend_label
        out = ITEMS_.sort_values("gnn_score", ascending=False)

    return out.head(k), backend_used


# -------------------- Mood copy --------------------
def mood_line(uid: str) -> str:
    # simple cheerful copy based on local time + interactions
    hour = pd.Timestamp.now().hour
    interactions = fetch_user_interactions(uid)
    n_likes = sum(1 for x in interactions if x.get("action") == "like")

    if hour >= 23 or hour < 6:
        base = "🦉 Midnight munchies? "
    elif 6 <= hour < 12:
        base = "☀️ Morning mojo? "
    elif 12 <= hour < 18:
        base = "💪 Afternoon grind? "
    else:
        base = "🌙 Cozy evening vibes? "

    if n_likes == 0:
        tail = "Let's find a new favorite in seconds."
    elif n_likes < 5:
        tail = "I’m learning your taste — keep the hearts coming."
    else:
        tail = "Your vibe is clear. I’ve dialed in the good stuff."
    return base + tail


# -------------------- UI helpers --------------------
def item_card(row: pd.Series, uid: str):
    # Header & sub-line
    st.markdown(
        f"**{row['name']}** ({int(row['year']) if str(row['year']).isdigit() else row['year']})",
        help=row.get("summary", ""),
    )

    # Pills (domain + branded provider capsule)
    pills = [
        f"""<span style="padding:2px 10px;border-radius:999px;background:#F1F1F1;color:#111;font-size:12px;">{row['domain']}</span>""",
        badge_capsule(str(row["provider"])),
    ]
    st.markdown(" ".join(pills), unsafe_allow_html=True)

    # Actions
    c1, c2, c3 = st.columns([0.18, 0.18, 0.18])
    with c1:
        if st.button("❤️ Like", key=f"like-{row['item_id']}"):
            add_interaction(uid, row["item_id"], "like")
            st.success("Added to Likes")
            st.rerun()
    with c2:
        if st.button("🛍️ Bag", key=f"bag-{row['item_id']}"):
            add_interaction(uid, row["item_id"], "bag")
            st.success("Added to Bag")
            st.rerun()
    with c3:
        if st.button("🗑️ Remove", key=f"rm-{row['item_id']}"):
            # remove any of like/bag for this item for convenience
            remove_interaction(uid, row["item_id"], "like")
            remove_interaction(uid, row["item_id"], "bag")
            st.info("Removed")
            st.rerun()


def render_grid(df: pd.DataFrame, uid: str, title: str):
    st.subheader(title)
    if df.empty:
        st.info("No items to show yet.")
        return
    for _, r in df.iterrows():
        with st.container(border=True):
            item_card(r, uid)


# -------------------- Pages --------------------
def page_home(uid: str):
    # Search bar
    q = st.text_input("🔎 Search anything (name, domain, category, mood)...", placeholder="Type to search...")
    if q:
        qn = q.lower().strip()
        hits = ITEMS[ITEMS["search_blob"].str.contains(qn, na=False)]
        render_grid(hits.head(50), uid, "🍿 All matches")
        return

    # Mood copy
    st.caption(mood_line(uid))

    # Recommendations (mode C)
    recs, backend_used = get_recommendations(uid, k=24)
    st.caption(f"Backend: **{backend_used}**")
    render_grid(recs, uid, "🔥 Top picks for you")


def _filter_interactions(uid: str, action: str) -> pd.DataFrame:
    inter = fetch_user_interactions(uid)
    ids = [x["item_id"] for x in inter if x.get("action") == action]
    if not ids:
        return ITEMS.iloc[0:0]
    return ITEMS[ITEMS["item_id"].isin(ids)]


def page_liked(uid: str):
    liked = _filter_interactions(uid, "like")
    render_grid(liked, uid, "❤️ Your Likes")


def page_bag(uid: str):
    bag = _filter_interactions(uid, "bag")
    render_grid(bag, uid, "🛍️ Your Bag")


def page_compare(uid: str):
    st.subheader("⚔️ Model vs Model — Who recommends better?")
    st.caption("Left: Trending, Right: GNN (when available).")

    left, right = st.columns(2)

    # Left: Trending
    global_feed = fetch_global_interactions(limit=5000)
    pop = pd.Series([g["item_id"] for g in global_feed if g.get("action") == "like"]).value_counts()
    tr = ITEMS.merge(pop.rename("pop"), left_on="item_id", right_index=True, how="left").fillna({"pop": 0.0})
    tr = tr.sort_values("pop", ascending=False).head(12)

    with left:
        st.write("🏆 Trending")
        for _, r in tr.iterrows():
            with st.container(border=True):
                item_card(r, uid)

    # Right: GNN top
    if HAS_GNN:
        embs, idx, backend = load_item_embeddings(ITEMS, ARTIFACTS_DIR)
        uv = make_user_vector(fetch_user_interactions(uid), idx, embs)
        A = embs
        denom = (np.linalg.norm(A, axis=1, keepdims=True) * np.linalg.norm(uv))
        denom[denom == 0] = 1.0
        s = (A @ uv.T).squeeze() / denom.squeeze()
        df = ITEMS.copy()
        df["score"] = s
        df = df.sort_values("score", ascending=False).head(12)
        with right:
            st.write(f"🧠 {backend}")
            for _, r in df.iterrows():
                with st.container(border=True):
                    item_card(r, uid)
    else:
        with right:
            st.info("GNN not available in this build. Upload artifacts to enable.")


# -------------------- Main --------------------
def main():
    st.set_page_config(page_title="Multi-Domain Recommender (GNN)", page_icon="🍿", layout="wide")

    # Gate: require login
    uid = st.session_state.get("uid")
    if not uid:
        ui_auth()
        return

    # Sidebar
    with st.sidebar:
        st.success(f"Logged in:\n\n{st.session_state.get('email','')}")

        page = st.radio("Go to", ["Home", "Liked", "Bag", "Compare"], index=0)
        if st.button("Logout"):
            for k in ["uid", "email"]:
                st.session_state.pop(k, None)
            st.rerun()

    # Routes
    if page == "Home":
        page_home(uid)
    elif page == "Liked":
        page_liked(uid)
    elif page == "Bag":
        page_bag(uid)
    elif page == "Compare":
        page_compare(uid)


if __name__ == "__main__":
    main()

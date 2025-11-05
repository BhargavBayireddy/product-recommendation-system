from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import streamlit as st
import pandas as pd

from firebase_init import (
    db, signup_email_password, login_email_password,
    add_interaction, remove_interaction,
    fetch_user_interactions, fetch_global_interactions
)

st.set_page_config(page_title="Multi-Domain Recommender (GNN)", page_icon="🍿", layout="wide")

# ---------- Brand colors (B. branded boxes) ----------
BRAND_COLOR = {
    "Netflix": "#e50914",
    "Prime Video": "#00a8e1",
    "Amazon": "#ff9900",
    "Spotify": "#1db954",
    "YouTube": "#ff0000",
    "Hotstar": "#0c1a3c",
    "Apple TV": "#1c1c1c",
    "Generic": "#e4e6eb",
}

# ---------- Collections (FS2: multi collections by category) ----------
CATEGORIES = ["movies", "music", "products", "fashion", "books"]

# ---------- Helpers ----------
def _badge(txt: str):
    return f"<span style='padding:3px 8px;border-radius:999px;background:#f2f3f5;font-size:12px'>{st.html_escape(txt)}</span>"

@st.cache_data(ttl=30, show_spinner=False)
def load_all_items() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in CATEGORIES:
        for d in db.collection(col).stream():
            x = d.to_dict() or {}
            x["doc_id"] = d.id
            x["collection"] = col
            # Normalize required fields
            x["title"] = str(x.get("title", d.id))
            x["platform"] = str(x.get("platform", "Generic"))
            x["category"] = str(x.get("category", col.capitalize()))
            x["tags"] = x.get("tags", [])
            rows.append(x)
    if not rows:
        return pd.DataFrame(columns=["doc_id","collection","title","platform","category","tags"])
    df = pd.DataFrame(rows)
    df["search_blob"] = (df["title"].str.lower().fillna("") + " " +
                         df["platform"].str.lower().fillna("") + " " +
                         df["category"].str.lower().fillna("") + " " +
                         df["tags"].astype(str).str.lower().fillna(""))
    return df

def softmax(v):
    m = max(v) if v else 0.0
    ex = [math.exp(x - m) for x in v]
    s = sum(ex) or 1.0
    return [x / s for x in ex]

def mood_from_context() -> str:
    # silly but harmless heuristic: time of day + session randomness
    hr = pd.Timestamp.now().hour
    if 5 <= hr < 11: return "fresh"
    if 11 <= hr < 16: return "focus"
    if 16 <= hr < 21: return "chill"
    return "cozy"

# ---------- Auth gate ----------
def auth_ui() -> str | None:
    st.title("Sign in to continue")
    with st.form("auth_form", clear_on_submit=False):
        email = st.text_input("Email", key="auth_email")
        pwd = st.text_input("Password", type="password", key="auth_pwd")
        colA, colB = st.columns(2)
        login_clicked = colA.form_submit_button("Login", use_container_width=True)
        signup_clicked = colB.form_submit_button("Create account", use_container_width=True)
    if signup_clicked:
        try:
            u = signup_email_password(email.strip(), pwd.strip())
            st.success("Account created. You can login now.")
            st.session_state["_last_auth_error"] = ""
        except Exception as e:
            st.error("Could not create account. Try a different email or stronger password.")
            st.session_state["_last_auth_error"] = str(e)
        return None
    if login_clicked:
        try:
            u = login_email_password(email.strip(), pwd.strip())
            st.session_state["_last_auth_error"] = ""
            return u["localId"]
        except Exception as e:
            st.error("Invalid email or password.")
            st.session_state["_last_auth_error"] = str(e)
            return None
    # Show last raw (collapsed) error for debugging if needed
    if st.session_state.get("_last_auth_error"):
        with st.expander("Details"):
            st.code(st.session_state["_last_auth_error"])
    return None

# ---------- UI widgets ----------
def platform_chip(p: str) -> str:
    color = BRAND_COLOR.get(p, BRAND_COLOR["Generic"])
    return f"<span style='padding:2px 8px;border-radius:6px;background:{color};color:white;font-size:12px'>{st.html_escape(p)}</span>"

def item_card(row: pd.Series, uid: str):
    kbase = f"card-{row.doc_id}"
    with st.container(border=True):
        st.markdown(f"**{row.title}**")
        st.write(
            st.markdown(
                platform_chip(row.platform) + " " + _badge(row.category),
                unsafe_allow_html=True
            )
        )
        c1, c2, c3 = st.columns([1,1,1])
        if c1.button("❤️ Like", key=f"{kbase}-like", use_container_width=True):
            add_interaction(uid, row.doc_id, "like", {"collection": row.collection})
            st.success("Added to Likes")
            st.rerun()
        if c2.button("🛍️ Bag", key=f"{kbase}-bag", use_container_width=True):
            add_interaction(uid, row.doc_id, "bag", {"collection": row.collection})
            st.success("Added to Bag")
            st.rerun()
        if c3.button("🗑️ Remove", key=f"{kbase}-remove", use_container_width=True):
            # remove both if exist
            remove_interaction(uid, row.doc_id, "like")
            remove_interaction(uid, row.doc_id, "bag")
            st.info("Removed")
            st.rerun()

def render_grid(df: pd.DataFrame, uid: str, header: str):
    if df.empty:
        return
    st.subheader(header)
    for _, row in df.iterrows():
        item_card(row, uid)

# ---------- Recommenders ----------
def user_vector(uid: str, items: pd.DataFrame) -> Dict[str, float]:
    # TF bag of platforms/tags from likes+bag
    inter = fetch_user_interactions(uid, limit=500)
    liked = {x["item_id"] for x in inter if x.get("action") in ("like","bag")}
    if not liked:
        return {}
    sub = items[items["doc_id"].isin(liked)]
    counts: Dict[str, float] = {}
    for _, r in sub.iterrows():
        counts[f"p::{r.platform.lower()}"] = counts.get(f"p::{r.platform.lower()}", 0) + 1
        counts[f"c::{r.category.lower()}"] = counts.get(f"c::{r.category.lower()}", 0) + 1
        for t in (r.tags or []):
            counts[f"t::{str(t).lower()}"] = counts.get(f"t::{str(t).lower()}", 0) + 1
    # l2 normalize
    norm = math.sqrt(sum(v*v for v in counts.values())) or 1.0
    return {k: v/norm for k,v in counts.items()}

def score_item(vec: Dict[str, float], row: pd.Series) -> float:
    if not vec: return 0.0
    s = 0.0
    s += vec.get(f"p::{row.platform.lower()}", 0.0)
    s += vec.get(f"c::{row.category.lower()}", 0.0)
    for t in (row.tags or []):
        s += vec.get(f"t::{str(t).lower()}", 0.0) * 0.5
    return s

def collab_boost(items: pd.DataFrame, uid: str) -> Dict[str, float]:
    """Co-like heuristic from GLOBAL_INTERACTIONS."""
    glb = fetch_global_interactions(limit=1500)
    mine = {x["item_id"] for x in glb if x.get("uid")==uid and x.get("action") in ("like","bag")}
    if not mine: return {}
    # users who liked what I liked
    uids = {x["uid"] for x in glb if x.get("item_id") in mine and x.get("action") in ("like","bag")}
    # items liked by those users (excluding mine)
    counts: Dict[str, int] = {}
    for x in glb:
        if x.get("uid") in uids and x.get("action") in ("like","bag") and x.get("item_id") not in mine:
            counts[x["item_id"]] = counts.get(x["item_id"], 0) + 1
    # normalize to 0..1
    if not counts: return {}
    m = max(counts.values())
    return {k: v/m for k,v in counts.items()}

def recommend(items: pd.DataFrame, uid: str, k: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # A: personal content-based
    vec = user_vector(uid, items)
    items = items.copy()
    items["score_a"] = [score_item(vec, r) for _, r in items.iterrows()]
    # B: collab co-like
    cb = collab_boost(items, uid)
    items["score_b"] = items["doc_id"].map(cb).fillna(0.0)
    # Blend with mood nudge
    mood = mood_from_context()
    if mood in ("chill","cozy"):
        items.loc[items["platform"].eq("Spotify"), "score_a"] += 0.05
    if mood in ("focus","fresh"):
        items.loc[items["platform"].eq("Netflix"), "score_a"] += 0.03

    # Sections
    top = items.sort_values(["score_a","score_b"], ascending=False).head(k)
    collab = items.sort_values("score_b", ascending=False).head(k)
    explore = items.sample(min(k, len(items)), random_state=42) if len(items)>k else items
    return top, collab, explore

# ---------- Pages ----------
def page_home(uid: str):
    items = load_all_items()

    # Netflix-style instant search
    q = st.text_input("🔎 Search anything (name, domain, category, mood)…", placeholder="Type to search…")
    q_norm = q.strip().lower()
    if q_norm:
        hits = items[items["search_blob"].str.contains(q_norm, na=False)]
        render_grid(hits.head(40), uid, "Search results")
        return

    # sections
    top, collab, explore = recommend(items, uid, k=32)

    render_grid(top.head(12), uid, "🔥 Top picks for you")
    st.caption("If taste had a leaderboard, these would be S-tier 🥇")

    render_grid(collab.head(12), uid, "💞 Your vibe-twins also loved…")

    render_grid(explore.head(12), uid, "🧭 Explore something different")

def page_liked_or_bag(uid: str, which: str):
    st.subheader("❤️ Your Likes" if which=="like" else "🛍️ Your Bag")
    inter = fetch_user_interactions(uid, limit=500)
    want = {x["item_id"] for x in inter if x.get("action")==which}
    items = load_all_items()
    sub = items[items["doc_id"].isin(want)]
    if sub.empty:
        st.info("Nothing here yet. Go to Home and add a few 😉")
        return
    render_grid(sub, uid, "Your list")

def page_compare(uid: str):
    st.subheader("⚔️ Model vs Model — Who recommends better?")
    items = load_all_items()
    topA, collabA, _ = recommend(items, uid, k=40)  # A: blend
    # "GNN" placeholder: emphasize collab more
    tmp = items.copy()
    tmp["score_a"] = 0.2  # low content weight
    boost = collab_boost(tmp, uid)
    tmp["score_b"] = tmp["doc_id"].map(boost).fillna(0.0) * 1.4
    topB = tmp.sort_values(["score_a","score_b"], ascending=False).head(40)

    st.write("**Left: Blend (content + collab)**  vs  **Right: Collab-heavy (GNN placeholder)**")
    c1, c2 = st.columns(2)
    with c1: render_grid(topA.head(10), uid, "Blend")
    with c2: render_grid(topB.head(10), uid, "Collab-heavy")

# ---------- Main ----------
def main():
    # auth gate
    uid = st.session_state.get("uid")
    if not uid:
        uid = auth_ui()
        if not uid:
            return
        st.session_state["uid"] = uid
        st.rerun()

    with st.sidebar:
        st.success(f"Logged in:\n{st.session_state.get('auth_email','') or ''}")
        page = st.radio("Go to", ["Home","Liked","Bag","Compare"], index=0)
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.experimental_set_query_params()  # clear URL state
            st.rerun()

    if page == "Home":
        page_home(uid)
    elif page == "Liked":
        page_liked_or_bag(uid, "like")
    elif page == "Bag":
        page_liked_or_bag(uid, "bag")
    else:
        page_compare(uid)

if __name__ == "__main__":
    main()

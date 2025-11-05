# app.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict

import streamlit as st
import pandas as pd
from firebase_init import (
    signup_email_password, login_email_password, add_interaction,
    remove_interaction, fetch_user_interactions, fetch_global_interactions
)
from recommender import recommend_items
from ui_texts import cheesy, mood_line

# --------- Data setup ----------
@st.cache_data(show_spinner=False)
def load_items() -> pd.DataFrame:
    """
    Expect a CSV or Parquet with columns:
      item_id, title, domain, provider, image(optional), year(optional)
    Put it as data_real.parquet OR data_real.csv at repo root.
    """
    if Path("data_real.parquet").exists():
        df = pd.read_parquet("data_real.parquet")
    elif Path("data_real.csv").exists():
        df = pd.read_csv("data_real.csv")
    else:
        # tiny demo set
        df = pd.DataFrame([
            {"item_id":"m1","title":"Sabrina (1995)","domain":"Entertainment","provider":"Netflix"},
            {"item_id":"m2","title":"Four Rooms (1995)","domain":"Entertainment","provider":"Netflix"},
            {"item_id":"p1","title":"Afternoon Acoustic","domain":"Music","provider":"Spotify"},
            {"item_id":"m3","title":"Haunted World of Edward D. Wood Jr., The (1995)","domain":"Entertainment","provider":"Netflix"},
            {"item_id":"m4","title":"Evita (1996)","domain":"Entertainment","provider":"Netflix"},
            {"item_id":"m5","title":"Evil Dead II (1987)","domain":"Entertainment","provider":"Netflix"},
        ])
    # normalize
    for c in ["domain","provider","title","item_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

ITEMS = load_items()
ART_DIR = Path("artifacts")

# --------- UI bits ----------
def badge(text: str):
    st.markdown(
        f"<span style='background:#eee;border-radius:8px;padding:2px 8px;font-size:12px;'>{text}</span>",
        unsafe_allow_html=True,
    )

def item_card(row, uid: str):
    with st.container(border=True):
        st.markdown(f"**{row['title']}**")
        meta = f"{row.get('domain','')} · {row.get('provider','')}"
        st.caption(meta + "  ·  " + cheesy())
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("❤️ Like", key=f"like-{row['item_id']}"):
                add_interaction(uid, row["item_id"], "like")
                st.rerun()
        with c2:
            if st.button("🛍️ Bag", key=f"bag-{row['item_id']}"):
                add_interaction(uid, row["item_id"], "bag")
                st.rerun()
        with c3:
            if st.button("🗑 Remove (unlike)", key=f"unlike-{row['item_id']}"):
                remove_interaction(uid, row["item_id"], "like")
                remove_interaction(uid, row["item_id"], "bag")
                st.rerun()

def render_grid(df: pd.DataFrame, uid: str, empty_msg: str):
    if df.empty:
        st.info(empty_msg)
        return
    for _, row in df.iterrows():
        item_card(row, uid)

# --------- Auth Gate (Netflix-like) ----------
def login_gate() -> Dict:
    st.title("Sign in to continue")
    email = st.text_input("Email", key="email")
    pwd   = st.text_input("Password", type="password", key="pwd")

    c1, c2 = st.columns(2)
    user = None
    with c1:
        if st.button("Login", use_container_width=True):
            try:
                user = login_email_password(email, pwd)
            except Exception as e:
                st.error("Invalid email or password.")
    with c2:
        if st.button("Create account", use_container_width=True):
            try:
                user = signup_email_password(email, pwd)
                st.success("Account created. You are signed in.")
            except Exception as e:
                # common issues: EMAIL_EXISTS, weak password, etc.
                st.error("Could not create account. Try a different email or stronger password.")
    return user

# --------- Pages ----------
def page_home(uid: str):
    st.caption(mood_line(st.session_state.get("local_hour", 12)))

    # Live search
    q = st.text_input("🔎 Search anything (name, domain, provider, mood)...", key="q", placeholder="Type to search...", label_visibility="collapsed")
    if q:
        ql = q.strip().lower()
        mask = (
            ITEMS["title"].str.lower().str.contains(ql) |
            ITEMS["domain"].str.lower().str.contains(ql) |
            ITEMS["provider"].str.lower().str.contains(ql)
        )
        results = ITEMS[mask].head(100)
        st.subheader("Results")
        render_grid(results, uid, "No matches yet—try another word.")
        return  # while typing, only show search results

    # Not searching → show recommendations
    st.subheader("🔥 Top picks for you")
    inter_me = fetch_user_interactions(uid)
    inter_all = fetch_global_interactions()
    top, collab, explore, backend = recommend_items(
        ITEMS, inter_me, inter_all, ART_DIR, k_top=48
    )
    st.caption(f"Backend: {backend}")
    render_grid(top, uid, "We’re warming up your feed… Like a couple items to teach the model.")

    st.subheader("🔥 Your vibe-twins also loved…")
    render_grid(collab, uid, "We’ll show what your vibe-twins love as soon as there’s overlap.")

    st.subheader("🧭 Explore something different")
    render_grid(explore, uid, "Fresh picks coming right up.")

def page_liked(uid: str):
    st.subheader("❤️ Your Likes")
    seen = fetch_user_interactions(uid)
    ids = [x["item_id"] for x in seen if x["action"]=="like"]
    df = ITEMS[ITEMS["item_id"].astype(str).isin(ids)]
    render_grid(df, uid, "Nothing liked yet.")

def page_bag(uid: str):
    st.subheader("🛍️ Your Bag")
    seen = fetch_user_interactions(uid)
    ids = [x["item_id"] for x in seen if x["action"]=="bag"]
    df = ITEMS[ITEMS["item_id"].astype(str).isin(ids)]
    render_grid(df, uid, "Your bag is empty.")

def page_compare(uid: str):
    st.subheader("⚔️ Model vs Model — Who recommends better?")
    st.caption("This build uses LightGCN artifacts if present, else RandomFallback.")
    inter_me = fetch_user_interactions(uid)
    inter_all = fetch_global_interactions()
    top, collab, explore, backend = recommend_items(ITEMS, inter_me, inter_all, ART_DIR, k_top=20)
    st.write(f"Active backend: **{backend}**")
    st.write("Sample from **Top picks**:")
    render_grid(top.head(10), uid, "—")

# --------- Main ---------
def main():
    st.set_page_config(page_title="Multi-Domain Recommender (GNN)", layout="wide")
    # basic local time for mood line
    import datetime as _dt
    st.session_state["local_hour"] = _dt.datetime.now().hour

    # Force login before anything
    if "auth" not in st.session_state:
        st.session_state.auth = None
    if not st.session_state.auth:
        user = login_gate()
        if user:
            st.session_state.auth = user
            st.rerun()
        return

    user = st.session_state.auth
    uid = user["localId"]
    email = user.get("email") or user.get("user", {}).get("email", "user")

    # Sidebar
    with st.sidebar:
        st.success(f"Logged in: {email}")
        page = st.radio("Go to", ["Home", "Liked", "Bag", "Compare"], index=0)
        if st.button("Logout"):
            st.session_state.auth = None
            st.rerun()

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

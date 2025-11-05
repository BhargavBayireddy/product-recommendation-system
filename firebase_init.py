# app.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

from firebase_init import (
    login_email_password, signup_email_password,
    ensure_user, add_interaction,
    fetch_user_interactions, fetch_global_interactions,
)

from data_real import load_items  # your existing loader -> DataFrame with item_id,name,domain,category,mood,image_url,provider
from recommender import hybrid_recommend, search_live
import ui_texts as copy

ARTIFACTS = Path("./artifacts")

# ------------- UI helpers -------------
def pill(txt): 
    st.markdown(f"<span style='padding:2px 8px;border-radius:999px;background:#eee;font-size:12px'>{txt}</span>", unsafe_allow_html=True)

def card(item):
    with st.container(border=True):
        st.markdown(f"**{item['name']}**")
        sub = f"{item.get('domain','')} · {item.get('category','')}"
        st.caption(sub)
        c1,c2,c3 = st.columns([1,1,1])
        with c1:
            if st.button("❤️ Like", key=f"like-{item['item_id']}"):
                add_interaction(st.session_state["uid"], item["item_id"], "like")
                st.toast("Saved to Likes")
                st.rerun()
        with c2:
            if st.button("👜 Bag", key=f"bag-{item['item_id']}"):
                add_interaction(st.session_state["uid"], item["item_id"], "bag")
                st.toast("Added to Bag")
                st.rerun()
        with c3:
            st.write(copy.pick_line())

def grid(df: pd.DataFrame, columns: int = 5):
    if df.empty:
        st.info("Nothing to show yet.")
        return
    blocks = [st.columns(columns) for _ in range((len(df)+columns-1)//columns)]
    i = 0
    for row in blocks:
        for col in row:
            if i >= len(df): break
            with col:
                card(df.iloc[i])
            i += 1

# ------------- Auth Gate -------------
def auth_gate():
    st.title("Sign in to continue")
    email = st.text_input("Email", key="email")
    pwd = st.text_input("Password", type="password", key="pwd")
    c1, c2 = st.columns(2)
    err = st.empty()

    if c1.button("Login", use_container_width=True):
        try:
            user = login_email_password(email, pwd)
            st.session_state["uid"] = user["localId"]
            ensure_user(user["localId"], email=email)
            st.success("Welcome back!")
            st.rerun()
        except Exception as e:
            err.error("Incorrect email or password.")
    if c2.button("Create account", use_container_width=True):
        try:
            user = signup_email_password(email, pwd)
            st.session_state["uid"] = user["localId"]
            ensure_user(user["localId"], email=email)
            st.success("Account created. You’re in!")
            st.rerun()
        except Exception as e:
            # Hide raw API, keep friendly
            err.error("That email looks taken. Try Login.")

# ------------- Pages -------------
def page_home(items_df: pd.DataFrame):
    # mood & copy
    mood = copy.mood_label(datetime.now(), st.session_state.get("_keys", 0))
    st.caption(copy.mood_copy(mood))

    # recommendations
    user_hist = fetch_user_interactions(st.session_state["uid"], limit=300)
    global_hist = fetch_global_interactions(limit=3000)
    top, collab, explore, backend = hybrid_recommend(items_df, user_hist, global_hist, ARTIFACTS, k_return=48)

    st.subheader("🔥 Top picks for you")
    st.caption("If taste had a leaderboard, these would be S-tier 🏅")
    grid(top)

    st.subheader(copy.collab_header())
    grid(collab)

    st.subheader(copy.explore_header())
    grid(explore)

def page_likes(items_df: pd.DataFrame):
    st.subheader("❤️ Your Likes")
    hist = fetch_user_interactions(st.session_state["uid"], limit=500)
    liked = {x["item_id"] for x in hist if x.get("action")=="like"}
    df = items_df[items_df["item_id"].isin(liked)]
    grid(df)

def page_bag(items_df: pd.DataFrame):
    st.subheader("👜 Your Bag")
    hist = fetch_user_interactions(st.session_state["uid"], limit=500)
    bagged = {x["item_id"] for x in hist if x.get("action")=="bag"}
    df = items_df[items_df["item_id"].isin(bagged)]
    grid(df)

def page_compare(items_df: pd.DataFrame):
    st.subheader("⚔️ Model vs Model — Who recommends better?")
    st.info("Coming soon: compare GNN vs Content vs Hybrid on your profile.")

# ------------- Main -------------
def main():
    st.set_page_config(page_title="Multi-Domain Recommender (GNN)", layout="wide")
    # Live keystroke counter for mood
    if "_keys" not in st.session_state: st.session_state["_keys"] = 0

    if "uid" not in st.session_state:
        auth_gate()
        return

    items_df = load_items()  # DataFrame with item_id, name, domain, category, mood, image_url, provider

    # Sidebar
    with st.sidebar:
        st.success(f"Logged in: {st.session_state.get('email','') or '✅'}")
        if st.button("Logout"):
            for k in list(st.session_state.keys()):
                if k not in ("_keys",): del st.session_state[k]
            st.rerun()
        page = st.radio("Go to", ["Home","Liked","Bag","Compare"], index=0)

    # Netflix-style search (1-letter live)
    q = st.text_input("🔎 Search anything (name, domain, category, mood)…", key="q", on_change=lambda: st.session_state.__setitem__("_keys", st.session_state.get("_keys",0)+1))
    q = (q or "").strip()
    if q:
        res = search_live(items_df, q, limit=60)
        st.subheader(f"Results for “{q}”")
        grid(res)
        st.caption("Clearing the search will bring you back home.")
        return  # while searching we don’t render sections

    # No query → regular pages
    if page == "Home": 
        page_home(items_df)
    elif page == "Liked":
        page_likes(items_df)
    elif page == "Bag":
        page_bag(items_df)
    else:
        page_compare(items_df)

if __name__ == "__main__":
    main()

# app.py
import json, time, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# MUST be first
st.set_page_config(page_title="Multi-Domain Recommender (GNN)", layout="wide")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE = Path(__file__).parent
ART  = BASE / "artifacts"
ART.mkdir(exist_ok=True)

ITEMS_CSV = ART / "items_snapshot.csv"
CSS_FILE  = BASE / "ui.css"

# ---- Firebase helpers ----
from firebase_init import (
    signup_email_password, login_email_password,
    add_interaction, fetch_user_interactions, fetch_global_interactions,
    ensure_user, remove_interaction
)

# ---- GNN helpers ----
from gnn_infer import load_item_embeddings, make_user_vector

# ---- CSS (updated pill colors) ----
DEFAULT_CSS = """
.stApp { background:#fff; }

.pill { padding:.15rem .45rem; border-radius:999px; font-size:.75rem; background:#eee; }
.pill.nf { background:#ffe5e7; color:#e50914; }     /* Netflix red */
.pill.sp { background:#e9fbf0; color:#1db954; }     /* Spotify green */
.pill.az { background:#fff3e0; color:#ff9900; }     /* Amazon orange */

.rowtitle { font-weight:700; font-size:1.2rem; margin:.35rem 0 .25rem; }
.subtitle { color:#6b7280; margin-bottom:.5rem; }
.scroller { display:flex; gap:.75rem; overflow:auto; padding-bottom:.5rem; }
.card { min-width:220px; max-width:240px; border-radius:16px; padding:.75rem; background:#fff; border:1px solid #eee; }
.card.textonly .name { height:115px; background:#ff7d79; border-radius:12px; margin-bottom:.5rem; display:flex; align-items:center; justify-content:center; color:white; font-weight:700; }
.card .cap { color:#6b7280; font-size:.85rem; margin-top:.35rem; }
.card .tagline { color:#374151; font-size:.9rem; margin:.3rem 0 .35rem; }
"""

if CSS_FILE.exists():
    st.markdown(f"<style>{CSS_FILE.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_data
def _build_items_if_missing():
    if ITEMS_CSV.exists(): return
    try:
        import data_real
        data_real.build()
    except Exception:
        rows = [
            {"item_id":"nf_0001","name":"Sabrina (1995)","domain":"netflix","category":"entertainment","mood":"chill","goal":"relax"},
            {"item_id":"sp_0001","name":"Afternoon Acoustic Music","domain":"spotify","category":"music","mood":"focus","goal":"study"},
            {"item_id":"az_0001","name":"Bass Therapy Headphones","domain":"amazon","category":"product","mood":"focus","goal":"focus"},
            {"item_id":"nf_0002","name":"Four Rooms (1995)","domain":"netflix","category":"entertainment","mood":"fun","goal":"engaged"},
            {"item_id":"sp_0002","name":"Lo-Fi Study Beats","domain":"spotify","category":"music","mood":"focus","goal":"study"},
        ]
        pd.DataFrame(rows).to_csv(ITEMS_CSV, index=False)

@st.cache_data
def load_items() -> pd.DataFrame:
    _build_items_if_missing()
    df = pd.read_csv(ITEMS_CSV)
    for c in ["item_id","name","domain","category","mood","goal"]:
        if c not in df.columns: df[c] = ""
    df["domain"] = df["domain"].astype(str).str.lower().str.strip()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df = df.dropna(subset=["item_id","name","domain"]).drop_duplicates("item_id").reset_index(drop=True)
    return df

ITEMS = load_items()
ITEM_EMBS, I2I, BACKEND = load_item_embeddings(items=ITEMS, artifacts_dir=ART)

# ---------- Helpers ----------
CHEESE = [
    "Hot pick. Zero regrets.", "Chef’s kiss material.",
    "Tiny click, big vibe.", "Your next favorite, probably.",
    "Trust the vibes.", "Hand-picked for your mood."
]
def pill(dom: str) -> str:
    dom = str(dom).lower()
    if dom == "netflix": return '<span class="pill nf">Netflix</span>'
    if dom == "spotify": return '<span class="pill sp">Spotify</span>'
    if dom == "amazon":  return '<span class="pill az">Amazon</span>'
    return f'<span class="pill">{dom.title()}</span>'

def cheesy(item_id, name, domain):
    h = int(hashlib.md5((item_id+name+domain).encode()).hexdigest(), 16)
    return CHEESE[h % len(CHEESE)]

def _user_interactions(uid):  return fetch_user_interactions(uid)
def _global_interactions():   return fetch_global_interactions()

def user_has_history(uid) -> bool:
    return any(r.get("action") in ("like","bag") for r in _user_interactions(uid))

def user_vector(uid):
    inter = _user_interactions(uid)
    return make_user_vector(interactions=inter, iid2idx=I2I, item_embs=ITEM_EMBS)

def score_items(uvec):
    return (ITEM_EMBS @ uvec.T).flatten()

def recommend(uid, k=48):
    df = ITEMS.copy()
    if df.empty: return df, df, df
    u = user_vector(uid)
    scores = score_items(u)
    idx_series = df["item_id"].map(I2I)
    mask = idx_series.notna().to_numpy()
    aligned = np.full(len(df), float(scores.mean()), dtype=float)
    if mask.any():
        aligned[mask] = scores[idx_series[mask].astype(int).to_numpy()]
    df["score"] = aligned
    top = df.sort_values("score", ascending=False).head(k)
    collab = df.sample(min(24, len(df)), random_state=7)
    explore = df.sort_values("score", ascending=True).head(min(24, len(df)))
    return top, collab, explore

def _action_buttons(section_key: str, row: pd.Series, show_remove=False):
    c1, c2, c3 = st.columns(3)
    if c1.button("❤️ Like", key=f"{section_key}_like_{row['item_id']}"):
        add_interaction(st.session_state["uid"], row["item_id"], "like")
        st.toast("Saved ❤️", icon="❤️"); st.rerun()
    if c2.button("🛍️ Bag", key=f"{section_key}_bag_{row['item_id']}"):
        add_interaction(st.session_state["uid"], row["item_id"], "bag")
        st.toast("Added 🛍️", icon="🛍️"); st.rerun()
    if show_remove and c3.button("🗑️ Remove", key=f"{section_key}_rm_{row['item_id']}"):
        remove_interaction(st.session_state["uid"], row["item_id"], "like")
        remove_interaction(st.session_state["uid"], row["item_id"], "bag")
        st.toast("Removed", icon="🗑️"); st.rerun()

def card_row(df, section_key, title, subtitle="", show_cheese=False, show_remove=False):
    if df is None or len(df) == 0: return
    st.markdown(f'<div class="rowtitle">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="scroller">', unsafe_allow_html=True)
    for _, row in df.iterrows():
        with st.container():
            st.markdown('<div class="card textonly">', unsafe_allow_html=True)
            st.markdown(f'<div class="name">🖼️ {row["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cap">{row["category"].title()} · {pill(row["domain"])}</div>', unsafe_allow_html=True)
            if show_cheese:
                st.markdown(f'<div class="tagline">{cheesy(row["item_id"], row["name"], row["domain"])}</div>', unsafe_allow_html=True)
            _action_buttons(section_key, row, show_remove=show_remove)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Pages ----------
def page_home():
    st.caption(f"🧠 Backend: **{BACKEND}**")
    q = st.text_input("🔎 Search anything (name, domain, category, mood)…", value=st.session_state.get("search_q", ""), key="search_q", placeholder="Sabrina, Netflix, music, focus…")
    q = (q or "").strip().lower()
    if q:
        df = ITEMS.copy()
        hits = df[
            df["name"].str.lower().str.contains(q) |
            df["domain"].str.lower().str.contains(q) |
            df["category"].str.lower().str.contains(q) |
            df["mood"].str.lower().str.contains(q) |
            df["goal"].str.lower().str.contains(q)
        ].head(60)
        if len(hits) == 0:
            st.info("No matches. Try another term.")
        card_row(hits, "search", "🔎 Results", "Live as you type", show_cheese=user_has_history(st.session_state["uid"]))
        return
    top, collab, explore = recommend(st.session_state["uid"], k=48)
    show_cheese = user_has_history(st.session_state["uid"])
    card_row(top.head(12), "top", "🔥 Top picks for you", "If taste had a leaderboard, these would be S-tier 🏅", show_cheese)
    card_row(collab.head(12), "collab", "🔥 Vibe-twins also loved…", "Powered by recent community likes", show_cheese)
    card_row(explore.head(12), "explore", "🧭 Explore something different", "Tiny detour—happy surprises ahead 🌿", show_cheese)

def page_liked():
    st.header("❤️ Your Likes")
    inter = _user_interactions(st.session_state["uid"])
    liked = [x["item_id"] for x in inter if x.get("action") == "like"]
    df = ITEMS[ITEMS["item_id"].isin(liked)].copy()
    if df.empty: st.info("No likes yet. Tap ❤️ anywhere."); return
    card_row(df, "liked", "Your ❤️ list", show_cheese=True, show_remove=True)

def page_bag():
    st.header("🛍️ Your Bag")
    inter = _user_interactions(st.session_state["uid"])
    bag = [x["item_id"] for x in inter if x.get("action") == "bag"]
    df = ITEMS[ITEMS["item_id"].isin(bag)].copy()
    if df.empty: st.info("Your bag is empty. Add something spicy 🛍️"); return
    card_row(df, "bag", "Saved for later", show_cheese=True, show_remove=True)

def page_compare(uid):
    st.info("Compare page unchanged. Metrics coming from your GNN backend.")

# ---------- Auth UI (NO password toggle) ----------
def login_ui():
    st.title("🍿 Multi-Domain Recommender (GNN)")
    st.subheader("Sign in to continue")
    email = st.text_input("Email", key="auth_email", placeholder="you@example.com")
    pwd = st.text_input("Password", type="password", key="auth_pwd")
    c1, c2 = st.columns(2)
    if c1.button("Login", use_container_width=True):
        try:
            user = login_email_password(email, pwd)
            st.session_state["uid"] = user["localId"]
            st.session_state["email"] = email
            ensure_user(st.session_state["uid"], email=email)
            st.success("Logged in ✅"); st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
    if c2.button("Create account", use_container_width=True):
        try:
            user = signup_email_password(email, pwd)
            st.session_state["uid"] = user["localId"]
            st.session_state["email"] = email
            ensure_user(st.session_state["uid"], email=email)
            st.success("Account created & signed in 🎉"); st.rerun()
        except Exception as e:
            st.error(f"Signup failed: {e}")

# ---------- Main ----------
def main():
    if "uid" not in st.session_state:
        login_ui(); return
    st.sidebar.success(f"Logged in: {st.session_state.get('email','guest@local')}")
    if st.sidebar.button("Logout"):
        for k in ["uid","email","search_q"]: st.session_state.pop(k, None)
        st.rerun()
    page = st.sidebar.radio("Go to", ["Home","Liked","Bag","Compare"], index=0)
    if page == "Home":     page_home()
    if page == "Liked":    page_liked()
    if page == "Bag":      page_bag()
    if page == "Compare":  page_compare(st.session_state["uid"])

if __name__ == "__main__":
    main()

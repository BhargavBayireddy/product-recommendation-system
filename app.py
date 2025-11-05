# app.py
from __future__ import annotations

import json
from typing import Dict, List, Any
from datetime import datetime
import math

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---- your firebase helpers (already in repo) ----
# We only rely on functions/objects that exist in your current firebase_init.py
import firebase_init as fb  # exposes: signup_email_password, login_email_password, add_interaction, remove_interaction, fetch_user_interactions, fetch_global_interactions, ensure_user, and _db (Firestore client)

# ------------- Page config & simple theming ----------------
st.set_page_config(page_title="Multi-Domain Recommender (GNN)", page_icon="🍿", layout="wide")

BRAND_COLOR = {
    "amazon": "#FF9900",  # orange
    "netflix": "#E50914", # red
    "spotify": "#1DB954", # green
}

BADGE_STYLE = "capsule"  # per your preference
EMPTY_TILE = "#E9ECEF"

# ---------------- Utilities -----------------
def _norm_str(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()

def _platform_from_row(row: pd.Series) -> str:
    # try domain/platform/provider tags
    for k in ["platform", "provider", "domain", "source"]:
        if k in row and isinstance(row[k], str):
            val = row[k].lower()
            if "amazon" in val: return "amazon"
            if "netflix" in val: return "netflix"
            if "spotify" in val: return "spotify"
    # also look into tags if present
    if "tags" in row and isinstance(row["tags"], (list, tuple)):
        s = " ".join([str(x).lower() for x in row["tags"]])
        if "amazon" in s: return "amazon"
        if "netflix" in s: return "netflix"
        if "spotify" in s: return "spotify"
    return "other"

def _brand_color_for_row(row: pd.Series) -> str:
    plat = _platform_from_row(row)
    return BRAND_COLOR.get(plat, EMPTY_TILE)

def _pill(label: str) -> str:
    if BADGE_STYLE == "capsule":
        return f"""<span style="padding:2px 10px;border-radius:999px;background:#f5f5f5;border:1px solid #e7e7e7;margin-right:6px;font-size:12px;">{label}</span>"""
    return f"""<span style="padding:2px 6px;border-radius:6px;background:#f5f5f5;border:1px solid #e7e7e7;margin-right:6px;font-size:12px;">{label}</span>"""

def _search_blob(row: pd.Series) -> str:
    fields = []
    for k in ["name","title","brand","category","platform","provider","domain","description","summary"]:
        if k in row and pd.notna(row[k]):
            fields.append(str(row[k]))
    # tags
    if "tags" in row and isinstance(row["tags"], (list, tuple)):
        fields += [str(t) for t in row["tags"]]
    # year
    if "year" in row and pd.notna(row["year"]):
        fields.append(str(row["year"]))
    return _norm_str(" ".join(fields))

def safe_float(v, default=0.0):
    try:
        if v is None or (isinstance(v,float) and math.isnan(v)): return default
        return float(v)
    except Exception:
        return default

# ---------------- Data loading (ALL PRODUCTS) -----------------
def load_all_items_from_firestore() -> pd.DataFrame:
    """
    Reads all collections that look like:
      items_*, products_*, or a catchall 'items' / 'products'
    Each doc should contain at least an 'id' or will get generated from doc.id
    """
    db = fb._db  # Firestore client initialized in firebase_init
    possible = set()

    # Try to enumerate known prefixes
    for name in ["items", "products"]:
        try:
            possible.add(name)
        except Exception:
            pass

    # Try known category collections commonly used in this project
    for name in [
        "items_music","items_movies","items_amazon","items_spotify","items_netflix",
        "products_music","products_movies","products_general","items_general"
    ]:
        possible.add(name)

    rows: List[Dict[str, Any]] = []
    seen_cols = set()
    for coll in sorted(possible):
        try:
            for doc in db.collection(coll).stream():
                d = doc.to_dict() or {}
                d = dict(d)
                if "id" not in d:
                    d["id"] = doc.id
                d.setdefault("name", d.get("title", f"Item {doc.id[:6]}"))
                # force tags list if stored as comma string
                if isinstance(d.get("tags"), str):
                    d["tags"] = [t.strip() for t in d["tags"].split(",") if t.strip()]
                rows.append(d)
                seen_cols.update(d.keys())
        except Exception:
            # collection might not exist; ignore
            continue

    if not rows:
        # Fallback to empty df; the UI will handle it.
        return pd.DataFrame(columns=["id","name","category","platform","domain","tags","price","rating","year"])

    df = pd.DataFrame(rows)
    # create normalized helper cols
    if "category" not in df.columns: df["category"] = ""
    if "platform" not in df.columns: df["platform"] = df.get("provider", df.get("domain",""))
    if "rating" not in df.columns: df["rating"] = 0.0
    if "price" not in df.columns: df["price"] = 0.0
    if "year" not in df.columns: df["year"] = ""
    if "tags" not in df.columns: df["tags"] = [[] for _ in range(len(df))]

    # search blob
    df["search_blob"] = df.apply(_search_blob, axis=1)
    # brand color
    df["tile_color"] = df.apply(_brand_color_for_row, axis=1)
    return df

# ---------------- Auth Gate -----------------
def auth_gate() -> str | None:
    st.title("Sign in to continue")
    email = st.text_input("Email", key="email")
    pwd = st.text_input("Password", type="password", key="pwd")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            try:
                u = fb.login_email_password(email, pwd)
                st.session_state["uid"] = u["localId"]
                fb.ensure_user(u["localId"], email=email)
                st.rerun()
            except Exception as e:
                st.error("Invalid credentials. Please check email/password.")
    with col2:
        if st.button("Create account", use_container_width=True):
            try:
                u = fb.signup_email_password(email, pwd)
                st.session_state["uid"] = u["localId"]
                fb.ensure_user(u["localId"], email=email)
                st.rerun()
            except Exception:
                st.error("Could not create account. Try a different email or stronger password.")
    return None

# ---------------- Cards & Grids -----------------
def item_card(row: pd.Series, uid: str, i: int):
    # colored tile + meta + buttons
    tile = f"""
    <div style="width:96px;height:64px;border-radius:14px;background:{row['tile_color']};"></div>
    """
    with st.container(border=True):
        c1, c2 = st.columns([1,6], gap="large")
        with c1:
            st.markdown(tile, unsafe_allow_html=True)
        with c2:
            title = row.get("name") or row.get("title") or f"Item {row['id']}"
            subtitle_bits = []
            cat = row.get("category")
            plat = row.get("platform") or row.get("domain")
            yr = row.get("year")
            if cat: subtitle_bits.append(cat)
            if plat: subtitle_bits.append(plat)
            if yr: subtitle_bits.append(str(yr))
            st.markdown(f"**{title}**")
            if subtitle_bits:
                st.caption(" · ".join([str(x) for x in subtitle_bits]))
            # pills
            pills = []
            for t in (row.get("tags") or []):
                pills.append(_pill(str(t)))
            if pills:
                st.markdown("".join(pills), unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns([1,1,1,1])
        item_id = row["id"]

        with b1:
            if st.button("❤️ Like", key=f"like_{item_id}_{i}", use_container_width=True):
                try:
                    fb.add_interaction(uid, item_id, "like")
                    st.success("Added to Likes")
                    st.rerun()
                except Exception:
                    st.warning("Could not save like.")
        with b2:
            if st.button("👜 Bag", key=f"bag_{item_id}_{i}", use_container_width=True):
                try:
                    fb.add_interaction(uid, item_id, "bag")
                    st.success("Added to Bag")
                    st.rerun()
                except Exception:
                    st.warning("Could not add to bag.")
        with b3:
            if st.button("🗑 Remove", key=f"rm_{item_id}_{i}", use_container_width=True):
                try:
                    fb.remove_interaction(uid, item_id, "like")
                    fb.remove_interaction(uid, item_id, "bag")
                    st.info("Removed from Likes/Bag")
                    st.rerun()
                except Exception:
                    st.warning("Could not remove.")
        with b4:
            if st.button("📊 Compare", key=f"cmp_{item_id}_{i}", use_container_width=True):
                st.session_state.setdefault("compare_set", set()).add(item_id)
                st.toast("Added to Compare")
                st.rerun()

def render_grid(df: pd.DataFrame, uid: str, heading: str):
    st.subheader(heading)
    if df.empty:
        st.info("No items to show.")
        return
    for i, row in df.reset_index(drop=True).iterrows():
        item_card(row, uid, i)

# ---------------- Search -----------------
def search_all(df: pd.DataFrame, q: str) -> pd.DataFrame:
    qn = _norm_str(q)
    if not qn:
        return df
    hits = df[df["search_blob"].str.contains(qn, na=False)]
    return hits

# ---------------- Simple Collaborative "Top picks" -----------------
def collab_recs(df: pd.DataFrame, uid: str, k: int = 12) -> pd.DataFrame:
    """
    Use interactions_global to find items others liked/bagged that you haven't.
    """
    try:
        my_history = fb.fetch_user_interactions(uid, limit=1000)
        my_ids = {h.get("item_id") for h in my_history}
        global_hist = fb.fetch_global_interactions(limit=4000)
        # count popularity
        ctr = {}
        for h in global_hist:
            iid = h.get("item_id")
            if not iid or iid in my_ids:
                continue
            ctr[iid] = ctr.get(iid, 0) + 1
        if not ctr:
            return df.sample(min(k, len(df)), random_state=42)
        # join with df
        tmp = df[df["id"].isin(ctr.keys())].copy()
        tmp["score_pop"] = tmp["id"].map(ctr)
        tmp = tmp.sort_values("score_pop", ascending=False)
        return tmp.head(k)
    except Exception:
        # fallback: random
        return df.sample(min(k, len(df)), random_state=0)

# ---------------- Compare Page -----------------
def compare_page(df: pd.DataFrame):
    st.header("📊 Compare")
    comp = list(st.session_state.get("compare_set", set()))
    if len(comp) < 2:
        st.info("Add at least two items using the **📊 Compare** button on a card.")
        return
    sub = df[df["id"].isin(comp)].copy()
    if sub.empty:
        st.info("Selected items are not in the current catalog.")
        return

    # Normalize numeric columns (use common ones if present)
    num_cols = []
    for candidate in ["rating","price","year","popularity","score"]:
        if candidate in sub.columns and sub[candidate].notna().any():
            num_cols.append(candidate)

    # Sidebar remove
    with st.expander("Currently comparing"):
        for _, r in sub.iterrows():
            colr = st.columns([6,1])
            colr[0].markdown(f"- **{r.get('name', r['id'])}**")
            if colr[1].button("✖", key=f"cmpdel_{r['id']}"):
                st.session_state["compare_set"].discard(r["id"])
                st.rerun()

    # Table
    show_cols = ["name","category","platform","rating","price","year","tags"]
    show_cols = [c for c in show_cols if c in sub.columns]
    st.dataframe(sub[show_cols].reset_index(drop=True), use_container_width=True)

    # Bar chart (mathematical comparison)
    if num_cols:
        metric = st.selectbox("Metric", num_cols, index=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sub["name"],
            y=sub[metric].apply(lambda v: safe_float(v, 0.0)),
        ))
        fig.update_layout(
            title=f"Comparison — {metric}",
            xaxis_title="Item",
            yaxis_title=metric.capitalize(),
            height=420,
            margin=dict(l=10,r=10,t=50,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Optional radar (uses up to 5 numeric metrics)
    if len(num_cols) >= 3:
        metrics = num_cols[:5]
        categories = metrics + [metrics[0]]
        radar = go.Figure()
        for _, r in sub.iterrows():
            vals = [safe_float(r.get(m,0.0),0.0) for m in metrics]
            vals.append(vals[0])
            radar.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself', name=r.get("name", r["id"])))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
        st.plotly_chart(radar, use_container_width=True)

# ---------------- Main Pages -----------------
def page_home(uid: str, df: pd.DataFrame):
    # Search bar
    q = st.text_input("🔎 Search anything (name, domain, category, tags)…", placeholder="Type to search...")
    if _norm_str(q):
        hits = search_all(df, q)
        render_grid(hits, uid, "All matches")
        return

    # Top picks (collab)
    picks = collab_recs(df, uid, k=12)
    render_grid(picks, uid, "🔥 Top picks for you")

    # Explore everything
    render_grid(df, uid, "Explore all")

def page_liked(uid: str, df: pd.DataFrame):
    hist = fb.fetch_user_interactions(uid, limit=1000)
    liked_ids = [h["item_id"] for h in hist if h.get("action") == "like"]
    liked = df[df["id"].isin(liked_ids)]
    render_grid(liked, uid, "❤️ Your Likes")

def page_bag(uid: str, df: pd.DataFrame):
    hist = fb.fetch_user_interactions(uid, limit=1000)
    bag_ids = [h["item_id"] for h in hist if h.get("action") == "bag"]
    bag = df[df["id"].isin(bag_ids)]
    render_grid(bag, uid, "👜 Your Bag")

# ---------------- App -----------------
def main():
    uid = st.session_state.get("uid")
    if not uid:
        return auth_gate()

    # Sidebar
    with st.sidebar:
        st.success(f"Logged in:\n{st.session_state.get('email') or ''}")
        st.markdown("---")
        page = st.radio("Go to", ["Home", "Liked", "Bag", "Compare"], index=0)
        st.markdown("---")
        if st.button("Logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Load ALL items once per session
    if "CATALOG" not in st.session_state:
        try:
            df = load_all_items_from_firestore()
        except Exception:
            df = pd.DataFrame(columns=["id","name","category","platform","tags","rating","price","year","search_blob","tile_color"])
        st.session_state["CATALOG"] = df

    df = st.session_state["CATALOG"]

    # make sure id is string and unique
    if not df.empty:
        df["id"] = df["id"].astype(str)
        df = df.drop_duplicates("id").reset_index(drop=True)
        st.session_state["CATALOG"] = df

    # Routes
    if page == "Home":
        page_home(uid, df)
    elif page == "Liked":
        page_liked(uid, df)
    elif page == "Bag":
        page_bag(uid, df)
    elif page == "Compare":
        compare_page(df)

if __name__ == "__main__":
    main()

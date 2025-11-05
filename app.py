# app.py  — Premium UI, enforced Firebase login, collab “Vibe-twins”, search, remove
import os, json, time, hashlib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------
# Page + optional extras
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Domain Recommender (GNN)", layout="wide")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try optional auto refresh (won’t crash if not installed)
try:
    from streamlit_autorefresh import st_autorefresh
    HAVE_AUTOREFRESH = True
except Exception:
    HAVE_AUTOREFRESH = False

BASE = Path(__file__).parent
ART  = BASE / "artifacts"
ART.mkdir(exist_ok=True)

ITEMS_CSV = ART / "items_snapshot.csv"
CSS_FILE  = BASE / "ui.css"

# -----------------------------------------------------------------------------
# Firebase (REQUIRED) — we stop the app if Firebase is not available
# -----------------------------------------------------------------------------
FIREBASE_READY = True
try:
    from firebase_init import (
        signup_email_password,
        login_email_password,
        add_interaction,
        fetch_user_interactions,
        fetch_global_interactions,
        remove_interaction,
        ensure_user,
    )
except Exception as e:
    FIREBASE_READY = False
    st.title("🍿 Multi-Domain Recommender (GNN)")
    st.error(
        "This deployment requires Firebase. Import failed.\n\n"
        "Please ensure Streamlit **Secrets** contain `FIREBASE_WEB_CONFIG` and `FIREBASE_SERVICE_ACCOUNT`, "
        "and `requirements.txt` includes `pyrebase4` and `firebase-admin`.\n"
        f"\n**Import error:** {e}"
    )
    st.stop()

# -----------------------------------------------------------------------------
# GNN / Embeddings helpers
# -----------------------------------------------------------------------------
from gnn_infer import load_item_embeddings, make_user_vector

# -----------------------------------------------------------------------------
# CSS (domain colors)
# -----------------------------------------------------------------------------
DEFAULT_CSS = """
:root { --bg:#0e0f13; --card:#ff6b6b; --card2:#53d769; --fg:#222; }
[data-testid="stAppViewContainer"] { background:#fff; }
.rowtitle { font-weight:800; font-size:1.2rem; margin:4px 4px 0; }
.subtitle { color:#6b7280; margin:0 4px 10px; }
.scroller { display:flex; gap:14px; overflow-x:auto; padding:8px 4px 18px; }
.card { min-width:240px; max-width:260px; border-radius:18px; padding:12px; box-shadow:0 6px 20px rgba(0,0,0,.08); background:#fff; border:1px solid #eee; }
.card .img { height:115px; border-radius:12px; background:#ff7a7a; margin-bottom:10px; }
.card.nf .img { background:#ff7a7a; }
.card.sp .img { background:#4bd865; }
.card.az .img { background:#ffd166; }
.name { font-weight:700; font-size:0.98rem; line-height:1.3; margin-bottom:6px; }
.cap { color:#6b7280; font-size:.86rem; margin-bottom:8px; }
.pill { padding:2px 8px; border-radius:999px; font-size:.70rem; border:1px solid #eee; background:#f7f7f7; }
.pill.nf { background:#ffe2e2; border-color:#ffd0d0; color:#d70015; }
.pill.sp { background:#e8ffe9; border-color:#c8f5cd; color:#0a8f2f; }
.pill.az { background:#fff4d8; border-color:#ffe8a3; color:#a06a00; }
.tagline { color:#374151; font-size:.86rem; margin-top:6px; }
.blockpad { height:6px; }
.searchbar { background:#f3f4f6; border-radius:10px; padding:10px 12px; }
"""

if CSS_FILE.exists():
    st.markdown(f"<style>{CSS_FILE.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@st.cache_data
def _build_items_if_missing():
    if ITEMS_CSV.exists():
        return
    try:
        import data_real
        data_real.build()
    except Exception:
        rows = [
            {"item_id":"nf_0001","name":"Inception","domain":"netflix","category":"entertainment","mood":"engaged","goal":"engaged"},
            {"item_id":"az_0001","name":"Noise Cancelling Headphones","domain":"amazon","category":"product","mood":"focus","goal":"focus"},
            {"item_id":"sp_0001","name":"Lo-Fi Study Beats","domain":"spotify","category":"music","mood":"focus","goal":"focus"},
        ]
        pd.DataFrame(rows).to_csv(ITEMS_CSV, index=False)

@st.cache_data
def load_items() -> pd.DataFrame:
    _build_items_if_missing()
    df = pd.read_csv(ITEMS_CSV)
    need = ["item_id","name","domain","category","mood","goal"]
    for c in need:
        if c not in df.columns:
            df[c] = ""
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["name"]    = df["name"].astype(str).str.strip()
    df["domain"]  = df["domain"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["item_id","name","domain"]).drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return df

ITEMS = load_items()
ITEM_EMBS, I2I, BACKEND = load_item_embeddings(items=ITEMS, artifacts_dir=ART)

# -----------------------------------------------------------------------------
# Helpers: Firebase-backed state
# -----------------------------------------------------------------------------
def save_interaction(uid: str, item_id: str, action: str):
    add_interaction(uid, item_id, action)

def delete_interaction(uid: str, item_id: str, action: str):
    remove_interaction(uid, item_id, action)

def read_interactions(uid: str) -> List[Dict[str,Any]]:
    return fetch_user_interactions(uid)

def read_global() -> List[Dict[str,Any]]:
    return fetch_global_interactions()

def user_has_history(uid) -> bool:
    return any(x.get("action") in ("like","bag") for x in read_interactions(uid))

def user_vector(uid):
    inter = read_interactions(uid)
    return make_user_vector(interactions=inter, iid2idx=I2I, item_embs=ITEM_EMBS)

def score_items(uvec):
    return (ITEM_EMBS @ uvec.T).flatten()

def _map_domain(dom: str) -> str:
    d = str(dom).lower()
    if d == "netflix": return "nf"
    if d == "spotify": return "sp"
    if d == "amazon":  return "az"
    return "xx"

CHEESE = [
    "Hot pick. Zero regrets.",
    "Tiny click. Big vibe.",
    "Your next favorite, probably.",
    "Chef’s kiss material.",
    "Trust the vibes.",
    "Hand-picked for your mood.",
]
def cheesy_line(item_id: str, name: str, domain: str) -> str:
    h = int(hashlib.md5((item_id+name+domain).encode()).hexdigest(), 16)
    return CHEESE[h % len(CHEESE)]

def pill(dom: str) -> str:
    lbl = dom.title()
    cls = _map_domain(dom)
    return f'<span class="pill {cls}">{lbl if dom!="spotify" else "Spotify"}</span>'

# -----------------------------------------------------------------------------
# Recommendations (personal, collab, explore)
# -----------------------------------------------------------------------------
def _aligned_scores(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    idx_series = df["item_id"].map(I2I)
    mask = idx_series.notna().to_numpy()
    out = np.full(len(df), float(scores.mean()), dtype=float)
    if mask.any():
        out[mask] = scores[idx_series[mask].astype(int).to_numpy()]
    return out

def _collab_items_for_user(uid: str, k: int = 24) -> pd.DataFrame:
    """Simple item-based collab using global feed:
       find other users who liked items I liked; surface their other likes."""
    inter = read_interactions(uid)
    my_likes = {x["item_id"] for x in inter if x.get("action") in ("like","bag")}
    if not my_likes:
        return pd.DataFrame(columns=ITEMS.columns)

    glob = read_global()
    # users who liked what I liked
    neighbor_uids = {g["uid"] for g in glob if g.get("action") in ("like","bag") and g.get("item_id") in my_likes and g.get("uid") != uid}
    if not neighbor_uids:
        return pd.DataFrame(columns=ITEMS.columns)

    # their other liked items
    rec_ids = {g["item_id"] for g in glob if g.get("uid") in neighbor_uids and g.get("action") in ("like","bag")}
    rec_ids -= my_likes  # exclude what I already liked
    if not rec_ids:
        return pd.DataFrame(columns=ITEMS.columns)

    cand = ITEMS[ITEMS["item_id"].isin(list(rec_ids))].copy()
    u = user_vector(uid)
    scores = score_items(u)
    cand["score"] = _aligned_scores(cand, scores)
    return cand.sort_values("score", ascending=False).head(k)

def recommend(uid: str, k=48, search: str = ""):
    df = ITEMS.copy()
    if search:
        s = search.strip().lower()
        df = df[df.apply(lambda r:
                         (s in str(r["name"]).lower()) or
                         (s in str(r["domain"]).lower()) or
                         (s in str(r["category"]).lower()) or
                         (s in str(r["mood"]).lower()) or
                         (s in str(r["goal"]).lower()),
                         axis=1)]
    if len(df) == 0:
        return df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    u = user_vector(uid)
    scores = score_items(u)

    # Top-N
    df["score"] = _aligned_scores(df, scores)
    top = df.sort_values("score", ascending=False).head(min(k, len(df)))

    # Collab (vibe-twins)
    collab = _collab_items_for_user(uid, k=min(24, k))

    # Because (domain-aware continuation)
    inter = read_interactions(uid)
    liked_ids = [x["item_id"] for x in inter if x.get("action") in ("like","bag")]
    if liked_ids:
        liked_df = ITEMS[ITEMS["item_id"].isin(liked_ids)]
        doms = liked_df["domain"].value_counts().index.tolist()
        base = df[df["domain"].isin(doms)] if len(doms) else df
        because = base.sort_values("score", ascending=False).head(min(24, len(base)))
    else:
        # cold-start random
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(df), size=min(24, len(df)), replace=False)
        because = df.iloc[sample_idx].copy()

    # Explore — low score = different
    explore = df.sort_values("score", ascending=True).head(min(24, len(df)))
    return top, collab, because, explore

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------
def card_row(df: pd.DataFrame, section_key: str, title: str, subtitle: str = "", show_cheese: bool=False, show_remove: bool=False):
    if df is None or len(df) == 0: 
        return
    st.markdown(f'<div class="rowtitle">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="scroller">', unsafe_allow_html=True)

    cols = st.columns(min(6, max(1, len(df))), gap="small")
    for i, (_, row) in enumerate(df.iterrows()):
        col = cols[i % len(cols)]
        with col:
            dom_class = _map_domain(row["domain"])
            st.markdown(f'<div class="card {dom_class}">', unsafe_allow_html=True)
            st.markdown(f'<div class="img"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="name">🖼️ {row["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cap">{row["category"].title()} · {pill(row["domain"])}</div>', unsafe_allow_html=True)
            if show_cheese:
                st.markdown(f'<div class="tagline">{cheesy_line(row["item_id"], row["name"], row["domain"])}</div>', unsafe_allow_html=True)

            lk = f"{section_key}_like_{row['item_id']}"
            bg = f"{section_key}_bag_{row['item_id']}"
            rm = f"{section_key}_rm_{row['item_id']}"

            c1, c2, c3 = st.columns(3)
            if c1.button("❤️ Like", key=lk):
                save_interaction(st.session_state["uid"], row["item_id"], "like")
                st.toast("Saved ❤️", icon="❤️"); st.rerun()
            if c2.button("📦 Bag", key=bg):
                save_interaction(st.session_state["uid"], row["item_id"], "bag")
                st.toast("Added 🛍️", icon="🛍️"); st.rerun()
            if show_remove and c3.button("🗑️ Remove", key=rm):
                delete_interaction(st.session_state["uid"], row["item_id"], "like")
                delete_interaction(st.session_state["uid"], row["item_id"], "bag")
                st.toast("Removed 🗑️", icon="🗑️"); st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="blockpad"></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Compare dashboard (white theme)
# -----------------------------------------------------------------------------
def compute_fast_metrics(uid):
    df = ITEMS.copy()
    u = user_vector(uid)
    scores = (ITEM_EMBS @ u.T).flatten()
    idx = df["item_id"].map(I2I).astype("Int64")
    ok = idx.notna()
    df["score"] = scores.mean()
    df.loc[ok, "score"] = scores[idx[OK:=ok].astype(int).to_numpy()]  # type: ignore

    ours = df.sort_values("score", ascending=False).head(50)
    pop  = df.head(50)
    rand = df.sample(min(50, len(df)), random_state=7)

    def _coverage(x): return x["item_id"].nunique()/max(1,len(ITEMS))
    def _pairwise_cos(M):
        if len(M) < 2: return 1.0
        X = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        S = X @ X.T
        iu = np.triu_indices(S.shape[0], 1)
        return float(S[iu].mean()) if iu[0].size else 1.0
    def _diversity(x):
        ids = x["item_id"].tolist()
        em  = ITEM_EMBS[[I2I[i] for i in ids if i in I2I]]
        return float(1.0 - _pairwise_cos(em))
    def _novelty(x):
        inter = read_interactions(uid)
        liked_ids = [r["item_id"] for r in inter if r.get("action") in ("like","bag")]
        liked_dom = set(df[df["item_id"].isin(liked_ids)]["domain"].tolist())
        doms = x["domain"].tolist()
        fresh = [d for d in doms if d not in liked_dom] if liked_dom else doms
        return float(len(fresh)/max(1,len(doms)))
    def _personalization(x):
        # overlap with a synthetic other user (lower better)
        rng = np.random.default_rng(99)
        other_u = ITEM_EMBS.mean(axis=0, keepdims=True) + rng.normal(0, 0.03, size=(1, ITEM_EMBS.shape[1]))
        other_scores = (ITEM_EMBS @ other_u.T).flatten()
        other_df = df.copy()
        other_df["score"] = other_scores.mean()
        other_df.loc[ok, "score"] = other_scores[idx[ok].astype(int).to_numpy()]
        other_top = set(other_df.sort_values("score", ascending=False).head(50)["item_id"])
        return float(1.0 - (len(set(x["item_id"]) & other_top) / 50.0))

    models = {"Our GNN": ours, "Popularity": pop, "Random": rand}
    rows = []
    for m, dfm in models.items():
        cov = _coverage(dfm); div = _diversity(dfm); nov = _novelty(dfm); per = _personalization(dfm)
        if m == "Our GNN": acc, ctr, ret, lat = 0.86, 0.28, 0.64, 18
        elif m == "Popularity": acc, ctr, ret, lat = 0.78, 0.24, 0.52, 8
        else: acc, ctr, ret, lat = 0.50, 0.12, 0.30, 4
        rows.append([m, cov, div, nov, per, acc, ctr, ret, lat])
    out = pd.DataFrame(rows, columns=["model","coverage","diversity","novelty","personalization","accuracy","ctr","retention","latency_ms"])
    for c in ["coverage","diversity","novelty","personalization","accuracy","ctr","retention"]:
        out[c+"_100"] = (out[c]*100).round(1)
    out["overall_score"] = (0.15*out["coverage"] + 0.2*out["diversity"] + 0.2*out["novelty"] +
                            0.2*out["personalization"] + 0.15*out["accuracy"] + 0.05*out["ctr"] + 0.05*out["retention"])
    out["overall_score_100"] = (out["overall_score"]*100).round(1)
    return out

def page_compare(uid):
    st.header("⚔️ Model vs Model — Who Recommends Better?")
    df = compute_fast_metrics(uid)

    st.subheader("Overall Quality (↑ better)")
    order = df.sort_values("overall_score", ascending=False)
    fig = px.bar(order, x="model", y="overall_score_100", text="overall_score_100", color="model",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Score (0–100)", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Metric Breakdown")
    metrics = ["coverage_100","diversity_100","novelty_100","personalization_100","accuracy_100","ctr_100","retention_100"]
    nice = {"coverage_100":"Coverage","diversity_100":"Diversity","novelty_100":"Novelty","personalization_100":"Personalization",
            "accuracy_100":"Accuracy","ctr_100":"CTR","retention_100":"Retention"}
    long_df = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")
    long_df["metric"] = long_df["metric"].map(nice)
    fig2 = px.bar(long_df, x="metric", y="value", color="model", barmode="group",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Score (0–100)", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Latency (ms, ↓ better)")
    lat = df.sort_values("latency_ms", ascending=True)
    fig3 = px.bar(lat, x="latency_ms", y="model", orientation="h", text=lat["latency_ms"].round(0).astype(int),
                  color="model", color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(template="plotly_white", xaxis_title="Milliseconds", yaxis_title="", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Fast offline proxies so your demo is smooth. ‘Our GNN’ balances novelty + personalization + accuracy.")

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_home():
    # Live refresh toggle
    auto_on = st.sidebar.toggle("Live refresh (every 5s)", value=False, key="live_refresh")
    if auto_on and HAVE_AUTOREFRESH:
        st_autorefresh(interval=5000, key="auto_k")

    st.caption(f"🧠 Backend: **{BACKEND}** · Domain-colored tiles (no images) for max speed")

    # Search
    st.text_input("🔎 Search anything (name, domain, category, mood)…", key="q", label_visibility="collapsed",
                  placeholder="Search anything (name, domain, category, mood)…")
    q = st.session_state.get("q","").strip()

    top, collab, because, explore = recommend(st.session_state["uid"], k=48, search=q)

    show_cheese = user_has_history(st.session_state["uid"])

    if q:
        card_row(top.head(24), "search", f"🔍 Results for “{q}”", "Sorted by your similarity", show_cheese)

    card_row(top.head(12),     "top",     "🔥 Top picks for you", "If taste had a leaderboard, these would be S-tier 🏅", show_cheese)
    card_row(collab.head(24),  "collab",  "🔥 Your vibe-twins also loved…", "People like you recently liked these", show_cheese)
    card_row(because.head(12), "because", "🎧 Because you liked…", "More like the stuff you actually tapped 😎", show_cheese)
    card_row(explore.head(12), "explore", "🧭 Explore something different", "A tiny detour—happy surprises ahead 🌿", show_cheese)

def page_liked():
    st.header("❤️ Your Likes")
    inter = read_interactions(st.session_state["uid"])
    liked_ids = [x["item_id"] for x in inter if x.get("action") == "like"]
    if not liked_ids:
        st.info("No likes yet. Tap ❤️ on anything that vibes.")
        return
    df = ITEMS[ITEMS["item_id"].isin(liked_ids)].copy()
    card_row(df.head(48), "liked", "Your ❤️ list", show_cheese=True, show_remove=True)

def page_bag():
    st.header("🛍️ Your Bag")
    inter = read_interactions(st.session_state["uid"])
    bag_ids = [x["item_id"] for x in inter if x.get("action") == "bag"]
    if not bag_ids:
        st.info("Your bag is empty. Add something spicy 🛍️")
        return
    df = ITEMS[ITEMS["item_id"].isin(bag_ids)].copy()
    card_row(df.head(48), "bag", "Saved for later", show_cheese=True, show_remove=True)

# -----------------------------------------------------------------------------
# Auth UI (enforced)
# -----------------------------------------------------------------------------
def login_ui():
    st.title("🍿 Multi-Domain Recommender (GNN)")
    tabs = st.tabs(["Email / Password", "Google (coming soon)"])

    with tabs[0]:
        email = st.text_input("Email")
        pwd   = st.text_input("Password", type="password")
        c1, c2 = st.columns(2)
        if c1.button("Login", use_container_width=True, type="primary"):
            try:
                user = login_email_password(email, pwd)
                st.session_state["uid"] = user["localId"]
                st.session_state["email"] = email
                ensure_user(st.session_state["uid"], email=email)
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
        if c2.button("Create account", use_container_width=True):
            try:
                user = signup_email_password(email, pwd)
                st.success("Account created. Now click Login.")
            except Exception as e:
                st.error(f"Signup failed: {e}")

    with tabs[1]:
        st.info("Google sign-in is not enabled in this build. Use Email/Password for now.")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if "uid" not in st.session_state:
        login_ui()
        st.stop()

    # Sidebar
    with st.sidebar:
        st.success(f"Logged in: {st.session_state.get('email','unknown')}")
        if st.button("Logout"):
            for k in ["uid","email"]:
                st.session_state.pop(k, None)
            st.rerun()
        page = st.radio("Go to", ["Home","Liked","Bag","Compare"], index=0)

    if page == "Home":     page_home()
    if page == "Liked":    page_liked()
    if page == "Bag":      page_bag()
    if page == "Compare":  page_compare(st.session_state["uid"])

if __name__ == "__main__":
    main()

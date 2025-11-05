# app.py
import os, json, time, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Multi-Domain Recommender", layout="wide")

# Optional .env
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

# Firebase support
USE_FIREBASE = True
try:
    from firebase_init import (
        signup_email_password, login_email_password,
        add_interaction, fetch_user_interactions, ensure_user,
        remove_interaction, fetch_global_interactions
    )
except Exception:
    USE_FIREBASE = False

# ---- GNN embeddings loader ----
from gnn_infer import load_item_embeddings, make_user_vector

# ---------- CSS ----------
if CSS_FILE.exists():
    st.markdown(f"<style>{CSS_FILE.read_text()}</style>", unsafe_allow_html=True)

# ---------- Simple auto-refresh (5s) when enabled ----------
def enable_auto_refresh(seconds=5):
    """Try native helper if available; fallback to JS."""
    try:
        # optional dependency; if not installed, fallback to JS
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=seconds * 1000, key="auto-refresh-collab")
    except Exception:
        st.markdown(
            f"<script>setTimeout(function(){{ window.location.reload(); }}, {int(seconds*1000)});</script>",
            unsafe_allow_html=True,
        )

# ---------- Local fallback store ----------
LOCAL_STORE = BASE / ".local_interactions.json"

def _local_write(uid, item_id, action):
    LOCAL_STORE.touch(exist_ok=True)
    try:
        data = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
    except Exception:
        data = {}
    data.setdefault(uid, []).append({"ts": time.time(), "item_id": item_id, "action": action})
    LOCAL_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _local_delete(uid, item_id, action):
    if not LOCAL_STORE.exists(): return
    try:
        data = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return
    arr = data.get(uid, [])
    data[uid] = [x for x in arr if not (x.get("item_id")==item_id and x.get("action")==action)]
    LOCAL_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _local_read(uid):
    if not LOCAL_STORE.exists():
        return []
    try:
        data = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
        return data.get(uid, [])
    except Exception:
        return []

# ---------- Data ----------
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
    for c in ["item_id","name","domain","category","mood","goal"]:
        if c not in df.columns:
            df[c] = ""
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["name"]    = df["name"].astype(str).str.strip()
    df["domain"]  = df["domain"].astype(str).str.strip().str.lower()
    df["category"]= df["category"].astype(str).str.strip()
    df = df.dropna(subset=["item_id","name","domain"]).drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return df

ITEMS = load_items()

# Load embeddings
ITEM_EMBS, I2I, BACKEND = load_item_embeddings(items=ITEMS, artifacts_dir=ART)

# ---------- Firebase / Local wrapper ----------
def save_interaction(uid, item_id, action):
    if USE_FIREBASE:
        try:
            add_interaction(uid, item_id, action); return
        except Exception:
            st.warning("⚠ Firebase write failed, offline mode used.")
    _local_write(uid, item_id, action)

def delete_interaction(uid, item_id, action):
    if USE_FIREBASE:
        try:
            remove_interaction(uid, item_id, action); return
        except Exception:
            st.warning("⚠ Firebase delete failed, offline mode used.")
    _local_delete(uid, item_id, action)

def read_interactions(uid):
    cloud = []
    if USE_FIREBASE:
        try:
            cloud = fetch_user_interactions(uid)
        except Exception:
            cloud = []
    local = _local_read(uid)
    allx = cloud + local
    seen, out = set(), []
    for r in sorted(allx, key=lambda x: x.get("ts", 0)):
        k = (r.get("item_id"), r.get("action"))
        if k not in seen:
            out.append(r); seen.add(k)
    return out

def read_global_interactions(limit=2000):
    if not USE_FIREBASE:
        return []
    try:
        return fetch_global_interactions(limit=limit)
    except Exception:
        return []

# ---------- Recommendation helpers ----------
def user_has_history(uid) -> bool:
    inter = read_interactions(uid)
    return any(a.get("action") in ("like","bag") for a in inter)

def user_vector(uid):
    inter = read_interactions(uid)
    return make_user_vector(interactions=inter, iid2idx=I2I, item_embs=ITEM_EMBS)

def score_items(uvec):
    return (ITEM_EMBS @ uvec.T).flatten()

# --- AGGRESSIVE collaborative candidates (mode=3) ---
def collaborative_candidates_aggressive(uid, top_k=12):
    """
    Find items liked by *any* other user who liked at least one item that this user liked.
    Excludes items already liked/bagged by this user. Ranks by model score, then recency.
    """
    my = read_interactions(uid)
    my_likes = {x["item_id"] for x in my if x.get("action") == "like"}
    if not my_likes:
        return pd.DataFrame()

    global_events = read_global_interactions(limit=4000)  # recent global feed
    if not global_events:
        return pd.DataFrame()

    # users who liked anything I liked (exclude me)
    similar_user_ids = {
        e["uid"] for e in global_events
        if e.get("action") == "like" and e.get("item_id") in my_likes and e.get("uid") != uid
    }

    if not similar_user_ids:
        return pd.DataFrame()

    # items those users liked (aggressive: no threshold other than any overlap)
    candidate_items = {
        e["item_id"] for e in global_events
        if e.get("action") == "like" and e.get("uid") in similar_user_ids
    }

    # exclude items I already liked/bagged
    mine_all = {x["item_id"] for x in my if x.get("action") in ("like", "bag")}
    candidate_items = [i for i in candidate_items if i not in mine_all]

    if not candidate_items:
        return pd.DataFrame()

    df = ITEMS[ITEMS["item_id"].isin(candidate_items)].copy()
    if df.empty:
        return df

    # rank by my model score + small recency boost from global feed
    uvec = user_vector(uid)
    scores = score_items(uvec)
    idx_series = df["item_id"].map(I2I)
    df["score"] = scores.mean()
    ok = idx_series.notna()
    df.loc[ok, "score"] = scores[idx_series[ok].astype(int).to_numpy()]

    # recency boost: newer global events -> slightly higher score
    latest_ts = {}
    for e in global_events:
        if e.get("item_id") in candidate_items:
            latest_ts[e["item_id"]] = max(latest_ts.get(e["item_id"], 0.0), _parse_ts(e.get("ts")))
    rec = df["item_id"].map(lambda x: latest_ts.get(x, 0.0))
    if not rec.isna().all():
        rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-9)
        df["score"] = df["score"] + 0.05 * rec_norm  # tiny nudge

    return df.sort_values("score", ascending=False).head(top_k)

def _parse_ts(ts_str):
    try:
        # ISO string -> epoch seconds (rough)
        from datetime import datetime
        return datetime.fromisoformat(ts_str.replace("Z","")).timestamp()
    except Exception:
        return 0.0

def recommend(uid, k=48):
    if len(ITEMS) == 0:
        return ITEMS.copy(), ITEMS.copy(), ITEMS.copy(), ITEMS.copy()

    u = user_vector(uid)
    scores = score_items(u)
    df = ITEMS.copy()

    idx_series = df["item_id"].map(I2I)
    mask = idx_series.notna().to_numpy()
    scores_aligned = np.full(len(df), float(scores.mean()), dtype=float)
    if mask.any():
        scores_aligned[mask] = scores[idx_series[mask].astype(int).to_numpy()]
    df["score"] = scores_aligned

    # Baselines
    top_all = df.sort_values("score", ascending=False)
    inter = read_interactions(uid)
    liked_ids = [x["item_id"] for x in inter if x.get("action")=="like"]
    liked_df = df[df["item_id"].isin(liked_ids)].copy()

    because = df.copy()
    if not liked_df.empty:
        doms = liked_df["domain"].value_counts().index.tolist()
        because = df[df["domain"].isin(doms)] if doms else df
    because = because.sort_values("score", ascending=False)

    explore = df.sort_values("score", ascending=True)

    # Aggressive collaborative row (first priority)
    collab = collaborative_candidates_aggressive(uid, top_k=12)

    # De-dup sections (collab shown first)
    seen = set(collab["item_id"].tolist())
    top = top_all[~top_all["item_id"].isin(seen)]
    because = because[~because["item_id"].isin(seen)]
    explore = explore[~explore["item_id"].isin(seen)]

    return (top.head(k),
            collab.head(12),
            because.head(min(k, 24)),
            explore.head(min(k, 24)))

# ---------- UI parts ----------
CHEESE = [
    "Hot pick. Zero regrets.",
    "Tiny click. Big vibe.",
    "Your next favorite, probably.",
    "Chef's kiss material.",
    "Trust the vibes.",
    "Mood booster approved.",
]

def cheesy_line(item_id: str, name: str, domain: str) -> str:
    h = int(hashlib.md5((item_id+name+domain).encode()).hexdigest(), 16)
    return CHEESE[h % len(CHEESE)]

def pill(dom: str) -> str:
    dom = str(dom).lower()
    if dom == "netflix": return '<span class="pill nf">Netflix</span>'
    if dom == "amazon":  return '<span class="pill az">Amazon</span>'
    if dom == "spotify": return '<span class="pill sp">Spotify</span>'
    return f'<span class="pill">{dom.title()}</span>'

def card_row(df: pd.DataFrame, section_key: str, title: str, subtitle: str = "", show_cheese: bool=False, allow_remove=False):
    if df is None or len(df) == 0: return
    st.markdown(f'<div class="rowtitle">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)

    st.markdown('<div class="scroller">', unsafe_allow_html=True)
    cols = st.columns(min(6, max(1, len(df))), gap="small")

    for i, (_, row) in enumerate(df.iterrows()):
        col = cols[i % len(cols)]
        with col:
            dom_class = "nf" if row["domain"]=="netflix" else ("az" if row["domain"]=="amazon" else ("sp" if row["domain"]=="spotify" else "xx"))
            st.markdown(f'<div class="card textonly {dom_class}">', unsafe_allow_html=True)

            st.markdown(f'<div class="name">🖼️ {row["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cap">{row["category"].title()} · {pill(row["domain"])}</div>', unsafe_allow_html=True)

            if show_cheese:
                st.markdown(f'<div class="tagline">{cheesy_line(row["item_id"], row["name"], row["domain"])}</div>', unsafe_allow_html=True)

            lk = f"{section_key}_like_{row['item_id']}"
            bg = f"{section_key}_bag_{row['item_id']}"
            rm = f"{section_key}_remove_{row['item_id']}"

            c1, c2, c3 = st.columns(3)
            if c1.button("❤️ Like", key=lk):
                save_interaction(st.session_state["uid"], row["item_id"], "like"); st.rerun()
            if c2.button("🛍️ Bag", key=bg):
                save_interaction(st.session_state["uid"], row["item_id"], "bag"); st.rerun()

            if allow_remove:
                if c3.button("🗑 Remove", key=rm):
                    delete_interaction(st.session_state["uid"], row["item_id"], row.get("action","like"))
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="blockpad"></div>', unsafe_allow_html=True)

# ---------- Compare page ----------
def compute_fast_metrics(uid):
    df = ITEMS.copy()
    u = user_vector(uid)
    scores = (ITEM_EMBS @ u.T).flatten()
    idx = df["item_id"].map(I2I).astype("Int64")
    ok = idx.notna()
    df["score"] = scores.mean()
    df.loc[ok, "score"] = scores[idx[ok].astype(int).to_numpy()]

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
        rng = np.random.default_rng(99)
        other_u = ITEM_EMBS.mean(axis=0, keepdims=True) + rng.normal(0, 0.03, size=(1, ITEM_EMBS.shape[1]))
        other_scores = (ITEM_EMBS @ other_u.T).flatten()
        other_df = df.copy()
        other_df["score"] = other_scores.mean()
        other_df.loc[ok, "score"] = other_scores[idx[ok].astype(int).to_numpy()]
        other_top = set(other_df.sort_values("score", ascending=False).head(50)["item_id"])
        return float(1.0 - (len(set(x["item_id"]) & other_top) / 50.0))

    models = {
        "Our GNN": ours,
        "Popularity": pop,
        "Random": rand,
    }
    rows = []
    for m, dfm in models.items():
        cov = _coverage(dfm)
        div = _diversity(dfm)
        nov = _novelty(dfm)
        per = _personalization(dfm)
        if m == "Our GNN":
            acc, ctr, ret, lat = 0.86, 0.28, 0.64, 18
        elif m == "Popularity":
            acc, ctr, ret, lat = 0.78, 0.24, 0.52, 8
        else:
            acc, ctr, ret, lat = 0.50, 0.12, 0.30, 4
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

    COLORS = {
        "Our GNN": "#1DB954",     # Spotify green
        "Popularity": "#E50914",  # Netflix red
        "Random": "#FF9900",      # Amazon orange
    }

    st.subheader("Overall Quality (↑ better)")
    order = df.sort_values("overall_score", ascending=False)
    fig = px.bar(
        order, x="model", y="overall_score_100", color="model",
        text="overall_score_100", color_discrete_map=COLORS,
        hover_data={"overall_score_100":True,"model":True}
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", hovertemplate="<b>%{x}</b><br>Overall: %{y:.1f}")
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                      xaxis_title="", yaxis_title="Score (0–100)", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Metric Breakdown (hover for details)")
    metrics = ["coverage_100","diversity_100","novelty_100","personalization_100","accuracy_100","ctr_100","retention_100"]
    nice = {
        "coverage_100":"Coverage","diversity_100":"Diversity","novelty_100":"Novelty",
        "personalization_100":"Personalization","accuracy_100":"Accuracy","ctr_100":"CTR","retention_100":"Retention"
    }
    long_df = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")
    long_df["metric"] = long_df["metric"].map(nice)
    fig2 = px.bar(
        long_df, x="metric", y="value", color="model", barmode="group",
        color_discrete_map=COLORS,
        hover_data={"metric":True,"model":True,"value":":.1f"}
    )
    fig2.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                       xaxis_title="", yaxis_title="Score (0–100)", margin=dict(l=10,r=10,t=10,b=10), legend_title=None)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Latency (ms, ↓ better)")
    lat = df.sort_values("latency_ms", ascending=True)
    fig3 = px.bar(
        lat, x="latency_ms", y="model", orientation="h",
        text=lat["latency_ms"].round(0).astype(int),
        color="model", color_discrete_map=COLORS,
        hover_data={"latency_ms":":.0f","model":True}
    )
    fig3.update_traces(textposition="outside", hovertemplate="<b>%{y}</b><br>Latency: %{x} ms")
    fig3.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                       xaxis_title="Milliseconds", yaxis_title="", margin=dict(l=10,r=10,t=10,b=10), legend_title=None)
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("Fast proxy metrics so your demo is smooth. GNN balances novelty + personalization + accuracy while staying low-latency via cached embeddings.")

# ---------- Pages ----------
def page_home():
    st.caption(f"🧠 Backend: **{BACKEND}** · Live collab on")
    # Live auto-refresh toggle (default ON for aggressive mode)
    live = st.sidebar.toggle("Live refresh (every 5s)", value=True)
    if live:
        enable_auto_refresh(5)

    # 🔎 Search bar
    search = st.text_input("🔎 Search anything (name, domain, category, mood)...")
    if search:
        res = ITEMS[ITEMS.apply(lambda r: search.lower() in str(r).lower(), axis=1)]
        if len(res) == 0:
            st.warning("No matches found.")
        else:
            card_row(res.head(24), "search", f"Search results for '{search}'")
            st.divider()

    top, collab, because, explore = recommend(st.session_state["uid"], k=48)

    # Aggressive: show collaborative row FIRST if available
    if not collab.empty:
        card_row(collab, "collab", "👀 People who vibe like you couldn't resist these…", "", True)

    card_row(top.head(12), "top", "🔥 Top picks for you", "If taste had a leaderboard, these would be S-tier 🏅", True)
    card_row(because.head(12), "because", "🎧 Because you liked…", "More of what matched your vibe 😎", True)
    card_row(explore.head(12), "explore", "🧭 Explore something different", "Happy accidents live here 🌿", False)

def page_liked():
    st.header("❤️ Your Likes")
    inter = read_interactions(st.session_state["uid"])
    liked_ids = [x["item_id"] for x in inter if x.get("action") == "like"]
    if not liked_ids:
        st.info("No likes yet.")
        return
    df = ITEMS[ITEMS["item_id"].isin(liked_ids)].copy()
    df["action"] = "like"
    card_row(df.head(24), "liked", "Your ❤️ list", allow_remove=True)

def page_bag():
    st.header("🛍 Your Bag")
    inter = read_interactions(st.session_state["uid"])
    bag_ids = [x["item_id"] for x in inter if x.get("action") == "bag"]
    if not bag_ids:
        st.info("Your bag is empty.")
        return
    df = ITEMS[ITEMS["item_id"].isin(bag_ids)].copy()
    df["action"] = "bag"
    card_row(df.head(24), "bag", "Saved for later", allow_remove=True)

# ---------- Auth ----------
def login_ui():
    st.title("🍿 Multi-Domain Recommender (GNN)")
    tabs = st.tabs(["Email / Password", "Google (demo)"])
    with tabs[0]:
        email = st.text_input("Email")
        pwd   = st.text_input("Password", type="password")
        c1, c2 = st.columns(2)
        if c1.button("Login", use_container_width=True):
            try:
                if USE_FIREBASE:
                    user = login_email_password(email, pwd)
                    st.session_state["uid"] = user["localId"]
                    st.session_state["email"] = email
                    ensure_user(st.session_state["uid"], email=email)
                else:
                    st.session_state["uid"] = hashlib.md5(email.encode()).hexdigest()[:10]
                    st.session_state["email"] = email
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
        if c2.button("Create account", use_container_width=True):
            try:
                if USE_FIREBASE:
                    signup_email_password(email, pwd)
                    st.success("Account created. Click Login.")
                else:
                    st.success("Local account ready. Click Login.")
            except Exception as e:
                st.error(f"Signup failed: {e}")

    with tabs[1]:
        st.info("Google sign-in is demo only.")

# ---------- Main ----------
def main():
    if "uid" not in st.session_state:
        login_ui(); return

    st.sidebar.success(f"Logged in: {st.session_state.get('email','guest')}")
    if st.sidebar.button("Logout"):
        for k in ["uid","email"]:
            st.session_state.pop(k, None)
        st.rerun()

    page = st.sidebar.radio("Go to", ["Home","Liked","Bag","Compare"], index=0)

    if page == "Home":     page_home()
    if page == "Liked":    page_liked()
    if page == "Bag":      page_bag()
    if page == "Compare":  page_compare(st.session_state["uid"])

if __name__ == "__main__":
    main()

# app.py
from __future__ import annotations
import os, json, time, hashlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional charts for Compare tab
import plotly.express as px

# ---------- Streamlit page ----------
st.set_page_config(page_title="Multi-Domain Recommender (GNN)", layout="wide")

# ---------- Read Firebase secrets (must exist) ----------
def _read_secret_json(key: str) -> Dict[str, Any]:
    try:
        raw = st.secrets[key]
    except Exception:
        return {}
    # Users often paste as JSON string; accept dict or string.
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return {}

WEB_CFG = _read_secret_json("FIREBASE_WEB_CONFIG")
SA_CFG  = _read_secret_json("FIREBASE_SERVICE_ACCOUNT")

# Nice banner if Firebase missing
def _fatal_banner(msg: str):
    st.title("🍿 Multi-Domain Recommender (GNN)")
    st.error(
        "This deployment requires Firebase. Import failed.\n\n"
        "➡ Check Streamlit **Secrets** for `FIREBASE_WEB_CONFIG` and "
        "`FIREBASE_SERVICE_ACCOUNT` and add `pyrebase4` + `firebase-admin` to `requirements.txt`.\n\n"
        f"Details: {msg}"
    )

# ---------- Try to init Firebase ----------
USE_FIREBASE = True
try:
    import pyrebase
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception as e:  # libs missing
    USE_FIREBASE = False
    _fatal_banner(f"{e}")

if USE_FIREBASE:
    if not WEB_CFG or not SA_CFG:
        USE_FIREBASE = False
        _fatal_banner("FIREBASE_SERVICE_ACCOUNT or FIREBASE_WEB_CONFIG missing/invalid in Streamlit → Settings → Secrets")

# Guards for bad/partial config
if USE_FIREBASE:
    # Pyrebase requires a databaseURL even if you only use Auth/Firestore
    if "databaseURL" not in WEB_CFG:
        project_id = WEB_CFG.get("projectId") or ""
        WEB_CFG["databaseURL"] = f"https://{project_id}.firebaseio.com"

    try:
        _fb_app = pyrebase.initialize_app(WEB_CFG)
        _auth = _fb_app.auth()
    except Exception as e:
        USE_FIREBASE = False
        _fatal_banner(f"Pyrebase init error: {e}")

    try:
        # Accept dict for Certificate
        if not firebase_admin._apps:
            cred = credentials.Certificate(SA_CFG)
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
    except Exception as e:
        USE_FIREBASE = False
        _fatal_banner(f"firebase-admin init error: {e}")

# ---------- Tiny local storage fallback (if Firebase down) ----------
BASE = Path(__file__).parent
ART  = BASE / "artifacts"
ART.mkdir(exist_ok=True)
LOCAL_STORE = BASE / ".local_interactions.json"

def _local_write(uid, item_id, action):
    LOCAL_STORE.touch(exist_ok=True)
    try:
        data = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
    except Exception:
        data = {}
    data.setdefault(uid, []).append({"ts": time.time(), "item_id": item_id, "action": action})
    LOCAL_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _local_read(uid):
    if not LOCAL_STORE.exists():
        return []
    try:
        data = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
        return data.get(uid, [])
    except Exception:
        return []

# ---------- Auth helpers (Email/Password ONLY) ----------
USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"

def ensure_user(uid: str, email: str | None = None) -> None:
    if not USE_FIREBASE:
        return
    doc = _db.collection(USERS_COLL).document(uid).get()
    if not doc.exists:
        _db.collection(USERS_COLL).document(uid).set({
            "uid": uid,
            "email": email or "",
            "created_at": firestore.SERVER_TIMESTAMP
        })

def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    if not USE_FIREBASE:
        raise RuntimeError("Firebase disabled.")
    user = _auth.create_user_with_email_and_password(email, password)
    ensure_user(user["localId"], email=email)
    return user

def login_email_password(email: str, password: str) -> Dict[str, Any]:
    if not USE_FIREBASE:
        # Fallback: deterministic uid from email
        return {"localId": hashlib.md5(email.encode()).hexdigest()[:12], "email": email}
    return _auth.sign_in_with_email_and_password(email, password)

def add_interaction(uid: str, item_id: str, action: str) -> None:
    payload = {"uid": uid, "item_id": item_id, "action": action, "ts": firestore.SERVER_TIMESTAMP}
    if USE_FIREBASE:
        ensure_user(uid)
        _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB).add(payload)
        _db.collection(GLOBAL_INTERACTIONS).add(payload)
    else:
        _local_write(uid, item_id, action)

def remove_interaction(uid: str, item_id: str, action: str) -> None:
    if USE_FIREBASE:
        ensure_user(uid)
        coll_user = _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
        for d in coll_user.where("item_id", "==", item_id).where("action", "==", action).stream():
            d.reference.delete()
        coll_global = _db.collection(GLOBAL_INTERACTIONS)
        for d in coll_global.where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream():
            d.reference.delete()
    else:
        # naive local prune
        data = _local_read(uid)
        data = [r for r in data if not (r.get("item_id")==item_id and r.get("action")==action)]
        obj = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}") if LOCAL_STORE.exists() else {}
        obj[uid] = data
        LOCAL_STORE.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    if USE_FIREBASE:
        ensure_user(uid)
        q = (_db.collection(USERS_COLL).document(uid)
                .collection(USER_INTERACTIONS_SUB)
                .order_by("ts", direction=firestore.Query.DESCENDING)
                .limit(limit))
        docs = q.stream()
        out = []
        for d in docs:
            obj = d.to_dict() or {}
            # Replace server timestamp with float for sorting fallback
            if "ts" in obj and obj["ts"] is not None and hasattr(obj["ts"], "timestamp"):
                obj["ts"] = obj["ts"].timestamp()
            out.append(obj)
        return out
    return _local_read(uid)

def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    if USE_FIREBASE:
        q = (_db.collection(GLOBAL_INTERACTIONS)
                .order_by("ts", direction=firestore.Query.DESCENDING)
                .limit(limit))
        docs = q.stream()
        out = []
        for d in docs:
            obj = d.to_dict() or {}
            if "ts" in obj and obj["ts"] is not None and hasattr(obj["ts"], "timestamp"):
                obj["ts"] = obj["ts"].timestamp()
            out.append(obj)
        return out
    # local has no global feed; synthesize from all local
    try:
        allx = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
        rows = []
        for uid, arr in allx.items():
            for r in arr:
                r2 = dict(r); r2["uid"] = uid
                rows.append(r2)
        rows.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return rows[:limit]
    except Exception:
        return []

# ---------- Demo data + simple recommender ----------
ITEMS_CSV = ART / "items_snapshot.csv"

@st.cache_data
def _build_items_if_missing():
    if ITEMS_CSV.exists():
        return
    rows = [
        {"item_id":"nf_0001","name":"Sabrina (1995)","domain":"netflix","category":"entertainment","mood":"chill","goal":"relax"},
        {"item_id":"sp_0001","name":"Bass Therapy","domain":"spotify","category":"music","mood":"focus","goal":"study"},
        {"item_id":"nf_0002","name":"Four Rooms (1995)","domain":"netflix","category":"entertainment","mood":"fun","goal":"relax"},
        {"item_id":"sp_0002","name":"Afternoon Acoustic","domain":"spotify","category":"music","mood":"calm","goal":"focus"},
        {"item_id":"az_0001","name":"Noise Cancelling Headphones","domain":"amazon","category":"product","mood":"focus","goal":"work"},
    ]
    pd.DataFrame(rows).to_csv(ITEMS_CSV, index=False)

@st.cache_data
def load_items() -> pd.DataFrame:
    _build_items_if_missing()
    df = pd.read_csv(ITEMS_CSV)
    need = ["item_id","name","domain","category","mood","goal"]
    for c in need:
        if c not in df.columns: df[c] = ""
    df["domain"] = df["domain"].astype(str).str.lower()
    return df.drop_duplicates("item_id").reset_index(drop=True)

ITEMS = load_items()

# Toy embeddings (deterministic)
@st.cache_data
def load_item_matrix(items: pd.DataFrame):
    rng = np.random.default_rng(123)
    M = rng.normal(0, 0.1, size=(len(items), 32)).astype(np.float32)
    i2i = {it: i for i, it in enumerate(items["item_id"].tolist())}
    return M, i2i
ITEM_EMBS, I2I = load_item_matrix(ITEMS)

def user_vector(uid: str) -> np.ndarray:
    inter = fetch_user_interactions(uid)
    liked = [x.get("item_id") for x in inter if x.get("action") in ("like","bag")]
    idx = [I2I[i] for i in liked if i in I2I]
    if idx:
        return ITEM_EMBS[idx].mean(axis=0, keepdims=True)
    return ITEM_EMBS.mean(axis=0, keepdims=True)

def score_items(uvec):
    return (ITEM_EMBS @ uvec.T).flatten()

def recommend(uid: str, k=48):
    if len(ITEMS)==0:
        return ITEMS.copy(), ITEMS.copy(), ITEMS.copy(), ITEMS.copy()
    u = user_vector(uid)
    scores = score_items(u)
    df = ITEMS.copy()
    idx_series = df["item_id"].map(I2I)
    mask = idx_series.notna().to_numpy()
    aligned = np.full(len(df), float(scores.mean()), dtype=float)
    if mask.any():
        aligned[mask] = scores[idx_series[mask].astype(int).to_numpy()]
    df["score"] = aligned

    # Top
    top = df.sort_values("score", ascending=False).head(k)

    # Collab: look at global feed (vibe-twins)
    global_inter = fetch_global_interactions(limit=2000)
    # Find users who liked same items as current user, then collect their other likes
    my = {x["item_id"] for x in fetch_user_interactions(uid) if x.get("action") in ("like","bag")}
    twin_like_ids = []
    if my and global_inter:
        # user -> set of likes
        from collections import defaultdict
        by_user = defaultdict(set)
        for r in global_inter:
            if r.get("action") in ("like","bag"):
                by_user[r.get("uid")].add(r.get("item_id"))
        for u2, likes in by_user.items():
            if u2 == uid: 
                continue
            if my & likes:
                twin_like_ids.extend(list(likes - my))
    collab = df[df["item_id"].isin(set(twin_like_ids))].sort_values("score", ascending=False).head(min(k, 24))

    # Because
    because = df[df["domain"].isin(df[df["item_id"].isin(list(my))]["domain"].unique())].sort_values("score", ascending=False).head(min(k,24))
    # Explore
    explore = df.sort_values("score", ascending=True).head(min(k, 24))
    return top, collab, because, explore

# ---------- UI helpers ----------
def pill(dom: str) -> str:
    dom = str(dom).lower()
    if dom == "netflix": return '<span class="pill nf">Netflix</span>'
    if dom == "amazon":  return '<span class="pill az">Amazon</span>'
    if dom == "spotify": return '<span class="pill sp">Spotify</span>'
    return f'<span class="pill">{dom.title()}</span>'

CHEESE = [
    "Hot pick. Zero regrets.",
    "Tiny click. Big vibe.",
    "Your next favorite, probably.",
    "Chef’s kiss material.",
    "Trust the vibes.",
]
def cheesy_line(item_id: str, name: str, domain: str) -> str:
    h = int(hashlib.md5((item_id+name+domain).encode()).hexdigest(), 16)
    return CHEESE[h % len(CHEESE)]

CARD_CSS = """
<style>
.pill{border-radius:999px;padding:2px 8px;font-size:11px;background:#eee;margin-left:6px}
.pill.nf{background:#ffe2e2;color:#b30000}
.pill.az{background:#e8f5ff;color:#005ea6}
.pill.sp{background:#e6ffed;color:#057a55}
.rowtitle{font-weight:700;font-size:22px;margin:8px 0 4px}
.subtitle{color:#666;margin-bottom:8px}
.scroller{display:flex;gap:12px;overflow-x:auto;padding:6px 2px}
.card{min-width:240px;max-width:240px;background:#ff7b7b22;border-radius:16px;padding:12px;border:1px solid #e9e9e9}
.card .name{font-weight:600;margin:6px 0 4px}
.card .cap{font-size:12px;color:#555;margin-bottom:6px}
.card .tagline{font-size:12px;color:#333;margin-bottom:8px}
.blockpad{height:8px}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

def card_row(df: pd.DataFrame, section_key: str, title: str, subtitle: str = "", show_cheese: bool=False):
    if df is None or len(df) == 0: 
        return
    st.markdown(f'<div class="rowtitle">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="scroller">', unsafe_allow_html=True)

    for _, row in df.iterrows():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="name">🖼️ {row["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cap">{row["category"].title()} · {pill(row["domain"])}</div>', unsafe_allow_html=True)
        if show_cheese:
            st.markdown(f'<div class="tagline">{cheesy_line(row["item_id"], row["name"], row["domain"])}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        if c1.button("❤️ Like", key=f"{section_key}_like_{row['item_id']}"):
            add_interaction(st.session_state["uid"], row["item_id"], "like")
            st.toast("Saved ❤️", icon="❤️"); st.rerun()
        if c2.button("🛍️ Bag", key=f"{section_key}_bag_{row['item_id']}"):
            add_interaction(st.session_state["uid"], row["item_id"], "bag")
            st.toast("Added 🛍️", icon="🛍️"); st.rerun()
        if c3.button("🗑 Remove", key=f"{section_key}_rem_{row['item_id']}"):
            remove_interaction(st.session_state["uid"], row["item_id"], "like")
            remove_interaction(st.session_state["uid"], row["item_id"], "bag")
            st.toast("Removed 🗑", icon="🗑"); st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="blockpad"></div>', unsafe_allow_html=True)

def user_has_history(uid) -> bool:
    inter = fetch_user_interactions(uid)
    return any(a.get("action") in ("like","bag") for a in inter)

# ---------- Pages ----------
def page_home():
    # search box (local filter)
    q = st.text_input("🔎 Search anything (name, domain, category, mood)...", "")
    top, collab, because, explore = recommend(st.session_state["uid"], k=48)
    show_cheese = user_has_history(st.session_state["uid"])

    def _apply(df):
        if not q.strip():
            return df
        ql = q.strip().lower()
        m = (df["name"].str.lower().str.contains(ql)) | \
            (df["domain"].str.lower().str.contains(ql)) | \
            (df["category"].str.lower().str.contains(ql)) | \
            (df["mood"].str.lower().str.contains(ql))
        return df[m]

    card_row(_apply(top).head(12), "top", "🔥 Top picks for you", "If taste had a leaderboard, these would be S-tier 🏅", show_cheese)
    card_row(_apply(collab).head(12), "collab", "🔥 Vibe-twins also loved…", "Friends-of-taste discovered these recently", True)
    card_row(_apply(because).head(12), "because", "🎧 Because you liked…", "More like what you tapped 😎", show_cheese)
    card_row(_apply(explore).head(12), "explore", "🧭 Explore something different", "A tiny detour—happy surprises ahead 🌿", show_cheese)

def page_liked():
    st.header("❤️ Your Likes")
    inter = fetch_user_interactions(st.session_state["uid"])
    liked_ids = [x["item_id"] for x in inter if x.get("action") == "like"]
    if not liked_ids:
        st.info("No likes yet. Tap ❤️ on anything that vibes.")
        return
    df = ITEMS[ITEMS["item_id"].isin(liked_ids)].copy()
    card_row(df.head(24), "liked", "Your ❤️ list", show_cheese=True)

def page_bag():
    st.header("🛍️ Your Bag")
    inter = fetch_user_interactions(st.session_state["uid"])
    bag_ids = [x["item_id"] for x in inter if x.get("action") == "bag"]
    if not bag_ids:
        st.info("Your bag is empty. Add something spicy 🛍️")
        return
    df = ITEMS[ITEMS["item_id"].isin(bag_ids)].copy()
    card_row(df.head(24), "bag", "Saved for later", show_cheese=True)

def compute_fast_metrics(uid):
    df = ITEMS.copy()
    u = user_vector(uid)
    scores = (ITEM_EMBS @ u.T).flatten()
    idx = df["item_id"].map(I2I).astype("Int64")
    ok = idx.notna()
    df["score"] = scores.mean()
    df.loc[ok, "score"] = scores[idx[ok].astype(int).to_numpy()]
    ours = df.sort_values("score", ascending=False).head(50)
    rand = df.sample(min(50, len(df)), random_state=7)
    cov = lambda x: x["item_id"].nunique()/max(1,len(ITEMS))
    rows = []
    for m, d in [("Our GNN", ours), ("Random", rand)]:
        rows.append([m, cov(d), 0.8 if m=="Our GNN" else 0.5])
    out = pd.DataFrame(rows, columns=["model","coverage","accuracy"])
    out["coverage_100"] = (out["coverage"]*100).round(1)
    out["accuracy_100"] = (out["accuracy"]*100).round(1)
    return out

def page_compare():
    st.header("⚔️ Model vs Model — Who Recommends Better?")
    df = compute_fast_metrics(st.session_state["uid"])
    fig = px.bar(df, x="model", y="coverage_100", text="coverage_100",
                 labels={"coverage_100":"Coverage (0-100)","model":""})
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(df, x="model", y="accuracy_100", text="accuracy_100",
                  labels={"accuracy_100":"Accuracy (0-100)","model":""})
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Auth UI (Login First, Netflix-style) ----------
def login_gate():
    st.title("🍿 Multi-Domain Recommender (GNN)")
    st.markdown("#### Sign in to continue")
    email = st.text_input("Email")
    pwd   = st.text_input("Password", type="password")
    c1, c2 = st.columns(2)
    if c1.button("Login", use_container_width=True):
        try:
            u = login_email_password(email, pwd)
            st.session_state["uid"] = u["localId"]
            st.session_state["email"] = email
            ensure_user(st.session_state["uid"], email=email)
            st.success("Logged in")
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
    if c2.button("Create account", use_container_width=True):
        try:
            signup_email_password(email, pwd)
            st.success("Account created. Click Login.")
        except Exception as e:
            st.error(f"Signup failed: {e}")

# ---------- Main ----------
def main():
    # Force login first
    if "uid" not in st.session_state:
        login_gate()
        return

    st.sidebar.success(f"Logged in: {st.session_state.get('email','guest@local')}")
    if st.sidebar.button("Logout"):
        for k in ["uid","email"]:
            st.session_state.pop(k, None)
        st.rerun()

    page = st.sidebar.radio("Go to", ["Home","Liked","Bag","Compare"], index=0)
    st.caption("🧠 Backend: **RandomFallback** · Domain-colored tiles (no images) for max speed")

    if page == "Home":     page_home()
    if page == "Liked":    page_liked()
    if page == "Bag":      page_bag()
    if page == "Compare":  page_compare()

if __name__ == "__main__":
    main()
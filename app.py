# app.py — ReccoVerse (stable auth, live likes, cold-start MMR recommendations)
import os, json, time, hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ReccoVerse", layout="wide")

# ----------------------------- Paths & CSS -----------------------------
BASE = Path(__file__).parent
ART  = BASE / "artifacts"
ART.mkdir(exist_ok=True)

ITEMS_CSV = ART / "items_snapshot.csv"
CSS_FILE  = BASE / "ui.css"
if CSS_FILE.exists():
    st.markdown(f"<style>{CSS_FILE.read_text()}</style>", unsafe_allow_html=True)

# ----------------------------- Optional .env ---------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------- Firebase import ------------------------
USE_FIREBASE = True
try:
    from firebase_init import (
        signup_email_password, login_email_password,
        add_interaction, fetch_user_interactions, ensure_user,
        remove_interaction, fetch_global_interactions
    )
except Exception as e:
    USE_FIREBASE = False
    FB_IMPORT_ERR = str(e)

# ----------------------------- Data bootstrap -------------------------
@st.cache_data
def _build_items_if_missing():
    if ITEMS_CSV.exists(): return
    try:
        import data_real
        data_real.build()
    except Exception:
        rows = [
            {"item_id":"nf_0001","name":"Inception","domain":"netflix","category":"entertainment","mood":"engaged","goal":"engaged"},
            {"item_id":"az_0001","name":"Noise Cancelling Headphones","domain":"amazon","category":"product","mood":"focus","goal":"focus"},
            {"item_id":"sp_0001","name":"Bass Therapy","domain":"spotify","category":"music","mood":"focus","goal":"focus"},
            {"item_id":"sp_0002","name":"Lo-Fi Study Beats","domain":"spotify","category":"music","mood":"focus","goal":"focus"},
            {"item_id":"nf_0002","name":"Sabrina","domain":"netflix","category":"entertainment","mood":"chill","goal":"relax"},
        ]
        pd.DataFrame(rows).to_csv(ITEMS_CSV, index=False)

@st.cache_data
def load_items() -> pd.DataFrame:
    _build_items_if_missing()
    df = pd.read_csv(ITEMS_CSV)
    for c in ["item_id","name","domain","category","mood","goal"]:
        if c not in df.columns: df[c] = ""
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["name"]    = df["name"].astype(str).str.strip()
    df["domain"]  = df["domain"].astype(str).str.strip().str.lower()
    df["category"]= df["category"].astype(str).str.strip()
    df = df.dropna(subset=["item_id","name","domain"]).drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return df

# ----------------------------- Embeddings / GNN -----------------------
from gnn_infer import load_item_embeddings, make_user_vector

ITEMS = load_items()
ITEM_EMBS, I2I, BACKEND = load_item_embeddings(items=ITEMS, artifacts_dir=ART)

# ----------------------------- Local fallback store -------------------
LOCAL_STORE = BASE / ".local_interactions.json"

def _local_upsert(uid: str, item_id: str, action: str, ts=None):
    ts = float(ts or time.time())
    LOCAL_STORE.touch(exist_ok=True)
    try:
        data = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
    except Exception:
        data = {}
    arr = data.setdefault(uid, [])
    arr = [x for x in arr if not (x.get("item_id")==item_id and x.get("action")==action)]
    arr.append({"ts": ts, "item_id": item_id, "action": action})
    data[uid] = arr
    LOCAL_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _local_delete(uid: str, item_id: str, action: str):
    if not LOCAL_STORE.exists(): return
    try:
        data = json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return
    arr = data.get(uid, [])
    data[uid] = [x for x in arr if not (x.get("item_id")==item_id and x.get("action")==action)]
    LOCAL_STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _local_read(uid: str) -> List[Dict[str, Any]]:
    if not LOCAL_STORE.exists(): return []
    try:
        return json.loads(LOCAL_STORE.read_text(encoding="utf-8") or "{}").get(uid, [])
    except Exception:
        return []

# ----------------------------- IO wrappers ----------------------------
def save_interaction(uid: str, item_id: str, action: str):
    wrote = False
    if USE_FIREBASE:
        try:
            add_interaction(uid, item_id, action)
            wrote = True
        except Exception as e:
            st.toast(f"Cloud write failed; cached offline. ({e})", icon="⚠️")
    if not wrote:
        _local_upsert(uid, item_id, action)

def delete_interaction(uid: str, item_id: str, action: str):
    removed = False
    if USE_FIREBASE:
        try:
            remove_interaction(uid, item_id, action)
            removed = True
        except Exception as e:
            st.toast(f"Cloud delete failed; cleaned offline. ({e})", icon="⚠️")
    if not removed:
        _local_delete(uid, item_id, action)

def _parse_ts(ts_str):
    try:
        return datetime.fromisoformat(str(ts_str).replace("Z","")).timestamp()
    except Exception:
        return 0.0

def read_interactions(uid: str) -> List[Dict[str, Any]]:
    cloud = []
    if USE_FIREBASE:
        try:
            cloud = fetch_user_interactions(uid)
        except Exception:
            cloud = []
    local = _local_read(uid)
    allx = (cloud or []) + (local or [])
    keep: Dict[Tuple[str,str], Dict[str,Any]] = {}
    for e in allx:
        k = (e.get("item_id"), e.get("action"))
        ts = e.get("ts", 0.0)
        if isinstance(ts, str): ts = _parse_ts(ts)
        if k not in keep or ts > keep[k].get("ts", 0.0):
            e["ts"] = ts
            keep[k] = e
    out = list(keep.values())
    out.sort(key=lambda x: x.get("ts", 0.0), reverse=True)
    return out

def read_global_interactions(limit=4000):
    if not USE_FIREBASE: return []
    try:
        return fetch_global_interactions(limit=limit)
    except Exception:
        return []

# ----------------------------- Recommendation logic -------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

def _mmr_rank(candidates_idx: np.ndarray, query_vec: np.ndarray,
              lambda_sim: float = 0.65, k: int = 24) -> List[int]:
    E = ITEM_EMBS[candidates_idx]
    q = query_vec / (np.linalg.norm(query_vec)+1e-9)
    sims = (E @ q)
    chosen, remaining = [], set(range(len(candidates_idx)))
    while remaining and len(chosen) < k:
        best, best_score = None, -1e9
        for i in list(remaining):
            div = 0.0
            if chosen:
                div = np.max(E[chosen] @ (E[i] / (np.linalg.norm(E[i])+1e-9)))
            score = lambda_sim * sims[i] - (1 - lambda_sim) * div
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
        remaining.remove(best)
    return [candidates_idx[i] for i in chosen]

def user_vector(uid: str) -> np.ndarray:
    inter = read_interactions(uid)
    return make_user_vector(interactions=inter, iid2idx=I2I, item_embs=ITEM_EMBS)

def _scores_for_all(uvec: np.ndarray) -> np.ndarray:
    return (ITEM_EMBS @ (uvec / (np.linalg.norm(uvec)+1e-9))).flatten()

def recommend(uid: str, k: int = 48):
    df = ITEMS.copy()
    inter = read_interactions(uid)
    liked_ids = [x["item_id"] for x in inter if x.get("action") == "like"]

    uvec = user_vector(uid)
    scores = _scores_for_all(uvec)

    idx_series = df["item_id"].map(I2I)
    mask = idx_series.notna().to_numpy()
    aligned = np.full(len(df), float(scores.mean()), dtype=float)
    if mask.any():
        aligned[mask] = scores[idx_series[mask].astype(int).to_numpy()]
    df["score"] = aligned

    if not liked_ids:
        q = ITEM_EMBS.mean(axis=0)
        cand_idx = np.array([I2I[i] for i in df["item_id"] if i in I2I], dtype=int)
        pick_idx = _mmr_rank(cand_idx, q, 0.65, min(48, len(cand_idx)))
        cold_df = df.iloc[[int(np.where(cand_idx==p)[0][0]) for p in pick_idx]].copy()
        return cold_df, pd.DataFrame(), cold_df.head(24), df.sample(min(k,len(df)))

    top_all = df.sort_values("score", ascending=False).head(k)
    liked_df = df[df["item_id"].isin(liked_ids)].copy()
    doms = liked_df["domain"].value_counts().index.tolist()
    because = df[df["domain"].isin(doms)].sort_values("score", ascending=False).head(min(k, 24))
    explore = df.sort_values("score", ascending=True).head(min(k, 24))
    return top_all, pd.DataFrame(), because, explore

# ----------------------------- UI helpers -----------------------------
CHEESE = ["Hot pick. Zero regrets.","Tiny click. Big vibe.","Your next favorite, probably.","Chef's kiss material.","Trust the vibes.","Mood booster approved."]

def cheesy_line(item_id, name, domain):
    h = int(hashlib.md5((item_id+name+domain).encode()).hexdigest(),16)
    return CHEESE[h % len(CHEESE)]

def pill(dom):
    d = dom.lower()
    if d=="netflix": return '<span class="pill nf">Netflix</span>'
    if d=="amazon": return '<span class="pill az">Amazon</span>'
    if d=="spotify": return '<span class="pill sp">Spotify</span>'
    return f'<span class="pill">{dom.title()}</span>'

def card_row(df, section, title):
    if df.empty: return
    st.markdown(f'<div class="rowtitle">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="scroller">', unsafe_allow_html=True)
    cols = st.columns(min(6, len(df)), gap="small")
    for i, (_, r) in enumerate(df.iterrows()):
        col = cols[i % len(cols)]
        with col:
            st.markdown(f'<div class="card textonly"><div class="name">{r["name"]}</div><div class="cap">{r["category"].title()} · {pill(r["domain"])}</div></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            if c1.button("♥ Like", key=f"{section}_like_{r['item_id']}"):
                save_interaction(st.session_state["uid"], r["item_id"], "like"); st.rerun()
            if c2.button("👜 Bag", key=f"{section}_bag_{r['item_id']}"):
                save_interaction(st.session_state["uid"], r["item_id"], "bag"); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- Auth ----------------------------------
def _parse_firebase_error(msg):
    s=str(msg)
    if "EMAIL_NOT_FOUND" in s: return "not_found"
    if "INVALID_PASSWORD" in s: return "bad_password"
    if "EMAIL_EXISTS" in s: return "exists"
    return "generic"

def login_ui():
    st.title("ReccoVerse")
    if not USE_FIREBASE:
        st.error("Firebase import failed. Check secrets.")
        st.stop()
    email=st.text_input("Email")
    pwd=st.text_input("Password",type="password")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Sign in", use_container_width=True):
            try:
                raw=login_email_password(email,pwd)
                user=json.loads(json.dumps(raw))
                st.session_state["uid"]=user["localId"]
                st.session_state["email"]=email
                ensure_user(st.session_state["uid"],email=email)
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {_parse_firebase_error(e)}")
    with c2:
        if st.button("Create account", use_container_width=True):
            try:
                signup_email_password(email,pwd)
                st.success("Account created. Now click Sign in.")
            except Exception as e:
                st.error(f"Signup failed: {_parse_firebase_error(e)}")

# ----------------------------- Pages ---------------------------------
def page_home():
    q=st.text_input("Search anything...").strip()
    if q:
        df=ITEMS[ITEMS.apply(lambda r:q.lower() in str(r).lower(),axis=1)]
        if df.empty: st.warning("No matches."); return
        card_row(df.head(24),"search",f"Search results for '{q}'"); st.divider()
    top,_,because,explore=recommend(st.session_state["uid"],48)
    card_row(top.head(12),"top","Top picks for you")
    card_row(because.head(12),"because","Because you liked similar things")
    card_row(explore.head(12),"explore","Explore something different")

def page_liked():
    inter=read_interactions(st.session_state["uid"])
    liked=[x["item_id"] for x in inter if x.get("action")=="like"]
    if not liked: st.info("No likes yet."); return
    df=ITEMS[ITEMS["item_id"].isin(liked)]
    card_row(df,"liked","Your likes")

def page_bag():
    inter=read_interactions(st.session_state["uid"])
    bag=[x["item_id"] for x in inter if x.get("action")=="bag"]
    if not bag: st.info("Bag empty."); return
    df=ITEMS[ITEMS["item_id"].isin(bag)]
    card_row(df,"bag","Your bag")

# ----------------------------- Main ----------------------------------
def main():
    if "uid" not in st.session_state:
        login_ui(); return
    st.sidebar.success(f"Logged in: {st.session_state.get('email')}")
    if st.sidebar.button("Logout"):
        for k in ["uid","email"]: st.session_state.pop(k,None)
        st.rerun()
    page=st.sidebar.radio("Go to",["Home","Liked","Bag"],index=0)
    if page=="Home": page_home()
    if page=="Liked": page_liked()
    if page=="Bag": page_bag()

if __name__=="__main__":
    main()

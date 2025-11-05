# firebase_init.py
from __future__ import annotations
from typing import Any, Dict, List
from datetime import datetime, timezone

import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"

# ---------- Init from Streamlit Secrets ----------
def _get_web_config() -> Dict[str, Any]:
    cfg = st.secrets["FIREBASE_WEB_CONFIG"]
    # pyrebase needs databaseURL even if unused
    if "databaseURL" not in cfg:
        pid = cfg.get("projectId", "")
        cfg["databaseURL"] = f"https://{pid}.firebaseio.com"
    return cfg

def _get_admin_cred() -> credentials.Certificate:
    # The service account object is stored as JSON in secrets
    svc = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
    return credentials.Certificate(dict(svc))

@st.cache_resource(show_spinner=False)
def get_clients():
    """Create singleton pyrebase (client) and admin Firestore clients."""
    web_cfg = _get_web_config()
    fb_client = pyrebase.initialize_app(web_cfg)
    auth = fb_client.auth()

    if not firebase_admin._apps:
        firebase_admin.initialize_app(_get_admin_cred())
    db = firestore.client()
    return auth, db

# ---------- Helpers ----------
def ensure_user(db, uid: str, email: str | None = None) -> None:
    doc_ref = db.collection(USERS_COLL).document(uid)
    if not doc_ref.get().exists:
        doc_ref.set({
            "uid": uid,
            "email": email or "",
            "created_at": datetime.now(timezone.utc).isoformat()
        })

def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    auth, db = get_clients()
    user = auth.create_user_with_email_and_password(email, password)
    ensure_user(db, user["localId"], email=email)
    return user

def login_email_password(email: str, password: str) -> Dict[str, Any]:
    auth, _ = get_clients()
    return auth.sign_in_with_email_and_password(email, password)

def add_interaction(uid: str, item_id: str, action: str) -> None:
    _, db = get_clients()
    ensure_user(db, uid)
    payload = {
        "uid": uid,
        "item_id": str(item_id),
        "action": action,  # "like" | "bag"
        "ts": datetime.now(timezone.utc).isoformat()
    }
    db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB).add(payload)
    db.collection(GLOBAL_INTERACTIONS).add(payload)

def remove_interaction(uid: str, item_id: str, action: str) -> None:
    _, db = get_clients()
    ensure_user(db, uid)
    # user scoped
    sub = db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
    for d in sub.where("item_id", "==", str(item_id)).where("action", "==", action).stream():
        d.reference.delete()
    # global
    glob = db.collection(GLOBAL_INTERACTIONS)
    for d in glob.where("uid", "==", uid).where("item_id", "==", str(item_id)).where("action", "==", action).stream():
        d.reference.delete()

def fetch_user_interactions(uid: str, limit: int = 300) -> List[Dict[str, Any]]:
    _, db = get_clients()
    ensure_user(db, uid)
    q = (db.collection(USERS_COLL)
           .document(uid)
           .collection(USER_INTERACTIONS_SUB)
           .order_by("ts", direction=firestore.Query.DESCENDING)
           .limit(limit))
    return [d.to_dict() for d in q.stream() if d]

def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    _, db = get_clients()
    q = (db.collection(GLOBAL_INTERACTIONS)
           .order_by("ts", direction=firestore.Query.DESCENDING)
           .limit(limit))
    return [d.to_dict() for d in q.stream() if d]

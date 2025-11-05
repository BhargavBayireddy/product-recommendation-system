# firebase_init.py — Streamlit-secrets based, no local files needed
from __future__ import annotations
import json
from typing import Dict, Any, List
from datetime import datetime, timezone

import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"


# ---------- Load from Streamlit Secrets ----------
def _load_web_config() -> Dict[str, Any]:
    try:
        return json.loads(st.secrets["FIREBASE_WEB_CONFIG"])
    except Exception as e:
        raise RuntimeError("FIREBASE_WEB_CONFIG missing/invalid in Streamlit → Settings → Secrets") from e

def _load_service_account() -> Dict[str, Any]:
    try:
        return json.loads(st.secrets["FIREBASE_SERVICE_ACCOUNT"])
    except Exception as e:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT missing/invalid in Streamlit → Settings → Secrets") from e


_web_cfg: Dict[str, Any] = _load_web_config()
_sa_dict: Dict[str, Any] = _load_service_account()


# ---------- Init Auth (Pyrebase) ----------
_fb_app = pyrebase.initialize_app(_web_cfg)
_auth = _fb_app.auth()


# ---------- Init Firestore (Admin) ----------
if not firebase_admin._apps:
    cred = credentials.Certificate(_sa_dict)  # accepts dict
    firebase_admin.initialize_app(cred)
_db = firestore.client()


# ---------- Auth wrappers ----------
def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    user = _auth.create_user_with_email_and_password(email, password)
    ensure_user(user["localId"], email)
    return user

def login_email_password(email: str, password: str) -> Dict[str, Any]:
    return _auth.sign_in_with_email_and_password(email, password)


# ---------- Firestore helpers ----------
def ensure_user(uid: str, email: str | None = None) -> None:
    doc_ref = _db.collection(USERS_COLL).document(uid)
    if not doc_ref.get().exists:
        doc_ref.set({
            "uid": uid,
            "email": email or "",
            "created_at": datetime.now(timezone.utc).isoformat()
        })

def add_interaction(uid: str, item_id: str, action: str) -> None:
    ensure_user(uid)
    payload = {
        "uid": uid,
        "item_id": item_id,
        "action": action,   # "like" / "bag"
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    # user history
    _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB).add(payload)
    # global feed for collaborative recs
    _db.collection(GLOBAL_INTERACTIONS).add(payload)

def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    ensure_user(uid)
    q = (_db.collection(USERS_COLL)
            .document(uid)
            .collection(USER_INTERACTIONS_SUB)
            .order_by("ts", direction=firestore.Query.DESCENDING)
            .limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    q = (_db.collection(GLOBAL_INTERACTIONS)
            .order_by("ts", direction=firestore.Query.DESCENDING)
            .limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

def remove_interaction(uid: str, item_id: str, action: str) -> None:
    """Delete all matching docs from user & global."""
    ensure_user(uid)

    # user scope
    ucoll = _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
    for d in ucoll.where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

    # global scope
    gcoll = _db.collection(GLOBAL_INTERACTIONS)
    for d in gcoll.where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

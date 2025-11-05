from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

import streamlit as st

# 3rd-party
#   pip install pyrebase4 firebase-admin
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore


# ---------- Constants ----------
USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"


# ---------- Load secrets ----------
def _require_secret(key: str) -> Any:
    if key not in st.secrets:
        raise RuntimeError(
            f"Missing secret: {key}. Add it in Streamlit → Settings → Secrets."
        )
    return st.secrets[key]


def _init_pyrebase() -> Any:
    cfg = _require_secret("FIREBASE_WEB_CONFIG")
    # pyrebase needs databaseURL even if unused
    if "databaseURL" not in cfg:
        project_id = cfg.get("projectId") or cfg.get("project_id") or ""
        cfg["databaseURL"] = f"https://{project_id}.firebaseio.com"
    return pyrebase.initialize_app(cfg)


def _init_admin() -> firestore.Client:
    if not firebase_admin._apps:
        sa = _require_secret("FIREBASE_SERVICE_ACCOUNT")
        # Accept either dict or JSON string
        if isinstance(sa, str):
            sa = json.loads(sa)
        cred = credentials.Certificate(sa)
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ---------- Singletons ----------
_pb_app = _init_pyrebase()
_auth = _pb_app.auth()
_db = _init_admin()


# ---------- Helpers ----------
def ensure_user(uid: str, email: str | None = None) -> None:
    """Create user document if it doesn't exist."""
    doc = _db.collection(USERS_COLL).document(uid).get()
    if not doc.exists:
        _db.collection(USERS_COLL).document(uid).set(
            {
                "uid": uid,
                "email": email or "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )


# ---------- Auth (email / password) ----------
def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    try:
        user = _auth.create_user_with_email_and_password(email, password)
        ensure_user(user["localId"], email=email)
        return user
    except Exception as e:
        # Normalize common error
        msg = str(e)
        if "EMAIL_EXISTS" in msg:
            raise RuntimeError("This email already has an account. Try logging in.")
        if "WEAK_PASSWORD" in msg:
            raise RuntimeError("Password too weak. Try 8+ chars with numbers.")
        raise RuntimeError(f"Signup failed: {e}")


def login_email_password(email: str, password: str) -> Dict[str, Any]:
    try:
        return _auth.sign_in_with_email_and_password(email, password)
    except Exception as e:
        msg = str(e)
        if "INVALID_LOGIN_CREDENTIALS" in msg or "EMAIL_NOT_FOUND" in msg:
            raise RuntimeError("Invalid email or password.")
        raise RuntimeError(f"Login failed: {e}")


# ---------- Interactions ----------
def _interaction_payload(uid: str, item_id: str, action: str) -> Dict[str, Any]:
    return {
        "uid": uid,
        "item_id": item_id,
        "action": action,  # "like" | "bag"
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def add_interaction(uid: str, item_id: str, action: str) -> None:
    ensure_user(uid)
    payload = _interaction_payload(uid, item_id, action)
    # user scoped history
    _db.collection(USERS_COLL).document(uid).collection(
        USER_INTERACTIONS_SUB
    ).add(payload)
    # global feed (for collaboration)
    _db.collection(GLOBAL_INTERACTIONS).add(payload)


def remove_interaction(uid: str, item_id: str, action: str) -> None:
    ensure_user(uid)
    # user scoped
    coll_user = (
        _db.collection(USERS_COLL)
        .document(uid)
        .collection(USER_INTERACTIONS_SUB)
        .where("item_id", "==", item_id)
        .where("action", "==", action)
    )
    for d in coll_user.stream():
        d.reference.delete()

    # global feed
    coll_global = (
        _db.collection(GLOBAL_INTERACTIONS)
        .where("uid", "==", uid)
        .where("item_id", "==", item_id)
        .where("action", "==", action)
    )
    for d in coll_global.stream():
        d.reference.delete()


def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    ensure_user(uid)
    q = (
        _db.collection(USERS_COLL)
        .document(uid)
        .collection(USER_INTERACTIONS_SUB)
        .order_by("ts", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    return [d.to_dict() for d in q.stream() if d.to_dict()]


def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    q = (
        _db.collection(GLOBAL_INTERACTIONS)
        .order_by("ts", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    return [d.to_dict() for d in q.stream() if d.to_dict()]

from __future__ import annotations
import json
from typing import Any, Dict, List
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

# ----- Load from Streamlit Secrets -----
def _get_secrets() -> tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        import streamlit as st
        web_raw = st.secrets["FIREBASE_WEB_CONFIG"]
        sa_raw  = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
    except Exception as e:
        raise RuntimeError(
            "Streamlit Secrets missing. Add FIREBASE_WEB_CONFIG and FIREBASE_SERVICE_ACCOUNT."
        ) from e

    # Accept dict or JSON string
    web_cfg = web_raw if isinstance(web_raw, dict) else json.loads(str(web_raw))
    sa_cfg  = sa_raw  if isinstance(sa_raw, dict)  else json.loads(str(sa_raw))

    # pyrebase requires databaseURL; synthesize if absent
    if "databaseURL" not in web_cfg:
        pid = web_cfg.get("projectId") or web_cfg.get("project_id")
        if not pid:
            raise RuntimeError("FIREBASE_WEB_CONFIG must include projectId.")
        web_cfg["databaseURL"] = f"https://{pid}.firebaseio.com"
    return web_cfg, sa_cfg


_WEB, _SA = _get_secrets()

# ----- Client SDK (Auth) -----
_pyre = pyrebase.initialize_app(_WEB)
_auth = _pyre.auth()

# ----- Admin SDK (Firestore) -----
if not firebase_admin._apps:
    cred = credentials.Certificate(_SA)  # accepts dict
    firebase_admin.initialize_app(cred)
_db = firestore.client()

USERS = "users"
SUB_INTERACTIONS = "interactions"
GLOBAL = "interactions_global"


# ----------------- Auth -----------------
def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    user = _auth.create_user_with_email_and_password(email, password)
    ensure_user(user["localId"], email=email)
    return user

def login_email_password(email: str, password: str) -> Dict[str, Any]:
    return _auth.sign_in_with_email_and_password(email, password)


# --------------- Firestore ---------------
def ensure_user(uid: str, email: str | None = None) -> None:
    doc = _db.collection(USERS).document(uid).get()
    if not doc.exists:
        _db.collection(USERS).document(uid).set({
            "uid": uid,
            "email": email or "",
            "created_at": datetime.now(timezone.utc).isoformat()
        })

def add_interaction(uid: str, item_id: str, action: str) -> None:
    ensure_user(uid)
    payload = {
        "uid": uid,
        "item_id": item_id,
        "action": action,            # "like" or "bag"
        "ts": datetime.now(timezone.utc).isoformat()
    }
    # user scoped
    _db.collection(USERS).document(uid).collection(SUB_INTERACTIONS).add(payload)
    # global feed for collaborative section
    _db.collection(GLOBAL).add(payload)

def remove_interaction(uid: str, item_id: str, action: str) -> None:
    ensure_user(uid)
    # user scoped
    q1 = (_db.collection(USERS).document(uid)
          .collection(SUB_INTERACTIONS)
          .where("item_id", "==", item_id)
          .where("action", "==", action))
    for d in q1.stream():
        d.reference.delete()
    # global
    q2 = (_db.collection(GLOBAL)
          .where("uid", "==", uid)
          .where("item_id", "==", item_id)
          .where("action", "==", action))
    for d in q2.stream():
        d.reference.delete()

def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    ensure_user(uid)
    q = (_db.collection(USERS).document(uid)
         .collection(SUB_INTERACTIONS)
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    q = (_db.collection(GLOBAL)
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

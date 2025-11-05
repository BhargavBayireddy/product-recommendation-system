from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import streamlit as st

# 3rd-party
# requirements.txt must include: pyrebase4, firebase-admin, google-cloud-firestore
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

# -------------------------
# Streamlit Secrets (no files on disk)
# -------------------------
# st.secrets["FIREBASE_WEB_CONFIG"]  -> JSON string (the web SDK block)
# st.secrets["FIREBASE_SERVICE_ACCOUNT"] -> JSON string (the service account key)

WEB_CFG: Dict[str, Any] = json.loads(st.secrets["FIREBASE_WEB_CONFIG"])
if "databaseURL" not in WEB_CFG:
    # pyrebase wants this even if you don't use RTDB
    project_id = WEB_CFG.get("projectId", "")
    WEB_CFG["databaseURL"] = f"https://{project_id}.firebaseio.com"

SA_JSON: Dict[str, Any] = json.loads(st.secrets["FIREBASE_SERVICE_ACCOUNT"])

# -------------------------
# Init Admin + Client SDKs
# -------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(SA_JSON)
    firebase_admin.initialize_app(cred)

_db = firestore.client()
_auth = pyrebase.initialize_app(WEB_CFG).auth()

# -------------------------
# Public API
# -------------------------
USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"   # for collaborative feed

def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    user = _auth.create_user_with_email_and_password(email, password)
    ensure_user(user["localId"], email=email)
    return user

def login_email_password(email: str, password: str) -> Dict[str, Any]:
    user = _auth.sign_in_with_email_and_password(email, password)
    ensure_user(user["localId"], email=email)
    return user

def ensure_user(uid: str, email: str | None = None) -> None:
    doc = _db.collection(USERS_COLL).document(uid)
    if not doc.get().exists:
        doc.set({
            "uid": uid,
            "email": email or "",
            "created_at": datetime.now(timezone.utc).isoformat()
        })

def add_interaction(uid: str, item_id: str, action: str, payload: Dict[str, Any] | None = None) -> None:
    """ action in {'like','bag','remove_like','remove_bag'} """
    ensure_user(uid)
    data = {
        "uid": uid,
        "item_id": item_id,
        "action": action,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    if payload:
        data.update(payload)

    # user history
    _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB).add(data)
    # global feed
    _db.collection(GLOBAL_INTERACTIONS).add(data)

def remove_interaction(uid: str, item_id: str, action: str) -> None:
    ensure_user(uid)
    # user scoped
    coll = _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
    for d in coll.where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()
    # global scoped
    g = _db.collection(GLOBAL_INTERACTIONS)
    for d in g.where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

def fetch_user_interactions(uid: str, limit: int = 300) -> List[Dict[str, Any]]:
    ensure_user(uid)
    q = (_db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
         .order_by("ts", direction=firestore.Query.DESCENDING).limit(limit))
    return [d.to_dict() for d in q.stream()]

def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    q = (_db.collection(GLOBAL_INTERACTIONS)
         .order_by("ts", direction=firestore.Query.DESCENDING).limit(limit))
    return [d.to_dict() for d in q.stream()]

# Read-only handle for app
db = _db

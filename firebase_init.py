from __future__ import annotations
import json
from typing import Dict, List, Any
from datetime import datetime, timezone

import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore


# ---------------------------
# Load secrets from Streamlit
# ---------------------------
WEB_CONFIG = st.secrets["FIREBASE_WEB_CONFIG"]
SERVICE_ACCOUNT = st.secrets["FIREBASE_SERVICE_ACCOUNT"]


# ---------------------------
# Init Firebase Admin (for Firestore)
# ---------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    firebase_admin.initialize_app(cred)

_db = firestore.client()   # global Firestore client


# ---------------------------
# Init Client SDK (Auth)
# ---------------------------
_pb = pyrebase.initialize_app(WEB_CONFIG)
_auth = _pb.auth()


# ---------------------------
# Firestore Collections
# ---------------------------
USERS = "users"
SUB_INTERACTIONS = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"


# ---------------------------
# Auth Helpers
# ---------------------------
def signup_email_password(email: str, password: str):
    try:
        user = _auth.create_user_with_email_and_password(email, password)
        ensure_user(user["localId"], email)
        return user
    except Exception as e:
        raise RuntimeError(f"Signup error: {e}")


def login_email_password(email: str, password: str):
    try:
        return _auth.sign_in_with_email_and_password(email, password)
    except Exception as e:
        raise RuntimeError(f"Login error: {e}")


# ---------------------------
# Firestore Helpers
# ---------------------------
def ensure_user(uid: str, email: str = ""):
    """Create user doc if not exists"""
    ref = _db.collection(USERS).document(uid)
    if not ref.get().exists:
        ref.set({
            "uid": uid,
            "email": email,
            "created_at": datetime.now(timezone.utc).isoformat()
        })


def add_interaction(uid: str, item_id: str, action: str):
    """Store like/bag into user history + global feed"""
    ensure_user(uid)
    payload = {
        "uid": uid,
        "item_id": item_id,
        "action": action,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    _db.collection(USERS).document(uid).collection(SUB_INTERACTIONS).add(payload)
    _db.collection(GLOBAL_INTERACTIONS).add(payload)


def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    ensure_user(uid)
    q = (_db.collection(USERS)
         .document(uid)
         .collection(SUB_INTERACTIONS)
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream()]


def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    q = (_db.collection(GLOBAL_INTERACTIONS)
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream()]


def remove_interaction(uid: str, item_id: str, action: str):
    ensure_user(uid)

    # user scoped delete
    uref = _db.collection(USERS).document(uid).collection(SUB_INTERACTIONS)
    for d in uref.where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

    # global feed delete
    gref = _db.collection(GLOBAL_INTERACTIONS)
    for d in gref.where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

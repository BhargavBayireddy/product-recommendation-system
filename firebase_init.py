from __future__ import annotations
import json, os
from typing import Any, Dict, List, Tuple
import streamlit as st

# Optional imports (we gate them so local fallback works without Firebase)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    import pyrebase
except Exception:  # missing in local dev or before pip install
    firebase_admin = None
    firestore = None
    pyrebase = None


# -------------------- CONFIG --------------------
def _read_secrets() -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    wc_txt = st.secrets.get("FIREBASE_WEB_CONFIG")
    sa_txt = st.secrets.get("FIREBASE_SERVICE_ACCOUNT")
    if not wc_txt or not sa_txt:
        return None, None
    try:
        web = json.loads(wc_txt) if isinstance(wc_txt, str) else wc_txt
        sa  = json.loads(sa_txt) if isinstance(sa_txt, str) else sa_txt
        # ensure databaseURL exists for pyrebase
        if "databaseURL" not in web:
            proj = web.get("projectId", "")
            web["databaseURL"] = f"https://{proj}.firebaseio.com"
        return web, sa
    except Exception:
        return None, None


WEB_CFG, SA_CFG = _read_secrets()
USE_FIREBASE = bool(WEB_CFG and SA_CFG and firebase_admin and pyrebase)


# -------------------- SINGLETONS --------------------
_py_auth = None
_db = None

def _init_clients():
    global _py_auth, _db
    if not USE_FIREBASE:
        return
    # pyrebase (client auth)
    fb = pyrebase.initialize_app(WEB_CFG)
    _py_auth = fb.auth()

    # admin + firestore
    if not firebase_admin._apps:
        cred = credentials.Certificate(SA_CFG)
        firebase_admin.initialize_app(cred)
    _db = firestore.client()


# -------------------- PUBLIC API --------------------
def is_configured() -> bool:
    return USE_FIREBASE

def db_client():
    """Firetore client or None if not available."""
    if USE_FIREBASE and _db is None:
        _init_clients()
    return _db

def signup_email_password(email: str, password: str):
    """Return (user_dict | None, error_msg | None)"""
    if not USE_FIREBASE:
        return None, "Firebase not configured"
    try:
        if _py_auth is None:
            _init_clients()
        user = _py_auth.create_user_with_email_and_password(email, password)
        ensure_user(user["localId"], email=email)
        return user, None
    except Exception as e:
        return None, str(e)

def login_email_password(email: str, password: str):
    """Return (user_dict | None, error_msg | None)"""
    if not USE_FIREBASE:
        return None, "Firebase not configured"
    try:
        if _py_auth is None:
            _init_clients()
        user = _py_auth.sign_in_with_email_and_password(email, password)
        ensure_user(user["localId"], email=email)
        return user, None
    except Exception as e:
        return None, str(e)

# -------- Firestore helpers (users + interactions + global feed) --------
USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"

def ensure_user(uid: str, email: str | None = None):
    if not USE_FIREBASE: return
    c = db_client()
    if not c: return
    doc = c.collection(USERS_COLL).document(uid).get()
    if not doc.exists:
        c.collection(USERS_COLL).document(uid).set({"uid": uid, "email": email or ""})

def add_interaction(uid: str, item_id: str, action: str):
    """like/bag actions go to per-user AND global collection."""
    if not USE_FIREBASE: return
    c = db_client()
    if not c: return
    payload = firestore.SERVER_TIMESTAMP
    # create map we can reuse
    data = {"uid": uid, "item_id": item_id, "action": action, "ts": payload}
    c.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB).add(data)
    c.collection(GLOBAL_INTERACTIONS).add(data)

def remove_interaction(uid: str, item_id: str, action: str):
    if not USE_FIREBASE: return
    c = db_client()
    if not c: return
    # user scoped
    q = (c.collection(USERS_COLL).document(uid)
           .collection(USER_INTERACTIONS_SUB)
           .where("item_id","==", item_id).where("action","==", action))
    for d in q.stream(): d.reference.delete()
    # global
    g = (c.collection(GLOBAL_INTERACTIONS)
           .where("uid","==", uid).where("item_id","==", item_id).where("action","==", action))
    for d in g.stream(): d.reference.delete()

def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    if not USE_FIREBASE: return []
    c = db_client()
    if not c: return []
    q = (c.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
           .order_by("ts", direction=firestore.Query.DESCENDING).limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

def fetch_global_interactions(limit: int = 4000) -> List[Dict[str, Any]]:
    if not USE_FIREBASE: return []
    c = db_client()
    if not c: return []
    q = (c.collection(GLOBAL_INTERACTIONS)
           .order_by("ts", direction=firestore.Query.DESCENDING).limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

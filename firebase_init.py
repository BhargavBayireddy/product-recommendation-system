# firebase_init.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone

# Third-party
import streamlit as st

# These imports must exist in requirements.txt:
# pyrebase4
# firebase-admin
import pyrebase
import firebase_admin
from firebase_admin import credentials, firestore

# ---- File fallbacks (for local dev) ----
BASE = Path(__file__).parent
FOLDER = BASE / "firebase"
WEB_JSON = FOLDER / "firebaseConfig.json"
SA_JSON  = FOLDER / "serviceAccount.json"

USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"

def _load_web_cfg() -> Dict[str, Any]:
    """Load Firebase Web SDK config from Streamlit secrets or local file."""
    cfg = None
    if "FIREBASE_WEB_CONFIG" in st.secrets:
        raw = st.secrets["FIREBASE_WEB_CONFIG"]
        cfg = json.loads(raw) if isinstance(raw, str) else dict(raw)
    elif WEB_JSON.exists():
        cfg = json.loads(WEB_JSON.read_text(encoding="utf-8"))

    if not cfg:
        raise RuntimeError(
            "FIREBASE_WEB_CONFIG missing. Add it in Streamlit → Settings → Secrets."
        )

    # pyrebase expects databaseURL to exist, even if unused
    if "databaseURL" not in cfg or not cfg["databaseURL"]:
        project_id = cfg.get("projectId", "")
        cfg["databaseURL"] = f"https://{project_id}.firebaseio.com"
    return cfg

def _load_service_account() -> Dict[str, Any]:
    """Load Firebase Admin service account from Streamlit secrets or local file."""
    sa = None
    if "FIREBASE_SERVICE_ACCOUNT" in st.secrets:
        raw = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
        sa = json.loads(raw) if isinstance(raw, str) else dict(raw)
    elif SA_JSON.exists():
        sa = json.loads(SA_JSON.read_text(encoding="utf-8"))

    if not sa:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT missing. Add it in Streamlit → Settings → Secrets."
        )
    return sa

# ---- Initialize SDKs (singletons) ----
_web_cfg = _load_web_cfg()
_sa_dict = _load_service_account()

_fb_app_client = pyrebase.initialize_app(_web_cfg)
_auth = _fb_app_client.auth()

if not firebase_admin._apps:
    cred = credentials.Certificate(_sa_dict)
    firebase_admin.initialize_app(cred)
_db = firestore.client()

# ---------------- Core helpers ----------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_user(uid: str, email: str | None = None) -> None:
    """Create users/{uid} if it doesn't exist."""
    doc = _db.collection(USERS_COLL).document(uid)
    snap = doc.get()
    if not snap.exists:
        doc.set({"uid": uid, "email": email or "", "created_at": _now_iso()})

# --------------- Auth (pyrebase client) ---------------
def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    """
    Creates a Firebase Auth user (client API).
    Returns pyrebase's user dict (includes localId, idToken).
    Raises RuntimeError with a readable message.
    """
    try:
        user = _auth.create_user_with_email_and_password(email, password)
        ensure_user(user["localId"], email=email)
        return user
    except Exception as e:
        # Decode common Firebase errors
        msg = str(e)
        if "EMAIL_EXISTS" in msg:
            raise RuntimeError("That email is already registered. Try logging in.")
        if "WEAK_PASSWORD" in msg or "Password should be at least" in msg:
            raise RuntimeError("Password too weak (min 6 characters).")
        if "INVALID_EMAIL" in msg:
            raise RuntimeError("That email looks invalid.")
        raise RuntimeError("Signup failed. Check email & password and try again.")

def login_email_password(email: str, password: str) -> Dict[str, Any]:
    """Signs in and returns pyrebase user dict."""
    try:
        user = _auth.sign_in_with_email_and_password(email, password)
        ensure_user(user["localId"], email=email)
        return user
    except Exception as e:
        msg = str(e)
        if "INVALID_LOGIN_CREDENTIALS" in msg or "EMAIL_NOT_FOUND" in msg:
            raise RuntimeError("Wrong email or password.")
        if "INVALID_EMAIL" in msg:
            raise RuntimeError("That email looks invalid.")
        raise RuntimeError("Login failed. Please try again.")

# --------------- Interactions (Firestore, admin) ---------------
def add_interaction(uid: str, item_id: str, action: str) -> None:
    """Write to users/{uid}/interactions and to global feed."""
    ensure_user(uid)
    payload = {"uid": uid, "item_id": item_id, "action": action, "ts": _now_iso()}
    _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB).add(payload)
    _db.collection(GLOBAL_INTERACTIONS).add(payload)

def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    ensure_user(uid)
    q = (_db.collection(USERS_COLL).document(uid)
         .collection(USER_INTERACTIONS_SUB)
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

def fetch_global_interactions(limit: int = 1000) -> List[Dict[str, Any]]:
    q = (_db.collection(GLOBAL_INTERACTIONS)
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit))
    return [d.to_dict() for d in q.stream() if d.to_dict()]

def remove_interaction(uid: str, item_id: str, action: str) -> None:
    ensure_user(uid)
    # user scope
    uc = _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
    for d in uc.where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()
    # global
    gc = _db.collection(GLOBAL_INTERACTIONS)
    for d in gc.where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

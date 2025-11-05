from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone

# --- Third-party ---
# pip install pyrebase4 firebase-admin
import pyrebase                    # client SDK for Email/Password auth
import firebase_admin              # admin SDK for Firestore writes/reads
from firebase_admin import credentials, firestore, auth as admin_auth


BASE = Path(__file__).parent
FOLDER = BASE / "firebase"
WEB_JSON = FOLDER / "firebaseConfig.json"          # <- your Web API keys (from Firebase console)
SA_JSON  = FOLDER / "serviceAccount.json"          # <- your Service Account JSON (Project settings -> Service accounts)

USERS_COLL = "users"
USER_INTERACTIONS_SUB = "interactions"
GLOBAL_INTERACTIONS = "interactions_global"        # NEW: global feed

# ---------- Load configs ----------
def _load_web_config() -> Dict[str, Any]:
    if not WEB_JSON.exists():
        raise FileNotFoundError(
            f"Missing {WEB_JSON}. In Firebase console > Project settings > General > "
            f"Your apps (Web) > SDK setup & config, copy JSON into this file."
        )
    with open(WEB_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # pyrebase expects a databaseURL key even if you only use Auth/Firestore.
    if "databaseURL" not in cfg:
        project_id = cfg.get("projectId") or ""
        cfg["databaseURL"] = f"https://{project_id}.firebaseio.com"
    return cfg


def _init_admin() -> firestore.Client:
    """Initialize firebase_admin once and return Firestore client."""
    if not firebase_admin._apps:
        if not SA_JSON.exists():
            raise FileNotFoundError(
                f"Missing {SA_JSON}. In Firebase console > Project settings > "
                f"Service accounts > Generate new private key and save here as serviceAccount.json"
            )
        cred = credentials.Certificate(str(SA_JSON))
        firebase_admin.initialize_app(cred)
    return firestore.client()


# Global singletons
_web_cfg = _load_web_config()
_fb_app  = pyrebase.initialize_app(_web_cfg)
_auth    = _fb_app.auth()
_db = _init_admin()  # Firestore client


# ------------------------------------------------------------
# Email / Password auth (client) via pyrebase
# ------------------------------------------------------------
def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    """Create user using Firebase Auth (client API)."""
    try:
        user = _auth.create_user_with_email_and_password(email, password)
        ensure_user(user["localId"], email=email)
        return user
    except Exception as e:
        raise RuntimeError(f"Signup error: {e}")


def login_email_password(email: str, password: str) -> Dict[str, Any]:
    """Sign in with email/password"""
    try:
        user = _auth.sign_in_with_email_and_password(email, password)
        return user
    except Exception as e:
        raise RuntimeError(f"Login error: {e}")


# ------------------------------------------------------------
# Firestore helper primitives
# ------------------------------------------------------------
def ensure_user(uid: str, email: str | None = None) -> None:
    """Create user doc if missing."""
    doc_ref = _db.collection(USERS_COLL).document(uid)
    snap = doc_ref.get()
    if not snap.exists:
        doc_ref.set({
            "uid": uid,
            "email": email or "",
            "created_at": datetime.now(timezone.utc).isoformat()
        })


def add_interaction(uid: str, item_id: str, action: str) -> None:
    """
    Write interaction to:
      - users/{uid}/interactions/{auto-id}
      - interactions_global/{auto-id}   (NEW)
    """
    ensure_user(uid)
    payload = {
        "uid": uid,
        "item_id": item_id,
        "action": action,  # e.g., like / bag
        "ts": datetime.now(timezone.utc).isoformat()
    }
    # user history
    _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB).add(payload)
    # global feed
    _db.collection(GLOBAL_INTERACTIONS).add(payload)


def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Fetch recent interactions for a user"""
    ensure_user(uid)
    q = (_db.collection(USERS_COLL)
             .document(uid)
             .collection(USER_INTERACTIONS_SUB)
             .order_by("ts", direction=firestore.Query.DESCENDING)
             .limit(limit))
    docs = q.stream()
    return [d.to_dict() for d in docs if d.to_dict()]


def fetch_global_interactions(limit: int = 2000) -> List[Dict[str, Any]]:
    """NEW: Fetch recent interactions across all users (for collaborative recs)."""
    q = (_db.collection(GLOBAL_INTERACTIONS)
             .order_by("ts", direction=firestore.Query.DESCENDING)
             .limit(limit))
    docs = q.stream()
    return [d.to_dict() for d in docs if d.to_dict()]


def remove_interaction(uid: str, item_id: str, action: str) -> None:
    """
    Delete matching interaction docs from:
      - users/{uid}/interactions
      - interactions_global
    Removes ALL docs that match (uid, item_id, action).
    """
    ensure_user(uid)

    # user scoped
    coll_user = _db.collection(USERS_COLL).document(uid).collection(USER_INTERACTIONS_SUB)
    for d in coll_user.where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

    # global feed
    coll_global = _db.collection(GLOBAL_INTERACTIONS)
    for d in coll_global.where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

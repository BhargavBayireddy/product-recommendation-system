from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
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

# ---------- Load configs ----------
def _load_web_config() -> Dict[str, Any]:
    if not WEB_JSON.exists():
        raise FileNotFoundError(
            f"Missing {WEB_JSON}. In Firebase console > Project settings > General > "
            f"Your apps (Web) > SDK setup & config, copy JSON into this file."
        )
    with open(WEB_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)
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
_db      = _init_admin()  # Firestore client


# ------------------------------------------------------------
# Auth
# ------------------------------------------------------------
def signup_email_password(email: str, password: str) -> Dict[str, Any]:
    try:
        user = _auth.create_user_with_email_and_password(email, password)
        ensure_user(user["localId"], email=email)
        return user
    except Exception as e:
        raise RuntimeError(f"Signup error: {e}")


def login_email_password(email: str, password: str) -> Dict[str, Any]:
    try:
        user = _auth.sign_in_with_email_and_password(email, password)
        return user
    except Exception as e:
        raise RuntimeError(f"Login error: {e}")


# ------------------------------------------------------------
# Firestore helpers
# ------------------------------------------------------------
def ensure_user(uid: str, email: str | None = None) -> None:
    doc_ref = _db.collection("users").document(uid)
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
      - users/{uid}/interactions
      - interactions_global  (so other users' recs learn from it)
    """
    ensure_user(uid)
    payload = {
        "uid": uid,
        "item_id": item_id,
        "action": action,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    _db.collection("users").document(uid).collection("interactions").add(payload)
    _db.collection("interactions_global").add(payload)   # NEW: global log


def fetch_user_interactions(uid: str, limit: int = 200) -> List[Dict[str, Any]]:
    ensure_user(uid)
    q = (_db.collection("users")
             .document(uid)
             .collection("interactions")
             .order_by("ts", direction=firestore.Query.DESCENDING)
             .limit(limit))
    docs = q.stream()
    return [d.to_dict() for d in docs if d.to_dict()]


def remove_interaction(uid: str, item_id: str, action: str) -> None:
    """
    Delete matching interaction docs from:
      - users/{uid}/interactions
      - interactions_global (for that uid)
    Removes ALL docs that match (uid, item_id, action).
    """
    ensure_user(uid)

    # user subcollection
    coll_user = _db.collection("users").document(uid).collection("interactions")
    for d in coll_user.where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()

    # global collection (only those created by this uid)
    coll_global = _db.collection("interactions_global")
    for d in coll_global.where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream():
        d.reference.delete()


# ------------------------------------------------------------
# GLOBAL collaborative helpers (aggressive mode)
# ------------------------------------------------------------
def fetch_global_likes_for_items(item_ids: List[str], limit_users_per_item: int = 100) -> Dict[str, Set[str]]:
    """
    For each item in item_ids, return a set of uids who liked/bagged it.
    """
    out: Dict[str, Set[str]] = {}
    coll = _db.collection("interactions_global")
    for iid in item_ids:
        uids: Set[str] = set()
        q = coll.where("item_id", "==", iid).where("action", "in", ["like", "bag"]).order_by("ts", direction=firestore.Query.DESCENDING).limit(limit_users_per_item)
        for d in q.stream():
            obj = d.to_dict() or {}
            u = obj.get("uid"); 
            if u: uids.add(u)
        out[iid] = uids
    return out


def fetch_items_liked_by_users(uids: List[str], exclude_items: Set[str], per_user_limit: int = 100) -> List[str]:
    """
    Get items those users liked/bagged, excluding exclude_items.
    """
    out: List[str] = []
    coll = _db.collection("interactions_global")
    for u in uids:
        q = coll.where("uid", "==", u).where("action", "in", ["like", "bag"]).order_by("ts", direction=firestore.Query.DESCENDING).limit(per_user_limit)
        for d in q.stream():
            obj = d.to_dict() or {}
            iid = obj.get("item_id")
            if iid and iid not in exclude_items:
                out.append(iid)
    return out

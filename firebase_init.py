# firebase_init.py — Firebase auth + Firestore, with a robust local mock fallback.
# Put credentials in .streamlit/secrets.toml:
#
# [FIREBASE_WEB_CONFIG]
# apiKey = "..."
# authDomain = "..."
# projectId = "..."
# storageBucket = "..."
# messagingSenderId = "..."
# appId = "..."
#
# [FIREBASE_SERVICE_ACCOUNT]
# type = "service_account"
# project_id = "..."
# private_key_id = "..."
# private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
# client_email = "firebase-adminsdk-...@<project>.iam.gserviceaccount.com"
# client_id = "..."
# auth_uri = "https://accounts.google.com/o/oauth2/auth"
# token_uri = "https://oauth2.googleapis.com/token"
# auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
# client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
#
# Dependencies: pyrebase4, firebase-admin, google-cloud-firestore

from typing import Tuple, Dict, List
import time
import streamlit as st

FIREBASE_READY = False
_py = None
_admin = None
_db = None
_auth = None
MOCK_STORE: Dict[str, List[Dict]] = {}  # uid -> list of interactions
MOCK_USERS: Dict[str, Dict] = {}        # uid -> {email, pwd}
MOCK_POP: List[Dict] = []               # global interactions (popularity)

def _init_firebase():
    global FIREBASE_READY, _py, _admin, _db, _auth
    try:
        web_cfg = st.secrets.get("FIREBASE_WEB_CONFIG", None)
        svc = st.secrets.get("FIREBASE_SERVICE_ACCOUNT", None)
        if not web_cfg or not svc:
            FIREBASE_READY = False
            return

        import pyrebase  # pyrebase4
        import firebase_admin
        from firebase_admin import credentials, firestore

        _py = pyrebase.initialize_app(dict(web_cfg))
        _auth = _py.auth()

        if not firebase_admin._apps:
            cred = credentials.Certificate(dict(svc))
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
        FIREBASE_READY = True
    except Exception:
        FIREBASE_READY = False

_init_firebase()

def is_mock_backend() -> bool:
    return not FIREBASE_READY

# ------------- Public API -------------

def signup_email_password(email: str, password: str) -> Tuple[bool, str]:
    """Returns (ok, uid_or_error)."""
    if FIREBASE_READY:
        try:
            user = _auth.create_user_with_email_and_password(email, password)
            uid = user["localId"]
            return True, uid
        except Exception as e:
            return False, f"Signup failed: {e}"
    # Mock
    uid = f"mock-{hash(email) & 0xfffffff}"
    if uid in MOCK_USERS:
        return False, "User already exists."
    MOCK_USERS[uid] = {"email": email, "password": password}
    return True, uid

def login_email_password(email: str, password: str) -> Tuple[bool, str]:
    if FIREBASE_READY:
        try:
            user = _auth.sign_in_with_email_and_password(email, password)
            return True, user["localId"]
        except Exception as e:
            return False, f"Login failed: {e}"
    # Mock
    for uid, row in MOCK_USERS.items():
        if row["email"] == email and row["password"] == password:
            return True, uid
    return False, "Invalid credentials (mock)."

def ensure_user(uid: str, email: str) -> None:
    if FIREBASE_READY:
        try:
            _db.collection("users").document(uid).set({"email": email, "created_at": time.time()}, merge=True)
        except Exception:
            pass
    else:
        MOCK_USERS.setdefault(uid, {"email": email, "password": "mock"})

def add_interaction(uid: str, item_id: str, action: str) -> None:
    """Actions: like, unlike, bag, remove_bag"""
    ts = time.time()
    rec = {"uid": uid, "item_id": item_id, "action": action, "ts": ts}
    if FIREBASE_READY:
        try:
            _db.collection("interactions").add(rec)
        except Exception:
            pass
    else:
        MOCK_STORE.setdefault(uid, [])
        MOCK_STORE[uid].append(rec)
        # approximate popularity
        if action in ("like", "bag"):
            MOCK_POP.append(rec)

def fetch_user_interactions(uid: str) -> List[Dict]:
    if FIREBASE_READY:
        try:
            q = _db.collection("interactions").where("uid", "==", uid).order_by("ts", direction="DESCENDING").limit(500)
            docs = q.stream()
            return [d.to_dict() for d in docs]
        except Exception:
            return []
    else:
        return MOCK_STORE.get(uid, [])

def fetch_global_interactions(limit: int = 300) -> List[Dict]:
    if FIREBASE_READY:
        try:
            q = _db.collection("interactions").order_by("ts", direction="DESCENDING").limit(limit)
            docs = q.stream()
            return [d.to_dict() for d in docs]
        except Exception:
            return []
    else:
        return MOCK_POP[-limit:]

import json
import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth, firestore

# ---------------- FIREBASE WEB CONFIG ----------------
try:
    web_config_raw = st.secrets["FIREBASE_WEB_CONFIG"]
    # Parse JSON string → Python dict
    FIREBASE_WEB_CONFIG = json.loads(web_config_raw) if isinstance(web_config_raw, str) else web_config_raw
except Exception as e:
    st.error(f"Firebase web config missing or invalid: {e}")
    raise

# ---------------- FIREBASE SERVICE ACCOUNT ----------------
try:
    svc_raw = st.secrets["FIREBASE_SERVICE_ACCOUNT"]
    FIREBASE_SERVICE_ACCOUNT = json.loads(svc_raw) if isinstance(svc_raw, str) else svc_raw
except Exception as e:
    st.error(f"Firebase service account missing or invalid: {e}")
    raise

# ---------------- INITIALIZE CLIENTS ----------------
try:
    firebase = pyrebase.initialize_app(FIREBASE_WEB_CONFIG)
    auth_client = firebase.auth()
    db = firebase.database()
except Exception as e:
    st.error(f"Pyrebase init failed: {e}")
    raise

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred)
    firestore_client = firestore.client()
except Exception as e:
    st.warning(f"Firebase Admin init failed: {e}")
    firestore_client = None


# ---------------- AUTH HELPERS ----------------
def signup_email_password(email, password):
    """Sign up new user."""
    return auth_client.create_user_with_email_and_password(email, password)

def login_email_password(email, password):
    """Login user."""
    return auth_client.sign_in_with_email_and_password(email, password)

def ensure_user(uid, email):
    """Ensure user doc exists in Firestore."""
    if firestore_client is None:
        return
    doc_ref = firestore_client.collection("users").document(uid)
    if not doc_ref.get().exists:
        doc_ref.set({"email": email, "created_at": firestore.SERVER_TIMESTAMP})

def add_interaction(uid, item_id, action):
    """Store a like/bag action."""
    if firestore_client is None:
        return
    firestore_client.collection("interactions").add({
        "uid": uid,
        "item_id": item_id,
        "action": action,
        "ts": firestore.SERVER_TIMESTAMP
    })

def fetch_user_interactions(uid):
    """Fetch all user interactions."""
    if firestore_client is None:
        return []
    docs = firestore_client.collection("interactions").where("uid", "==", uid).stream()
    return [{"item_id": d.to_dict()["item_id"], "action": d.to_dict()["action"], "ts": d.to_dict()["ts"].isoformat()} for d in docs]

def fetch_global_interactions(limit=2000):
    """Fetch global interactions."""
    if firestore_client is None:
        return []
    docs = firestore_client.collection("interactions").limit(limit).stream()
    return [d.to_dict() for d in docs]

def remove_interaction(uid, item_id, action):
    """Remove interaction."""
    if firestore_client is None:
        return
    docs = firestore_client.collection("interactions").where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action).stream()
    for d in docs:
        firestore_client.collection("interactions").document(d.id).delete()

def signup_email_password(email, password):
    try:
        return auth_client.create_user_with_email_and_password(email, password)
    except Exception as e:
        if "EMAIL_EXISTS" in str(e):
            st.warning("This email is already registered. Please sign in instead.")
        else:
            st.error(f"Signup failed: {e}")
        return None

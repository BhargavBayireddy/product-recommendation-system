# firebase_init.py
# Unified Firebase Auth + Firestore helper for Streamlit Cloud

import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import streamlit as st

# --- Load Secrets ---
fb_web = st.secrets["FIREBASE_WEB_CONFIG"]
fb_svc = st.secrets["FIREBASE_SERVICE_ACCOUNT"]

# --- Pyrebase (Auth, Realtime DB style) ---
pb = pyrebase.initialize_app(fb_web)
pb_auth = pb.auth()

def signup_email_password(email, password):
    return pb_auth.create_user_with_email_and_password(email, password)

def login_email_password(email, password):
    return pb_auth.sign_in_with_email_and_password(email, password)

# --- Firebase Admin (Firestore) ---
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(json.dumps(fb_svc)))
    firebase_admin.initialize_app(cred)

_db = firestore.client()

def get_firestore_client():
    """Return Firestore db instance so other modules can use it."""
    return _db

# --- Firestore helpers (optional but used by app.py) ---
def ensure_user(uid, email=None):
    _db.collection("users").document(uid).set(
        {"email": email or "", "created": firestore.SERVER_TIMESTAMP},
        merge=True
    )

def add_interaction(uid, item_id, action):
    _db.collection("interactions").add({
        "uid": uid,
        "item_id": item_id,
        "action": action,
        "ts": firestore.SERVER_TIMESTAMP
    })

def remove_interaction(uid, item_id, action):
    q = (_db.collection("interactions")
         .where("uid", "==", uid)
         .where("item_id", "==", item_id)
         .where("action", "==", action))
    for doc in q.stream():
        doc.reference.delete()

def fetch_user_interactions(uid):
    q = _db.collection("interactions").where("uid", "==", uid).stream()
    return [{"uid": uid, **d.to_dict()} for d in q]

def fetch_global_interactions(limit=2000):
    q = (_db.collection("interactions")
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit)
         .stream())
    return [{"id": d.id, **d.to_dict()} for d in q]

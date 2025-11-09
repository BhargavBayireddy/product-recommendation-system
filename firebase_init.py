# firebase_init.py
# Works on Streamlit Cloud with secrets:
# [FIREBASE_SERVICE_ACCOUNT] ...  and [FIREBASE_WEB_CONFIG] ...
import time
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pyrebase

# ---- Admin SDK (service account from Streamlit Secrets) ----
if "firebase_app" not in st.session_state:
    sa_fields = dict(st.secrets["FIREBASE_SERVICE_ACCOUNT"])
    cred = credentials.Certificate(sa_fields)
    st.session_state["firebase_app"] = firebase_admin.initialize_app(cred)
    st.session_state["db"] = firestore.client()

db = st.session_state["db"]

# ---- Web SDK (email/password via pyrebase4) ----
def _get_pyrebase():
    cfg = dict(st.secrets["FIREBASE_WEB_CONFIG"])
    return pyrebase.initialize_app({
        "apiKey": cfg["apiKey"],
        "authDomain": cfg["authDomain"],
        "projectId": cfg["projectId"],
        "storageBucket": cfg.get("storageBucket",""),
        "messagingSenderId": cfg.get("messagingSenderId",""),
        "appId": cfg["appId"],
        "databaseURL": ""  # not using RTDB
    })

def signup_email_password(email: str, password: str):
    return _get_pyrebase().auth().create_user_with_email_and_password(email, password)

def login_email_password(email: str, password: str):
    return _get_pyrebase().auth().sign_in_with_email_and_password(email, password)

def verify_id_token(id_token: str):
    return auth.verify_id_token(id_token)

# ---- Firestore helpers (optional) ----
def ensure_user(uid: str, email: str = ""):
    ref = db.collection("users").document(uid)
    if not ref.get().exists:
        ref.set({"created_at": time.time(), "email": email})

def add_interaction(uid: str, item_id: str, action: str):
    db.collection("interactions").document().set(
        {"uid": uid, "item_id": item_id, "action": action, "ts": time.time()}
    )

def remove_interaction(uid: str, item_id: str, action: str):
    q = db.collection("interactions").where("uid","==",uid).where("item_id","==",item_id).where("action","==",action).stream()
    for doc in q:
        db.collection("interactions").document(doc.id).delete()

def fetch_user_interactions(uid: str):
    col = db.collection("interactions").where("uid","==",uid).order_by("ts", direction=firestore.Query.DESCENDING).limit(500)
    return [x.to_dict() for x in col.stream()]

def fetch_global_interactions(limit: int = 2000):
    col = db.collection("interactions").order_by("ts", direction=firestore.Query.DESCENDING).limit(limit)
    return [x.to_dict() for x in col.stream()]

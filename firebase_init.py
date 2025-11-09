# firebase_init.py
import json
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase
import streamlit as st

fb_web = st.secrets["FIREBASE_WEB_CONFIG"]
fb_svc = st.secrets["FIREBASE_SERVICE_ACCOUNT"]

pb = pyrebase.initialize_app(fb_web)
pb_auth = pb.auth()

def signup_email_password(email, password):
    return pb_auth.create_user_with_email_and_password(email, password)

def login_email_password(email, password):
    return pb_auth.sign_in_with_email_and_password(email, password)

if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(json.dumps(fb_svc)))
    firebase_admin.initialize_app(cred)

_db = firestore.client()

def get_firestore_client():
    return _db

def ensure_user(uid, email=None):
    _db.collection("users").document(uid).set(
        {"email": email or "", "created": firestore.SERVER_TIMESTAMP},
        merge=True
    )

def add_interaction(uid, item_id, action):
    _db.collection("interactions").add({
        "uid": uid, "item_id": item_id, "action": action,
        "ts": firestore.SERVER_TIMESTAMP
    })

def remove_interaction(uid, item_id, action):
    q = (_db.collection("interactions")
         .where("uid","==",uid).where("item_id","==",item_id).where("action","==",action))
    for doc in q.stream():
        doc.reference.delete()

def fetch_user_interactions(uid):
    q = _db.collection("interactions").where("uid","==",uid).stream()
    return [{"uid": uid, **d.to_dict()} for d in q]

def fetch_global_interactions(limit=2000):
    q = (_db.collection("interactions")
         .order_by("ts", direction=firestore.Query.DESCENDING)
         .limit(limit).stream())
    return [{"id": d.id, **d.to_dict()} for d in q]

import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import time

# ---- Load secrets from Streamlit ----
import streamlit as st
FB_WEB = st.secrets["FIREBASE_WEB_CONFIG"]
FB_SA  = st.secrets["FIREBASE_SERVICE_ACCOUNT"]

# ---- Init Pyrebase (Auth + Realtime DB if needed) ----
pb = pyrebase.initialize_app(FB_WEB)
pb_auth = pb.auth()

# ---- Init Admin SDK (Firestore) ----
if not firebase_admin._apps:
    cred = credentials.Certificate(json.loads(json.dumps(FB_SA)))
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ================================
#  AUTH HELPERS
# ================================
def signup_email_password(email, password):
    return pb_auth.create_user_with_email_and_password(email, password)

def login_email_password(email, password):
    return pb_auth.sign_in_with_email_and_password(email, password)

def ensure_user(uid, email=None):
    ref = db.collection("users").document(uid)
    if not ref.get().exists:
        ref.set({"email": email, "created": time.time()})

# ================================
#  INTERACTIONS (likes, bag, etc)
# ================================
def add_interaction(uid, item_id, action):
    ev = {"uid": uid, "item_id": item_id, "action": action, "ts": time.time()}
    db.collection("interactions").add(ev)

def remove_interaction(uid, item_id, action):
    q = db.collection("interactions").where("uid", "==", uid).where("item_id", "==", item_id).where("action", "==", action)
    for doc in q.get():
        doc.reference.delete()

def fetch_user_interactions(uid):
    q = db.collection("interactions").where("uid", "==", uid).get()
    return [x.to_dict() for x in q]

def fetch_global_interactions(limit=2000):
    q = db.collection("interactions").order_by("ts", direction=firestore.Query.DESCENDING).limit(limit).get()
    return [x.to_dict() for x in q]

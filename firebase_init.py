# firebase_init.py — Firebase + Twilio OTP + robust local mock
# Works standalone without real Firebase; real OTP via Twilio optional.

from typing import Tuple, Dict, List, Optional
import time, random, string
import streamlit as st

FIREBASE_READY = False
_py = None
_admin = None
_db = None
_auth = None
_admin_auth = None

MOCK_STORE: Dict[str, List[Dict]] = {}
MOCK_USERS: Dict[str, Dict] = {}
MOCK_POP: List[Dict] = []
OTP_STORE: Dict[str, Dict] = {}

_TWILIO_READY = False
_twilio_client = None
_twilio_from = None

def _init_twilio():
    global _TWILIO_READY, _twilio_client, _twilio_from
    try:
        cfg = st.secrets.get("TWILIO", None)
        if not cfg:
            _TWILIO_READY = False
            return
        from twilio.rest import Client
        _twilio_client = Client(cfg["ACCOUNT_SID"], cfg["AUTH_TOKEN"])
        _twilio_from = cfg["FROM"]
        _TWILIO_READY = True
    except Exception:
        _TWILIO_READY = False

def _init_firebase():
    global FIREBASE_READY, _py, _admin, _db, _auth, _admin_auth
    try:
        web_cfg = st.secrets.get("FIREBASE_WEB_CONFIG", None)
        svc = st.secrets.get("FIREBASE_SERVICE_ACCOUNT", None)
        if not web_cfg or not svc:
            FIREBASE_READY = False
            return
        import pyrebase
        import firebase_admin
        from firebase_admin import credentials, firestore, auth as admin_auth
        _py = pyrebase.initialize_app(dict(web_cfg))
        _auth = _py.auth()
        if not firebase_admin._apps:
            cred = credentials.Certificate(dict(svc))
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
        _admin_auth = admin_auth
        FIREBASE_READY = True
    except Exception:
        FIREBASE_READY = False

_init_firebase()
_init_twilio()

def email_exists(email: str) -> bool:
    if FIREBASE_READY and _admin_auth:
        try:
            _admin_auth.get_user_by_email(email)
            return True
        except Exception:
            return False
    for _, u in MOCK_USERS.items():
        if u.get("email") == email:
            return True
    return False

def signup_email_password(email: str, password: str) -> Tuple[bool, str]:
    if FIREBASE_READY:
        try:
            user = _auth.create_user_with_email_and_password(email, password)
            return True, user["localId"]
        except Exception as e:
            return False, str(e)
    if email_exists(email):
        return False, "User already exists."
    uid = f"mock-{abs(hash(email)) & 0xfffffff}"
    MOCK_USERS[uid] = {"email": email, "password": password}
    return True, uid

def login_email_password(email: str, password: str) -> Tuple[bool, str]:
    if FIREBASE_READY:
        try:
            user = _auth.sign_in_with_email_and_password(email, password)
            return True, user["localId"]
        except Exception as e:
            return False, str(e)
    for uid, u in MOCK_USERS.items():
        if u.get("email") == email:
            if u.get("password") == password:
                return True, uid
            return False, "INVALID_PASSWORD"
    return False, "EMAIL_NOT_FOUND"

def ensure_user(uid: str, email: Optional[str] = None, phone: Optional[str] = None):
    if uid not in MOCK_USERS:
        MOCK_USERS[uid] = {}
    if email: MOCK_USERS[uid]["email"] = email
    if phone: MOCK_USERS[uid]["phone"] = phone

def _gen_otp(n=6) -> str:
    return "".join(random.choices(string.digits, k=n))

def send_phone_otp(phone: str) -> Tuple[bool, str]:
    code = _gen_otp()
    OTP_STORE[phone] = {"code": code, "ts": time.time()}
    if _TWILIO_READY:
        try:
            _twilio_client.messages.create(to=phone, from_=_twilio_from, body=f"Your ReccoVerse OTP: {code}")
            return True, "OTP sent via SMS."
        except Exception as e:
            return False, str(e)
    return True, code  # mock/dev mode

def verify_phone_otp(phone: str, code: str) -> Tuple[bool, str]:
    obj = OTP_STORE.get(phone)
    if not obj:
        return False, "Request OTP first."
    if time.time() - obj["ts"] > 300:
        return False, "OTP expired."
    if code != obj["code"]:
        return False, "Invalid OTP."
    uid = f"phone-{phone}"
    ensure_user(uid, phone=phone)
    return True, uid

def add_interaction(uid: str, item_id: str, action: str):
    rec = {"uid": uid, "item_id": item_id, "action": action, "ts": time.time()}
    MOCK_STORE.setdefault(uid, []).append(rec)
    if action in ("like", "bag"):
        MOCK_POP.append(rec)

def fetch_user_interactions(uid: str) -> List[Dict]:
    return MOCK_STORE.get(uid, [])

def fetch_global_interactions(limit=300) -> List[Dict]:
    return MOCK_POP[-limit:]

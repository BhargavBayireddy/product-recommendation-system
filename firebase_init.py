from typing import Tuple, Dict, List, Optional
import time, random, string
import streamlit as st

FIREBASE_READY = False
_py = None
_admin = None
_db = None
_auth = None
_admin_auth = None

# ---- Local mock stores ----
MOCK_STORE: Dict[str, List[Dict]] = {}   # uid -> interactions[{uid,item_id,action,ts}]
MOCK_USERS: Dict[str, Dict] = {}         # uid -> {email,password} or {phone}
MOCK_POP: List[Dict] = []                # recent like/bag records
OTP_STORE: Dict[str, Dict] = {}          # phone -> {code, ts}

# ---- Optional Twilio ----
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

        import pyrebase  # pyrebase4
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

def is_mock_backend() -> bool:
    return not FIREBASE_READY

# ----------------- EMAIL HELPERS -----------------

def email_exists(email: str) -> bool:
    """Check whether an email has a Firebase user. Mock supported."""
    if FIREBASE_READY and _admin_auth is not None:
        try:
            _admin_auth.get_user_by_email(email)
            return True
        except Exception:
            return False
    # mock
    for _, u in MOCK_USERS.items():
        if u.get("email") == email:
            return True
    return False

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
            # Let caller differentiate wrong password vs not found using email_exists
            return False, f"{e}"
    # Mock
    for uid, row in MOCK_USERS.items():
        if row.get("email") == email:
            if row.get("password") == password:
                return True, uid
            return False, "INVALID_PASSWORD"
    return False, "EMAIL_NOT_FOUND"

def ensure_user(uid: str, email: Optional[str] = None, phone: Optional[str] = None) -> None:
    if FIREBASE_READY and _db is not None:
        try:
            doc = {"created_at": time.time()}
            if email: doc["email"] = email
            if phone: doc["phone"] = phone
            _db.collection("users").document(uid).set(doc, merge=True)
        except Exception:
            pass
    else:
        if uid not in MOCK_USERS:
            MOCK_USERS[uid] = {}
        if email: MOCK_USERS[uid]["email"] = email
        if phone: MOCK_USERS[uid]["phone"] = phone

# ----------------- OTP HELPERS -----------------

def _gen_otp(n: int = 6) -> str:
    return "".join(random.choices(string.digits, k=n))

def send_phone_otp(phone: str) -> Tuple[bool, str]:
    """
    If Twilio secrets exist, send SMS; else show code to the dev (mock).
    Returns (ok, message_or_code).
    """
    code = _gen_otp()
    OTP_STORE[phone] = {"code": code, "ts": time.time()}

    if _TWILIO_READY:
        try:
            _twilio_client.messages.create(
                to=phone, from_=_twilio_from, body=f"Your ReccoVerse OTP: {code}"
            )
            return True, "OTP sent via SMS."
        except Exception as e:
            return False, f"Failed to send SMS: {e}"
    # Mock/dev
    return True, code  # Expose code in UI for local/testing

def verify_phone_otp(phone: str, code: str) -> Tuple[bool, str]:
    obj = OTP_STORE.get(phone)
    if not obj:
        return False, "Request OTP first."
    if time.time() - obj["ts"] > 5 * 60:
        return False, "OTP expired. Please resend."
    if code != obj["code"]:
        return False, "Invalid OTP."
    # Create/log user
    uid = f"phone-{phone}"
    ensure_user(uid, phone=phone)
    return True, uid

# ----------------- INTERACTIONS -----------------

def add_interaction(uid: str, item_id: str, action: str) -> None:
    ts = time.time()
    rec = {"uid": uid, "item_id": item_id, "action": action, "ts": ts}
    if FIREBASE_READY and _db is not None:
        try:
            _db.collection("interactions").add(rec)
        except Exception:
            pass
    else:
        MOCK_STORE.setdefault(uid, []).append(rec)
        if action in ("like", "bag"):
            MOCK_POP.append(rec)

def fetch_user_interactions(uid: str) -> List[Dict]:
    if FIREBASE_READY and _db is not None:
        try:
            q = _db.collection("interactions").where("uid", "==", uid).order_by("ts", direction="DESCENDING").limit(500)
            return [d.to_dict() for d in q.stream()]
        except Exception:
            return []
    return MOCK_STORE.get(uid, [])

def fetch_global_interactions(limit: int = 300) -> List[Dict]:
    if FIREBASE_READY and _db is not None:
        try:
            q = _db.collection("interactions").order_by("ts", direction="DESCENDING").limit(limit)
            return [d.to_dict() for d in q.stream()]
        except Exception:
            return []
    return MOCK_POP[-limit:]

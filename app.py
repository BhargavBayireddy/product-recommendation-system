# ==========================================================
# ReccoVerse v5 — Netflix-Style Multidomain AI Recommender
# ==========================================================
import streamlit as st
import pandas as pd, numpy as np, requests, uuid, random

# -------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")

# -------------------- STYLE -------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
 background:radial-gradient(circle at 30% 20%,#020617,#000);
 color:#fff;font-family:'Poppins',sans-serif;}
h1,h2,h3{color:#fff;font-weight:700;}
.stButton>button{
 background:linear-gradient(90deg,#00c6ff,#ff0080);
 border:none;border-radius:30px;font-weight:600;
 padding:.4rem 1rem;color:white;transition:.3s;}
.stButton>button:hover{transform:scale(1.05);}
input,textarea{background:rgba(255,255,255,.07)!important;color:#fff!important;}
.badge{background:rgba(255,255,255,.1);padding:.2rem .6rem;
 border-radius:999px;font-size:.75rem;}
.rowtitle{font-size:1.3rem;margin-top:1.5rem;}
</style>
""", unsafe_allow_html=True)

# -------------------- API KEY ------------------------------
TMDB_KEY = st.secrets.get("TMDB_API_KEY","57b87af46cd78b943c23b3b94c68cfef")

# -------------------- FETCHERS -----------------------------
@st.cache_data(show_spinner=False)
def tmdb_movies(n=40):
    url=f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_KEY}&language=en-US&page=1"
    try:
        res=requests.get(url,timeout=8).json()
        rows=[{
            "id":f"mv_{m['id']}",
            "title":m["title"],
            "category":"Movies",
            "genre":"Film",
            "image":f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else "",
            "source":"TMDB"} for m in res["results"][:n]]
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def deezer_tracks(n=30):
    try:
        r=requests.get(f"https://api.deezer.com/chart/0/tracks?limit={n}",timeout=8).json()
        return pd.DataFrame([{
            "id":f"mu_{t['id']}",
            "title":t["title_short"],
            "category":"Music",
            "genre":t["artist"]["name"],
            "image":t["album"]["cover_medium"],
            "source":"Deezer"} for t in r["data"]])
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def dummy_products(n=40):
    try:
        r=requests.get(f"https://dummyjson.com/products?limit={n}",timeout=8).json()
        rows=[]
        for p in r["products"]:
            cat=p["category"].title()
            domain="Beauty" if "beauty" in cat.lower() else \
                    "Fashion" if any(k in cat.lower() for k in["shirt","shoe","bag","dress"]) else \
                    "Tech"
            rows.append({
                "id":f"pr_{p['id']}",
                "title":p["title"],
                "category":domain,
                "genre":cat,
                "image":p["thumbnail"],
                "source":"DummyJSON"})
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

@st.cache_data(show_spinner="Loading AI datasets…")
def load_data():
    return pd.concat([tmdb_movies(),deezer_tracks(),dummy_products()],
                     ignore_index=True).drop_duplicates("title").reset_index(drop=True)

# -------------------- NOVELTY / EXPLAIN --------------------
def novelty_score(title, liked):
    if not liked: return random.uniform(0.4,0.8)
    overlap=sum(1 for t in liked if any(w in title.lower() for w in t.lower().split()))
    return max(0.2,1-overlap/len(liked))

def why_this(title, liked):
    rel=[t for t in liked if any(w in title.lower() for w in t.lower().split())]
    return rel[:3] if rel else random.sample(liked,min(len(liked),2)) if liked else []

# -------------------- CARD RENDER --------------------------
def render_card(r, liked_set, bag_set, liked_titles):
    st.image(r.image or "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80", use_column_width=True)
    st.markdown(f"**{r.title}**  \n_{r.category} • {r.genre}_")
    nov=novelty_score(r.title,liked_titles)
    st.markdown(f"<div class='badge'>🧬 Novelty {int(nov*100)}%</div>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    uid=str(uuid.uuid4())[:8]
    with c1:
        if st.button(("❤️" if r.id in liked_set else "♡ Like"), key=f"l_{r.id}_{uid}"):
            (liked_set.remove(r.id) if r.id in liked_set else liked_set.add(r.id)); st.rerun()
    with c2:
        if st.button(("👜" if r.id in bag_set else "➕ Bag"), key=f"b_{r.id}_{uid}"):
            (bag_set.remove(r.id) if r.id in bag_set else bag_set.add(r.id)); st.rerun()
    with st.expander("🔍 Why this was recommended"):
        reasons=why_this(r.title,[t for t in liked_titles])
        if reasons: [st.markdown(f"- Related to **{x}**") for x in reasons]
        else: st.markdown("Diverse pick by Quanta MMR engine.")

# -------------------- ROW LAYOUT ---------------------------
def render_row(df, title, liked, bag, liked_titles):
    st.markdown(f"<div class='rowtitle'>{title}</div>", unsafe_allow_html=True)
    cols=st.columns(5)
    for i,(_,r) in enumerate(df.head(5).iterrows()):
        with cols[i]: render_card(r,liked,bag,liked_titles)

# -------------------- LOGIN -------------------------------
def login_screen():
    st.markdown("<h1 style='text-align:center;'>🎬 ReccoVerse</h1>", unsafe_allow_html=True)
    st.video("https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4")
    email=st.text_input("Email")
    pwd=st.text_input("Password",type="password")
    if st.button("Login / Sign-Up",use_container_width=True):
        if email and pwd:
            st.session_state["authed"]=True
            st.session_state["user"]=email
            st.session_state["liked"]=set()
            st.session_state["bag"]=set()
            st.success(f"Welcome {name_from_email(email)}!")
            st.rerun()
        else: st.error("Enter email and password.")
    st.stop()

def name_from_email(e): return e.split("@")[0].capitalize()

# -------------------- MAIN DASHBOARD -----------------------
def main_app():
    st.sidebar.header("ReccoVerse")
    if st.sidebar.button("Sign Out"): st.session_state.clear(); st.rerun()

    query=st.sidebar.text_input("🔍 Search anything...")
    surprise=st.sidebar.checkbox("🎢 Surprise Mode",False)

    df=load_data()
    liked_titles=[df.loc[df.id==x,"title"].values[0] for x in st.session_state["liked"] if x in df.id.values]
    if query: df=df[df["title"].str.lower().str.contains(query.lower())]

    if surprise: df=df.sample(frac=1).reset_index(drop=True)

    st.markdown("## 🍿 Top Picks For You")
    render_row(df[df.category=="Movies"],"🎬 Popular Movies",st.session_state["liked"],st.session_state["bag"],liked_titles)
    render_row(df[df.category=="Music"],"🎵 Top Music Tracks",st.session_state["liked"],st.session_state["bag"],liked_titles)
    render_row(df[df.category.isin(["Beauty","Fashion","Tech"])],
               "🛍️ Trending Products",st.session_state["liked"],st.session_state["bag"],liked_titles)
    if st.session_state["liked"]:
        liked_df=df[df.id.isin(st.session_state["liked"])]
        render_row(liked_df,"💖 Because You Liked These",st.session_state["liked"],st.session_state["bag"],liked_titles)

# -------------------- ENTRYPOINT ---------------------------
if "authed" not in st.session_state: st.session_state["authed"]=False
if not st.session_state["authed"]: login_screen()
else: main_app()

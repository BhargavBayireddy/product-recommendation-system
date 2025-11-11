import streamlit as st
import pandas as pd, requests, uuid, random

st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")

# -------------- GLOBAL STYLE --------------
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
.badge{background:rgba(255,255,255,.1);padding:.2rem .6rem;
 border-radius:999px;font-size:.75rem;}
.progress{height:8px;border-radius:10px;overflow:hidden;background:#222;}
.bar{height:8px;background:linear-gradient(90deg,#00f5a0,#00d9ff);}
</style>
""", unsafe_allow_html=True)

TMDB_KEY = st.secrets.get("TMDB_API_KEY","57b87af46cd78b943c23b3b94c68cfef")

# -------- SESSION INIT --------
if "liked" not in st.session_state: st.session_state["liked"]=set()
if "bag" not in st.session_state: st.session_state["bag"]=set()
if "authed" not in st.session_state: st.session_state["authed"]=False

# -------- DATA FETCHERS -------
@st.cache_data
def get_movies():
    res=requests.get(f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_KEY}&language=en-US&page=1").json()
    return pd.DataFrame([{
        "id":f"mv_{m['id']}",
        "title":m["title"],
        "category":"Movies",
        "genre":"Film",
        "image":f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else ""
    } for m in res["results"]])

@st.cache_data
def get_music():
    res=requests.get("https://api.deezer.com/chart/0/tracks?limit=20").json()
    return pd.DataFrame([{
        "id":f"mu_{t['id']}",
        "title":t["title_short"],
        "category":"Music",
        "genre":t["artist"]["name"],
        "image":t["album"]["cover_medium"]
    } for t in res["data"]])

@st.cache_data
def get_products():
    res=requests.get("https://dummyjson.com/products?limit=20").json()
    return pd.DataFrame([{
        "id":f"pr_{p['id']}",
        "title":p["title"],
        "category":"Products",
        "genre":p["category"],
        "image":p["thumbnail"]
    } for p in res["products"]])

@st.cache_data
def load_all():
    return pd.concat([get_movies(),get_music(),get_products()],ignore_index=True)

# -------- AI NOVELTY ----------
def novelty(title,liked_titles):
    if not liked_titles: return random.uniform(0.7,1.0)
    overlap=sum(1 for t in liked_titles if any(w in title.lower() for w in t.lower().split()))
    return max(0.1,1-overlap/len(liked_titles))

# -------- LOGIN ---------------
def login_screen():
    st.markdown("<h1 style='text-align:center;'>🎬 ReccoVerse</h1>", unsafe_allow_html=True)
    st.video("https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4")
    email=st.text_input("Email")
    pwd=st.text_input("Password",type="password")
    if st.button("Login / Sign-Up",use_container_width=True):
        if email and pwd:
            st.session_state["authed"]=True
            st.session_state["user"]=email
            st.success(f"Welcome {email.split('@')[0].capitalize()}!")
            st.rerun()
        else:
            st.error("Please enter all fields.")
    st.stop()

# -------- CARD ----------------
def card(row):
    liked= row.id in st.session_state.liked
    bagged= row.id in st.session_state.bag
    st.image(row.image or "https://placehold.co/400x600/000/fff", use_column_width=True)
    st.markdown(f"**{row.title}**  \n_{row.category} • {row.genre}_")
    liked_titles=[row.title for id in st.session_state.liked]
    nov=round(novelty(row.title,liked_titles)*100)
    st.markdown(f"<div class='badge'>🧬 Novelty {nov}%</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='progress'><div class='bar' style='width:{nov}%;'></div></div>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    key=str(uuid.uuid4())[:8]
    with c1:
        if st.button(("❤️" if liked else "♡ Like"), key=f"l_{row.id}_{key}"):
            st.session_state.liked.symmetric_difference_update([row.id]); st.rerun()
    with c2:
        if st.button(("👜" if bagged else "➕ Bag"), key=f"b_{row.id}_{key}"):
            st.session_state.bag.symmetric_difference_update([row.id]); st.rerun()

# -------- MAIN APP ------------
def main():
    st.sidebar.header("ReccoVerse")
    if st.sidebar.button("Sign Out"): st.session_state.clear(); st.rerun()
    search=st.sidebar.text_input("Search anything...")
    df=load_all()
    if search: df=df[df.title.str.lower().str.contains(search.lower())]
    st.markdown("## 🍿 Top Picks For You")
    for cat in df.category.unique():
        st.markdown(f"### {cat}")
        cols=st.columns(5)
        for i,(_,r) in enumerate(df[df.category==cat].head(5).iterrows()):
            with cols[i]: card(r)
    if st.session_state.liked:
        liked_df=df[df.id.isin(st.session_state.liked)]
        st.markdown("### 💖 Because You Liked These")
        cols=st.columns(5)
        for i,(_,r) in enumerate(liked_df.head(5).iterrows()):
            with cols[i]: card(r)

# -------- RUN -----------------
if not st.session_state.authed: login_screen()
else: main()

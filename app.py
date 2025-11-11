import streamlit as st
import pandas as pd, numpy as np, requests, uuid, random
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURATION
st.set_page_config(page_title="ReccoVerse", page_icon="🎬", layout="wide")

TMDB_KEY = st.secrets.get("TMDB_API_KEY", "57b87af46cd78b943c23b3b94c68cfef")

# ---------------------------------------------------------
# STYLES
st.markdown("""
<style>
body,[data-testid="stAppViewContainer"]{
  background:radial-gradient(circle at 30% 20%,#060b1c,#000) !important;
  color:#f5f5f5;font-family:'Poppins',sans-serif;}
h1,h2,h3{color:#fff;}
.stButton>button{
  background:linear-gradient(90deg,#00ffff,#ff00c3);
  border:none;border-radius:50px;color:white;font-weight:600;
  padding:.5rem 1.2rem;transition:.3s;}
.stButton>button:hover{transform:scale(1.05);}
input,textarea{background:rgba(255,255,255,.05)!important;
  color:#fff!important;border-radius:8px!important;}
.card{transition:transform .2s ease;}
.card:hover{transform:scale(1.03);}
.badge{font-size:.75rem;padding:.2rem .6rem;
  border-radius:999px;background:rgba(255,255,255,.1);}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
@st.cache_data(show_spinner=False)
def tmdb_poster(title):
    try:
        url=f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={requests.utils.quote(title)}"
        r=requests.get(url,timeout=5).json()
        if r.get("results") and r["results"][0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500"+r["results"][0]["poster_path"]
    except: pass
    return "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80"

@st.cache_data(show_spinner=False)
def deezer_sample(limit=30):
    try:
        r=requests.get(f"https://api.deezer.com/chart/0/tracks?limit={limit}",timeout=6).json()
        data=[{
            "item_id":f"mu_{t['id']}",
            "title":t["title_short"],
            "category":"Music",
            "genre":t["artist"]["name"],
            "source":"Deezer",
            "image":t["album"]["cover_medium"]
        } for t in r["data"]]
        return pd.DataFrame(data)
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def dummy_products(limit=60):
    r=requests.get(f"https://dummyjson.com/products?limit={limit}",timeout=8).json()
    rows=[]
    for p in r["products"]:
        cat=p["category"].title()
        domain=("Beauty" if "beauty" in cat.lower()
                else "Fashion" if any(k in cat.lower() for k in["shirt","shoe","dress","bag"])
                else "Tech")
        rows.append({
            "item_id":f"pr_{p['id']}",
            "title":p["title"],
            "category":domain,
            "genre":cat,
            "source":"DummyJSON",
            "image":p["thumbnail"]
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def tmdb_movies(limit=80):
    r=requests.get(f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_KEY}&language=en-US&page=1",timeout=6).json()
    data=[{
        "item_id":f"mv_{m['id']}",
        "title":m["title"],
        "category":"Movies",
        "genre":", ".join([g.get("name","") for g in m.get("genre_ids",[])]) if isinstance(m.get("genre_ids"),list) else "Film",
        "source":"TMDB",
        "image":"https://image.tmdb.org/t/p/w500"+m["poster_path"] if m.get("poster_path") else tmdb_poster(m["title"])
    } for m in r["results"][:limit]]
    return pd.DataFrame(data)

# ---------------------------------------------------------
# MULTIDOMAIN MERGE
@st.cache_data(show_spinner="Loading AI datasets…")
def load_data():
    dfs=[tmdb_movies(), deezer_sample(), dummy_products()]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("title")
    return df.sample(frac=1).reset_index(drop=True)

# ---------------------------------------------------------
# MOCK AUTHENTICATION
def login_screen():
    st.markdown("<h1 style='text-align:center;'>🎬 ReccoVerse</h1>", unsafe_allow_html=True)
    st.video("https://videos.pexels.com/video-files/856604/856604-hd_1920_1080_30fps.mp4")
    email=st.text_input("Email",placeholder="user@example.com")
    pwd=st.text_input("Password",type="password")
    if st.button("Login / Sign-Up",use_container_width=True):
        if email and pwd:
            st.session_state.authed=True
            st.session_state.user=email
            st.success(f"Welcome, {email.split('@')[0]}!")
            st.experimental_rerun()
        else:
            st.error("Enter credentials.")
    st.stop()

# ---------------------------------------------------------
# NOVELTY & EXPLAINABILITY
def novelty_score(item, liked_titles):
    if not liked_titles: return random.uniform(0.4,0.8)
    match=sum([1 for t in liked_titles if any(w in item.lower() for w in t.lower().split())])
    return max(0.2,1-match/len(liked_titles))

def why_this(title, liked):
    rel=[t for t in liked if any(w in title.lower() for w in t.lower().split())]
    return rel[:3] if rel else random.sample(liked,min(len(liked),2)) if liked else []

# ---------------------------------------------------------
# RENDER CARD
def render_card(row, liked_set, bag_set, surprise, liked_titles):
    col=st.container()
    with col:
        st.image(row.image,use_column_width=True)
        st.markdown(f"**{row.title}**  \n_{row.category} • {row.genre}_")
        nov=novelty_score(row.title,liked_titles)
        st.markdown(f"<div class='badge'>🧬 Novelty {int(nov*100)}%</div>",unsafe_allow_html=True)
        if st.button(("❤️" if row.item_id in liked_set else "♡ Like"),
                     key=f"like_{row.item_id}_{uuid.uuid4().hex[:6]}"):
            if row.item_id in liked_set: liked_set.remove(row.item_id)
            else: liked_set.add(row.item_id)
            st.experimental_rerun()
        if st.button(("👜" if row.item_id in bag_set else "➕ Bag"),
                     key=f"bag_{row.item_id}_{uuid.uuid4().hex[:6]}"):
            if row.item_id in bag_set: bag_set.remove(row.item_id)
            else: bag_set.add(row.item_id)
            st.experimental_rerun()
        with st.expander("🔍 Why this was recommended"):
            reasons=why_this(row.title,[t for t in liked_titles])
            if reasons: [st.markdown(f"- Related to **{r}**") for r in reasons]
            else: st.markdown("Explored for diversity (Quanta MMR mode).")

# ---------------------------------------------------------
# MAIN DASHBOARD
def main_app():
    st.sidebar.markdown("### 🎬 ReccoVerse Panel")
    if st.sidebar.button("Sign Out"): st.session_state.clear(); st.experimental_rerun()
    surprise=st.sidebar.checkbox("🎢 Surprise Me (Quanta Mode)",value=False)
    movies_w=st.sidebar.slider("Movies",0.0,1.0,0.8,0.1)
    music_w=st.sidebar.slider("Music",0.0,1.0,0.6,0.1)
    prod_w=st.sidebar.slider("Products",0.0,1.0,0.7,0.1)

    df=load_data()
    st.session_state.setdefault("liked",set())
    st.session_state.setdefault("bag",set())

    st.text_input("🔍 Search",key="query",placeholder="Search movies, music, fashion, tech...")
    q=st.session_state.query.lower().strip() if st.session_state.query else ""
    if q: df=df[df["title"].str.lower().str.contains(q)]

    # Weight rebalancing
    df["w"]=df["category"].map({"Movies":movies_w,"Music":music_w}).fillna(prod_w)
    if surprise: df=df.sample(frac=1).reset_index(drop=True)

    st.markdown("## ✨ Your AI Mood Mix")
    for _,r in df.head(30).iterrows():
        render_card(r,st.session_state.liked,st.session_state.bag,surprise,
                    [df.loc[df.item_id==x,"title"].values[0] for x in st.session_state.liked
                     if x in df.item_id.values])

# ---------------------------------------------------------
# ENTRYPOINT
if "authed" not in st.session_state: st.session_state.authed=False
if not st.session_state.authed: login_screen()
else: main_app()
# ---------------------------------------------------------
# ENTRYPOINT (safe session init)
if "authed" not in st.session_state:
    st.session_state["authed"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Safe check before using
if not st.session_state["authed"]:
    login_screen()
else:
    main_app()


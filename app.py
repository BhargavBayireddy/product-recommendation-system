# app.py — ReccoVerse (cinematic AI-powered hybrid recommender)
import os, io, json, time, zipfile, gzip, base64, requests
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from firebase_init import (
    signup_email_password, login_email_password,
    add_interaction, fetch_user_interactions, fetch_global_interactions,
    ensure_user, email_exists, send_phone_otp, verify_phone_otp, FIREBASE_READY
)
from gnn_infer import load_item_embeddings, make_user_vector, recommend_items, diversity_personalization_novelty, cold_start_mmr

APP_NAME="ReccoVerse"
ART=Path("artifacts"); REFRESH_MS=5000; CARD_COLS=5
st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide")

def inject_css():
    st.markdown("""
    <style>
    .hero-wrap{position:relative;height:48vh;min-height:380px;border-radius:28px;overflow:hidden;border:1px solid #1e2738;}
    .hero-video{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;filter:contrast(1.05)brightness(.8);}
    .hero-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;
    background:radial-gradient(900px 420px at 20% 10%,rgba(73,187,255,.10),transparent 60%),
    radial-gradient(900px 420px at 80% 10%,rgba(255,73,146,.08),transparent 60%);}
    .brand{font-size:3.1rem;font-weight:900;letter-spacing:.02em;text-shadow:0 0 18px rgba(73,187,255,.28);}
    .tagline{opacity:.88;margin-top:.35rem;}
    input[type="text"],input[type="password"]{background:rgba(255,255,255,.07)!important;color:#e7e7ea!important;border:1px solid rgba(255,255,255,.25)!important;border-radius:8px!important;padding:.6rem .8rem!important;font-size:1rem!important;}
    input::placeholder{color:rgba(255,255,255,.55);}
    button[kind="primary"],.stButton>button{background:linear-gradient(90deg,#2979ff,#00bfa5)!important;color:white!important;border-radius:999px!important;font-weight:600!important;border:none!important;}
    </style>
    """,unsafe_allow_html=True)
inject_css()

def hero_with_video():
    local=ART/"hero.mp4"
    if local.exists():
        b64=base64.b64encode(local.read_bytes()).decode("utf-8")
        src=f"data:video/mp4;base64,{b64}"
    else:
        src="https://assets.mixkit.co/videos/preview/mixkit-artificial-intelligence-visualization-984-large.mp4"
    st.markdown(f"""
    <div class='hero-wrap'>
      <video class='hero-video' autoplay muted loop playsinline>
        <source src='{src}' type='video/mp4'>
      </video>
      <div class='hero-overlay'>
        <div class='brand'>ReccoVerse</div>
        <div class='tagline'>AI-curated picks across Movies • Music • Products</div>
      </div>
    </div>
    """,unsafe_allow_html=True)

def init_state():
    s=st.session_state
    for k,v in dict(authed=False,uid=None,email=None,items_df=None,embeddings=None,id_to_idx=None,liked=set(),bag=set(),
                   auth_mode="email",otp_sent=False,otp_phone="",otp_hint="").items():
        s.setdefault(k,v)
init_state()

@st.cache_data(show_spinner="Loading datasets…")
def load_multidomain_online():
    frames=[]
    try:
        z=requests.get("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",timeout=12)
        with zipfile.ZipFile(io.BytesIO(z.content)) as f:
            with f.open("ml-latest-small/movies.csv") as c: movies=pd.read_csv(c)
        frames.append(pd.DataFrame({"item_id":"mv_"+movies.movieId.astype(str),
            "title":movies.title,"provider":"Netflix",
            "genre":movies.genres.str.split("|").str[0],
            "image":"https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80&w=1200",
            "text":movies.title+" "+movies.genres}).sample(200))
    except: pass
    try:
        music=pd.read_csv("https://raw.githubusercontent.com/yg397/music-recommender-dataset/master/data.csv").dropna().sample(200)
        frames.append(pd.DataFrame({"item_id":"mu_"+music.artist.astype(str)+"_"+music.track.astype(str),
            "title":music.track,"provider":"Spotify","genre":"Music",
            "image":"https://images.unsplash.com/photo-1511379938547-c1f69419868d?q=80&w=1200",
            "text":music.artist+" "+music.track}))
    except: pass
    try:
        r=requests.get("https://datarepo.s3.amazonaws.com/beauty_5.json.gz",timeout=12)
        import gzip; lines=[json.loads(l) for l in gzip.decompress(r.content).splitlines()[:2000]]
        beauty=pd.DataFrame(lines)
        frames.append(pd.DataFrame({"item_id":"pr_"+beauty.asin.astype(str),"title":beauty.title,
            "provider":"Amazon","genre":"Beauty",
            "image":"https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?q=80&w=1200",
            "text":beauty.title+" "+beauty.get("description","").astype(str)}).dropna().sample(200))
    except: pass
    df=pd.concat(frames,ignore_index=True); df.drop_duplicates("title",inplace=True); return df

def auth_page():
    hero_with_video(); st.write("")
    left,right=st.columns([1,1])
    with left:
        st.markdown("#### Join with")
        c1,c2=st.columns(2)
        if c1.button("📧 Email"): st.session_state.auth_mode="email"
        if c2.button("📱 Mobile (OTP)"): st.session_state.auth_mode="phone"
        st.caption(f"Backend: {'Firebase' if FIREBASE_READY else 'Local mock'}")
    with right:
        if st.session_state.auth_mode=="email":
            email=st.text_input("Email"); pwd=st.text_input("Password",type="password")
            col1,col2=st.columns(2)
            if col1.button("Continue"):
                if email_exists(email):
                    ok,uid=login_email_password(email,pwd)
                    if ok:
                        st.session_state.update(dict(authed=True,uid=uid,email=email)); ensure_user(uid,email=email); st.success("Logged in!"); st.rerun()
                    else: st.error("Incorrect email or password.")
                else: st.warning("Account not found. Please create one.")
            if col2.button("Create Account"):
                ok,uid=signup_email_password(email,pwd)
                if ok:
                    st.session_state.update(dict(authed=True,uid=uid,email=email)); ensure_user(uid,email=email); st.success("Account created!"); st.rerun()
                else: st.error(uid)
        else:
            phone=st.text_input("Mobile number (+91XXXXXXXXXX)")
            otp=st.text_input("Enter OTP")
            c1,c2=st.columns(2)
            if c1.button("Send OTP"):
                ok,msg=send_phone_otp(phone); st.session_state.otp_sent=ok; st.session_state.otp_phone=phone; 
                st.info(f"OTP: {msg}" if msg.isdigit() else msg)
            if c2.button("Verify & Continue"):
                ok,uid=verify_phone_otp(st.session_state.otp_phone,otp)
                if ok:
                    st.session_state.update(dict(authed=True,uid=uid,email=f"{phone}@phone.local")); ensure_user(uid,phone=phone); st.success("Logged in with mobile."); st.rerun()
                else: st.error(uid)

def render_card(row,liked,bagged,uid):
    st.image(row.image,use_column_width=True)
    st.markdown(f"**{row.title}**  \n_{row.provider} • {row.genre}_")
    c1,c2=st.columns(2)
    if c1.button("❤️" if liked else "♡ Like",key=f"l{row.item_id}"):
        add_interaction(uid,row.item_id,"like" if not liked else "unlike")
        (st.session_state.liked.add if not liked else st.session_state.liked.discard)(row.item_id); st.rerun()
    if c2.button("👜" if bagged else "➕ Bag",key=f"b{row.item_id}"):
        add_interaction(uid,row.item_id,"bag" if not bagged else "remove_bag")
        (st.session_state.bag.add if not bagged else st.session_state.bag.discard)(row.item_id); st.rerun()

def section(title,df,ids,uid):
    st.markdown(f"### {title}"); cols=st.columns(CARD_COLS)
    for i,iid in enumerate(ids[:CARD_COLS*2+5]):
        row=df[df.item_id==iid]
        if not row.empty: render_card(row.iloc[0],iid in st.session_state.liked,iid in st.session_state.bag,uid)

def ensure_embs():
    if st.session_state.items_df is None:
        df=load_multidomain_online(); items,embs,idmap,A=load_item_embeddings(df,ART)
        st.session_state.update(dict(items_df=items,embeddings=embs,id_to_idx=idmap,A=A))

def home(uid):
    st.markdown(f"## {APP_NAME}"); ensure_embs()
    df,embs,idmap,A=st.session_state.items_df,st.session_state.embeddings,st.session_state.id_to_idx,st.session_state.A
    inter=fetch_user_interactions(uid)
    st.session_state.liked={x["item_id"] for x in inter if x["action"]=="like"}
    st.session_state.bag={x["item_id"] for x in inter if x["action"]=="bag"}
    userv=make_user_vector(st.session_state.liked,st.session_state.bag,idmap,embs)
    top=recommend_items(userv,embs,df,exclude=set(st.session_state.liked),topk=15,A=A)
    crowd=fetch_global_interactions()
    ppl=recommend_items(userv,embs,df,topk=12,A=A,crowd=crowd)
    sim=recommend_items(userv,embs,df,topk=12,A=A,force_content=True)
    cold=cold_start_mmr(df,embs,0.65,12)
    section("Top Picks For You",df,top,uid); section("People Like You Also Liked",df,ppl,uid)
    section("Because You Liked Similar Items",df,sim,uid); section("Explore Something Different",df,cold,uid)
    st.autorefresh(interval=REFRESH_MS)

def liked(uid): st.markdown("## ❤️ Liked Items"); ensure_embs(); section("Your Likes",st.session_state.items_df,list(st.session_state.liked),uid)
def bag(uid): st.markdown("## 👜 Your Bag"); ensure_embs(); section("Saved Items",st.session_state.items_df,list(st.session_state.bag),uid)

def navbar():
    st.sidebar.markdown("### ReccoVerse")
    page=st.sidebar.radio("Navigate",["Home","Liked Items","Bag"])
    if st.sidebar.button("Sign Out"):
        for k in ["authed","uid","email"]: st.session_state[k]=None
        st.session_state.liked.clear(); st.session_state.bag.clear(); st.session_state.authed=False; st.rerun()
    return page

def main():
    if not st.session_state.authed: auth_page(); return
    p=navbar(); uid=st.session_state.uid
    {"Home":home,"Liked Items":liked,"Bag":bag}[p](uid)

if __name__=="__main__": main()

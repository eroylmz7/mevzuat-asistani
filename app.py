import streamlit as st
import datetime
import pytz
import time
import os
from supabase import create_client, Client
from data_ingestion import process_pdfs
from generation import generate_answer

# --- 1. AYARLAR VE TASARIM ---
st.set_page_config(page_title="Mevzuat Asistanı", page_icon="🎓", layout="wide")

# Modern Tasarım CSS
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f9; }
    .user-card { text-align: center; padding: 1rem; background: #1E3A8A; color: white; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERİTABANI VE YARDIMCILAR ---
@st.cache_resource
def get_supabase_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = get_supabase_client()

def get_tr_time():
    return datetime.datetime.now(pytz.timezone('Europe/Istanbul'))

def daktilo_efekti(metin, alan=None):
    if alan is None: alan = st.empty()
    gecici_metin = ""
    for harf in metin:
        gecici_metin += harf
        alan.markdown(gecici_metin + "▌")
        time.sleep(0.01)
    alan.markdown(gecici_metin)

# --- 3. OTURUM YÖNETİMİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "vector_db" not in st.session_state:
    # Başlangıçta varsa yükle (Open files hatasını önlemek için cache'li yüklenebilir)
    st.session_state.vector_db = None

# --- 4. GİRİŞ EKRANI (Supabase) ---
if not st.session_state.logged_in:
    _, col_login, _ = st.columns([1, 1.2, 1])
    with col_login:
        st.markdown("<h1 style='text-align: center;'>🎓 Mevzuat Sistemi Giriş</h1>", unsafe_allow_html=True)
        with st.form("login"):
            u = st.text_input("Kullanıcı Adı")
            p = st.text_input("Şifre", type="password")
            if st.form_submit_button("Giriş Yap", type="primary"):
                res = supabase.table("kullanicilar").select("*").eq("username", u).eq("password", p).execute()
                if res.data:
                    st.session_state.logged_in = True
                    st.session_state.username = res.data[0]['username']
                    st.session_state.role = res.data[0]['role']
                    st.rerun()
                else:
                    st.error("Giriş başarısız!")
    st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown(f"<div class='user-card'><h3>{st.session_state.username.upper()}</h3></div>", unsafe_allow_html=True)
    
    st.subheader("📁 Veri Yönetimi")
    uploaded_files = st.file_uploader("PDF Yükleyin", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Veritabanını Güncelle", type="primary"):
        if uploaded_files:
            durum = st.empty()
            with st.spinner("İşleniyor..."):
                durum.info("📑 PDF'ler işleniyor...")
                st.session_state.vector_db = process_pdfs(uploaded_files)
                durum.success("✅ Veritabanı güncellendi!")
        else:
            st.warning("Lütfen dosya yükleyin.")

    st.divider()
    
    # Sohbet İndirme
    if st.session_state.messages:
        tr_saat = get_tr_time()
        log = f"🎓 SOHBET KAYDI - {tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*40 + "\n\n"
        for m in st.session_state.messages:
            log += f"[{m['role'].upper()}]: {m['content']}\n\n"
        st.download_button("📥 Sohbeti İndir", log, file_name=f"sohbet_{tr_saat.strftime('%H%M')}.txt")

    if st.button("🚪 Çıkış Yap"):
        st.session_state.logged_in = False
        st.rerun()

# --- 6. ANA EKRAN ---
st.title("🎓 Kampüs Mevzuat Asistanı")
tab1, tab2 = st.tabs(["💬 Sohbet", "📊 Analiz"])

with tab1:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.vector_db is not None:
                with st.spinner("Mevzuat taranıyor..."):
                    response = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
                    daktilo_efekti(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Lütfen önce sol menüden PDF yükleyip veritabanını güncelleyin.")

with tab2:
    st.subheader("📑 Doküman Analizi")
    if uploaded_files:
        st.write(f"Aktif Doküman Sayısı: {len(uploaded_files)}")
        for f in uploaded_files:
            st.write(f"- {f.name}")
    else:
        st.info("Henüz doküman yüklenmedi.")
import streamlit as st
import datetime
import pytz
import time
import os
from supabase import create_client, Client

# --- 1. KENDİ MODÜLLERİNİ İMPORT ET ---
try:
    from data_ingestion import process_pdfs 
    # generation.py içindeki generate_answer fonksiyonun hem yanıtı hem kaynakları dönmeli
    from generation import generate_answer 
except ImportError:
    st.error("Kritik Hata: Modüller (data_ingestion veya generation) bulunamadı!")

# --- 2. TASARIM VE SAYFA AYARLARI ---
st.set_page_config(page_title="Kampüs Mevzuat Asistanı", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f4f7f9; }
    .stTabs [aria-selected="true"] { background-color: #1E3A8A; color: white !important; font-weight: bold; }
    .user-card { text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1E3A8A, #3B82F6); color: white; border-radius: 12px; margin-bottom: 15px; }
    .stChatMessage { border-radius: 15px; padding: 12px; margin-bottom: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. VERİTABANI VE ZAMAN YÖNETİMİ ---
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
        time.sleep(0.005) # Hızlı daktilo efekti
    alan.markdown(gecici_metin)

# --- 4. OTURUM VE AUTH YÖNETİMİ ---
if "messages" not in st.session_state: st.session_state.messages = [] # Hafıza
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "auth_mode" not in st.session_state: st.session_state.auth_mode = "login"

# --- GİRİŞ VE KAYIT EKRANI (Supabase Entegre) ---
if not st.session_state.logged_in:
    _, col_auth, _ = st.columns([1, 1.2, 1])
    with col_auth:
        if st.session_state.auth_mode == "login":
            st.markdown("<h1 style='text-align: center;'>🎓 Giriş Yap</h1>", unsafe_allow_html=True)
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
                    else: st.error("❌ Kullanıcı adı veya şifre hatalı!")
            if st.button("Hesabın yok mu? Kayıt Ol"):
                st.session_state.auth_mode = "signup"
                st.rerun()
        else: # KAYIT MODU
            st.markdown("<h1 style='text-align: center;'>📝 Yeni Kayıt</h1>", unsafe_allow_html=True)
            with st.form("signup"):
                nu = st.text_input("Kullanıcı Adı")
                np = st.text_input("Şifre", type="password")
                if st.form_submit_button("Kaydı Tamamla"):
                    try:
                        supabase.table("kullanicilar").insert({"username": nu, "password": np, "role": "student"}).execute()
                        st.success("Kayıt başarılı! Giriş ekranına yönlendiriliyorsunuz.")
                        time.sleep(1.5)
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    except: st.error("Bu kullanıcı adı zaten alınmış.")
            if st.button("Giriş ekranına dön"):
                st.session_state.auth_mode = "login"
                st.rerun()
    st.stop()

# --- 5. SIDEBAR (KONTROL PANELİ) ---
with st.sidebar:
    st.markdown(f"<div class='user-card'><h3>{st.session_state.username.upper()}</h3><small>{st.session_state.role.upper()} YETKİSİ</small></div>", unsafe_allow_html=True)
    
    st.subheader("📁 Veri Yönetimi")
    uploaded_files = st.file_uploader("PDF Yükleyin", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Veritabanını Güncelle", type="primary"):
        if uploaded_files:
            durum = st.status("Mevzuat analiz ediliyor...", expanded=True)
            durum.write("📄 PDF içerikleri okunuyor...")
            st.session_state.vector_db = process_pdfs(uploaded_files)
            durum.write("🧠 Gemini 2.5 Flash tabanlı vektör hafızası güncelleniyor...")
            durum.update(label="✅ Veritabanı Güncellendi!", state="complete")
        else: st.warning("Lütfen dosya seçin.")

    st.divider()

    # SOHBET İNDİRME
    if st.session_state.messages:
        tr_now = get_tr_time()
        log = f"🎓 MEVZUAT ASİSTANI KAYDI - {tr_now.strftime('%d.%m.%Y %H:%M')}\n" + "="*45 + "\n\n"
        for m in st.session_state.messages:
            label = "ASİSTAN" if m["role"] == "assistant" else "ÖĞRENCİ"
            log += f"[{label}]: {m['content']}\n\n"
        st.download_button("📥 Sohbeti İndir (.txt)", log, file_name=f"kayit_{tr_now.strftime('%H%M')}.txt", use_container_width=True)

    if st.button("🚪 Güvenli Çıkış"):
        st.session_state.logged_in = False
        st.rerun()

# --- 6. ANA PANEL (SOHBET VE ANALİZ) ---
st.title("💬 Kampüs Mevzuat Sorgulama")
tab_chat, tab_analiz = st.tabs(["💬 Akıllı Sohbet", "📊 Doküman Analizi"])

with tab_chat:
    # Eski mesajları bas (Memory)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Mevzuat hakkında sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini 2.5 Flash taranıyor..."):
                # Yanıtı ve Kaynakları Üret
                # generate_answer fonksiyonuna k=50 ayarını generation.py içinde verdik
                result = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
                
                # Cevabı daktilo ile yaz
                daktilo_efekti(result["answer"])
                
                # Kaynakları Listele
                if result.get("sources"):
                    source_box = "\n\n📚 **Kaynaklar:**\n" + "\n".join([f"- {s}" for s in result["sources"]])
                    st.markdown(source_box)
                    full_resp = result["answer"] + source_box
                else: full_resp = result["answer"]
                
                st.session_state.messages.append({"role": "assistant", "content": full_resp})

with tab_analiz:
    st.subheader("📑 Mevcut Yönetmelik Analizi")
    if uploaded_files:
        st.info(f"Sistemde şu an {len(uploaded_files)} adet doküman taranabilir durumda.")
        for f in uploaded_files: st.write(f"✅ {f.name}")
    else: st.warning("Henüz doküman yüklenmedi.")
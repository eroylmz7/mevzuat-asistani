import streamlit as st
import datetime
import pytz
import time
import os
from supabase import create_client, Client

# --- KENDİ MODÜLLERİNİ İMPORT ET ---
try:
    from data_ingestion import process_pdfs 
    from generation import generate_answer 
except ImportError:
    st.error("⚠️ Hata: data_ingestion.py veya generation.py bulunamadı.")

# --- 1. SAYFA VE TEMA AYARLARI (KRİTİK KISIM) ---
st.set_page_config(
    page_title="Kampüs Mevzuat Asistanı", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# BURASI ARKA PLANI VE RENKLERİ DÜZELTİR
st.markdown("""
    <style>
    /* 1. Tüm Arka Planı Koyu Yap */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* 2. Sidebar (Sol Menü) Rengi */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* 3. Giriş Ekranı ve Kartlar */
    .user-card, .login-container {
        background: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* 4. Input Alanları (Giriş kutuları) */
    .stTextInput > div > div > input {
        background-color: #111827; 
        color: white;
        border: 1px solid #374151;
    }
    
    /* 5. Tab (Sekme) Tasarımı */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937;
        color: #9ca3af;
        border-radius: 8px 8px 0 0;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* 6. Butonlar */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERİTABANI VE ARAÇLAR ---
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
        time.sleep(0.003) 
    alan.markdown(gecici_metin)

# --- 3. OTURUM YÖNETİMİ ---
if "messages" not in st.session_state: st.session_state.messages = []
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "auth_mode" not in st.session_state: st.session_state.auth_mode = "login"

# --- 4. GİRİŞ VE KAYIT EKRANI (KOYU MOD UYUMLU) ---
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True) # Üstten boşluk
    _, col_login, _ = st.columns([1, 1.5, 1])
    
    with col_login:
        st.markdown("<h1 style='text-align: center;'>🎓 Mevzuat Giriş</h1>", unsafe_allow_html=True)
        
        # Giriş Formu Konteynerı
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            if st.session_state.auth_mode == "login":
                u = st.text_input("Kullanıcı Adı")
                p = st.text_input("Şifre", type="password")
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    if st.button("Giriş Yap", use_container_width=True):
                        res = supabase.table("kullanicilar").select("*").eq("username", u).eq("password", p).execute()
                        if res.data:
                            st.session_state.logged_in = True
                            st.session_state.username = res.data[0]['username']
                            st.session_state.role = res.data[0]['role']
                            st.rerun()
                        else: st.error("❌ Hatalı bilgiler!")
                with col_btn2:
                    if st.button("Kayıt Ol", use_container_width=True):
                        st.session_state.auth_mode = "signup"
                        st.rerun()

            else: # KAYIT MODU
                st.subheader("📝 Yeni Hesap Oluştur")
                nu = st.text_input("Belirleyeceğiniz Kullanıcı Adı")
                np = st.text_input("Şifreniz", type="password")
                
                if st.button("Kaydı Tamamla", use_container_width=True):
                    try:
                        supabase.table("kullanicilar").insert({"username": nu, "password": np, "role": "student"}).execute()
                        st.success("Kayıt başarılı! Giriş yapabilirsiniz.")
                        time.sleep(1.5)
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    except: st.error("Bu kullanıcı adı zaten alınmış!")
                
                if st.button("Geri Dön"):
                    st.session_state.auth_mode = "login"
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
        <div class="user-card">
            <h3>👤 {st.session_state.username.upper()}</h3>
            <p style='color:#9ca3af; margin:0;'>{st.session_state.role.capitalize()} Yetkisi</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📂 Veri Yönetimi")
    uploaded_files = st.file_uploader("PDF Dosyalarını Seçin", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Veritabanını Güncelle", type="primary"):
        if uploaded_files:
            durum = st.status("Analiz başlatıldı...", expanded=True)
            durum.write("📄 PDF içerikleri taranıyor...")
            st.session_state.vector_db = process_pdfs(uploaded_files)
            durum.write("🧠 Yapay zeka hafızası güncelleniyor...")
            durum.update(label="✅ Hazır! Sorularınızı sorabilirsiniz.", state="complete")
        else: st.warning("Lütfen dosya yükleyin.")

    st.divider()

    if st.session_state.messages:
        tr_now = get_tr_time()
        log = f"🎓 SOHBET KAYDI - {tr_now.strftime('%d.%m.%Y %H:%M')}\n" + "="*40 + "\n\n"
        for m in st.session_state.messages:
            log += f"[{m['role'].upper()}]: {m['content']}\n\n"
        st.download_button("📥 Sohbeti İndir", log, file_name=f"chat_{tr_now.strftime('%H%M')}.txt", use_container_width=True)

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

    if prompt := st.chat_input("Mevzuat hakkında merak ettiklerinizi sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini 2.5 Flash mevzuatı tarıyor..."):
                # generation.py'den gelen fonksiyon (hem cevap hem kaynak döner)
                sonuc = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
                
                # Sadece cevabı daktilo efektiyle yaz
                daktilo_efekti(sonuc["answer"])
                
                # Kaynakları göster
                if sonuc.get("sources"):
                    kaynak_metni = "\n\n📚 **Kaynaklar:**\n" + "\n".join([f"- {k}" for k in sonuc["sources"]])
                    st.markdown(kaynak_metni)
                    tam_cevap = sonuc["answer"] + kaynak_metni
                else:
                    tam_cevap = sonuc["answer"]
                
                st.session_state.messages.append({"role": "assistant", "content": tam_cevap})

with tab2:
    st.subheader("📑 Yüklenen Dokümanlar")
    if uploaded_files:
        st.info(f"{len(uploaded_files)} adet doküman analiz edildi.")
        for f in uploaded_files: st.write(f"✅ {f.name}")
    else: st.warning("Doküman bulunamadı.")
import streamlit as st
import datetime
import pytz
import os
from supabase import create_client, Client

# --- 1. SAYFA VE GÖRSEL AYARLAR ---
st.set_page_config(
    page_title="Mevzuat Asistanı", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Profesyonel Tasarım için CSS
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .main { background-color: #f8f9fa; }
    
    /* Giriş Kartı Tasarımı */
    .login-container {
        padding: 2rem;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buton Tasarımları */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        transition: all 0.3s;
    }
    
    /* Sidebar Profil Alanı */
    .user-profile {
        text-align: center;
        padding: 1rem;
        background: #1E3A8A;
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERİTABANI VE ZAMAN YÖNETİMİ ---
@st.cache_resource
def get_supabase_client():
    """Bağlantıyı bir kez açar, kaynak tüketimini (Too many files hatası) önler."""
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = get_supabase_client()

def login_user(username, password):
    """Kullanıcıyı Supabase üzerinden sorgular."""
    try:
        res = supabase.table("kullanicilar").select("*").eq("username", username).eq("password", password).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None

def get_tr_time():
    """Sunucu nerede olursa olsun Türkiye saatini döner."""
    return datetime.datetime.now(pytz.timezone('Europe/Istanbul'))

# --- 3. OTURUM VE GİRİŞ KONTROLÜ ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- GİRİŞ EKRANI ---
if not st.session_state.logged_in:
    _, col_mid, _ = st.columns([1, 1.2, 1])
    with col_mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<h1 style='text-align: center;'>🎓 Mevzuat Asistanı</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: gray;'>Devlet ve Üniversite Mevzuat Sorgulama Sistemi</p>", unsafe_allow_html=True)
            
            with st.form("login_form"):
                u = st.text_input("Kullanıcı Adı", placeholder="admin")
                p = st.text_input("Şifre", type="password", placeholder="••••••")
                submit = st.form_submit_button("Sisteme Giriş Yap")
                
                if submit:
                    user_data = login_user(u, p)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.username = user_data['username']
                        st.session_state.role = user_data['role']
                        st.rerun()
                    else:
                        st.error("⚠️ Kullanıcı adı veya şifre hatalı!")
    st.stop()

# --- 4. ANA PANEL (SIDEBAR) ---
with st.sidebar:
    # Kullanıcı Kartı
    st.markdown(f"""
        <div class="user-profile">
            <h3 style='margin:0;'>{st.session_state.username.upper()}</h3>
            <small>{st.session_state.role.upper()} YETKİSİ</small>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # PDF Yükleme Alanı
    st.subheader("📁 Veri Yönetimi")
    uploaded_files = st.file_uploader("Mevzuat PDF'lerini Yükleyin", accept_multiple_files=True, type=['pdf'])
    
    # PDF İşleme Butonu (Senin mevcut PDF fonksiyonuna bağla)
    if st.button("Veritabanını Güncelle", type="primary"):
        with st.spinner("Mevzuat analiz ediliyor..."):
            # BURAYA: data_ingestion.py içindeki fonksiyonunu çağır
            # Örn: process_pdfs(uploaded_files)
            st.success("Veritabanı güncellendi!")

    st.divider()

    # Sohbet İndirme Bölümü (Gelişmiş Versiyon)
    tr_now = get_tr_time()
    if len(st.session_state.messages) > 0:
        log_content = f"🎓 MEVZUAT ASİSTANI SOHBET KAYDI\nTarih: {tr_now.strftime('%d.%m.%Y %H:%M')}\n" + "="*40 + "\n\n"
        for m in st.session_state.messages:
            label = "ASİSTAN" if m["role"] == "assistant" else "ÖĞRENCİ"
            log_content += f"[{label}]: {m['content']}\n{'-'*20}\n"
        
        st.download_button(
            label="📥 Sohbet Geçmişini İndir",
            data=log_content,
            file_name=f"Mevzuat_Kayit_{tr_now.strftime('%d_%m_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Çıkış Butonu
    if st.button("🚪 Güvenli Çıkış", type="secondary"):
        st.session_state.logged_in = False
        st.rerun()

# --- 5. ANA PANEL (SOHBET ARAYÜZÜ) ---
st.title("💬 Mevzuat Sorgulama Paneli")
st.write(f"Hoş geldin, **{st.session_state.username}**. Mevzuat hakkında her şeyi sorabilirsin.")

# Mesaj Geçmişini Görüntüle
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı Sorgu Girişi
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Kullanıcı mesajını ekrana bas
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan Yanıtı (Senin RAG/LLM fonksiyonuna bağla)
    with st.chat_message("assistant"):
        with st.spinner("Mevzuat taranıyor..."):
            # BURAYA: generation.py içindeki asistan yanıt fonksiyonunu bağla
            # Örn: response = generate_answer(prompt)
            response = "Bu bir örnek yanıttır. Lütfen LLM fonksiyonunuzu buraya bağlayın." 
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
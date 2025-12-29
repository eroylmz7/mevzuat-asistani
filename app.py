import streamlit as st
import datetime
import pytz
import time
from supabase import create_client

# Modülleri yükle
try:
    from data_ingestion import process_pdfs 
    from generation import generate_answer 
except ImportError:
    st.error("⚠️ Modüller yüklenemedi!")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kampüs Mevzuat Asistanı", page_icon="🎓", layout="wide")

# CSS DÜZELTMELERİ (Chat Bar Aşağıda, Renkler Düzgün)
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    
    /* Sekme (Tab) Tasarımı */
    .stTabs [data-baseweb="tab-list"] button {
        flex: 1; /* Sekmeleri eşit genişlikte yap */
        background-color: #1f2937;
        color: white;
        border-radius: 5px;
        margin: 2px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        font-weight: bold;
    }
    
    /* Kullanıcı Kartı */
    .user-card {
        padding: 15px;
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Chat Input'u aşağı sabitleme (Streamlit default ama garanti olsun) */
    .stChatInput { position: fixed; bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- VERİTABANI ---
@st.cache_resource
def get_supabase_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = get_supabase_client()

def get_tr_time():
    return datetime.datetime.now(pytz.timezone('Europe/Istanbul'))

def daktilo_efekti(metin):
    alan = st.empty()
    gecici = ""
    for h in metin:
        gecici += h
        alan.markdown(gecici + "▌")
        time.sleep(0.003)
    alan.markdown(gecici)

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "sorgu_sayaci" not in st.session_state: st.session_state.sorgu_sayaci = 0 # Admin analizi için

# --- GİRİŞ EKRANI (TAB YAPISI - İSTEK 4) ---
if not st.session_state.logged_in:
    st.markdown("<br><br><h1 style='text-align: center;'>🎓 Kampüs Asistanı</h1>", unsafe_allow_html=True)
    
    _, col_main, _ = st.columns([1, 1.5, 1])
    with col_main:
        # İki Sekmeli Yapı: Giriş Yap | Kayıt Ol
        tab_login, tab_signup = st.tabs(["🔑 Giriş Yap", "📝 Kayıt Ol"])
        
        with tab_login:
            with st.form("login_form"):
                u = st.text_input("Kullanıcı Adı")
                p = st.text_input("Şifre", type="password")
                if st.form_submit_button("Giriş Yap", type="primary"):
                    res = supabase.table("kullanicilar").select("*").eq("username", u).eq("password", p).execute()
                    if res.data:
                        st.session_state.logged_in = True
                        st.session_state.username = res.data[0]['username']
                        st.session_state.role = res.data[0]['role']
                        st.rerun()
                    else: st.error("Bilgiler hatalı!")

        with tab_signup:
            with st.form("signup_form"):
                new_u = st.text_input("Belirleyeceğiniz Kullanıcı Adı")
                new_p = st.text_input("Yeni Şifre", type="password")
                if st.form_submit_button("Kayıt Ol"):
                    try:
                        supabase.table("kullanicilar").insert({"username": new_u, "password": new_p, "role": "student"}).execute()
                        st.success("Kayıt Başarılı! 'Giriş Yap' sekmesinden girebilirsiniz.")
                    except: st.error("Bu kullanıcı adı dolu.")
    st.stop()

# --- SIDEBAR (ANALİZ VE YÖNETİM - İSTEK 3) ---
with st.sidebar:
    # Kullanıcı Bilgisi
    st.markdown(f"""
        <div class="user-card">
            <h3>{st.session_state.username.upper()}</h3>
            <small>{st.session_state.role.upper()} HESABI</small>
        </div>
    """, unsafe_allow_html=True)

    # Sadece ADMİN Analizleri Görür
    if st.session_state.role == 'admin':
        st.subheader("📊 Sistem Analizi")
        st.info(f"Toplam Sorgu: {st.session_state.sorgu_sayaci}")
        st.markdown("**Son Merak Edilenler:**")
        # Gerçek bir veritabanı tablosu olsaydı buradan çekerdik
        # Şimdilik session içindeki son soruları gösterelim
        if len(st.session_state.messages) > 0:
            son_sorular = [m['content'] for m in st.session_state.messages if m['role'] == 'user'][-3:]
            for s in son_sorular:
                st.caption(f"🔹 {s[:40]}...")
        st.divider()

    # PDF Yükleme (Herkes veya sadece Admin)
    st.subheader("📁 Veri Tabanı")
    uploaded_files = st.file_uploader("PDF Ekle", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Veritabanını Güncelle", type="primary"):
        if uploaded_files:
            durum = st.status("İşleniyor...", expanded=True)
            st.session_state.vector_db = process_pdfs(uploaded_files)
            durum.update(label="✅ Veritabanı Güncel!", state="complete")
    
    st.divider()
    if st.button("Çıkış Yap"):
        st.session_state.logged_in = False
        st.rerun()

# --- ANA SOHBET EKRANI (İSTEK 1 & 2 & 5) ---
st.title("💬 Mevzuat Asistanı")

# Mesajları Göster
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input (En altta sabit)
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # 1. Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.sorgu_sayaci += 1 # Analiz sayacını artır
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Asistan cevabı
    with st.chat_message("assistant"):
        with st.spinner("Araştırılıyor..."):
            sonuc = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
            
            # Cevabı yaz
            daktilo_efekti(sonuc["answer"])
            
            # Kaynakları Şık Göster (İstek 2)
            if sonuc["sources"]:
                st.markdown("---")
                st.caption("📚 **Referans Dokümanlar:**")
                # Her kaynağı yan yana etiket gibi göstermek için columns
                cols = st.columns(len(sonuc["sources"]))
                for idx, src in enumerate(sonuc["sources"]):
                    # Dosya adı ve sayfa numarasını temiz göster
                    # Örn: lisans_yonetmeligi.pdf (Sayfa 5)
                    st.success(f"📄 {src}")
            
            # Tam cevabı hafızaya kaydet
            full_resp = sonuc["answer"]
            if sonuc["sources"]:
                full_resp += "\n\n📚 Kaynaklar:\n" + "\n".join(sonuc["sources"])
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
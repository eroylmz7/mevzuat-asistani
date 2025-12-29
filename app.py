import streamlit as st
import datetime
import pytz
import time
from supabase import create_client

# --- MODÜLLER ---
try:
    from data_ingestion import process_pdfs 
    from generation import generate_answer 
except ImportError:
    st.error("⚠️ Modüller yüklenemedi! (data_ingestion.py veya generation.py eksik)")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kampüs Mevzuat Asistanı", page_icon="🎓", layout="wide")

# --- CSS TASARIMI ---
st.markdown("""
    <style>
    /* Genel Koyu Tema */
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    
    /* Kullanıcı Kartı */
    .user-card {
        padding: 15px;
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Analiz Kutusu (Sidebar İçi) */
    .stats-box {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* Dikey Kaynak Kutucukları (İstek 2) */
    .source-item {
        display: block; /* Alt alta dizilmesi için */
        background-color: #1f2937;
        color: #d1d5db;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9em;
        margin-bottom: 6px; /* Kutular arası boşluk */
        border-left: 4px solid #3b82f6; /* Sol tarafa mavi çizgi */
    }
    
    /* Butonlar */
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- YARDIMCI FONKSİYONLAR ---
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

# --- STATE YÖNETİMİ ---
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Kampüs mevzuatı, dersler veya yönetmelikler hakkında ne öğrenmek istersiniz?"}]

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "sorgu_sayaci" not in st.session_state: st.session_state.sorgu_sayaci = 0
if "analiz_acik" not in st.session_state: st.session_state.analiz_acik = False # Analiz kutusu durumu

# --- GİRİŞ EKRANI ---
if not st.session_state.logged_in:
    st.markdown("<br><br><h1 style='text-align: center;'>🎓 Kampüs Asistanı</h1>", unsafe_allow_html=True)
    
    _, col_main, _ = st.columns([1, 1.5, 1])
    with col_main:
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
                        st.success("Kayıt Başarılı! Giriş yapabilirsiniz.")
                    except: st.error("Bu kullanıcı adı dolu.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
        <div class="user-card">
            <h3>{st.session_state.username.upper()}</h3>
            <small>{st.session_state.role.upper()} HESABI</small>
        </div>
    """, unsafe_allow_html=True)

    # 1. ANALİZ BUTONU (İstek 1: Sidebar içinde açılır/kapanır yapı)
    if st.session_state.role == 'admin':
        if st.button("📊 Analizi Gör / Gizle"):
            st.session_state.analiz_acik = not st.session_state.analiz_acik
        
        if st.session_state.analiz_acik:
            st.markdown("""
                <div class="stats-box">
                    <h4 style="margin:0; color:#3b82f6;">Sistem Özeti</h4>
                    <hr style="margin:5px 0; border-color:#374151;">
            """, unsafe_allow_html=True)
            st.write(f"🔹 **Toplam Sorgu:** {st.session_state.sorgu_sayaci}")
            st.write(f"🔹 **Mesaj Sayısı:** {len(st.session_state.messages)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.divider()

    # 2. PDF YÜKLEME
    st.subheader("📁 Veri Tabanı")
    uploaded_files = st.file_uploader("PDF Ekle", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Veritabanını Güncelle", type="primary"):
        if uploaded_files:
            durum = st.status("İşleniyor...", expanded=True)
            st.session_state.vector_db = process_pdfs(uploaded_files)
            durum.update(label="✅ Veritabanı Güncel!", state="complete")
    
    st.divider()

    # 3. İNDİR VE TEMİZLE
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.messages:
            tr_saat = get_tr_time()
            log = f"🎓 SOHBET KAYDI\n{tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n"
            for m in st.session_state.messages:
                log += f"[{m['role'].upper()}]: {m['content']}\n"
            
            st.download_button(
                label="📥 İndir",
                data=log,
                file_name=f"sohbet_{tr_saat.strftime('%H%M')}.txt",
                mime="text/plain"
            )
    with c2:
        if st.button("🗑️ Temizle"):
            st.session_state.messages = [{"role": "assistant", "content": "Sohbet temizlendi. Nasıl yardımcı olabilirim?"}]
            st.session_state.sorgu_sayaci = 0
            st.rerun()

    if st.button("🚪 Çıkış Yap"):
        st.session_state.logged_in = False
        st.rerun()

# --- ANA SOHBET EKRANI ---
st.title("💬 Mevzuat Asistanı")

# Mesajları Listele
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.sorgu_sayaci += 1
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Araştırılıyor..."):
            sonuc = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
            
            daktilo_efekti(sonuc["answer"])
            
            # KAYNAKLARI ALT ALTA GÖSTER (İstek 2)
            if sonuc["sources"]:
                st.markdown("---")
                st.caption("📚 **Referans Kaynaklar:**")
                
                # HTML ile alt alta kutucuklar
                html_sources = ""
                for src in sonuc["sources"]:
                    # Her kaynak bir 'source-item' div'i içinde
                    html_sources += f'<div class="source-item">📄 {src}</div>'
                
                st.markdown(html_sources, unsafe_allow_html=True)
            
            # Hafızaya kaydet
            full_resp = sonuc["answer"]
            if sonuc["sources"]:
                full_resp += "\n\nKaynaklar:\n" + "\n".join(sonuc["sources"])
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
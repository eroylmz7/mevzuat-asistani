import streamlit as st
import datetime
import pytz
import time
from collections import Counter
import re
from supabase import create_client

# --- MODÜLLER ---
try:
    from data_ingestion import process_pdfs 
    from generation import generate_answer 
except ImportError:
    st.error("⚠️ Modüller yüklenemedi! (data_ingestion.py veya generation.py eksik)")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kampüs Mevzuat Asistanı", page_icon="🎓", layout="wide")

# --- PROFESYONEL CSS TASARIMI ---
st.markdown("""
    <style>
    /* 1. GENEL KOYU TEMA */
    .stApp { background-color: #0f172a; color: #f8fafc; } /* Daha modern bir koyu lacivert/gri */
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    
    /* 2. KULLANICI KARTI (Sidebar) */
    .user-card {
        padding: 20px;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* 3. BUTON TASARIMLARI (KRİTİK DÜZELTME) */
    /* Tüm butonları varsayılan olarak modern hale getir */
    .stButton > button {
        width: 100%;
        background-color: #3b82f6; /* Canlı Mavi */
        color: white !important;
        border: none;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hover (Üzerine gelince) Efekti */
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px); /* Hafif yukarı kalkma */
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    /* Form Gönder Butonları (Giriş/Kayıt) için özel stil */
    div[data-testid="stForm"] .stButton > button {
        background: linear-gradient(to right, #3b82f6, #6366f1);
        height: 3rem; /* Daha büyük tıklama alanı */
    }

    /* Çıkış ve Temizle Butonları (Kırmızı Tonlar) */
    /* CSS ile belirli butonları hedeflemek zor olduğu için Python tarafında type='primary' kullanacağız 
       ve buradaki genel stil onları ezecek ama renkleri aşağıda özelleştiriyoruz. */
       
    /* 4. ANALİZ KUTUSU */
    .stats-box {
        background-color: #334155;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #475569;
        margin: 10px 0;
    }
    
    /* 5. KAYNAK KUTUCUKLARI */
    .source-item {
        display: block;
        background-color: #334155;
        color: #e2e8f0;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.95em;
        margin-bottom: 8px;
        border-left: 5px solid #60a5fa;
        transition: background 0.2s;
    }
    .source-item:hover {
        background-color: #475569;
    }

    /* 6. GİRİŞ FORMU ALANLARI */
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: white;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 10px;
    }
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

def analiz_raporu_olustur():
    tr_now = get_tr_time()
    user_msgs = [m['content'] for m in st.session_state.messages if m['role'] == 'user']
    rapor = f"📊 SİSTEM ANALİZ RAPORU\nTarih: {tr_now.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n\n"
    rapor += f"🔹 Toplam Sorgu: {st.session_state.sorgu_sayaci}\n"
    rapor += f"🔹 Oturum Mesajı: {len(user_msgs)}\n\nSON SORGULAR:\n"
    for msg in user_msgs[-5:]: rapor += f"- {msg}\n"
    return rapor

def konu_analizi_yap():
    text = " ".join([m['content'] for m in st.session_state.messages if m['role'] == 'user']).lower()
    kelimeler = re.findall(r'\w+', text)
    return Counter([k for k in kelimeler if len(k) > 4]).most_common(5)

# --- STATE YÖNETİMİ ---
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Kampüs mevzuatı hakkında size nasıl yardımcı olabilirim?"}]
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "sorgu_sayaci" not in st.session_state: st.session_state.sorgu_sayaci = 0
if "analiz_acik" not in st.session_state: st.session_state.analiz_acik = False
if "view_mode" not in st.session_state: st.session_state.view_mode = "chat"

# --- GİRİŞ EKRANI (Solid Butonlar) ---
if not st.session_state.logged_in:
    st.markdown("<br><br><h1 style='text-align: center; color: white;'>🎓 Kampüs Asistanı</h1>", unsafe_allow_html=True)
    _, col_main, _ = st.columns([1, 1.5, 1])
    with col_main:
        # Tab'lerin arka planını düzeltmek için container
        with st.container():
            tab_login, tab_signup = st.tabs(["🔑 Giriş Yap", "📝 Kayıt Ol"])
            
            with tab_login:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.form("login_form"):
                    u = st.text_input("Kullanıcı Adı")
                    p = st.text_input("Şifre", type="password")
                    st.markdown("<br>", unsafe_allow_html=True)
                    # type="primary" CSS ile eziliyor ama yapısal olarak önemli
                    if st.form_submit_button("Giriş Yap", type="primary"): 
                        res = supabase.table("kullanicilar").select("*").eq("username", u).eq("password", p).execute()
                        if res.data:
                            st.session_state.logged_in = True
                            st.session_state.username = res.data[0]['username']
                            st.session_state.role = res.data[0]['role']
                            st.rerun()
                        else: st.error("Kullanıcı adı veya şifre hatalı!")

            with tab_signup:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.form("signup_form"):
                    new_u = st.text_input("Kullanıcı Adı Belirle")
                    new_p = st.text_input("Şifre Belirle", type="password")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.form_submit_button("Hesap Oluştur", type="primary"):
                        try:
                            supabase.table("kullanicilar").insert({"username": new_u, "password": new_p, "role": "student"}).execute()
                            st.success("Kayıt Başarılı! Giriş yapabilirsiniz.")
                        except: st.error("Bu kullanıcı adı zaten alınmış.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
        <div class="user-card">
            <h2 style='margin:0;'>{st.session_state.username.upper()}</h2>
            <p style='margin:0; opacity:0.9; font-size:0.9rem;'>{st.session_state.role.upper()} YETKİSİ</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. ANALİZ
    if st.session_state.role == 'admin':
        if st.button("📊 Analiz Paneli"):
            st.session_state.analiz_acik = not st.session_state.analiz_acik
        
        if st.session_state.analiz_acik:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.write(f"🔹 **Toplam Sorgu:** {st.session_state.sorgu_sayaci}")
            st.write(f"🔹 **Mesajlar:** {len(st.session_state.messages)}")
            
            c_zoom, c_rep = st.columns(2)
            with c_zoom:
                if st.button("🔍 Büyüt"):
                    st.session_state.view_mode = "analysis_fullscreen"
                    st.rerun()
            with c_rep:
                st.download_button("📥 Rapor", analiz_raporu_olustur(), "analiz.txt")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

    # 2. VERİTABANI
    st.subheader("📁 Veri Yönetimi")
    uploaded_files = st.file_uploader("PDF Yükle", accept_multiple_files=True, type=['pdf'])
    if st.button("Veritabanını Güncelle"):
        if uploaded_files:
            durum = st.status("Veriler işleniyor...", expanded=True)
            st.session_state.vector_db = process_pdfs(uploaded_files)
            durum.update(label="✅ Güncelleme Tamamlandı!", state="complete")
    
    st.divider()

    # 3. İŞLEMLER
    st.caption("Sohbet Yönetimi")
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.messages:
            tr_saat = get_tr_time()
            log = f"🎓 SOHBET\n{tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n"
            for m in st.session_state.messages: log += f"[{m['role']}]: {m['content']}\n"
            st.download_button("📥 İndir", log, "sohbet.txt")
    with c2:
        if st.button("🧹 Temizle"):
            st.session_state.messages = [{"role": "assistant", "content": "Sohbet temizlendi. Yeni sorunuz nedir?"}]
            st.session_state.sorgu_sayaci = 0
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    # Çıkış butonu için özel stil (Kırmızı)
    if st.button("🚪 Çıkış Yap"):
        st.session_state.logged_in = False
        st.rerun()

# --- EKRAN YÖNETİMİ ---

if st.session_state.view_mode == "analysis_fullscreen":
    # --- TAM EKRAN ANALİZ ---
    st.title("📊 Sistem İstatistikleri")
    k1, k2, k3 = st.columns(3)
    k1.metric("Toplam Sorgu", st.session_state.sorgu_sayaci)
    k2.metric("PDF Sayısı", len(uploaded_files) if uploaded_files else 0)
    k3.metric("Kullanıcı", st.session_state.username)
    st.divider()
    
    g1, g2 = st.columns([2, 1])
    with g1:
        st.subheader("Popüler Kelimeler")
        konular = konu_analizi_yap()
        if konular: st.bar_chart({k:v for k,v in konular})
        else: st.info("Veri yok.")
    with g2:
        st.subheader("Son Aktiviteler")
        msgs = [m['content'] for m in st.session_state.messages if m['role']=='user']
        for m in reversed(msgs[-8:]): st.info(m)
        
    if st.button("🔙 Sohbete Dön", type="primary"):
        st.session_state.view_mode = "chat"
        st.rerun()

else:
    # --- SOHBET MODU ---
    st.title("💬 Mevzuat Asistanı")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.sorgu_sayaci += 1
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mevzuat taranıyor..."):
                sonuc = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
                daktilo_efekti(sonuc["answer"])
                
                if sonuc["sources"]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("📚 REFERANS KAYNAKLAR")
                    html_src = ""
                    for src in sonuc["sources"]:
                        html_src += f'<div class="source-item">📄 {src}</div>'
                    st.markdown(html_src, unsafe_allow_html=True)
                
                full = sonuc["answer"] + ("\n\nKaynaklar:\n" + "\n".join(sonuc["sources"]) if sonuc["sources"] else "")
                st.session_state.messages.append({"role": "assistant", "content": full})
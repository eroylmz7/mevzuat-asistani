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
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    /* Dikey Kaynak Kutucukları */
    .source-item {
        display: block;
        background-color: #1f2937;
        color: #d1d5db;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9em;
        margin-bottom: 6px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Buton Grubu Düzenlemesi */
    .btn-group {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    
    /* Standart Butonlar */
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 500; }
    
    /* Çıkış Butonu (Biraz daha farklı dursun) */
    div[data-testid="stVerticalBlock"] > div:last-child button {
        border-color: #ef4444;
        color: #ef4444;
    }
    div[data-testid="stVerticalBlock"] > div:last-child button:hover {
        background-color: #ef4444;
        color: white;
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
    """Analiz verilerini indirilebilir metne çevirir."""
    tr_now = get_tr_time()
    user_msgs = [m['content'] for m in st.session_state.messages if m['role'] == 'user']
    
    rapor = f"📊 SİSTEM ANALİZ RAPORU\n"
    rapor += f"Tarih: {tr_now.strftime('%d.%m.%Y %H:%M')}\n"
    rapor += "="*30 + "\n\n"
    rapor += f"🔹 Toplam Sorgu Sayısı: {st.session_state.sorgu_sayaci}\n"
    rapor += f"🔹 Aktif Oturum Mesajları: {len(user_msgs)}\n\n"
    rapor += "🔹 SON SORULAN BAŞLIKLAR:\n"
    for msg in user_msgs[-5:]:
        rapor += f" - {msg}\n"
    return rapor

def konu_analizi_yap():
    """Basitçe mesajlardaki anahtar kelimeleri sayar."""
    text = " ".join([m['content'] for m in st.session_state.messages if m['role'] == 'user']).lower()
    # Basit bir filtreleme (bağlaçları çıkarabilirsin)
    kelimeler = re.findall(r'\w+', text)
    anahtar_kelimeler = [k for k in kelimeler if len(k) > 4] # 4 harften uzun kelimeler
    return Counter(anahtar_kelimeler).most_common(5)

# --- STATE YÖNETİMİ ---
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Kampüs mevzuatı, dersler veya yönetmelikler hakkında ne öğrenmek istersiniz?"}]

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "sorgu_sayaci" not in st.session_state: st.session_state.sorgu_sayaci = 0
if "analiz_acik" not in st.session_state: st.session_state.analiz_acik = False
if "view_mode" not in st.session_state: st.session_state.view_mode = "chat" # chat veya analysis_fullscreen

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

    # 1. ANALİZ BUTONU (Gelişmiş)
    if st.session_state.role == 'admin':
        if st.button("📊 Analiz Paneli"):
            st.session_state.analiz_acik = not st.session_state.analiz_acik
        
        if st.session_state.analiz_acik:
            with st.container():
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.write(f"🔹 **Toplam Sorgu:** {st.session_state.sorgu_sayaci}")
                st.write(f"🔹 **Oturum Mesajı:** {len(st.session_state.messages)}")
                
                # Konu Analizi (Mini)
                konular = konu_analizi_yap()
                if konular:
                    st.caption("🔥 **Popüler Konular:**")
                    for k, v in konular[:3]:
                        st.markdown(f"- *{k.capitalize()}* ({v})")
                
                st.markdown("---")
                
                # Tam Ekran ve İndir Butonları
                c_a1, c_a2 = st.columns(2)
                with c_a1:
                    if st.button("🔍 Büyüt"):
                        st.session_state.view_mode = "analysis_fullscreen"
                        st.rerun()
                with c_a2:
                    st.download_button(
                        label="📥 Rapor",
                        data=analiz_raporu_olustur(),
                        file_name="sistem_analizi.txt",
                        mime="text/plain"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # 2. VERİTABANI
    st.subheader("📁 Veri Yönetimi")
    uploaded_files = st.file_uploader("PDF Yükle", accept_multiple_files=True, type=['pdf'])
    if st.button("Veritabanını Güncelle", type="primary"):
        if uploaded_files:
            durum = st.status("İşleniyor...", expanded=True)
            st.session_state.vector_db = process_pdfs(uploaded_files)
            durum.update(label="✅ Güncel!", state="complete")
    
    st.divider()

    # 3. SOHBET İŞLEMLERİ (Yeni Düzen)
    st.caption("Sohbet İşlemleri")
    col_dl, col_clr = st.columns(2)
    with col_dl:
        if st.session_state.messages:
            tr_saat = get_tr_time()
            log = f"🎓 SOHBET KAYDI\n{tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n"
            for m in st.session_state.messages:
                log += f"[{m['role'].upper()}]: {m['content']}\n"
            st.download_button(label="📥 İndir", data=log, file_name="sohbet.txt")
    with col_clr:
        if st.button("🧹 Temizle"):
            st.session_state.messages = [{"role": "assistant", "content": "Sohbet temizlendi. Yardımcı olabileceğim başka bir konu var mı?"}]
            st.session_state.sorgu_sayaci = 0
            st.session_state.view_mode = "chat" # Chat ekranına dön
            st.rerun()

    # 4. ÇIKIŞ
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Çıkış Yap"):
        st.session_state.logged_in = False
        st.session_state.view_mode = "chat"
        st.rerun()

# --- ANA EKRAN KONTROLÜ ---

if st.session_state.view_mode == "analysis_fullscreen":
    # --- TAM EKRAN ANALİZ MODU ---
    st.title("📊 Detaylı Sistem Analizi")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Sorgu", st.session_state.sorgu_sayaci, "+1")
    col2.metric("Yüklenen Doküman", len(uploaded_files) if uploaded_files else 0)
    col3.metric("Aktif Kullanıcı", st.session_state.username)
    
    st.divider()
    
    c_chart, c_list = st.columns([2, 1])
    
    with c_chart:
        st.subheader("📈 Konu Dağılımı")
        konular = konu_analizi_yap()
        if konular:
            # Basit bir bar chart (Streamlit native)
            chart_data = {k: v for k, v in konular}
            st.bar_chart(chart_data)
        else:
            st.info("Analiz için yeterli veri yok.")
            
    with c_list:
        st.subheader("📝 Son Sorgular")
        user_msgs = [m['content'] for m in st.session_state.messages if m['role'] == 'user']
        for msg in reversed(user_msgs[-10:]):
            st.markdown(f"- {msg}")

    st.divider()
    if st.button("🔙 Sohbete Dön", type="primary"):
        st.session_state.view_mode = "chat"
        st.rerun()

else:
    # --- SOHBET MODU ---
    st.title("💬 Mevzuat Asistanı")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.sorgu_sayaci += 1
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Araştırılıyor..."):
                sonuc = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
                daktilo_efekti(sonuc["answer"])
                
                # KAYNAKLAR (Alt Alta Şık Liste)
                if sonuc["sources"]:
                    st.markdown("---")
                    st.caption("📚 **Referans Kaynaklar:**")
                    html_sources = ""
                    for src in sonuc["sources"]:
                        html_sources += f'<div class="source-item">📄 {src}</div>'
                    st.markdown(html_sources, unsafe_allow_html=True)
                
                full_resp = sonuc["answer"]
                if sonuc["sources"]:
                    full_resp += "\n\nKaynaklar:\n" + "\n".join(sonuc["sources"])
                st.session_state.messages.append({"role": "assistant", "content": full_resp})
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

# --- CSS İYİLEŞTİRMELERİ ---
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
    }
    
    /* Kaynak Gösterimi (Daha kompakt ve şık) */
    .source-tag {
        display: inline-block;
        background-color: #1f2937;
        color: #9ca3af;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid #374151;
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
# Başlangıç mesajı eklendi (İstek 3)
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Kampüs mevzuatı, dersler veya yönetmelikler hakkında ne öğrenmek istersiniz?"}]

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "sorgu_sayaci" not in st.session_state: st.session_state.sorgu_sayaci = 0

# --- GİRİŞ EKRANI (TABLI YAPI) ---
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
                        st.success("Kayıt Başarılı! 'Giriş Yap' sekmesinden girebilirsiniz.")
                    except: st.error("Bu kullanıcı adı dolu.")
    st.stop()

# --- SIDEBAR (SOL MENÜ) ---
with st.sidebar:
    # Kullanıcı Kartı
    st.markdown(f"""
        <div class="user-card">
            <h3>{st.session_state.username.upper()}</h3>
            <small>{st.session_state.role.upper()} HESABI</small>
        </div>
    """, unsafe_allow_html=True)

    # 1. NAVİGASYON (İstek 2: Analiz kısmı seçilebilir oldu)
    secilen_mod = st.radio("Mod Seçiniz:", ["💬 Sohbet Asistanı", "📊 Sistem Analizi"])
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

    # 3. SOHBETİ İNDİR (İstek 5: Çıkış'ın hemen üstünde)
    if st.session_state.messages:
        tr_saat = get_tr_time()
        log = f"🎓 MEVZUAT SOHBET KAYDI - {tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*40 + "\n\n"
        for m in st.session_state.messages:
            log += f"[{m['role'].upper()}]: {m['content']}\n\n"
        
        st.download_button(
            label="📥 Sohbeti İndir (.txt)",
            data=log,
            file_name=f"sohbet_{tr_saat.strftime('%H%M')}.txt",
            mime="text/plain"
        )

    # 4. ÇIKIŞ YAP
    if st.button("🚪 Çıkış Yap"):
        st.session_state.logged_in = False
        st.rerun()

# --- ANA EKRAN YÖNETİMİ ---

if secilen_mod == "💬 Sohbet Asistanı":
    # --- SOHBET MODU ---
    st.title("💬 Mevzuat Asistanı")

    # Mesajları Listele
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat Input (İstek 1: CSS kaldırıldığı için artık tam genişlikte)
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        # 1. Kullanıcı mesajı
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.sorgu_sayaci += 1
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Asistan cevabı
        with st.chat_message("assistant"):
            with st.spinner("Araştırılıyor..."):
                sonuc = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
                
                # Cevabı yaz
                daktilo_efekti(sonuc["answer"])
                
                # Kaynakları Kompakt Göster (İstek 4)
                if sonuc["sources"]:
                    st.markdown("---")
                    st.caption("📚 **Referans Kaynaklar:**")
                    
                    # HTML ile yan yana şık etiketler oluşturuyoruz
                    html_sources = ""
                    for src in sonuc["sources"]:
                        html_sources += f'<span class="source-tag">📄 {src}</span>'
                    st.markdown(html_sources, unsafe_allow_html=True)
                
                # Hafızaya kaydet
                full_resp = sonuc["answer"]
                if sonuc["sources"]:
                    full_resp += "\n\nKaynaklar: " + ", ".join(sonuc["sources"])
                st.session_state.messages.append({"role": "assistant", "content": full_resp})

elif secilen_mod == "📊 Sistem Analizi":
    # --- ANALİZ MODU (Sadece Admin veya Herkes?) ---
    # Eğer sadece admin görsün istiyorsan: if st.session_state.role == 'admin': altına alabilirsin.
    
    st.title("📊 Sistem Analiz Paneli")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Toplam Yapılan Sorgu:** {st.session_state.sorgu_sayaci}")
    with col2:
        st.success(f"**Aktif Doküman Sayısı:** {len(uploaded_files) if uploaded_files else 0}")

    st.divider()
    
    st.subheader("📌 Son Yapılan Sorgular (Oturum Bazlı)")
    if len(st.session_state.messages) > 1:
        # Sadece user mesajlarını al
        user_msgs = [m['content'] for m in st.session_state.messages if m['role'] == 'user']
        for i, msg in enumerate(reversed(user_msgs)):
            st.markdown(f"**{i+1}.** {msg}")
    else:
        st.caption("Henüz bir sorgu yapılmadı.")
    
    if st.session_state.role != 'admin':
        st.warning("Not: Daha detaylı analizler için Yönetici yetkisi gereklidir.")
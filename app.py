import streamlit as st
import datetime
import pytz
import time
from supabase import create_client, Client

# --- 1. AYARLAR VE TASARIM ---
st.set_page_config(page_title="Mevzuat Asistanı", page_icon="🎓", layout="wide")

# Modern CSS
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; }
    .user-profile { text-align: center; padding: 1rem; background: #1E3A8A; color: white; border-radius: 10px; margin-bottom: 10px; }
    [data-testid="stSidebarNav"] { display: none; } /* Sidebar navigasyonunu gizle */
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERİTABANI VE YARDIMCI ARAÇLAR ---
@st.cache_resource
def get_supabase_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

supabase = get_supabase_client()

def get_tr_time():
    return datetime.datetime.now(pytz.timezone('Europe/Istanbul'))

def daktilo_efekti(metin, alan=None):
    if alan is None:
        alan = st.empty()
    gecici_metin = ""
    for harf in metin:
        gecici_metin += harf
        alan.markdown(gecici_metin + "▌")
        time.sleep(0.01)
    alan.markdown(gecici_metin)

# --- 3. OTURUM YÖNETİMİ VE HAFIZA ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "messages" not in st.session_state:
    # Asistanın hafızası için başlangıç mesajı
    st.session_state.messages = []

# --- GİRİŞ EKRANI ---
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<h1 style='text-align: center;'>🔐 Mevzuat Sistemi</h1>", unsafe_allow_html=True)
        with st.form("login"):
            u = st.text_input("Kullanıcı Adı")
            p = st.text_input("Şifre", type="password")
            if st.form_submit_button("Giriş Yap"):
                res = supabase.table("kullanicilar").select("*").eq("username", u).eq("password", p).execute()
                if res.data:
                    st.session_state.logged_in = True
                    st.session_state.username = res.data[0]['username']
                    st.session_state.role = res.data[0]['role']
                    st.rerun()
                else:
                    st.error("Giriş bilgileri hatalı!")
    st.stop()

# --- 4. SIDEBAR (SOL PANEL) ---
with st.sidebar:
    st.markdown(f"<div class='user-profile'><h3>{st.session_state.username.upper()}</h3><small>{st.session_state.role.upper()}</small></div>", unsafe_allow_html=True)
    
    st.subheader("📁 Veri Kaynakları")
    uploaded_files = st.file_uploader("PDF Dosyalarını Seçin", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Veritabanını Güncelle", type="primary"):
        if uploaded_files:
            bilgi = st.empty()
            with st.spinner("İşleniyor..."):
                bilgi.info("📂 1. Dosyalar taranıyor...")
                time.sleep(1)
                bilgi.info("🧠 2. Mevzuat hafızaya alınıyor...")
                # process_pdfs(uploaded_files) -> Kendi fonksiyonunu bağla
                time.sleep(1)
                bilgi.success("✅ Sistem güncel!")
        else:
            st.warning("Lütfen dosya yükleyin.")

    st.divider()

    # SOHBET İNDİRME
    if len(st.session_state.messages) > 0:
        tr_saat = get_tr_time()
        sohbet_metni = f"🎓 MEVZUAT ASİSTANI KAYDI\n{tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*40 + "\n\n"
        for m in st.session_state.messages:
            sohbet_metni += f"[{m['role'].upper()}]: {m['content']}\n\n"
        
        st.download_button("📥 Sohbeti İndir", sohbet_metni, file_name=f"sohbet_{tr_saat.strftime('%H%M')}.txt")

    if st.button("Çıkış Yap"):
        st.session_state.logged_in = False
        st.rerun()

# --- 5. ANA EKRAN (SEKMELİ YAPI) ---
st.title("🎓 Kampüs Mevzuat Asistanı")

tab1, tab2 = st.tabs(["💬 Sohbet Modu", "📊 Doküman Analizi"])

# --- TAB 1: SOHBET MODU (HAFIZALI) ---
with tab1:
    # Geçmiş mesajları yükle
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Mevzuat hakkında bir soru sorun..."):
        # Kullanıcı mesajı
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan yanıtı
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                # HAFIZA BURADA ÇALIŞIR: 
                # yanıt = chat_engine.chat(prompt) -> Önceki mesajları da gönderir
                response = "Önceki söylediklerinizi de hatırlayarak söylüyorum ki; yönetmeliğin 5. maddesine göre bu işlem mümkündür." 
                
                daktilo_efekti(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- TAB 2: ANALİZ SEKMESİ ---
with tab2:
    st.subheader("📄 Yüklenen Dokümanların Analizi")
    if uploaded_files:
        st.write(f"Toplam {len(uploaded_files)} doküman sisteme yüklü.")
        # Burada dokümanların özetini veya istatistiklerini gösterebilirsin
        col1, col2 = st.columns(2)
        with col1:
            st.info("📌 En Çok Sorgulanan Maddeler")
            st.write("- Sınav Yönetmeliği\n- Disiplin Kuralları")
        with col2:
            st.info("💡 Otomatik Özet")
            st.write("Bu dokümanlar 2024-2025 eğitim yılını kapsamaktadır.")
    else:
        st.warning("Analiz için henüz doküman yüklenmedi.")
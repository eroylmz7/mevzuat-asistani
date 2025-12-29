import streamlit as st
import datetime
import pytz
from supabase import create_client, Client

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kampüs Asistanı", layout="wide")

# --- VERİTABANI BAĞLANTISI ---
@st.cache_resource
def get_supabase_client():
    # Bu fonksiyon bağlantıyı bir kez kurar, 'too many open files' hatasını önler
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase_client()

# --- YARDIMCI FONKSİYONLAR ---
def login_user(username, password):
    """Veritabanından kullanıcıyı sorgular."""
    try:
        response = supabase.table("kullanicilar").select("*").eq("username", username).eq("password", password).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        st.error(f"Veritabanı bağlantı hatası: {e}")
        return None

def get_tr_time():
    """Her zaman Türkiye saatini döner."""
    return datetime.datetime.now(pytz.timezone('Europe/Istanbul'))

# --- GİRİŞ KONTROLÜ ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🎓 Mevzuat Asistanı - Giriş")
    with st.form("login_form"):
        u = st.text_input("Kullanıcı Adı")
        p = st.text_input("Şifre", type="password")
        if st.form_submit_button("Giriş Yap"):
            user_data = login_user(u, p)
            if user_data:
                st.session_state.logged_in = True
                st.session_state.username = user_data['username']
                st.session_state.role = user_data['role']
                st.rerun()
            else:
                st.error("Kullanıcı adı veya şifre hatalı!")
    st.stop()

# --- ANA UYGULAMA PANELİ ---
st.sidebar.success(f"Hoş geldin, {st.session_state.username}")

# Mevcut RAG ve mesajlaşma kodlarınızın burada olduğunu varsayıyorum...
# st.session_state.messages içindeki mesajları kullanıyoruz.

# --- SOHBET İNDİRME BUTONU ---
st.sidebar.markdown("---")
tr_now = get_tr_time()
tarih_str = tr_now.strftime("%d.%m.%Y %H:%M")

if "messages" in st.session_state and len(st.session_state.messages) > 0:
    indirilecek_metin = f"🎓 MEVZUAT ASİSTANI - SOHBET KAYDI\nTarih: {tarih_str}\n" + "="*40 + "\n\n"
    
    for m in st.session_state.messages:
        rol = "ASİSTAN" if m["role"] == "assistant" else "ÖĞRENCİ"
        indirilecek_metin += f"[{rol}]: {m['content']}\n"
        indirilecek_metin += "-"*20 + "\n"

    st.sidebar.download_button(
        label="📂 Sohbet Geçmişini İndir",
        data=indirilecek_metin,
        file_name=f"Mevzuat_Asistani_{tr_now.strftime('%d_%m_%H_%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )

if st.sidebar.button("Çıkış Yap"):
    st.session_state.logged_in = False
    st.rerun()
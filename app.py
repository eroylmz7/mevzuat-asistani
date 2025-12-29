import streamlit as st
import datetime
import pytz
import time
import pandas as pd
import os
import asyncio 
from supabase import create_client

# --- KRİTİK HATA DÜZELTİCİ ---
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Kampüs Mevzuat Asistanı", page_icon="🎓", layout="wide")

# --- MODÜLLERİ GÜVENLİ YÜKLEME ---
try:
    from langchain_pinecone import PineconeVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from data_ingestion import process_pdfs 
    from generation import generate_answer 
except ImportError as e:
    st.error(f"⚠️ Kritik Başlatma Hatası: {e}")
    st.stop()

# --- CSS TASARIMI ---
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    .user-card { padding: 20px; background: linear-gradient(135deg, #2563eb, #1d4ed8); border-radius: 12px; color: white; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
    .stButton > button { width: 100%; background-color: #3b82f6; color: white !important; border: none; padding: 0.7rem 1rem; font-weight: 600; border-radius: 8px; transition: all 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stButton > button:hover { background-color: #2563eb; transform: translateY(-2px); box-shadow: 0 6px 8px rgba(0,0,0,0.2); }
    .file-item { background-color: #334155; padding: 8px; border-radius: 5px; margin-bottom: 5px; font-size: 0.9em; border-left: 3px solid #10b981; }
    .source-item { display: block; background-color: #334155; color: #e2e8f0; padding: 10px 15px; border-radius: 8px; font-size: 0.95em; margin-bottom: 8px; border-left: 5px solid #60a5fa; }
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
    rapor = f"📊 SİSTEM RAPORU\n{tr_now.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n\n"
    rapor += f"🔹 Toplam Sorgu: {st.session_state.sorgu_sayaci}\n"
    rapor += f"🔹 Mesajlar: {len(user_msgs)}\n"
    return rapor

# --- BULUT BAĞLANTISI ---
@st.cache_resource
def get_cloud_db():
    try:
        os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        index_name = "mevzuat-asistani"
        vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding_model)
        return vector_store
    except Exception as e:
        print(f"Pinecone Hatası: {e}")
        return None

# --- STATE ---
if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Kampüs mevzuatı hakkında size nasıl yardımcı olabilirim?"}]
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "sorgu_sayaci" not in st.session_state: st.session_state.sorgu_sayaci = 0
if "analiz_acik" not in st.session_state: st.session_state.analiz_acik = False
if "view_mode" not in st.session_state: st.session_state.view_mode = "chat"

if "vector_db" not in st.session_state or st.session_state.vector_db is None:
    st.session_state.vector_db = get_cloud_db()

# --- GİRİŞ EKRANI ---
if not st.session_state.logged_in:
    st.markdown("<br><br><h1 style='text-align: center; color: white;'>🎓 Kampüs Asistanı</h1>", unsafe_allow_html=True)
    _, col_main, _ = st.columns([1, 1.5, 1])
    with col_main:
        with st.container():
            tab_login, tab_signup = st.tabs(["🔑 Giriş Yap", "📝 Kayıt Ol"])
            with tab_login:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.form("login_form"):
                    u = st.text_input("Kullanıcı Adı")
                    p = st.text_input("Şifre", type="password")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.form_submit_button("Giriş Yap", type="primary"): 
                        res = supabase.table("kullanicilar").select("*").eq("username", u).eq("password", p).execute()
                        if res.data:
                            st.session_state.logged_in = True
                            st.session_state.username = res.data[0]['username']
                            st.session_state.role = res.data[0]['role']
                            st.session_state.view_mode = "chat" 
                            st.rerun()
                        else: st.error("Hatalı giriş!")
            with tab_signup:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.form("signup_form"):
                    new_u = st.text_input("Kullanıcı Adı")
                    new_p = st.text_input("Şifre", type="password")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.form_submit_button("Hesap Oluştur", type="primary"):
                        try:
                            supabase.table("kullanicilar").insert({"username": new_u, "password": new_p, "role": "student"}).execute()
                            st.success("Başarılı! Giriş yapabilirsiniz.")
                        except: st.error("Kullanıcı adı alınmış.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    rol_txt = "YÖNETİCİ" if st.session_state.role == "admin" else "ÖĞRENCİ"
    st.markdown(f"""<div class="user-card"><h2 style='margin:0;'>{st.session_state.username.upper()}</h2><p style='margin:0; opacity:0.9; font-size:0.9rem;'>{rol_txt} HESABI</p></div>""", unsafe_allow_html=True)

    if st.session_state.role == 'admin':
        if st.button("📊 Analiz Paneli"): st.session_state.analiz_acik = not st.session_state.analiz_acik
        if st.session_state.analiz_acik:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.write(f"🔹 **Sorgu:** {st.session_state.sorgu_sayaci}")
            st.download_button("📥 Rapor", analiz_raporu_olustur(), "analiz.txt", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("📁 Veri Yönetimi")
        uploaded_files = st.file_uploader("PDF Yükle", accept_multiple_files=True, type=['pdf'])
        if st.button("Veritabanını Güncelle"):
            if uploaded_files:
                durum = st.status("Sistem güncelleniyor...", expanded=True)
                st.session_state.vector_db = process_pdfs(uploaded_files)
                durum.update(label="✅ Güncelleme Tamamlandı!", state="complete")
        
        # --- YÜKLÜ DOSYALARI LİSTELE ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("📚 SİSTEMDEKİ BELGELER")
        try:
            docs = supabase.table("dokumanlar").select("*").execute()
            if docs.data:
                for d in docs.data:
                    st.markdown(f'<div class="file-item">📄 {d["dosya_adi"]}</div>', unsafe_allow_html=True)
            else:
                st.info("Henüz belge yüklenmemiş.")
        except:
            st.error("Liste alınamadı.")
        # -------------------------------
        st.divider()

    st.caption("İşlemler")
    if st.session_state.messages:
        tr_saat = get_tr_time()
        log = f"🎓 SOHBET\n{tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n"
        for m in st.session_state.messages: log += f"[{m['role']}]: {m['content']}\n"
        st.download_button("📥 Sohbeti İndir", log, "chat.txt", use_container_width=True)
    st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Temizle", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Sohbet temizlendi."}]
        st.session_state.sorgu_sayaci = 0
        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- ÇIKIŞ YAP ---
    if st.button("🚪 Çıkış", type="secondary", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Kampüs mevzuatı hakkında size nasıl yardımcı olabilirim?"}]
        st.session_state.view_mode = "chat"
        st.session_state.sorgu_sayaci = 0
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

# --- EKRANLAR ---
# (Burada kod kısalığı için analiz ekranını sadeleştirdim, istersen ekleyebilirsin)
st.title("💬 Mevzuat Asistanı")
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.sorgu_sayaci += 1
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vector_db is None:
            st.error("⚠️ Bulut veritabanına bağlanılamadı.")
        else:
            with st.spinner("Mevzuat taranıyor..."):
                try:
                    sonuc = generate_answer(prompt, st.session_state.vector_db, st.session_state.messages)
                    daktilo_efekti(sonuc["answer"])
                    if sonuc["sources"]:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.caption("📚 KAYNAKLAR")
                        html_src = ""
                        for src in sonuc["sources"]:
                            html_src += f'<div class="source-item">📄 {src}</div>'
                        st.markdown(html_src, unsafe_allow_html=True)
                    full = sonuc["answer"] + ("\n\nKaynaklar:\n" + "\n".join(sonuc["sources"]) if sonuc["sources"] else "")
                    st.session_state.messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    st.error(f"Hata: {e}")
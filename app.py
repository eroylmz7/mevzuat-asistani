import subprocess
import sys

# --- ZOMBİ DOSYA TEMİZLEYİCİ ---
# Sunucuda takılı kalan hatalı paketi zorla siliyoruz.
try:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pinecone-plugin-inference"])
    print("✅ Hatalı eklenti silindi.")
except Exception:
    pass
# -------------------------------
import streamlit as st
import datetime
import pytz
import time
import pandas as pd
import os
from supabase import create_client

# --- SAYFA ---
st.set_page_config(page_title="Kampüs Mevzuat Asistanı", page_icon="🎓", layout="wide")

# --- HATA YAKALAYICI IMPORT ---
try:
    from langchain_pinecone import PineconeVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # Önce modülleri import etmeyi dene
    import data_ingestion
    import generation
    # Sonra fonksiyonları çek
    from data_ingestion import process_pdfs 
    from generation import generate_answer 
except ImportError as e:
    st.error(f"⚠️ Kritik Başlatma Hatası: {e}")
    st.warning("Bu hata genellikle 'requirements.txt' uyumsuzluğundan veya 'generation.py' dosyasının bozuk olmasından kaynaklanır.")
    st.stop()

# --- CSS (Aynı Tasarım) ---
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    .user-card { padding: 20px; background: linear-gradient(135deg, #2563eb, #1d4ed8); border-radius: 12px; color: white; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
    .stButton > button { width: 100%; background-color: #3b82f6; color: white !important; border: none; padding: 0.7rem 1rem; font-weight: 600; border-radius: 8px; transition: all 0.2s; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stButton > button:hover { background-color: #2563eb; transform: translateY(-2px); box-shadow: 0 6px 8px rgba(0,0,0,0.2); }
    .stDownloadButton > button { width: 100%; background-color: #475569; color: white !important; border-radius: 8px; font-weight: 500; }
    .stats-box { background-color: #334155; padding: 15px; border-radius: 10px; border: 1px solid #475569; margin: 10px 0; }
    .source-item { display: block; background-color: #334155; color: #e2e8f0; padding: 10px 15px; border-radius: 8px; font-size: 0.95em; margin-bottom: 8px; border-left: 5px solid #60a5fa; }
    </style>
    """, unsafe_allow_html=True)

# --- YARDIMCI ---
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
    rapor = f"📊 RAPOR - {tr_now.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n"
    rapor += f"Sorgu Sayısı: {st.session_state.sorgu_sayaci}\nMesaj Sayısı: {len(user_msgs)}\n"
    return rapor

def detayli_konu_analizi():
    user_msgs = [m['content'].lower() for m in st.session_state.messages if m['role'] == 'user']
    if not user_msgs: return pd.DataFrame()
    kategoriler = {
        "Sınav & Not": ["sınav", "vize", "final", "büt", "not", "ortalama"],
        "Mezuniyet & Kredi": ["mezun", "kredi", "akts", "diploma"],
        "Staj & İş": ["staj", "iş yeri", "pratik", "uygulama"],
        "Ders & Kayıt": ["kayıt", "ders", "seçmeli", "zorunlu"],
        "İzin & Hak": ["izin", "mazeret", "dondurma", "rapor"]
    }
    sonuclar = {k: 0 for k in kategoriler.keys()}
    for msg in user_msgs:
        for kat, keywords in kategoriler.items():
            if any(k in msg for k in keywords):
                sonuclar[kat] += 1
                break
    df = pd.DataFrame(list(sonuclar.items()), columns=["Konu", "Adet"])
    df = df[df["Adet"] > 0].sort_values(by="Adet", ascending=False)
    if not df.empty: df["Oran"] = (df["Adet"] / len(user_msgs)) * 100
    return df

# --- BULUT DB ---
@st.cache_resource
def get_cloud_db():
    try:
        # Import buraya taşındı (Lazy Loading)
        from langchain_pinecone import PineconeVectorStore
        os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        # Mevcut indexe bağlan
        return PineconeVectorStore.from_existing_index(index_name="mevzuat-asistani", embedding=embedding_model)
    except Exception as e:
        print(f"DB Hatası: {e}")
        return None

# --- STATE ---
if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Kampüs mevzuatı hakkında size nasıl yardımcı olabilirim?"}]
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "sorgu_sayaci" not in st.session_state: st.session_state.sorgu_sayaci = 0
if "analiz_acik" not in st.session_state: st.session_state.analiz_acik = False
if "view_mode" not in st.session_state: st.session_state.view_mode = "chat"
if "vector_db" not in st.session_state or st.session_state.vector_db is None:
    st.session_state.vector_db = get_cloud_db()

# --- GİRİŞ ---
if not st.session_state.logged_in:
    st.markdown("<br><h1 style='text-align: center; color: white;'>🎓 Kampüs Asistanı</h1>", unsafe_allow_html=True)
    _, col_main, _ = st.columns([1, 1.5, 1])
    with col_main:
        tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
        with tab1:
            with st.form("login"):
                u = st.text_input("Kullanıcı Adı")
                p = st.text_input("Şifre", type="password")
                if st.form_submit_button("Giriş", type="primary"):
                    res = supabase.table("kullanicilar").select("*").eq("username", u).eq("password", p).execute()
                    if res.data:
                        st.session_state.logged_in = True
                        st.session_state.username = res.data[0]['username']
                        st.session_state.role = res.data[0]['role']
                        st.session_state.view_mode = "chat"
                        st.rerun()
                    else: st.error("Hatalı!")
        with tab2:
            with st.form("signup"):
                nu = st.text_input("Kullanıcı Adı")
                np = st.text_input("Şifre", type="password")
                if st.form_submit_button("Kayıt Ol", type="primary"):
                    try:
                        supabase.table("kullanicilar").insert({"username": nu, "password": np, "role": "student"}).execute()
                        st.success("Kayıt Başarılı!")
                    except: st.error("Kullanıcı adı dolu.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    rol = "YÖNETİCİ" if st.session_state.role == "admin" else "ÖĞRENCİ"
    st.markdown(f"""<div class="user-card"><h3>{st.session_state.username.upper()}</h3><p>{rol} HESABI</p></div>""", unsafe_allow_html=True)
    
    if st.session_state.role == 'admin':
        if st.button("📊 Analiz", key="btn_analiz"): st.session_state.analiz_acik = not st.session_state.analiz_acik
        if st.session_state.analiz_acik:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.write(f"Sorgu: {st.session_state.sorgu_sayaci}")
            c1, c2 = st.columns(2)
            if c1.button("🔍 Büyüt", key="btn_fs"): 
                st.session_state.view_mode = "analysis_fullscreen"
                st.rerun()
            c2.download_button("📥 Rapor", analiz_raporu_olustur(), "r.txt", key="btn_dr")
            st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("📁 Veri")
        pdfs = st.file_uploader("PDF Yükle", accept_multiple_files=True, type=['pdf'], key="uploader")
        if st.button("Güncelle", key="btn_upd"):
            if pdfs:
                st.status("Yükleniyor...").update(label="✅ Tamam!", state="complete")
                st.session_state.vector_db = process_pdfs(pdfs)
        st.divider()
    
    if st.session_state.messages:
        chat_log = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("📥 Sohbeti İndir", chat_log, "chat.txt", key="btn_dl_chat", use_container_width=True)
    
    st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Temizle", key="btn_clr", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Temizlendi."}]
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Çıkış", key="btn_logout", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- EKRANLAR ---
if st.session_state.view_mode == "analysis_fullscreen" and st.session_state.role == 'admin':
    st.title("📊 İstatistikler")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sorgu", st.session_state.sorgu_sayaci)
    c2.metric("Durum", "Aktif")
    c3.metric("Kullanıcı", st.session_state.username)
    st.divider()
    g1, g2 = st.columns([2, 1])
    df = detayli_konu_analizi()
    if not df.empty: g1.dataframe(df, use_container_width=True, hide_index=True)
    else: g1.info("Veri yok")
    msgs = [m['content'] for m in st.session_state.messages if m['role']=='user']
    for m in reversed(msgs[-5:]): g2.code(m[:40]+"...", language="text")
    if st.button("🔙 Dön", key="btn_back"):
        st.session_state.view_mode = "chat"
        st.rerun()
else:
    st.title("💬 Mevzuat Asistanı")
    for m in st.session_state.messages: st.chat_message(m["role"]).markdown(m["content"])
    if p := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": p})
        st.session_state.sorgu_sayaci += 1
        st.chat_message("user").markdown(p)
        with st.chat_message("assistant"):
            if not st.session_state.vector_db:
                st.error("⚠️ Veritabanı bağlı değil.")
            else:
                with st.spinner("..."):
                    res = generate_answer(p, st.session_state.vector_db, st.session_state.messages)
                    daktilo_efekti(res["answer"])
                    if res["sources"]:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.caption("📚 Kaynaklar")
                        for s in res["sources"]: st.markdown(f'<div class="source-item">📄 {s}</div>', unsafe_allow_html=True)
                    full = res["answer"] + ("\n\nKaynaklar:\n" + "\n".join(res["sources"]) if res["sources"] else "")
                    st.session_state.messages.append({"role": "assistant", "content": full})
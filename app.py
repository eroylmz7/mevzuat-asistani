import streamlit as st
import datetime
import pytz
import time
import pandas as pd
import os
import asyncio 
from supabase import create_client
from data_ingestion import delete_document_cloud

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

# --- YENİ LOGLAMA SİSTEMİ (BULUTA KAYIT) ---
def log_kaydet(kullanici, soru, cevap):
    try:
        supabase.table("sorgu_loglari").insert({
            "kullanici_adi": kullanici,
            "soru": soru,
            "cevap": cevap
        }).execute()
    except Exception as e:
        print(f"Log Hatası: {e}")

# --- YENİ ANALİZ SİSTEMİ (BULUTTAN OKUMA) ---
def admin_analiz_getir():
    try:
        # Tüm logları çek
        response = supabase.table("sorgu_loglari").select("*").execute()
        df = pd.DataFrame(response.data)
        return df
    except:
        return pd.DataFrame()

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
if "username" not in st.session_state: st.session_state.username = ""
if "role" not in st.session_state: st.session_state.role = ""
if "analiz_acik" not in st.session_state: st.session_state.analiz_acik = False
# Mesaj geçmişi zaten vardır ama LangChain için chat_history de lazım
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

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

# Gerekli import (Dosyanın en başına eklemelisin)
from data_ingestion import process_pdfs, delete_document_cloud

# --- SIDEBAR BAŞLANGICI ---
with st.sidebar:
    # 1. KULLANICI KARTI
    rol_txt = "YÖNETİCİ" if st.session_state.role == "admin" else "ÖĞRENCİ"
    st.markdown(f"""<div class="user-card"><h2 style='margin:0;'>{st.session_state.username.upper()}</h2><p style='margin:0; opacity:0.9; font-size:0.9rem;'>{rol_txt} HESABI</p></div>""", unsafe_allow_html=True)

    # ========================================================
    #  YÖNETİCİ PANELİ (Sadece Admin Görür)
    # ========================================================
    if st.session_state.role == 'admin':
        if st.button("📊 Analiz Paneli"): st.session_state.analiz_acik = not st.session_state.analiz_acik
        
        # --- GELİŞMİŞ ANALİZ ---
        if st.session_state.analiz_acik:
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            df_log = admin_analiz_getir() # Bu fonksiyonun tanımlı olduğunu varsayıyoruz
            
            if not df_log.empty:
                toplam_soru = len(df_log)
                aktif_kullanici = df_log['kullanici_adi'].nunique()
                
                st.write(f"🔹 **Toplam Soru:** {toplam_soru}")
                st.write(f"🔹 **Aktif Öğrenci:** {aktif_kullanici}")
                st.markdown("---")
                st.caption("Son 5 Soru:")
                st.dataframe(df_log[['kullanici_adi', 'soru']].tail(5), hide_index=True)
            else:
                st.write("Henüz veri yok.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # --- DOSYA YÖNETİMİ ---
        st.subheader("📁 Veri Yönetimi")
        
        # 1. Dosya Yükleme
        uploaded_files = st.file_uploader("PDF Yükle", accept_multiple_files=True, type=['pdf'])
        
        # İSİM GÜNCELLENDİ: "Veritabanına Belge Ekle"
        if st.button("Veritabanına Belge Ekle", type="primary"):
            if uploaded_files:
                durum = st.status("Sistem güncelleniyor...", expanded=True)
                # Cloud işlem fonksiyonun
                st.session_state.vector_db = process_pdfs(uploaded_files)
                durum.update(label="✅ Belgeler Eklendi!", state="complete")
                st.rerun()
        
        # 2. Yüklü Dosyaları Listeleme ve Silme (Supabase'den Çeker)
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("📚 SİSTEMDEKİ BELGELER (YÖNET)")
        
        try:
            # Supabase'den dosya listesini çek
            docs = supabase.table("dokumanlar").select("*").execute()
            
            if docs.data:
                for d in docs.data:
                    dosya_adi = d["dosya_adi"]
                    
                    # Yan yana düzen: Dosya Adı | Sil Butonu
                    col1, col2 = st.columns([0.85, 0.15])
                    
                    with col1:
                        st.markdown(f'<div style="font-size:0.9em; padding-top:5px;">📄 {dosya_adi}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Çöp Kutusu Butonu
                        if st.button("🗑️", key=f"del_btn_{dosya_adi}", help="Belgeyi Sil"):
                            st.session_state.delete_target = dosya_adi
                            st.rerun()
            else:
                st.info("Henüz belge yüklenmemiş.")
                
        except Exception as e:
            st.error(f"Liste alınamadı: {e}")

        # --- SİLME ONAY KUTUSU (Confirmation) ---
        if "delete_target" in st.session_state and st.session_state.delete_target:
            target_file = st.session_state.delete_target
            
            with st.container():
                st.warning(f"⚠️ **{target_file}** silinecek. Emin misiniz?")
                col_yes, col_no = st.columns(2)
                
                with col_yes:
                    if st.button("✅ EVET, SİL", use_container_width=True):
                        with st.spinner("Siliniyor..."):
                            # Data_ingestion.py'deki fonksiyonu çağır
                            success, msg = delete_document_cloud(target_file)
                            if success:
                                st.success(msg)
                                del st.session_state.delete_target
                                st.rerun()
                            else:
                                st.error(msg)
                
                with col_no:
                    if st.button("❌ VAZGEÇ", use_container_width=True):
                        del st.session_state.delete_target
                        st.rerun()

        st.divider()

    # ========================================================
    #  ÖĞRENCİ GÖRÜNÜMÜ (Salt Okunur Liste)
    # ========================================================
    else:
        # Öğrenci sadece listeyi görür, butonları görmez
        st.subheader("📚 Mevzuat Listesi")
        try:
            docs = supabase.table("dokumanlar").select("dosya_adi").execute()
            if docs.data:
                for d in docs.data:
                    st.markdown(f"🔹 *{d['dosya_adi']}*")
            else:
                st.caption("Yüklü belge yok.")
        except:
            st.caption("Liste yüklenemedi.")
        
        st.divider()

    # ========================================================
    # ORTAK BUTONLAR (Sohbet İndir, Çıkış vb.)
    # ========================================================
    st.caption("İşlemler")
    
    # Sohbet İndirme
    if st.session_state.messages:
        # get_tr_time fonksiyonun import edildiğini varsayıyoruz
        tr_saat = get_tr_time() 
        log = f"🎓 SOHBET\n{tr_saat.strftime('%d.%m.%Y %H:%M')}\n" + "="*30 + "\n"
        for m in st.session_state.messages: log += f"[{m['role']}]: {m['content']}\n"
        st.download_button("📥 Sohbeti İndir", log, "chat.txt", use_container_width=True)
    
    st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
    
    if st.button("🗑️ Temizle", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Sohbet temizlendi."}]
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Çıkış Yap
    if st.button("🚪 Çıkış", type="secondary", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- SOHBET EKRANI ---
st.title("💬 Mevzuat Asistanı")
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if "vector_store" not in st.session_state or st.session_state.vector_store is None:
             st.warning("⚠️ Veritabanı bağlantısı yok. Lütfen sol menüden belge yükleyin veya 'Veritabanını Güncelle' deyin.")
        else:
            with st.spinner("Gemini (Cloud) düşünüyor..."):
                try:
                    # Cevabı al
                    sonuc = generate_answer(prompt, st.session_state.vector_store, st.session_state.chat_history)
                    
                    answer_text = sonuc["answer"]
                    sources = sonuc["sources"]

                    # --- KRİTİK DÜZELTME: OLUMSUZ CEVAPSA KAYNAKLARI GİZLE ---
                    # Eğer cevapta "bilgi yok" türevi şeyler geçiyorsa kaynakları boşalt.
                    negative_keywords = ["bilgi bulunamadı", "bilgi yer almıyor", "bilgim yok", "dokümanlarda bu bilgi yok"]
                    
                    if any(keyword in answer_text.lower() for keyword in negative_keywords):
                        sources = [] # Kaynak listesini sıfırla

                    # Kaynakları HTML Bloğu Olarak Hazırla
                    sources_html = ""
                    if sources: # Sadece kaynak varsa kutuyu oluştur
                        sources_html += '<div class="source-container"><div class="source-header">📚 REFERANSLAR</div>'
                        for src in sources:
                            sources_html += f'<div class="source-item"><span class="source-icon">📄</span> {src}</div>'
                        sources_html += '</div>'
                    
                    # Cevap ve Kaynakları Birleştir
                    final_content = answer_text + sources_html
                    
                    # Ekrana Bas
                    st.markdown(final_content, unsafe_allow_html=True)
                    
                    # Hafızaya Kaydet
                    st.session_state.messages.append({"role": "assistant", "content": final_content})
                    
                except Exception as e:
                    st.error(f"Hata: {e}")
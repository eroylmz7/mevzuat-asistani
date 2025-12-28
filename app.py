# --- BU 3 SATIR ÇOK ÖNEMLİ (CLOUD VERİTABANI YAMASI) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -------------------------------------------------------

import streamlit as st
import os
import shutil
import asyncio
import gc
import json
import time
import datetime
import chromadb
from dotenv import load_dotenv

# RAG ve LangChain Bileşenleri
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# Kendi fonksiyonlarımız
try:
    from data_ingestion import load_and_process_pdfs
except ImportError:
    st.error("⚠️ 'data_ingestion.py' dosyası bulunamadı!")

# -----------------------------------------------------------------------------
# 1. SAYFA VE TASARIM AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kampüs Asistanı", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()

# --- CSS TASARIMI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141E30) !important;
        background-size: 400% 400% !important;
        animation: gradient 15s ease infinite !important;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="column"]:nth-of-type(2), [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 12, 41, 0.9) !important;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AYARLAR
# -----------------------------------------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

PERSIST_DIRECTORY = "./chroma_db_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
USER_DB_FILE = "users.json"
LOG_FILE = "logs.json"

# --- ANALİZ ---
def log_query(username, question):
    entry = {
        "tarih": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kullanici": username,
        "soru": question
    }
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f: logs = json.load(f)
        except: logs = []
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f: json.dump(logs, f, ensure_ascii=False, indent=4)

def load_logs():
    if not os.path.exists(LOG_FILE): return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

# --- KULLANICI ---
def load_users():
    if not os.path.exists(USER_DB_FILE): return {}
    try:
        with open(USER_DB_FILE, "r") as f: return json.load(f)
    except: return {}

def save_users(users):
    with open(USER_DB_FILE, "w") as f: json.dump(users, f)

# -----------------------------------------------------------------------------
# 3. GİRİŞ SİSTEMİ
# -----------------------------------------------------------------------------
def login_system():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.session_state["role"] = None

    if st.session_state["logged_in"]: return True

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1]) 
    with col2:
        st.markdown("""<div style="text-align: center;"><h1 style="font-size: 4rem; margin-bottom: 0;">🎓</h1><h1 style="color: white; margin-top: -10px;">Kampüs Asistanı</h1></div>""", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
        
        with tab1:
            st.write("")
            username_in = st.text_input("Kullanıcı Adı", key="login_user", placeholder="Örn: admin")
            password_in = st.text_input("Şifre", type="password", key="login_pass", placeholder="••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Giriş Yap", use_container_width=True):
                users = load_users()
                if username_in in users and users[username_in]["password"] == password_in:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username_in
                    st.session_state["role"] = users[username_in]["role"]
                    st.success("Giriş Başarılı!")
                    time.sleep(0.5)
                    st.rerun()
                else: st.error("Hatalı kullanıcı adı veya şifre!")
        
        with tab2:
            st.write("")
            new_user = st.text_input("Yeni Kullanıcı Adı", key="reg_user")
            new_pass = st.text_input("Yeni Şifre", type="password", key="reg_pass")
            role = st.selectbox("Rol Seçiniz", ["Öğrenci", "Yönetici"], key="reg_role")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Hesap Oluştur", use_container_width=True):
                users = load_users()
                if new_user in users: st.warning("Bu isim zaten alınmış.")
                elif not new_user or not new_pass: st.warning("Alanlar boş bırakılamaz.")
                else:
                    users[new_user] = {"password": new_pass, "role": role}
                    save_users(users)
                    st.balloons()
                    st.success("Kayıt tamam! Giriş yapabilirsiniz.")
    return False

if not login_system(): st.stop()

# -----------------------------------------------------------------------------
# 4. BACKEND
# -----------------------------------------------------------------------------
@st.cache_resource
def get_vector_db():
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not os.path.exists(PERSIST_DIRECTORY): return None
    try:
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        return vectordb
    except Exception as e: return None

def get_llm_chain(vectordb):
    # SENİN İSTEDİĞİN MODEL AYARI BURADA:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    custom_template = """
    Sen üniversite mevzuatları konusunda uzman bir asistansın.
    Aşağıdaki sohbet geçmişini ve bağlamı (context) kullanarak soruyu cevapla.
     
    Kurallar:
    1. Sadece verilen bağlamı kullan. Uydurma cevap verme.
    2. Bağlamda bilgi yoksa "Yönetmeliklerde bu bilgiye rastlayamadım." de.
    3. Cevabı maddeleri referans göstererek ver.
     
    Sohbet Geçmişi:
    {chat_history}
     
    Bağlam:
    {context}
     
    Soru: {question}
    Cevap:
    """
    PROMPT = PromptTemplate(template=custom_template, input_variables=["chat_history", "context", "question"])
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 60}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=False
    )
    return qa_chain

# -----------------------------------------------------------------------------
# 5. ARAYÜZ
# -----------------------------------------------------------------------------

analytics_placeholder = None

with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚙️ Panel</h2>", unsafe_allow_html=True)
    
    if st.session_state["role"] == "Yönetici":
        st.info(f"Yönetici: {st.session_state['username']}")
        admin_tab1, admin_tab2 = st.tabs(["📂 PDF Yükle", "📊 Analiz"])
        
        with admin_tab1:
            uploaded_files = st.file_uploader("Dosya Ekle", accept_multiple_files=True, type="pdf")
            btn_disabled = not uploaded_files 
            
            if st.button("Veritabanını Güncelle", type="primary", use_container_width=True, disabled=btn_disabled):
                status_container = st.empty()
                try:
                    status_container.info("1. İşlem Başlatılıyor...")
                    if not os.path.exists("./veriler"): os.makedirs("./veriler")
                    else:
                        for f in os.listdir("./veriler"): os.remove(os.path.join("./veriler", f))
                    
                    status_container.info(f"2. {len(uploaded_files)} yeni dosya kaydediliyor...")
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            with open(os.path.join("./veriler", uploaded_file.name), "wb") as f: 
                                f.write(uploaded_file.getbuffer())
                    
                    status_container.info("3. Temizlik yapılıyor...")
                    st.cache_resource.clear()
                    if os.path.exists(PERSIST_DIRECTORY): shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
                    time.sleep(1)
                    
                    status_container.info("4. PDF'ler okunuyor...")
                    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                    chunks = load_and_process_pdfs()
                    
                    if chunks:
                        status_container.info("5. Veritabanı kuruluyor...")
                        Chroma.from_documents(chunks, embedding_model, persist_directory=PERSIST_DIRECTORY)
                        status_container.success("✅ GÜNCELLEME TAMAMLANDI!")
                        time.sleep(2)
                        st.rerun()
                    else: status_container.error("❌ HATA: Metin okunamadı.")
                except Exception as e: status_container.error(f"❌ HATA: {str(e)}")
        
        with admin_tab2:
            analytics_placeholder = st.empty()
            def render_analytics():
                logs = load_logs()
                with analytics_placeholder.container():
                    if logs:
                        st.metric("Toplam Soru", len(logs))
                        st.write("Son Sorulanlar:")
                        st.dataframe(logs[::-1], height=300)
                    else: st.info("Henüz veri yok.")
            render_analytics()
    else:
        st.info(f"Öğrenci: {st.session_state['username']}")

    st.markdown("---")
    if st.button("Çıkış Yap", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

# --- CHAT EKRANI ---
st.markdown("<h2 style='text-align: center;'>🎓 Mevzuat Asistanı</h2>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Hangi yönetmeliği merak ediyorsun?"}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "🧑‍🎓"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.chat_message("user", avatar="🧑‍🎓").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Loglama
    log_query(st.session_state["username"], prompt)
    if st.session_state["role"] == "Yönetici" and analytics_placeholder is not None:
        render_analytics()

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⚡ *Düşünüyor...*")
        try:
            vectordb = get_vector_db()
            if vectordb is None: st.error("⚠️ Veritabanı BOŞ.")
            else:
                qa_chain = get_llm_chain(vectordb)
                res = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                
                answer_text = res['answer']
                source_docs = res['source_documents']
                
                # Kaynakları düzenle
                source_map = {}
                for doc in source_docs:
                    source_name = os.path.basename(doc.metadata.get('source', 'Bilinmiyor'))
                    page_num = doc.metadata.get('page', 0) + 1
                    if source_name not in source_map: source_map[source_name] = set()
                    source_map[source_name].add(page_num)
                
                formatted_sources = []
                for name, pages in source_map.items():
                    sorted_pages = sorted(list(pages))
                    if len(sorted_pages) > 5: page_str = ", ".join(map(str, sorted_pages[:5])) + "..."
                    else: page_str = ", ".join(map(str, sorted_pages))
                    formatted_sources.append(f"**{name}** (Sayfalar: {page_str})")
                
                final_answer = f"{answer_text}\n\n---\n📚 **Kaynaklar:**\n" + "\n".join([f"- {s}" for s in formatted_sources])
                
                # Daktilo Efekti
                def stream_data():
                    for word in final_answer.split(" "):
                        yield word + " "
                        time.sleep(0.02)
                message_placeholder.write_stream(stream_data)

                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append((prompt, answer_text))

        except Exception as e: message_placeholder.error(f"Hata: {str(e)}")

# --- SOHBETİ İNDİR BUTONU ---
if st.session_state.messages and len(st.session_state.messages) > 1:
    st.markdown("---")
    chat_text = "🎓 MEVZUAT ASİSTANI - SOHBET KAYDI\n"
    chat_text += f"Tarih: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    chat_text += "-"*50 + "\n\n"
    
    for msg in st.session_state.messages:
        role = "ASİSTAN" if msg["role"] == "assistant" else "ÖĞRENCİ"
        content = msg["content"]
        chat_text += f"[{role}]: {content}\n\n"
        chat_text += "-"*30 + "\n\n"

    st.download_button(
        label="📥 Sohbeti İndir (.txt)",
        data=chat_text,
        file_name="sohbet_gecmisi.txt",
        mime="text/plain"
    )
# -----------------------------------------------------------------------------
# 1. BULUT VERİTABANI YAMASI (Mecburi - Dokunma)
# -----------------------------------------------------------------------------
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# -----------------------------------------------------------------------------
# KÜTÜPHANELER
# -----------------------------------------------------------------------------
import streamlit as st
import os
import shutil
import time
import json
from dotenv import load_dotenv

# Eski ve Sorunsuz Importlar
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# Kendi fonksiyonun
from data_ingestion import load_and_process_pdfs

# -----------------------------------------------------------------------------
# AYARLAR VE TASARIM
# -----------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Kampüs Asistanı", page_icon="🎓", layout="wide")

PERSIST_DIRECTORY = "./chroma_db_store"
USERS_FILE = "users.json"

# --- YARDIMCI FONKSİYONLAR ---

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

@st.cache_resource
def get_vector_db():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if os.path.exists(PERSIST_DIRECTORY):
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        if vectordb._collection.count() > 0:
            return vectordb
    return None

def get_llm_chain(vectordb):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    template = """
    Sen üniversite mevzuatları konusunda uzman, arkadaş canlısı bir asistansın.
    
    Kurallar:
    1. SADECE aşağıdaki bağlamı kullan.
    2. Cevabı maddeler halinde ve anlaşılır ver.
    3. Bilgi yoksa "Yönetmeliklerde bu bilgiye rastlayamadım." de.
    
    Bağlam: {context}
    Soru: {question}
    Geçmiş: {chat_history}
    
    Cevap:
    """
    PROMPT = PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

# -----------------------------------------------------------------------------
# ARAYÜZ (SIDEBAR)
# -----------------------------------------------------------------------------

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3406/3406987.png", width=80)
    st.title("🎓 Mevzuat Paneli")
    
    users_db = load_users()

    if not st.session_state.logged_in:
        # --- SEKME YAPISI (SENİN İSTEDİĞİN GİBİ) ---
        tab1, tab2 = st.tabs(["🔑 Giriş Yap", "📝 Kayıt Ol"])
        
        with tab1:
            st.subheader("Hoş Geldiniz")
            u_login = st.text_input("Kullanıcı Adı", key="login_user")
            p_login = st.text_input("Şifre", type="password", key="login_pass")
            
            if st.button("Giriş Yap", type="primary", use_container_width=True):
                if u_login in users_db and users_db[u_login]["password"] == p_login:
                    st.session_state.logged_in = True
                    st.session_state.username = u_login
                    st.session_state.role = users_db[u_login]["role"]
                    st.success("Giriş başarılı!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")

        with tab2:
            st.subheader("Yeni Hesap")
            new_user = st.text_input("Kullanıcı Adı Belirle", key="reg_user")
            new_pass = st.text_input("Şifre Belirle", type="password", key="reg_pass")
            
            if st.button("Kayıt Ol", type="secondary", use_container_width=True):
                if new_user and new_pass:
                    if new_user in users_db:
                        st.warning("Bu kullanıcı adı zaten alınmış.")
                    else:
                        users_db[new_user] = {"password": new_pass, "role": "student"}
                        save_users(users_db)
                        st.success("Kayıt başarılı! Şimdi 'Giriş Yap' sekmesinden girebilirsin.")
                else:
                    st.warning("Lütfen tüm alanları doldur.")

    else:
        # --- GİRİŞ YAPILINCA GÖRÜNEN KISIM ---
        st.success(f"👤 Aktif Kullanıcı: **{st.session_state.username}**")
        
        if st.session_state.get("role") == "admin":
            st.divider()
            st.markdown("### 🛠️ Yönetici Paneli")
            files = st.file_uploader("PDF Yükle", type=["pdf"], accept_multiple_files=True)
            
            if st.button("Sistemi Güncelle", type="primary"):
                if files:
                    if not os.path.exists("./veriler"): os.makedirs("./veriler")
                    for f in files:
                        with open(os.path.join("./veriler", f.name), "wb") as w: w.write(f.getbuffer())
                    
                    if os.path.exists(PERSIST_DIRECTORY): shutil.rmtree(PERSIST_DIRECTORY)
                    
                    with st.status("Veritabanı güncelleniyor...", expanded=True) as status:
                        st.write("📄 Dosyalar okunuyor...")
                        chunks = load_and_process_pdfs()
                        st.write("🧠 Vektör veritabanı kuruluyor...")
                        if chunks:
                            Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"), persist_directory=PERSIST_DIRECTORY)
                            status.update(label="✅ İşlem Başarılı!", state="complete")
                            time.sleep(1)
                            st.rerun()
        
        st.divider()
        if st.button("Çıkış Yap", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

# -----------------------------------------------------------------------------
# ANA SOHBET EKRANI
# -----------------------------------------------------------------------------

st.title("🏛️ Kampüs Mevzuat Asistanı")
st.markdown("Merhaba! Yönetmelikler hakkında aklına takılan her şeyi sorabilirsin.")

if st.session_state.logged_in:
    vectordb = get_vector_db()
    
    if vectordb:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Sorunuzu buraya yazın..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("⚡ *Düşünüyor...*")
                
                try:
                    chain = get_llm_chain(vectordb)
                    res = chain({"question": prompt, "chat_history": st.session_state.chat_history})
                    answer = res['answer']
                    
                    # Kaynakları Güzelleştirme
                    sources = []
                    seen = set()
                    for doc in res['source_documents']:
                        name = os.path.basename(doc.metadata.get('source', 'Belge'))
                        page = doc.metadata.get('page', 0) + 1
                        key = f"{name} (Sayfa {page})"
                        if key not in seen:
                            sources.append(key)
                            seen.add(key)
                    
                    final_text = f"{answer}\n\n📚 **Kaynaklar:**\n" + "\n".join([f"- {s}" for s in sources])
                    
                    # Daktilo Efekti
                    def stream():
                        for word in final_text.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    placeholder.write_stream(stream)
                    
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    st.session_state.chat_history.append((prompt, answer))
                
                except Exception as e:
                    placeholder.error(f"Hata oluştu: {e}")
    else:
        st.info("👋 Hoş geldin! Sistem şu an boş görünüyor. Lütfen yönetici hesabıyla giriş yapıp PDF yükleyin.")

else:
    st.warning("👈 Lütfen sol taraftaki panelden **Giriş Yapın** veya **Kayıt Olun**.")
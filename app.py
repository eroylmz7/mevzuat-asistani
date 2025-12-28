# -----------------------------------------------------------------------------
# 1. BULUT VERİTABANI YAMASI (EN ÜSTTE OLMALI)
# -----------------------------------------------------------------------------
import sys
import os

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# -----------------------------------------------------------------------------
# KÜTÜPHANELER
# -----------------------------------------------------------------------------
import streamlit as st
import shutil
import time
import json
import datetime
from dotenv import load_dotenv

# RAG ve LangChain Bileşenleri (KARARLI SÜRÜM AYARLARI)
from langchain_community.vectorstores import Chroma
# Yeni "langchain_huggingface" yerine eski "community" içinden çağırıyoruz:
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
# Prompt ve Chain'leri ana paketten çağırıyoruz (0.1.20 sürümü bunu destekler):
from langchain.prompts import PromptTemplate  
from langchain.chains import ConversationalRetrievalChain 

# Kendi fonksiyonlarımız
from data_ingestion import load_and_process_pdfs

# -----------------------------------------------------------------------------
# AYARLAR VE SABİTLER
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Kampüs Asistanı",
    page_icon="🎓",
    layout="wide"
)

PERSIST_DIRECTORY = "./chroma_db_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# -----------------------------------------------------------------------------
# FONKSİYONLAR
# -----------------------------------------------------------------------------

@st.cache_resource
def get_vector_db():
    # Model yükleme
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 1. YÖNTEM: Mevcut veritabanı kontrolü
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            print("💾 Mevcut veritabanı kontrol ediliyor...")
            vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
            if vectordb._collection.count() > 0:
                print("✅ Veritabanı sağlam.")
                return vectordb
        except Exception as e:
            print(f"⚠️ Hata: {e}")

    # 2. YÖNTEM: Otomatik Onarım (Auto-Healing)
    print("🔄 Veritabanı sıfırdan kuruluyor...")
    if os.path.exists("./veriler") and os.listdir("./veriler"):
        try:
            with st.spinner("Sistem hazırlanıyor (Bu işlem bir kez yapılır)..."):
                chunks = load_and_process_pdfs()
                if chunks:
                    vectordb = Chroma.from_documents(chunks, embedding, persist_directory=PERSIST_DIRECTORY)
                    print("✅ Kurulum tamamlandı!")
                    return vectordb
        except Exception as e:
            st.error(f"❌ Kurulum hatası: {e}")
            return None
    return None

def get_llm_chain(vectordb):
    # Gemini Ayarları
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    custom_template = """
    Sen üniversite mevzuatları konusunda uzman bir asistansın.
    Kurallar:
    1. SADECE verilen bağlamı kullan.
    2. Tarih ve gün hesaplamaları için kendi bilgini kullan.
    3. Bilgi yoksa "Bilmiyorum" de.
    
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
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=False
    )
    return qa_chain

# -----------------------------------------------------------------------------
# ARAYÜZ
# -----------------------------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("⚙️ Panel")
    users = {}
    try:
        if os.path.exists("users.json"):
            with open("users.json", "r") as f:
                users = json.load(f)
    except: pass

    if not st.session_state.logged_in:
        u = st.text_input("Kullanıcı Adı")
        p = st.text_input("Şifre", type="password")
        if st.button("Giriş Yap"):
            if u in users and users[u]["password"] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.role = users[u]["role"]
                st.rerun()
            else:
                st.error("Hatalı giriş!")
    else:
        st.info(f"Kullanıcı: {st.session_state.username}")
        if st.session_state.get("role") == "admin":
            st.divider()
            files = st.file_uploader("PDF Yükle", type=["pdf"], accept_multiple_files=True)
            if st.button("Güncelle"):
                if files:
                    if not os.path.exists("./veriler"): os.makedirs("./veriler")
                    for f in files:
                        with open(os.path.join("./veriler", f.name), "wb") as w:
                            w.write(f.getbuffer())
                    shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
                    st.rerun()
        if st.button("Çıkış"):
            st.session_state.logged_in = False
            st.rerun()

st.title("🎓 Mevzuat Asistanı")

if st.session_state.logged_in:
    vectordb = get_vector_db()
    if vectordb:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Sorunuzu yazın..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("⚡ *Düşünüyor...*")
                try:
                    chain = get_llm_chain(vectordb)
                    res = chain({"question": prompt, "chat_history": st.session_state.chat_history})
                    answer = res['answer']
                    
                    # Kaynakları formatla
                    sources = []
                    seen = set()
                    for doc in res['source_documents']:
                        name = os.path.basename(doc.metadata.get('source', 'Belge'))
                        page = doc.metadata.get('page', 0) + 1
                        key = f"{name} (S.{page})"
                        if key not in seen:
                            sources.append(key)
                            seen.add(key)
                    
                    final = f"{answer}\n\n📚 **Kaynaklar:**\n" + "\n".join([f"- {s}" for s in sources])
                    
                    # Streaming Efekti
                    def stream():
                        for word in final.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    placeholder.write_stream(stream)
                    
                    st.session_state.messages.append({"role": "assistant", "content": final})
                    st.session_state.chat_history.append((prompt, answer))
                except Exception as e:
                    placeholder.error(f"Hata: {e}")
    else:
        st.error("Veritabanı şu an hazır değil. Yönetici PDF yüklememiş olabilir.")
else:
    st.info("Lütfen giriş yapınız.")

# İndirme Butonu
if st.session_state.messages:
    st.markdown("---")
    txt = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in st.session_state.messages])
    st.download_button("Sohbeti İndir", txt, "chat.txt")
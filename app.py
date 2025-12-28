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

# Eski ve Sorunsuz Importlar (Görsel öğeler için gerekli)
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

# --- FONKSİYONLAR ---

@st.cache_resource
def get_vector_db():
    # Veritabanı var mı kontrol et
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if os.path.exists(PERSIST_DIRECTORY):
        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        if vectordb._collection.count() > 0:
            return vectordb
    return None

def get_llm_chain(vectordb):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    # O eski güzel cevap formatı
    template = """
    Sen üniversite mevzuatları konusunda uzman, yardımsever bir asistansın.
    
    Kurallar:
    1. SADECE aşağıdaki bağlamı kullan.
    2. Cevabı maddeler halinde, okunaklı ver.
    3. Bilgi yoksa kibarca "Yönetmeliklerde bulamadım" de.
    
    Bağlam: {context}
    Soru: {question}
    Geçmiş: {chat_history}
    
    Cevap:
    """
    PROMPT = PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True, # Kaynakları göstermek için şart
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

# -----------------------------------------------------------------------------
# ARAYÜZ (SIDEBAR)
# -----------------------------------------------------------------------------

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.title("🎓 Mevzuat Paneli")
    
    # Kullanıcı Verilerini Yükle
    users = {}
    if os.path.exists("users.json"):
        with open("users.json", "r") as f: users = json.load(f)
            
    if not st.session_state.logged_in:
        st.subheader("Giriş Yap")
        u = st.text_input("Kullanıcı Adı")
        p = st.text_input("Şifre", type="password")
        if st.button("Giriş Yap", type="primary"):
            if u in users and users[u]["password"] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.role = users[u]["role"]
                st.rerun()
            else: st.error("Hatalı kullanıcı adı veya şifre!")
    else:
        st.success(f"Hoş geldin, **{st.session_state.username}**")
        
        # --- YÖNETİCİ KISMI ---
        if st.session_state.get("role") == "admin":
            st.divider()
            st.markdown("### 🛠️ Yönetici Araçları")
            files = st.file_uploader("PDF Yükle", type=["pdf"], accept_multiple_files=True)
            
            if st.button("Veritabanını Güncelle", type="primary"):
                if files:
                    if not os.path.exists("./veriler"): os.makedirs("./veriler")
                    for f in files:
                        with open(os.path.join("./veriler", f.name), "wb") as w: w.write(f.getbuffer())
                    
                    if os.path.exists(PERSIST_DIRECTORY): shutil.rmtree(PERSIST_DIRECTORY)
                    
                    with st.status("PDF'ler işleniyor...", expanded=True) as status:
                        st.write("📄 Metinler okunuyor...")
                        chunks = load_and_process_pdfs()
                        st.write("🧠 Yapay zeka hafızası oluşturuluyor...")
                        if chunks:
                            Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"), persist_directory=PERSIST_DIRECTORY)
                            status.update(label="✅ İşlem Tamam!", state="complete", expanded=False)
                            time.sleep(1)
                            st.rerun()
        
        st.divider()
        if st.button("Çıkış Yap"):
            st.session_state.logged_in = False
            st.rerun()

# -----------------------------------------------------------------------------
# ANA SOHBET EKRANI (CHAT)
# -----------------------------------------------------------------------------

st.title("🏛️ Mevzuat Asistanı")
st.markdown("Üniversite yönetmelikleri hakkında her şeyi sorabilirsin.")

if st.session_state.logged_in:
    vectordb = get_vector_db()
    
    if vectordb:
        # Eski mesajları göster
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        # Yeni soru girişi
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
                    
                    final_text = f"{answer}\n\n📚 **Referanslar:**\n" + "\n".join([f"- {s}" for s in sources])
                    
                    # --- DAKTİLO EFEKTİ (Geri Döndü!) ---
                    def stream():
                        for word in final_text.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    placeholder.write_stream(stream)
                    # ------------------------------------

                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    st.session_state.chat_history.append((prompt, answer))
                
                except Exception as e:
                    placeholder.error(f"Bir hata oluştu: {e}")
    else:
        st.warning("⚠️ Sistem şu an boş. Lütfen yönetici panelinden PDF yükleyerek veritabanını oluşturun.")
else:
    st.info("👈 Lütfen sol panelden giriş yapınız.")
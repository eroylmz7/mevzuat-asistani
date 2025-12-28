# -----------------------------------------------------------------------------
# 1. BULUT VERİTABANI YAMASI (EN ÜSTTE OLMALI)
# -----------------------------------------------------------------------------
import sys
import os

try:
    # Bu kısım sadece Streamlit Cloud'da (Linux) çalışır
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Local bilgisayarda (Windows) bu kütüphane yoktur,
    # standart sqlite3 ile devam et.
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

# RAG ve LangChain Bileşenleri
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# Kendi fonksiyonlarımız (data_ingestion.py dosyanın olduğundan emin ol)
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
    """
    Veritabanını yükler. Eğer 'Doku Uyuşmazlığı' (Windows->Linux) yüzünden
    okuyamazsa, './veriler' klasöründeki PDF'lerden anında sıfırdan kurar.
    """
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 1. YÖNTEM: Mevcut veritabanını okumayı dene
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            print("💾 Mevcut veritabanı kontrol ediliyor...")
            vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
            
            # Basit bir okuma testi yapalım
            if vectordb._collection.count() > 0:
                print("✅ Veritabanı sağlam, yüklendi.")
                return vectordb
        except Exception as e:
            print(f"⚠️ Veritabanı okunamadı (OS Uyuşmazlığı): {e}")

    # 2. YÖNTEM: Okuyamadıysa veya yoksa SIFIRDAN KUR (Auto-Healing)
    print("🔄 Otomatik Onarım Modu: Veritabanı sıfırdan kuruluyor...")
    
    if os.path.exists("./veriler") and os.listdir("./veriler"):
        try:
            with st.spinner("Sistem ilk kez hazırlanıyor, lütfen bekleyiniz..."):
                # PDF'leri işle
                chunks = load_and_process_pdfs()
                if chunks:
                    # Sıfırdan veritabanı oluştur
                    vectordb = Chroma.from_documents(chunks, embedding, persist_directory=PERSIST_DIRECTORY)
                    print("✅ Otomatik kurulum tamamlandı!")
                    return vectordb
        except Exception as e:
            st.error(f"❌ Kritik Hata: Otomatik kurulum yapılamadı. {e}")
            return None
    else:
        # Veriler klasörü de boşsa yapacak bir şey yok
        return None

def get_llm_chain(vectordb):
    """
    Yapay Zeka ayarları ve Prompt şablonu.
    """
    # Gemini 1.5 Flash (Hızlı ve Ucuz)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    # --- AKILLI PROMPT (Tarih ve Gün Hesabı Yapabilen) ---
    custom_template = """
    Sen üniversite mevzuatları konusunda uzman, yardımsever bir asistansın.
    Aşağıdaki sohbet geçmişini ve bağlamı (context) kullanarak soruyu cevapla.
    
    Kurallar:
    1. Mevzuat maddeleri (süreler, cezalar, notlar) için SADECE verilen bağlamı kullan. Asla uydurma.
    2. Tarih hesaplamaları, "Hafta sonu iş günü müdür?", "Bugün pazartesi ise 5 gün sonra ne olur?" gibi mantık soruları için KENDİ GENEL BİLGİNİ kullan.
    3. Bağlamda bilgi yoksa "Yönetmeliklerde bu bilgiye rastlayamadım." de.
    4. Cevabı maddeleri referans göstererek ver.
    
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
# ARAYÜZ (SIDEBAR - GİRİŞ VE PANEL)
# -----------------------------------------------------------------------------

# Oturum Durumu Başlatma
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
    
    # Kullanıcı Verilerini Yükle
    users = {}
    try:
        with open("users.json", "r") as f:
            users = json.load(f)
    except FileNotFoundError:
        st.error("Kullanıcı veritabanı (users.json) bulunamadı.")

    # Giriş Ekranı
    if not st.session_state.logged_in:
        username_input = st.text_input("Kullanıcı Adı")
        password_input = st.text_input("Şifre", type="password")
        
        if st.button("Giriş Yap"):
            if username_input in users and users[username_input]["password"] == password_input:
                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.session_state.role = users[username_input]["role"]
                st.success(f"Hoş geldin {username_input}!")
                st.rerun()
            else:
                st.error("Hatalı kullanıcı adı veya şifre!")
    
    else:
        # Giriş Yapılmış Durum
        st.info(f"Öğrenci: {st.session_state.username}")
        st.caption("Soru sorarak yönetmelikleri öğrenebilirsin.")

        # --- YÖNETİCİ ÖZEL ALANI ---
        if st.session_state.role == "admin":
            st.divider()
            st.subheader("🔧 Yönetici Araçları")
            
            uploaded_files = st.file_uploader("PDF Yükle (Yönetmelik)", type=["pdf"], accept_multiple_files=True)
            
            if st.button("Veritabanını Güncelle"):
                if uploaded_files:
                    if not os.path.exists("./veriler"):
                        os.makedirs("./veriler")
                    
                    # Dosyaları kaydet
                    for file in uploaded_files:
                        with open(os.path.join("./veriler", file.name), "wb") as f:
                            f.write(file.getbuffer())
                    
                    st.toast("PDF'ler işleniyor, lütfen bekleyin...", icon="⏳")
                    
                    # Veritabanını sıfırla ve yeniden kur
                    if os.path.exists(PERSIST_DIRECTORY):
                        shutil.rmtree(PERSIST_DIRECTORY)
                    
                    chunks = load_and_process_pdfs()
                    if chunks:
                        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                        Chroma.from_documents(chunks, embedding, persist_directory=PERSIST_DIRECTORY)
                        st.success("✅ GÜNCELLEME TAMAMLANDI!")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.warning("Lütfen önce dosya seçin.")

        if st.button("Çıkış Yap"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.messages = []
            st.rerun()

# -----------------------------------------------------------------------------
# ANA SOHBET EKRANI
# -----------------------------------------------------------------------------

st.title("🎓 Mevzuat Asistanı")

if st.session_state.logged_in:
    # 1. Veritabanını Getir (Auto-Healing ile)
    vectordb = get_vector_db()

    if vectordb is None:
        st.error("🚨 Veritabanı şu an boş ve oluşturulamadı. Lütfen yöneticinin PDF yüklemesini bekleyin.")
    else:
        # 2. Sohbet Geçmişini Göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. Yeni Soru Girişi
        if prompt := st.chat_input("Sorunuzu yazın..."):
            # Kullanıcı mesajını ekle
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Cevap Üretimi
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⚡ *Düşünüyor...*")
                
                try:
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
                        page_str = ", ".join(map(str, sorted_pages))
                        formatted_sources.append(f"**{name}** (Sayfalar: {page_str})")
                    
                    final_answer = f"{answer_text}\n\n📚 **Kaynaklar:**\n" + "\n".join([f"- {s}" for s in formatted_sources])
                    
                    # --- DAKTİLO EFEKTİ (STREAMING) ---
                    def stream_data():
                        for word in final_answer.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                            
                    message_placeholder.write_stream(stream_data)
                    # ----------------------------------

                    # Geçmişe kaydet
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    st.session_state.chat_history.append((prompt, answer_text))
                
                except Exception as e:
                    message_placeholder.error(f"Bir hata oluştu: {str(e)}")

else:
    st.info("Lütfen sol taraftaki panelden giriş yapınız.")

# -----------------------------------------------------------------------------
# SOHBETİ İNDİR (SAYFANIN EN ALTI)
# -----------------------------------------------------------------------------
if st.session_state.messages:
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
        file_name="mevzuat_sohbeti.txt",
        mime="text/plain",
        use_container_width=True
    )
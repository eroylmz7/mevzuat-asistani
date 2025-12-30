import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from supabase import create_client
from pinecone import Pinecone
import streamlit as st

def process_pdfs(uploaded_files):
    # Supabase Bağlantısı (Kayıt Defteri İçin)
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    
    all_documents = []
    
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
        
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp_pdfs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nMadde", "\n\n", "\n", ". ", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        
        for doc in split_docs:
            doc.metadata["source"] = uploaded_file.name
            
        all_documents.extend(split_docs)
        os.remove(file_path)
        
        # --- YENİ EKLENEN KISIM: KAYIT DEFTERİNE YAZ ---
        try:
            # Önce bu isimde eski kayıt varsa silelim (Duplicate olmasın)
            supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
            # Yeni kaydı ekleyelim
            supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
        except Exception as e:
            print(f"Supabase kayıt hatası: {e}")
        # -----------------------------------------------

    if all_documents:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        vector_store = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            index_name="mevzuat-asistani"
        )
        return vector_store
    
    return None
def delete_document_cloud(file_name):
    """
    Belirtilen dosyayı hem Pinecone vektör veritabanından 
    hem de Supabase kayıtlarından siler.
    """
    try:
        # --- 1. ADIM: PINECONE'DAN SİL (Vektörler) ---
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        index_name = "mevzuat-asistani" # Index isminin doğruluğundan emin ol

        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)

        # 'source' metadata'sı dosya adına eşit olan vektörleri sil
        index.delete(filter={"source": file_name})
        
        # --- 2. ADIM: SUPABASE'DEN SİL (Liste Kaydı) ---
        # Eğer bunu yapmazsak sol menüde isim durmaya devam eder.
        try:
            supabase_url = st.secrets["SUPABASE_URL"]
            supabase_key = st.secrets["SUPABASE_KEY"]
            supabase = create_client(supabase_url, supabase_key)
            
            # 'dokumanlar' tablosundan dosya ismine göre satırı sil
            supabase.table("dokumanlar").delete().eq("dosya_adi", file_name).execute()
            
        except Exception as e:
            # Pinecone silindi ama Supabase silinemedi ise kullanıcıya söylemeyelim,
            # sistem çalışmaya devam etsin ama log düşsün.
            print(f"Supabase silme hatası: {e}")

        return True, f"{file_name} sistemden tamamen kaldırıldı."
        
    except Exception as e:
        return False, f"Silme işlemi sırasında hata oluştu: {str(e)}"

# --- Otomatik olarak veritabanına bağlantı sağlama ---

def connect_to_existing_index():
    """
    Dosya yüklemeden, sadece mevcut Pinecone index'ine bağlanır.
    """
    try:
        # API Anahtarlarını Al
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        
        # 1. Embedding Modelini Hazırla
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
        
        # 2. Mevcut Index'e Bağlan
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="mevzuat-asistani", # Senin index ismin
            embedding=embeddings
        )
        
        return vector_store
    except Exception as e:
        st.error(f"Otomatik bağlantı hatası: {e}")
        return None
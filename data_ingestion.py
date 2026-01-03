import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from supabase import create_client
from pinecone import Pinecone

# --- BELGE İŞLEME VE YÜKLEME ---
def process_pdfs(uploaded_files):
    # Supabase Bağlantısı
    try:
        supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception as e:
        st.error(f"Supabase bağlantı hatası: {e}")
        return None
    
    all_documents = []
    
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
        
    for uploaded_file in uploaded_files:
        try:
            # ---------------------------------------------------------
            # 1. ADIM: FİZİKSEL DOSYAYI STORAGE'A YÜKLE (EKSİK OLAN KISIM BUYDU)
            # ---------------------------------------------------------
            try:
                # Dosyayı okumadan önce başa saralım (Garanti olsun)
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                
                # 'belgeler' bucket'ına yükle
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name,
                    file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            except Exception as storage_err:
                print(f"Storage yükleme uyarısı ({uploaded_file.name}): {storage_err}")

            # ---------------------------------------------------------
            # 2. ADIM: VEKTÖR İŞLEME (MEVCUT KODUNUZ)
            # ---------------------------------------------------------
            # Dosyayı tekrar başa sar (Yukarıda read() yaptığımız için imleç sonda kaldı)
            uploaded_file.seek(0)
            
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,      
                chunk_overlap=250,    
                separators=["\nMADDE ", "\nMadde ", "\nGEÇİCİ MADDE", "\n\n", "\n", ". ", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)
            
            for doc in split_docs:
                doc.metadata["source"] = uploaded_file.name
            
            all_documents.extend(split_docs)
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # ---------------------------------------------------------
            # 3. ADIM: VERİTABANI TABLOSUNU GÜNCELLE
            # ---------------------------------------------------------
            try:
                # Önce tablo kaydını sil (varsa), sonra ekle
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except Exception as e:
                print(f"Supabase tablo kayıt hatası: {e}")
            
        except Exception as e:
            st.error(f"{uploaded_file.name} işlenirken hata: {e}")

    # ---------------------------------------------------------
    # 4. ADIM: PINECONE VEKTÖR YÜKLEME
    # ---------------------------------------------------------
    if all_documents:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'} # CPU zorlaması (Garanti olsun)
        )
        
        vector_store = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            index_name="mevzuat-asistani"
        )
        return vector_store
    
    return None

# --- BELGE SİLME (STORAGE DAHİL) ---
def delete_document_cloud(file_name):
    try:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        index_name = "mevzuat-asistani"

        # 1. Pinecone'dan sil (Vektörler)
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        index.delete(filter={"source": file_name})
        
        try:
            supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            
            # 2. Tablodan sil (Liste)
            supabase.table("dokumanlar").delete().eq("dosya_adi", file_name).execute()
            
            # 3. Storage'dan sil (Fiziksel Dosya) - YENİ EKLENDİ ✅
            supabase.storage.from_("belgeler").remove([file_name])
            
        except Exception as e:
            print(f"Supabase silme hatası: {e}")

        return True, f"{file_name} başarıyla silindi."
    except Exception as e:
        return False, f"Hata: {str(e)}"

# --- OTOMATİK BAĞLANTI ---
def connect_to_existing_index():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="mevzuat-asistani",
            embedding=embedding_model
        )
        return vector_store
    except Exception as e:
        st.error(f"Otomatik bağlantı hatası: {e}")
        return None
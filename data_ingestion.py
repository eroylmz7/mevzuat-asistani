import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Google Embeddings'i SİLDİK ❌
# HuggingFace'i EKLEDİK ✅
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
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,      # Biraz büyüttük (Madde bütünlüğü için)
                chunk_overlap=250,    # Örtüşmeyi artırdık (Bağlam kopmasın)
                separators=[
                    "\nMADDE ",       # En öncelikli bölme yeri (Büyük harf)
                    "\nMadde ",       # İkinci öncelik
                    "\nGEÇİCİ MADDE", # Geçici maddeler
                    "\n\n",           # Paragraflar
                    "\n",             # Satırlar
                    ". ",             # Cümleler
                    " ",              # Kelimeler
                    ""
                ]
            )
            split_docs = text_splitter.split_documents(documents)
            
            for doc in split_docs:
                doc.metadata["source"] = uploaded_file.name
            
            all_documents.extend(split_docs)
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Kayıt Defteri (Supabase) Güncelleme
            try:
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except Exception as e:
                print(f"Supabase kayıt hatası: {e}")
            
        except Exception as e:
            st.error(f"{uploaded_file.name} işlenirken hata: {e}")

    if all_documents:
        # ✅ BURASI ARTIK HUGGINGFACE
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        vector_store = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            index_name="mevzuat-asistani"
        )
        return vector_store
    
    return None

# --- BELGE SİLME ---
def delete_document_cloud(file_name):
    try:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        index_name = "mevzuat-asistani"

        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)

        index.delete(filter={"source": file_name})
        
        try:
            supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            supabase.table("dokumanlar").delete().eq("dosya_adi", file_name).execute()
        except:
            pass

        return True, f"{file_name} silindi."
    except Exception as e:
        return False, f"Hata: {str(e)}"

# --- OTOMATİK BAĞLANTI ---
def connect_to_existing_index():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="mevzuat-asistani",
            embedding=embedding_model
        )
        return vector_store
    except Exception as e:
        st.error(f"Otomatik bağlantı hatası: {e}")
        return None
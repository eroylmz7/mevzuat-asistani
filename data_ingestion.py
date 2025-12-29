import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from supabase import create_client

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
            chunk_size=400,
            chunk_overlap=100,
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
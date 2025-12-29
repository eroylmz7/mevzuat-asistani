import os
import streamlit as st
# DEĞİŞİKLİK BURADA: Artık PyMuPDFLoader kullanıyoruz
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_pdfs(uploaded_files):
    all_documents = []
    
    # Geçici klasör oluştur
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
        
    for uploaded_file in uploaded_files:
        # Dosyayı kaydet
        file_path = os.path.join("temp_pdfs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # --- DEĞİŞİKLİK BURADA ---
        # Daha akıllı ve düzenli okuyan yükleyici
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # HASSAS AYAR (Chunk Size: 400)
        # Yönetmelik maddelerini kaçırmamak için küçük parçalar
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\nMadde", "\n\n", "\n", ". ", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Kaynak ismini ekle
        for doc in split_docs:
            doc.metadata["source"] = uploaded_file.name
            
        all_documents.extend(split_docs)
        
        # Temizlik
        os.remove(file_path)

    # Pinecone'a Gönder
    if all_documents:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        vector_store = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            index_name="mevzuat-asistani"
        )
        return vector_store
    
    return None
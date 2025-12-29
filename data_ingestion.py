import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
# YENİ ADRES BURASI:
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_pdfs(uploaded_files):
    all_documents = []
    
    # Geçici klasör oluştur
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
        
    for uploaded_file in uploaded_files:
        # Dosyayı geçici olarak kaydet
        file_path = os.path.join("temp_pdfs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # PDF Yükle ve Parçala
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Metinleri Böl (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Kaynak ismini düzelt (temp/ dosya yolunu temizle)
        for doc in split_docs:
            doc.metadata["source"] = uploaded_file.name
            
        all_documents.extend(split_docs)
        
        # Temizlik: Dosyayı sil
        os.remove(file_path)

    # Pinecone'a Gönder
    if all_documents:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Vektör Veritabanını Oluştur/Güncelle
        vector_store = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            index_name="mevzuat-asistani"
        )
        return vector_store
    
    return None
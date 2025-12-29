import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_pdfs(uploaded_files):
    """
    PDF'leri okur, parçalar ve Pinecone Bulut Veritabanına yükler.
    """
    # API Key'i çevre değişkeni olarak ayarla (Kütüphane bunu otomatik okur)
    os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
    
    index_name = "mevzuat-asistani" # Pinecone'daki ismin aynısı olmalı
    
    # Embedding Model (384 boyutlu)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            documents.extend(docs)
        finally:
            os.remove(tmp_path)

    if not documents:
        return None

    # Parçalama
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Buluta Yükle
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name
    )
    
    return vector_store
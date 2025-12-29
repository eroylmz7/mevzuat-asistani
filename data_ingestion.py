import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_pdfs(uploaded_files):
    """
    Streamlit'ten gelen yüklenmiş dosyaları işler ve vektör veritabanını günceller.
    """
    documents = []
    # Embedding modelini başlat
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    for uploaded_file in uploaded_files:
        # Streamlit'ten gelen dosyayı geçici bir yere kaydet (Loader'ın okuyabilmesi için)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            os.remove(tmp_path) # Geçici dosyayı sil

    # Metinleri Parçala
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Vektör Veritabanını Oluştur/Güncelle
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db_store"
    )
    return vectordb
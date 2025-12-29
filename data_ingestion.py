import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_pdfs(uploaded_files):
    documents = []
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    for uploaded_file in uploaded_files:
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # KRİTİK DÜZELTME: Metaveriyi orijinal dosya adıyla güncelle
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
                
            documents.extend(docs)
        finally:
            os.remove(tmp_path)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Vektör DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db_store"
    )
    return vectordb
import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
# PyMuPDF yerine PDFPlumber geldi!
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from supabase import create_client
from pinecone import Pinecone

def process_pdfs(uploaded_files):
    # --- SUPABASE BAĞLANTISI ---
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
            # --- 1. STORAGE YÜKLEME (AYNI) ---
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name,
                    file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            except Exception as e:
                print(f"Storage uyarısı: {e}")

            # --- 2. BELGE İŞLEME (BURASI DEĞİŞTİ) ---
            uploaded_file.seek(0)
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 🔥 PDFPlumber: Tabloları ve sütunları anlar!
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            
            # --- 🔥 OTOMATİK BAŞLIK TESPİTİ ---
            # İlk sayfanın başını alıp her parçaya etiket olarak yapıştırıyoruz
            doc_real_title = "Belge Başlığı Bulunamadı"
            if documents and len(documents) > 0:
                raw_header = documents[0].page_content[:300].replace("\n", " ").strip()
                doc_real_title = raw_header

            # Tablolu veriler için 1000/500 stratejisi iyidir
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,      
                chunk_overlap=500,
                separators=["\nMADDE ", "\nMadde ", "\nGEÇİCİ MADDE", "\n\n", "\n", ". ", " "]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            for doc in split_docs:
                doc.metadata["source"] = uploaded_file.name
                # Dosya adını değil, İÇERİK BAŞLIĞINI ekliyoruz
                doc.page_content = f"BELGE KİMLİĞİ: {doc_real_title}\n---\n{doc.page_content}"
            
            all_documents.extend(split_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            # Tablo (Liste) güncelleme
            try:
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except: pass
            
        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

    # --- 3. PINECONE ---
    if all_documents:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            index_name="mevzuat-asistani"
        )
        return vector_store
    
    return None

# --- SİLME VE BAĞLANTI FONKSİYONLARI ---
# (Bu kısımlar önceki kodla aynı kalabilir)
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
            supabase.storage.from_("belgeler").remove([file_name])
        except Exception as e: print(f"Supabase silme hatası: {e}")
        return True, f"{file_name} başarıyla silindi."
    except Exception as e:
        return False, f"Hata: {str(e)}"

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
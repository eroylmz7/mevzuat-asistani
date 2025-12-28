import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from data_ingestion import load_and_process_pdfs

# Ayarlar
PERSIST_DIRECTORY = "./chroma_db_store"  # Veritabanının kaydedileceği klasör

def create_vector_db():
    print("🚀 Vektör Veritabanı oluşturma süreci başlıyor...")

    # 1. Verileri Hazırla
    chunks = load_and_process_pdfs()
    if not chunks:
        print("❌ İşlenecek veri bulunamadı. Lütfen 'veriler' klasörünü kontrol et.")
        return

    # 2. Embedding Modelini Yükle
    # Proje planında embedding kullanımı belirtilmiştir [cite: 21]
    print("🧠 Embedding modeli yükleniyor (HuggingFace)...")
    # Not: Plandaki 'BAAI/bge-m3' modeli büyük olabilir, başlangıç için 
    # Türkçe desteği çok daha güçlü olan model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 3. Vektör Veritabanını Oluştur ve Kaydet (Indexing)
    # Varsa eskisini temizle (temiz bir başlangıç için)
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f"🧹 Eski veritabanı temizlendi: {PERSIST_DIRECTORY}")

    print("💾 Vektörler oluşturuluyor ve ChromaDB'ye kaydediliyor...")
    
    # ChromaDB oluşturma [cite: 22]
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Belleğe kaydet (yeni sürümlerde otomatik olabilir ama garanti olsun)
    # db.persist() # Langchain güncel sürümlerinde otomatik yapılıyor.

    print(f"BAŞARILI! Veritabanı '{PERSIST_DIRECTORY}' klasörüne kaydedildi.")
    print(f"Toplam {len(chunks)} parça vektörleştirildi.")

if __name__ == "__main__":
    create_vector_db()
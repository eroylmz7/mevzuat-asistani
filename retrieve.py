import os
import sys
# Uyarıları gizlemek için (opsiyonel)
import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Veritabanı klasörü (vector_db.py ile aynı olmalı)
PERSIST_DIRECTORY = "./chroma_db_store"

def search_documents(query, k=20):
    """
    Kullanıcının sorusuna (query) en benzer metin parçalarını getirir.
    k: Getirilecek parça sayısı (Bu daha sonra değiştirilebilir.Yetmezse değiştireceğim.)
    """
    print(f"\n🔍 Soru: '{query}'")
    print("⏳ Veritabanı taranıyor...")

    # 1. Embedding Modelini Yükle (Sorguyu vektöre çevirmek için)
    # Veritabanını oluştururken kullandığımız modelin aynısı olmalı!
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Mevcut Veritabanına Bağlan
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"HATA: '{PERSIST_DIRECTORY}' klasörü bulunamadı. Önce vector_db.py çalıştırılmalı.")
        return

    try:
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=embedding_model
        )
    except Exception as e:
        print(f"HATA: Veritabanı yüklenemedi. Hata detayı: {e}")
        return

    # 3. Benzerlik Araması Yap (Similarity Search)
    # k=5: En alakalı 5 parçayı getir
    results = db.similarity_search(query, k=k)

    if not results:
        print("❌ Hiçbir sonuç bulunamadı.")
        return

    # 4. Sonuçları Ekrana Yazdır
    print(f"\n✅ Bulunan En Alakalı {len(results)} Parça:\n" + "="*50)
    
    for i, doc in enumerate(results, 1):
        # Metadata'dan kaynak ve sayfa bilgisini al
        source = doc.metadata.get("source", "Bilinmeyen Kaynak")
        # Dosya yolunu temizle, sadece dosya adını göster
        source_name = os.path.basename(source)
        page = doc.metadata.get("page", 0) + 1 # Sayfa numaraları 1'den başlasın diye
        content = doc.page_content.replace("\n", " ") # Okuması kolay olsun diye satır sonlarını temizle
        
        print(f"\n📄 SONUÇ #{i}")
        print(f"📌 Kaynak: {source_name} (Sayfa {page})")
        print(f"📝 İçerik: {content[:300]}...") # İlk 300 karakteri göster
        print("-" * 50)

if __name__ == "__main__":
    print("🤖 MEVZUAT ASİSTANI ARAMA MOTORU (Çıkmak için 'Q' basın)")
    # Kullanıcıdan terminal üzerinden soru alalım
    while True:
        user_query = input("\n❓ Sorunuzu yazın: ")
        if user_query.lower() in ['q', 'exit', 'çıkış']:
            print("👋 Görüşmek üzere!")
            break
        if user_query.strip() == "":
            continue
            
        search_documents(user_query)
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_process_pdfs():
    """
    ./veriler klasöründeki PDF'leri tek tek okur ve parçalar.
    """
    target_directory = "./veriler"
    documents = []

    # 1. Klasör Kontrolü
    if not os.path.exists(target_directory):
        print(f"HATA: '{target_directory}' klasörü bulunamadı!")
        return []

    # 2. Dosya Listesi
    files = [f for f in os.listdir(target_directory) if f.endswith(".pdf")]
    
    if not files:
        print(f"UYARI: '{target_directory}' klasöründe hiç PDF yok.")
        return []

    print(f"Bulunan PDF Dosyaları: {files}")

    # 3. Dosyaları Tek Tek Oku (En Garanti Yöntem)
    for file_name in files:
        file_path = os.path.join(target_directory, file_name)
        try:
            print(f"İşleniyor: {file_name}...")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"Başarılı: {file_name} ({len(docs)} sayfa)")
        except Exception as e:
            print(f"HATA: {file_name} okunurken sorun oluştu: {e}")

    # 4. Metinleri Parçala (Chunking)
    if not documents:
        print("HATA: Hiçbir belgeden metin çıkarılamadı.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Toplam {len(chunks)} parça metin oluşturuldu.")
    
    return chunks

if __name__ == "__main__":
    # Test için çalıştırılabilir
    load_and_process_pdfs()


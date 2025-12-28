import os
import sys
import time
import warnings
from dotenv import load_dotenv
import os
# HuggingFace'in gereksiz hata vermesini engelle
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Uyarıları temizle
warnings.filterwarnings("ignore")

# Gerekli Kütüphaneler
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Ayarları Yükle
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIRECTORY = "./chroma_db_store"

if not GOOGLE_API_KEY:
    print("❌ HATA: GOOGLE_API_KEY bulunamadı! .env dosyasını kontrol et.")
    sys.exit(1)

def ask_bot():
    print("\n⚙️  Mevzuat Asistanı Yükleniyor (Sunum Modu: 1.5 Flash)...")
    
    # 2. Embedding ve Veritabanı
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if not os.path.exists(PERSIST_DIRECTORY):
        print("HATA: Veritabanı yok. Önce vector_db.py çalıştır.")
        return

    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    
    # [TASARRUF MODU] k=1: Sadece en alakalı 1 paragrafı okur. 
    # Bu, sunum sırasında kota hatası almanı engeller.
    retriever = vectordb.as_retriever(search_kwargs={"k": 25})

    # 3. LLM AYARI
    # gemini-1.5-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        transport="rest"  # <--- İŞTE ÇÖZÜM BU! (İletişimi garantiye alır)
    )

   # 4. Prompt (Sistem İstemi) - DENGELİ VE AKILLI YAPI
    template = """
    Sen üniversite mevzuatları konusunda uzman bir asistansın.
    Aşağıda sana verilen metin parçalarını (context) dikkatlice oku ve sadece bu bilgilere dayanarak cevap ver.
    
    KURALLAR:
    1. ÖNCELİK: Eğer soru, sana verilen metinlerle (üniversite, yönetmelik, dersler, sınavlar vb.) tamamen alakasızsa (Örn: "Bugün hava nasıl?", "Messi mi Ronaldo mu?"), kesinlikle cevap uydurma. Sadece "Verilen dokümanlarda bu bilgi yer almamaktadır." de.
    
    2. DETAYLI ARAMA: Eğer soru mevzuatla ilgiliyse ama cevap metinde açıkça yazmıyorsa (ima ediliyorsa veya hesaplama gerektiriyorsa);
       - "Kanun No:..." gibi atıflar yerine, doğrudan sayı/yıl belirten ifadelere odaklan.
       - Sayılar yazı ile (yedi, dört) yazılmış olabilir, bunları rakama çevir (7, 4).
       - Metin parça parça olabilir, cümleleri birleştirerek mantık yürüt.

    3. Şüpheye düşersen, metinde en güçlü kanıtı sunan maddeyi referans göster.

    Bağlam (Context):
    {context}

    Soru:
    {question}

    Cevap:
    """
    
    QA_CHAIN_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    

    # 5. Zinciri (Chain) Oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    print("\n✅ ASİSTAN HAZIR! (Çıkmak için 'q' ya basın)")
    print("-" * 50)

    while True:
        user_input = input("\n❓ Sorunuz: ")
        if user_input.lower() in ['q', 'exit', 'çıkış']:
            print("👋 Görüşmek üzere!")
            break
        if not user_input.strip():
            continue
            
        print("🤖 Düşünüyor...")
        
    # --- GARANTİCİ MOD (65 Saniye Bekle - Kesin Çözüm) ---
        success = False
        retry_count = 0
        max_retries = 3 

        while not success and retry_count < max_retries:
            try:
                # Soruyu gönder
                result = qa_chain.invoke({"query": user_input})
                success = True 
            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "ResourceExhausted" in error_msg:
                    retry_count += 1
                    print(f"\n🛑 HIZ LİMİTİ DOLDU! Google '1 Dakika Bekle' diyor. ({retry_count}/{max_retries})")
                    
                    # 65 Saniye geri sayım (60 yetmeyebilir, 5 de bizden olsun)
                    for i in range(65, 0, -1):
                        sys.stdout.write(f"\r⏳ Mola: {i} sn...   ")
                        sys.stdout.flush()
                        time.sleep(1)
                    print("\n🚀 Süre doldu, tekrar deneniyor...")
                else:
                    print(f"\n❌ Farklı Bir Hata: {e}")
                    break 
        
        if success:
             # --- CEVAP BAŞARIYLA GELDİ ---
            answer = result["result"]
            source_docs = result["source_documents"]

            # 1. KISA DEBUG 
            print(f"\n🔍 İNCELENEN PARÇA SAYISI: {len(source_docs)}")
            print("🔍 İLK 3 PARÇANIN İÇERİĞİ:")
            for i, doc in enumerate(source_docs[:3]): 
                clean_text = doc.page_content.replace("\n", " ")
                print(f"📄 Parça {i+1}: {clean_text[:200]}...") 
            
            # 2. CEVAP
            print(f"\n💡 Cevap: {answer}")

            # 3. KAYNAKLAR
            print("\n📚 Kaynaklar:")
            seen_sources = set()
            for doc in source_docs:
                src = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
                pg = doc.metadata.get("page", 0) + 1
                source_id = f"{src} (Sayfa {pg})"
                if source_id not in seen_sources:
                    print(f"- {source_id}")
                    seen_sources.add(source_id)
            print("-" * 50)
        else:
            print("\n❌ Bu soru için Google API tamamen kilitlendi. Lütfen 5-10 dakika sonra tekrar deneyin veya Yeni Bir API Key alın.")

if __name__ == "__main__":
    ask_bot()
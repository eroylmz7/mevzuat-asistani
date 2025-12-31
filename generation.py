import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. AYARLAR ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. AKILLI ARAMA ÇEVİRMENİ ---
    # Kullanıcının sorusunu, veritabanındaki tabloları ve sayıları bulacak şekilde genişletiyoruz.
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    translation_prompt = f"""
    GÖREV: Kullanıcı sorusunu analiz et ve belge araması için en iyi anahtar kelimeleri üret.
    
    KURALLAR:
    1. LİTERATÜR ÇEVİRİSİ: "Vize" -> "Ara Sınav", "Final" -> "Yarıyıl Sonu Sınavı", "Büt" -> "Bütünleme".
    2. SAYISAL VERİ AVCLIĞI (KRİTİK): 
       - Eğer soru bir "Şart", "Koşul", "Nasıl olurum", "Onur Belgesi", "Geçme Notu" içeriyorsa;
       - Arama terimine mutlaka şunları ekle: "Not Ortalaması Tablosu", "GANO Puanı", "Sayısal Değerler", "Yüzdelik Dilim".
       - Amaç: Metin içinde gizlenmiş rakamları (3.00, 2.00, %70 vb.) bulmaktır.
    
    Soru: "{question}"
    Geliştirilmiş Arama Cümlesi:
    """
    
    try:
        # Çeviriyi yap
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        # Hem orijinal soruyu hem de akademik halini birleştir
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. GENİŞ KAPSAMLI ARAMA (Retrieval) ---
    try:
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query, 
            k=25,           # 15'ten 25'e çıkardık (Daha çok veri)
            fetch_k=80,    # Havuzu 100'e çıkardık (Daha geniş tarama)
            lambda_mult=0.5 # Çeşitlilik
        )
    except Exception as e:
        return {"answer": f"Arama hatası: {str(e)}", "sources": []}
    
    # --- 4. BAĞLAM OLUŞTURMA ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        # Yeni satır karakterlerini temizle ki tablo yapısı bozulmasın
        clean_content = doc.page_content.replace("\n", "  ").strip()
        context_text += f"\n[BÖLÜM {i+1}]: {clean_content}\n"
        
        # Kaynakça Listesi
        src = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{src} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. PROFESYONEL CEVAPLAYICI (Generator) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.2 # Düşük sıcaklık = Daha tutarlı cevaplar
    )
    
    final_template = f"""
    Sen Üniversite Mevzuat Asistanısın. Görevin, sağlanan belgeleri analiz ederek soruları yanıtlamaktır.
    
    BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- ⚠️ KESİN KURALLAR (HARFİYEN UYGULA) ---
    
    1. "BELGE PARÇASI" YASAGI:
       - Cevabında ASLA "Belge Parçası 9'a göre", "Bölüm 3'te yazdığı gibi", "Chunk 5" gibi ifadeler KULLANMA.
       - Bunun yerine: "Yönetmeliğe göre...", "İlgili madde uyarınca..." gibi profesyonel ifadeler kullan.
       
    2. TABLO VE SAYI OKUMA ZORUNLULUĞU:
       - Kullanıcı "Onur Öğrencisi", "Yatay Geçiş", "Mezuniyet" gibi statü şartlarını soruyorsa;
       - Metindeki sözel şartların (Disiplin cezası vb.) yanına MUTLAKA tablolardaki SAYISAL DEĞERLERİ (Örn: 3.00 - 3.49 arası, en az 2.50) ekle.
       - Eğer metinde "GANO" geçiyorsa, onun kaç olduğunu bulmadan cevabı bitirme.
       
    3. FORMAT:
       - Cevabı net, anlaşılır maddeler (Bullet Points) halinde ver.
       - Robotik değil, bir danışman gibi konuş.
       
    4. BİLİNMEYEN DURUM:
       - Eğer tüm aramalara rağmen bilgi yoksa, uydurma. Sadece "Verilen dokümanlarda bu bilgi yer almıyor." de.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
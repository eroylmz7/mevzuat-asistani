import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. AYARLAR ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. EVRENSEL ÇEVİRMEN (ARTIK KELİME EZBERLEMİYOR) ---
    # Gemini'ye diyoruz ki: "Sen Akademik Literatür Uzmanısın. Hangi kelimenin ne anlama geldiğini sen bul."
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    # BURASI DEĞİŞTİ: Artık "Vize=Ara Sınav" diye elle yazmıyoruz. Genel kural koyuyoruz.
    translation_prompt = f"""
    GÖREV: Kullanıcının sorusunu analiz et ve belge araması için "Resmi Mevzuat Literatürü"ne çevir.
    
    ANALİZ KURALLARI:
    1. HALK DİLİ -> RESMİ DİL: Kullanıcı "atılma", "kovulma", "büt", "vize", "dondurma" gibi günlük ifadeler kullanabilir. Sen bunları yönetmeliklerde geçen RESMİ KARŞILIKLARINA (Örn: İlişik Kesme, Bütünleme, Ara Sınav, Kayıt Dondurma) dönüştür.
    2. SAYISAL VERİ İPUÇLARI: Eğer soru "Ne kadar?", "Kaç?", "Şartı nedir?", "Süresi ne?" gibi nicelik soruyorsa; arama terimine "Süreleri", "Puanları", "Tablosu", "Oranları", "Kriterleri" gibi ifadeler ekle.
    
    Soru: "{question}"
    Akademik Arama Cümlesi:
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        # Hem orijinal soruyu hem de resmi halini arıyoruz (Garantici yaklaşım)
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. ARAMA (Retrieval - 20 PDF İÇİN GÜÇLENDİRİLDİ) ---
    try:
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query, 
            k=25,           # 25 parça getir (Geniş bağlam)
            fetch_k=100,    # 100 parça içinden seç (Çeşitlilik artar)
            lambda_mult=0.6 # Farklı konulardan da parça al
        )
    except Exception as e:
        return {"answer": f"Arama hatası: {str(e)}", "sources": []}
    
    # --- 4. BAĞLAM ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        clean_content = doc.page_content.replace("\n", " ").strip()
        context_text += f"\n[MADDE {i+1}]: {clean_content}\n"
        
        src = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{src} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. EVRENSEL CEVAPLAYICI (ARTIK HER YÖNETMELİĞE UYAR) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.2 
    )
    
    final_template = f"""
    Sen Üniversite Mevzuat Asistanısın. Görevin, yüklenen belgeleri (Yönetmelik, Yönerge, Esaslar) analiz ederek soruları cevaplamaktır.
    
    BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- ⚠️ EVRENSEL CEVAPLAMA KURALLARI ---
    
    1. TESPİT ET VE EŞLEŞTİR:
       - Kullanıcının kullandığı terimler belgede aynen geçmeyebilir. Bağlamı (Context) okuyarak doğru eşleşmeyi yap.
       - Örn: Kullanıcı "Yemekhane kartı" sorabilir, belgede "Akıllı Kart" yazabilir. Bunu sen eşleştir.
       
    2. SAYISAL HASSASİYET (TABLO OKUMA):
       - Soru bir kriter, hak, süre veya puan içeriyorsa (Örn: Geçme notu, Burs miktarı, İzin süresi);
       - Metin içindeki veya tablolardaki SAYISAL DEĞERLERİ (Rakam, Yüzde, Tarih) bulmadan cevap verme.
       - "Onur öğrencisi" gibi statüler sorulduğunda not aralıklarını (Örn: 3.00-3.49) mutlaka yaz.
       
    3. PROFESYONEL ÜSLUP:
       - "Belge Parçası 5'e göre" gibi ifadeler KULLANMA.
       - "Yönetmeliğin ilgili maddesine göre...", "Belirtilen esaslar uyarınca..." gibi ifadeler kullan.
       
    4. SINIRLAR:
       - Cevabı sadece verilen metne dayandır. Metinde yoksa "Dokümanlarda bu bilgi yer almıyor" de.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Cevap üretme hatası: {str(e)}", "sources": []}
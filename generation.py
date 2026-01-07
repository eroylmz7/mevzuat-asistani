import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. GÜVENLİK ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. ANALİST AJAN (Sorgu Zenginleştirme) ---
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    translation_prompt = f"""
    GÖREV: Kullanıcı sorusunu analiz et ve arama motoru için en kritik anahtar kelimeleri ekle.
    
    
    ANALİZ ADIMLARI:
    1. KONU TESPİTİ:
       - Akademik 1: "Tez", "Jüri", "Yüksek Lisans" -> "LİSANSÜSTÜ EĞİTİM"
       - Akademik 2: "Çap", "Yandal", "Yaz Okulu" -> "LİSANS EĞİTİMİ"
       - İdari: "Rektör", "Personel", "İzin", "Teşkilat", "Atama" -> "İDARİ MEVZUAT"
       - Disiplin: "Ceza", "Kopya", "Uzaklaştırma" -> "DİSİPLİN SUÇU"
       
    2. GÜNCELLİK VE DETAY:
       - Soru "Yayın şartı", "Mezuniyet kriteri" içeriyorsa -> "Senato Kararı", "Yayın Esasları", "Ek Madde" terimlerini ekle.
    
    Soru: "{question}"
    Geliştirilmiş Arama Sorgusu (Sadece terimler):
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. RETRIEVAL (KARARLI MOD) ---
    try:
        # Karmaşık if-else'i kaldırdık. Tek ve güçlü bir standart kullanacağız.
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query,
            k=15,             
            fetch_k=120,      
            lambda_mult=0.75  
        )
    except Exception as e:
        return {"answer": f"Veritabanı hatası: {str(e)}", "sources": []}
    
    # --- 4. AKILLI ETİKETLEME VE ÖNCELİKLENDİRME ---
    context_text = ""
    sources = []
    
    # generation.py içinde 'for doc in docs:' döngüsünün tamamını bununla değiştir:

    # --- 4. ETİKETLEME VE FORMATLAMA (SADE HALİ) ---
    context_text = ""
    sources = []

    for doc in docs:
        # Metni temizle
        content = doc.page_content.replace("\n", " ").strip()
        
        # Dosya adını al (Sadece kaynak göstermek için)
        filename = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        
        # Sayfa numarasını al
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1

        # --- LLM'E GİDECEK FORMAT ---
        # Artık "Öncelik", "Kapsam" vs. gibi yapay yönlendirmeler YOK.
        # LLM'e sadece saf metni veriyoruz, kararı o verecek.
        context_text += f"\n--- BELGE KAYNAĞI: {filename} (Sayfa {page}) ---\nİÇERİK: {content}\n"
        
        # Kullanıcıya gösterilecek kaynak listesi
        src_str = f"{filename} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. CEVAPLAYICI (HUKUKÇU MODU) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 # Yaratıcılık sıfır, sadece kanıt.
    )
    
    final_template = f"""
    Sen Bursa Uludağ Üniversitesi mevzuat asistanısın. 
    Elinizdeki belgeleri (context) kullanarak soruya (question) en doğru, resmi ve net cevabı ver.

    ELİNDEKİ BELGELER (context):
    {context_text}

    SORU: {question}

    --- 🧠 KARAR VERME VE CEVAPLAMA KURALLARI ---

    KURAL 1: BELGE TÜRÜ VE HİYERARŞİSİ (ETİKET YOK, MANTIK VAR) ⚖️
    - Hukukta "Özel Hüküm", "Genel Hüküm"den üstündür.
    - Eğer elindeki belgelerde bir çelişki görürsen:
      A) "Uygulama Esasları", "Yönerge" veya "Senato Kararı" gibi detaylı belgeler, genel "Yönetmelik"lerden daha önceliklidir. Onlardaki bilgiyi esas al.
      B) Daha yeni tarihli olan belgeyi (Eğer tarih varsa) esas al.

    KURAL 2: KAPSAM AYRIMI (ÇOK ÖNEMLİ)
    - Belge başlıklarına ve içeriğine bakarak kapsamı sen ayırt et:
      * Soru "Yüksek Lisans" veya "Doktora" ise ->  Lisansüstü belgelerinden cevap ver.
      * Soru "Lisans" veya "Önlisans" ise -> Lisansüstü belgelerinden cevap ver.
      * "Lisans" sorusuna "Lisansüstü" yönetmeliğinden cevap verme (veya tam tersi).

    KURAL 3: BİLGİ BİRLEŞTİRME VE SENTEZ
    - Kullanıcı "Mezuniyet şartları nelerdir?" gibi GENEL bir liste isterse:
    - Tek bir maddede toplu liste arama. Metin içine dağılmış bilgileri (AKTS, GANO, Süre, Zorunlu dersler) sen toplayıp BİRLEŞTİR.
    - "Belgelerde toplu liste yok" deyip kestirip atma. Dedektif gibi parçaları birleştir.

    KURAL 4: REFERANS FORMATI
    - Her bilginin sonuna, o bilgiyi hangi dosyadan aldığını parantez içinde ekle.

    KURAL 5: DÜRÜSTLÜK
    - Eğer bilgi metinlerde HİÇ YOKSA, uydurma. "Belgelerde bu bilgi bulunmamaktadır" de.

    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Cevap oluşturulurken hata: {str(e)}", "sources": []}
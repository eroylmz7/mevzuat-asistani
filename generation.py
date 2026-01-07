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
    GÖREV: Kullanıcı sorusunu analiz et ve arama motoru için SADECE GEREKLİYSE ek terim ekle.
    
    ANALİZ MANTIĞI (SADE):
    1. EĞER SORU "LİSANSÜSTÜ" (Master/Doktora) İLE İLGİLİYSE:
       - (İpuçları: Tez, Jüri, Yeterlik, Danışman, Enstitü, Seminer, TİK, ALES)
       - EKLE: "LİSANSÜSTÜ EĞİTİM YÖNETMELİĞİ"

    2. EĞER SORU "LİSANS" (Fakülte/MYO) İLE İLGİLİYSE:
       - (İpuçları: ÇAP, Yandal, Yaz Okulu, Tek Ders, Bütünleme, DC, DD, Azami Süre)
       - EKLE: "ÖNLİSANS VE LİSANS EĞİTİM YÖNETMELİĞİ"

    3. EĞER SORU "UYGULAMA / STAJ" İLE İLGİLİYSE (YENİ KURAL):
       - (İpuçları: Staj, İME, Uygulamalı Eğitim, İş Yeri Eğitimi, Grup)
       - EKLE: "UYGULAMALI EĞİTİM YÖNERGESİ"
       

    4. DİĞER DURUMLARDA:
       - Sadece "MEVZUAT" ekle.

    Soru: "{question}"
    Sadece eklenecek anahtar kelimeleri yaz:
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
            k=20,             
            fetch_k=150,      
            lambda_mult=0.7  
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
        src_str = filename
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. CEVAPLAYICI (HUKUKÇU MODU) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.0 # Yaratıcılık sıfır, sadece kanıt.
    )
    
    final_template = f"""
    Sen Bursa Uludağ Üniversitesi mevzuat asistanısın. 
    Elinizdeki belgeleri (context) kullanarak soruya (question) en doğru, resmi ve net cevabı ver.

    ELİNDEKİ BELGELER (context):
    {context_text}

    SORU: {question}

    --- 🧠 CEVAPLAMA KURALLARI ---

    KURAL 1: BELGE TÜRÜ VE HİYERARŞİSİ ⚖️
    - "Uygulama Esasları", "Yönerge" veya "Senato Kararı" gibi belgeler, o konudaki ÖZEL detayları içerir. 
    - Eğer "Yönetmelik" ile "Yönerge" arasında fark varsa, daha detaylı olan YÖNERGEYİ/ESASLARI baz al.
    - Örneğin "Staj" sorusunda "Uygulamalı Eğitim Yönergesi" önceliklidir.

    KURAL 2: SENTEZ VE BİRLEŞTİRME
    - "Lisans mezuniyet koşulları nelerdir ?" gibi geniş kapsamlı sorularda bilgiler parça parça olabilir (örn: Bir maddede süre, diğerinde AKTS yazar). Bunları birleştirerek bütünlüklü cevap ver.

    KURAL 3: REFERANS
    - Bilgiyi hangi dosyadan aldığını parantez içinde belirt. Örn: (uygulamali_egitimler.pdf)

    KURAL 4: DÜRÜSTLÜK
    - Bilgi yoksa uydurma, "Belgelerde bulunmamaktadır" de.

    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        
        # --- DEĞİŞİKLİK BURADA: CEVAP YOKSA KAYNAK GİZLE 🕵️‍♂️ ---
        # Eğer cevapta "bulunamadı", "yoktur" gibi şeyler geçiyorsa kaynakları boşalt.
        negative_signals = ["bulunmamaktadır", "bilgi yok", "rastlanmamıştır", "yer almamaktadır", "belirtilmemiştir"]
        
        if any(signal in answer.lower() for signal in negative_signals):
            final_sources = [] # Boş liste döndür (Böylece UI'da kutu çıkmaz)
        else:
            final_sources = sources[:5] # Sadece ilk 5 dosya adı

        return {"answer": answer, "sources": final_sources}

    except Exception as e:
        return {"answer": f"Cevap oluşturulurken hata: {str(e)}", "sources": []}
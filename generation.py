import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. GÜVENLİK VE AYARLAR ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. "DEDEKTİF" ÇEVİRMEN (SORUYU GENİŞLETİR) ---
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    # BURADAKİ YENİLİK: Soru "Yapabilir miyim?" ise, arkasına "Limitleri ve Kısıtlamaları" diye ekletiyoruz.
    translation_prompt = f"""
    GÖREV: Kullanıcı sorusunu, mevzuat veritabanında en detaylı sonucu bulacak şekilde "Akademik/Hukuki Arama Sorgusuna" dönüştür.
    
    ANALİZ STRATEJİSİ:
    1. EŞ ANLAMLILAR: "Vize" -> "Ara Sınav", "Af" -> "Öğrenci Affı", "Atılma" -> "İlişik Kesme".
    2. GİZLİ KISITLAMALAR (ÇOK ÖNEMLİ): 
       - Soru bir "İzin/Hak" içeriyorsa (Örn: "Ders saydırabilir miyim?", "Geçiş yapabilir miyim?");
       - Arama sorgusuna mutlaka şunları ekle: "Azami Kredi Sınırı", "Yüzde (%) Limiti", "Başvuru Şartları", "Kısıtlamaları", "Senato Esasları".
       - Amaç: Sadece "Evet yapılır" diyen maddeyi değil, "Ama şu kadar yapılır" diyen kısıtlama maddesini de bulmaktır.
    
    Soru: "{question}"
    Geliştirilmiş Arama Sorgusu:
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. GENİŞ AÇILI ARAMA (RETRIEVAL) ---
    try:
        # k=30 yaparak modelin "Çevresel Görüşünü" artırıyoruz.
        # Böylece cevap 5. sayfada, kısıtlaması 12. sayfadaysa ikisini de yakalar.
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query, 
            k=30,           
            fetch_k=100,    
            lambda_mult=0.5 
        )
    except Exception as e:
        return {"answer": f"Arama hatası: {str(e)}", "sources": []}
    
    # --- 4. BAĞLAM (CONTEXT) HAZIRLIĞI ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        # Satır sonlarını temizle ki tablolar bozulmasın
        clean_content = doc.page_content.replace("\n", " ").strip()
        context_text += f"\n[DOKÜMAN BÖLÜMÜ {i+1}]: {clean_content}\n"
        
        # Kaynak Adı Temizleme
        src = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{src} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. ŞÜPHECİ CEVAPLAYICI (GENERATOR) ---
    # Gemini'ye "Denetçi" (Auditor) rolü veriyoruz.
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.0 # Sıfır hata toleransı
    )
    
    final_template = f"""
    Sen, Üniversite Mevzuat Denetçisisin. Görevin, belgelerdeki kuralları en ince ayrıntısına kadar inceleyip kullanıcıya kesin ve eksiksiz bilgi vermektir.
    
    BELGELER (KANITLAR):
    {context_text}
    
    SORU: {question}
    
    --- 🧠 ANALİZ VE KONTROL SÜRECİ (DİKKATLE UYGULA) ---
    
    ADIM 1: TEMEL CEVABI BUL
    - Sorunun cevabı "Evet" mi, "Hayır" mı? Önce bunu belirle.
    
    ADIM 2: "AMA" KONTROLÜ (KISITLAMA AVCISI) 🕵️‍♂️
    - Eğer cevap "Evet" ise, hemen sevinme. Metinde şu kelimeleri tara: "Ancak", "Şartıyla", "En fazla", "En az", "%", "Oran", "Dahil edilmez".
    - ÖRNEK: "Ders saydırılır" yazıyorsa, hemen yanında "%50'sini geçemez" veya "Yönetim kurulu kararı gerekir" yazıyor mu? Varsa MUTLAKA ekle.
    
    ADIM 3: TARİH VE HİYERARŞİ KONTROLÜ
    - Eğer iki belge çelişiyorsa (Örn: Biri 2016, biri 2025 tarihli), her zaman YENİ TARİHLİ olan belgeyi esas al.
    - Metinde "Senato tarafından belirlenir" yazıyorsa ve elindeki belgelerde "Uygulama Esasları" veya "Senato Kararı" varsa, cevabı oradan çek.
    
    ADIM 4: NETLİK
    - Cevabında "Belge Parçası 5" gibi teknik terimler kullanma.
    - Cevaplayamadığın veya emin olmadığın durumlarda "Belgelerde net bir kısıtlama/oran belirtilmemiştir" de.
    
    --- CEVAP FORMATI ---
    Cevabı doğrudan kullanıcıya hitaben, profesyonel, açıklayıcı ve madde madde yaz.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]} # En alakalı 5 kaynağı göster
    except Exception as e:
        return {"answer": f"Cevap oluşturulurken hata: {str(e)}", "sources": []}
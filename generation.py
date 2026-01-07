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
        temperature=0.0 # Yaratıcılık sıfır, sadece kanıt.
    )
    
    final_template = f"""
    Sen Bursa Uludağ Üniversitesi mevzuat asistanısın. 
    Elinizdeki belgeleri (context) kullanarak soruya (question) en doğru, resmi ve net cevabı ver.

    ELİNDEKİ BELGELER (context):
    {context_text}

    SORU: {question}

    --- 🧠 KARAR VERME MEKANİZMASI (BU KURALLARA UY) ---

    KURAL 1: BELGE TÜRÜNÜ TANI
    - Soru "Akademik" (Öğrenci, Sınav) ise -> Akademik belgelere bak.
    - Soru "İdari" (Rektör, Personel, Teşkilat) ise -> İdari belgelere bak (Öğrenci yönetmeliğini karıştırma).

    KURAL 2: HİYERARŞİ VE GÜNCELLİK ⚖️
    - Eğer iki belge arasında çelişki varsa (Örn: Biri "X yapılabilir", diğeri "X yasaktır" diyorsa):
      A) Başlığında "🔥 [YÜKSEK ÖNCELİK]" yazan belgeye İTAAT ET. (O belge daha özel veya daha günceldir).
      B) "Özel Hüküm" (Yönerge/Esaslar), "Genel Hüküm"den (Yönetmelik) üstündür.

    KURAL 3: KAPSAM İZOLASYONU
    - Soru "Yüksek Lisans" ise -> "Doktora" başlıklarını GÖRMEZDEN GEL.
    - Soru "Doktora" ise -> "Yüksek Lisans" başlıklarını GÖRMEZDEN GEL.
    - Soru "Lisans" (Önlisans/Fakülte) ise -> "Lisansüstü" belgelerini GÖRMEZDEN GEL.
    
    KURAL 4: BİLGİ BİRLEŞTİRME VE SENTEZ
    - Kullanıcı "Mezuniyet şartları nelerdir?", "Yatay geçiş koşulları nelerdir?" gibi GENEL bir liste isterse:
    - Tek bir maddede "İşte liste budur" diye yazmayabilir.
    - Metin içindeki farklı maddelere dağılmış bilgileri (AKTS kredisi, GANO şartı, Süre şartı, Zorunlu dersler vb.) senin toplayıp BİRLEŞTİRMEN gerekir.
    - "Belgelerde toplu liste yok" deyip kestirip atma. Parçaları birleştirerek cevabı sen oluştur.

    KURAL 5: HALÜSİNASYON ENGELLEME
    - Yukarıdaki sentez kuralına rağmen, eğer parçalar da yoksa ve bilgi gerçekten metinde geçmiyorsa "Belgelerde bu bilgi bulunmamaktadır" de.
    - Tahmin yürütme, yorum yapma. Sadece metinde yazanı aktar.

    KURAL 6: REFERANS FORMATI
    - Cevap verirken, en son olarak bilgiyi hangi belgeden aldığını belirtmek için cümle sonuna (dosya_adi.pdf) formatını kullan.
    - Örnek: "Yüksek lisans için ALES puanı en az 55 olmalıdır. (lisansustu_yonetmeligi.pdf)"

    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Cevap oluşturulurken hata: {str(e)}", "sources": []}
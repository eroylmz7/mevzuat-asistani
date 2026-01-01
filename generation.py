import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. GÜVENLİK ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. ÇEVİRMEN VE "KİMLİK TESPİTİ" ---
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    # BURASI ÇOK ÖNEMLİ: Sorunun "Kime" ait olduğunu tespit ediyoruz.
    translation_prompt = f"""
    GÖREV: Kullanıcı sorusunu analiz et ve arama motoru için detaylandır.
    
    ANALİZ ADIMLARI:
    1. KİMLİK TESPİTİ: Soru "Lisans" öğrencisi için mi, "Yüksek Lisans/Doktora" öğrencisi için mi?
       - İpuçları: "Tez", "Danışman Atama", "Yeterlik", "Seminer", "Yayın Şartı" geçerse -> LİSANSÜSTÜ.
       - İpuçları: "ÇAP", "Yandal", "Yaz Okulu", "DC+", "DD+" geçerse -> LİSANS.
    2. EŞ ANLAMLILAR: "Büt" -> "Bütünleme", "Af" -> "Öğrenci Affı".
    3. SAYISAL VERİ: Soru bir süre (yıl/gün) veya puan soruyorsa, arama terimine "Süre Sınırı", "Azami Süre", "Geçerlilik" ekle.
    
    Soru: "{question}"
    Geliştirilmiş Arama Sorgusu:
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. RETRIEVAL (KAPASİTEYİ ARTIRDIK) ---
    try:
        # k=50 yapıyoruz. Neden?
        # Çünkü sistemde hem Lisans hem Lisansüstü belgeleri var. 
        # "Yatay Geçiş" arattığında ikisinden de 20'şer parça gelebilir. Hepsini alıp Prompt'a yollamalıyız.
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query, 
            k=50,           
            fetch_k=100,    
            lambda_mult=0.5 
        )
    except Exception as e:
        return {"answer": f"Arama hatası: {str(e)}", "sources": []}
    
    # --- 4. BAĞLAM (CONTEXT) HAZIRLIĞI ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        clean_content = doc.page_content.replace("\n", " ").strip()
        source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        
        # Modele hangi bilginin hangi dosyadan geldiğini açıkça söylüyoruz.
        context_text += f"\n[KAYNAK DOSYA: {source_name}] -> İÇERİK: {clean_content}\n"
        
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{source_name} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. CEVAPLAYICI (KAYNAK SEÇİCİ MODU) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.0
    )
    
    final_template = f"""
    Sen, Üniversite Mevzuat Uzmanısın. Elinde hem "LİSANS" hem de "LİSANSÜSTÜ" (Yüksek Lisans/Doktora) yönetmelikleri var.
    Görevin, soruya uygun olan DOĞRU yönetmeliği seçip oradan cevap vermektir.
    
    ELİNDEKİ BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- ⚠️ BELGE SEÇİM VE AYRIŞTIRMA KURALLARI (ÇOK KRİTİK) ---
    
    1. HEDEF KİTLE KONTROLÜ:
       - Soru "Yüksek Lisans", "Doktora", "Tez", "Yeterlik", "Danışman" veya "Yayın" içeriyorsa -> SADECE dosya adında "lisansustu" geçen belgelere bak. "lisans_yonetmeligi.pdf" dosyasını GÖRMEZDEN GEL.
       - Soru "Lisans", "Ön Lisans", "ÇAP", "Yandal" içeriyorsa -> "lisans_yonetmeligi.pdf" dosyasına bak.
       
    2. ÇELİŞKİ YÖNETİMİ:
       - Eğer "Lisans Yönetmeliği"nde süre 5 yıl, "Lisansüstü"nde süre sınırsız diyorsa; sorunun bağlamına göre doğru olanı seç. Karıştırma.
       - Emin değilsen: "Lisans yönetmeliğine göre şöyle, Lisansüstü yönetmeliğine göre böyledir" diye ayrım yaparak cevap ver.
       
    3. SAYISAL VERİ AVCILIĞI:
       - Soruda "Kaç yıl?", "Ne kadar süre?" varsa, metindeki "5 yıl", "3 ay", "Son ... yıl içinde" ifadelerini mutlaka bul.
    
    --- 🚫 FORMAT YASAKLARI ---
    - Cevap metninde "[KAYNAK DOSYA: ...]" gibi teknik etiketleri kullanıcıya gösterme.
    - Sadece profesyonel bir dille "Yönetmeliğe göre..." de.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Cevap oluşturma hatası: {str(e)}", "sources": []}
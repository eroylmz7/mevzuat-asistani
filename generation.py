import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. GÜVENLİK ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. ANALİST AJAN ---
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    # BURADA "ORTAK KONULAR" MANTIĞINI EKLİYORUZ
    translation_prompt = f"""
    GÖREV: Kullanıcı sorusunu analiz et ve arama motoru için zenginleştir.
    
    ANALİZ ADIMLARI:
    1. KİMLİK VE KONU TESPİTİ:
       - "LİSANSÜSTÜ": Soru "Tez", "Danışman", "Yeterlik", "Yayın Şartı", "Doktora" içeriyorsa.
       - "LİSANS": Soru "ÇAP", "Yandal", "DC+", "DD+" içeriyorsa.
       - "ORTAK/GENEL": Soru "Yatay Geçiş", "Muafiyet", "Kayıt Dondurma", "Devam Zorunluluğu", "İtiraz" gibi her iki seviyede de olan konuları içeriyorsa.
       
    2. ARAMA TERİMLERİ:
       - Soru bir "Zaman" veya "Yıl" soruyorsa (Örn: "Kaç yıl önce?"): Sorguya "Süre Sınırı", "Geçerlilik Süresi", "Zaman Aşımı", "Son ... yıl" terimlerini ekle.
    
    Soru: "{question}"
    Geliştirilmiş Arama Sorgusu:
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. RETRIEVAL (GENİŞ HAVUZ) ---
    try:
        # k=60 yapıyoruz ki hem Lisans hem Lisansüstü belgelerinden ilgili maddeler gelebilsin.
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query, 
            k=60,           
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
        source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor")).lower()
        
        # Dosya adına göre etiketleme
        if "lisansustu" in source_name:
            label = "LİSANSÜSTÜ YÖNETMELİĞİ"
        elif "lisans" in source_name and "lisansustu" not in source_name:
            label = "LİSANS YÖNETMELİĞİ"
        else:
            label = "DİĞER YÖNERGE"

        context_text += f"\n[KAYNAK: {label} ({source_name})] -> İÇERİK: {clean_content}\n"
        
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{source_name} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. CEVAPLAYICI (ESNEK MOD) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.0
    )
    
    final_template = f"""
    Sen, Üniversite Mevzuat Uzmanısın. Elindeki belgeleri analiz ederek soruya cevap ver.
    
    ELİNDEKİ BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- ⚠️ CEVAPLAMA STRATEJİSİ ---
    
    1. BELGE ÖNCELİĞİ (FİLTRELEME DEĞİL, ÖNCELİKLENDİRME):
       - Eğer kullanıcı soruda "Yüksek Lisans" veya "Doktora" dememişse bile; aradığı cevap (örneğin "5 yıl" kuralı) SADECE "LİSANSÜSTÜ" belgesinde yazıyorsa, o bilgiyi kullan ve kaynağını belirt.
       - "Görmezden gel" kuralını unut. Eğer bir belgede net bir sayısal kısıtlama (yıl, gün, puan) varsa, o bilgiyi kullanıcıya sun.
       
    2. AYRIM YAPMA:
       - Eğer hem Lisans hem Lisansüstü belgelerinde farklı bilgiler varsa, cevabı ayır:
         * **Lisans Yönetmeliğine Göre:** ...
         * **Lisansüstü Yönetmeliğine Göre:** ...
         
    3. SAYISAL DETAYLAR:
       - Soru "Kaç yıl?", "Ne zaman?" içeriyorsa; metindeki "5 yıl", "3 ay", "Son ... yıl içinde" ifadelerini mutlaka bul ve cevaba ekle.
    
    --- 🚫 FORMAT ---
    - "[KAYNAK: ...]" etiketlerini cevap metninde kullanma.
    - Kaynağı "Uludağ Üniversitesi Lisansüstü Eğitim Yönetmeliği'ne göre..." şeklinde cümle içinde geçir.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Cevap oluşturma hatası: {str(e)}", "sources": []}
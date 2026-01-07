import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

# --- YARDIMCI FONKSİYON: GEMINI RERANKER (AKILLI HAKEM) ---
def rerank_documents(query, docs, api_key):
    """
    Vektör veritabanından gelen kaba sonuçları (25 tane),
    Gemini'ye okutup 'Gerçekten alakalı mı?' diye puanlatır ve eler.
    """
    reranker_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Hızlı ve geniş context için ideal
        google_api_key=api_key,
        temperature=0.0
    )

    # Belgeleri numaralandırıp LLM'e sunuyoruz
    doc_text = ""
    for i, doc in enumerate(docs):
        # Dosya adını ve içeriği birleştiriyoruz
        source = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        doc_text += f"\n[ID: {i}] (Kaynak: {source}) -> {doc.page_content[:400]}...\n"

    rerank_prompt = f"""
    GÖREV: Aşağıdaki belge parçalarını analiz et ve kullanıcının sorusuyla EN ALAKALI olanları seç.

    SORU: "{query}"

    ADAY BELGELER:
    {doc_text}

    SEÇİM STRATEJİSİ (GENEL KURALLAR):
    1. **KAPSAM UYUMU:** Sorunun muhatabı kim? (Örn: Soru "Doktora" diyorsa, sadece "Lisans" ile ilgili belgeleri ELE. Soru "Yurt" diyorsa, "Eğitim" belgelerini ELE.)
    2. **İÇERİK EŞLEŞMESİ:** Belge, soruya cevap olabilecek somut bir hüküm, madde veya sayısal veri içeriyor mu? Boş veya alakasız giriş kısımlarını seçme.
    3. **HİYERARŞİ:** Eğer aynı konuda hem "Genel Yönetmelik" hem de "Uygulama Esasları/Yönerge" varsa, daha detaylı olan Yönergeyi/Esasları tercih et.
    
    ÇIKTI FORMATI (JSON):
    {{ "selected_indices": [0, 2, 5] }}
    """
    try:
        response = reranker_llm.invoke(rerank_prompt).content
        # JSON temizliği (Markdown backticklerini kaldır)
        cleaned_response = re.sub(r"```json|```", "", response).strip()
        selected_data = json.loads(cleaned_response)
        selected_indices = selected_data.get("selected_indices", [])
        
        # Eğer hiçbiri seçilmezse veya hata olursa (boş dönerse) ilk 5 belgeyi al (Fallback)
        if not selected_indices:
            return docs[:5]
            
        return [docs[i] for i in selected_indices if i < len(docs)]
    except:
        return docs[:5]

def generate_answer(question, vector_store):
    
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
    Soru: "{question}"

    GÖREV: Kullanıcı sorusunu analiz et ve arama motoru için SADECE GEREKLİYSE ek terim ekle.
    
    ANALİZ MANTIĞI (SADE):
    1. EĞER SORU "LİSANSÜSTÜ" İLE İLGİLİYSE:
       - (İpuçları: Tez, Jüri, Yeterlik, Danışman, Enstitü, Seminer, TİK, ALES)
       - EKLE: "LİSANSÜSTÜ EĞİTİM YÖNETMELİĞİ"

    2. EĞER SORU "LİSANS"  İLE İLGİLİYSE:
       - (İpuçları: ÇAP, Yandal, Yaz Okulu, Tek Ders, Bütünleme, DC, DD, Azami Süre)
       - EKLE: "LİSANS EĞİTİM YÖNETMELİĞİ"

    3. EĞER SORU "UYGULAMA / STAJ" İLE İLGİLİYSE (YENİ KURAL):
       - (İpuçları: Staj, İME, Uygulamalı Eğitim, İş Yeri Eğitimi, Grup)
       - EKLE: "UYGULAMALI EĞİTİM YÖNERGESİ"
       

    4. DİĞER DURUMLARDA:
       - Sadece "MEVZUAT" ekle.

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
        initial_docs = vector_store.max_marginal_relevance_search(
            hybrid_query,
            k=25,             
            fetch_k=300,      
            lambda_mult=0.6  
        )
    except Exception as e:
        return {"answer": f"Veritabanı hatası: {str(e)}", "sources": []}
    
  
# --- 3. RE-RANKING (AKILLI ELEME) 🔥 ---
    # 25 belgeyi al, Gemini'ye ver, en iyi 5 tanesini seçtir.
    # Bu aşama "Lisans vs Yüksek Lisans" karışıklığını %100 çözer.
    final_docs = rerank_documents(hybrid_query, initial_docs, google_api_key)

    # --- 4. FORMATLAMA ---
    context_text = ""
    sources = []

    for doc in final_docs:
        content = doc.page_content.replace("\n", " ").strip()
        filename = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        
        context_text += f"\n--- KAYNAK: {filename} (Sayfa {page}) ---\n{content}\n"
        if filename not in sources:
            sources.append(filename)

    # --- 5. CEVAPLAYICI (HUKUKÇU MODU) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.2 # Yaratıcılık 
    )
    
    final_template = f"""
    Sen Bursa Uludağ Üniversitesi mevzuat asistanısın. 
    Elinizdeki belgeleri  kullanarak soruya en doğru, resmi ve net cevabı ver.

    ELİNDEKİ BELGELER:
    {context_text}

    SORU: {question}

    ---  CEVAPLAMA KURALLARI ---

    KURAL 1: HİYERARŞİ 
    - Özel düzenleme > Genel düzenleme
    - Yönerge/Uygulama Esasları > Yönetmelik

    KURAL 2: SENTEZ VE BİRLEŞTİRME
    - Bilgiler parça parça olabilir (örn: Bir maddede süre, diğerinde AKTS yazar). Gerekirse bunları birleştirerek bütünlüklü cevap ver.
        Örnek: "lisans mezuniyet şartları nelerdir?" sorusu
    - Sayısal değerler (20 gün, %70, 240 AKTS gibi) özellikle dikkatli ara. Cevap sayısal bir değer gerektirebilir.

    KURAL 3: REFERANS
    - Bilgiyi hangi dosyadan aldığını parantez içinde belirt. Örn: (uygulamali_egitimler.pdf)

    KURAL 4: DÜRÜSTLÜK
    - Bilgi yoksa uydurma, "Belgelerde bulunmamaktadır" de.

    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        
        # --- DEĞİŞİKLİK BURADA: CEVAP YOKSA KAYNAK GİZLE  ---
        # Eğer cevapta "bulunamadı", "yoktur" gibi şeyler geçiyorsa kaynakları boşalt.
        negative_signals = ["bulunmamaktadır", "bilgi yok", "rastlanmamıştır", "yer almamaktadır", "belirtilmemiştir"]
        
        if any(signal in answer.lower() for signal in negative_signals):
            final_sources = [] # Boş liste döndür (Böylece UI'da kutu çıkmaz)
        else:
            final_sources = sources[:5] # Sadece ilk 5 dosya adı

        return {"answer": answer, "sources": final_sources}

    except Exception as e:
        return {"answer": f"Cevap oluşturulurken hata: {str(e)}", "sources": []}
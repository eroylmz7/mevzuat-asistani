import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

# --- 1. RERANKER (HAKEM) ---
# Bu kısım kalmalı çünkü Streamlit Cloud'un işlemcisi sınırlı. 
# 40 belgeyi birden okuyamaz, en iyi 5-10 tanesini seçmeli.
def rerank_documents(query, docs, api_key):
    reranker_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.0
    )

    doc_text = ""
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        clean_content = doc.page_content.replace("\n", " ").strip()
        # 1500 karaktere çıkardık ki bağlam kopmasın
        doc_text += f"\n[ID: {i}] (Kaynak: {source}) -> {clean_content[:1500]}...\n"

    rerank_prompt = f"""
    GÖREV: Kullanıcının sorusuna cevap verebilecek belge parçalarını seç.
    
    SORU: "{query}"
    
    ADAY BELGELER:
    {doc_text}
    
    SEÇİM KRİTERİ:
    - Belge sorudaki anahtar kelimeleri (Staj, Not, Yüzde, Bütünleme vb.) içeriyor mu?
    - Konuyla uzaktan yakından alakası varsa SEÇ. Çok sıkı eleme yapma.
    
    ÇIKTI (JSON):
    {{ "selected_indices": [0, 2, 5] }}
    """
    try:
        response = reranker_llm.invoke(rerank_prompt).content
        cleaned_response = re.sub(r"```json|```", "", response).strip()
        selected_data = json.loads(cleaned_response)
        selected_indices = selected_data.get("selected_indices", [])
        
        if not selected_indices:
            return docs[:5] # Hiçbir şey bulamazsa ilk 5'i döndür
        return [docs[i] for i in selected_indices if i < len(docs)]
    except:
        return docs[:5]

# --- 2. ANA FONKSİYON (SADELEŞTİRİLDİ) ---
def generate_answer(question, vector_store, chat_history):
    
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- ADIM 1: GENİŞ ARAMA (RETRIEVAL) ---
    # Router, HyDE vs. HEPSİNİ KALDIRDIK.
    # Sadece soruyu soruyoruz ama "k" değerini artırıyoruz.
    try:
        # k=45 yapıyoruz. "Ağı" geniş atıyoruz ki o staj maddesi mutlaka takılsın.
        docs = vector_store.max_marginal_relevance_search(
            question,
            k=45,             # 45 Belge getir (Eskiden 20 idi, yetmiyordu)
            fetch_k=200,      # 200 aday arasından seç
            lambda_mult=0.5   # Çeşitliliği artır
        )
    except Exception as e:
        return {"answer": f"Veritabanı hatası: {str(e)}", "sources": []}
    
    # --- ADIM 2: RERANKING (ELEME) ---
    # 45 belgeyi Gemini'ye verip "Bana en iyi 5-10 tanesini ver" diyoruz.
    final_docs = rerank_documents(question, docs, google_api_key)

    # --- ADIM 3: FORMATLAMA ---
    context_text = ""
    sources = []

    for doc in final_docs:
        content = doc.page_content.replace("\n", " ").strip()
        filename = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        
        context_text += f"\n--- KAYNAK: {filename} (Sayfa {page}) ---\n{content}\n"
        if filename not in sources:
            sources.append(filename)

    # --- ADIM 4: CEVAPLAYICI ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.3 # Biraz esneklik iyidir
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
    
    KURAL 3: SAYISAL VERİLER
    -Eğer soru "AA katsayısı" veya "Onur notu" gibi bir sayı soruyorsa, belgelerdeki tabloları veya sayı içeren maddeleri çok dikkatli oku.

    KURAL 3: REFERANS
    - Bilgiyi bulabildiysen cevap ile birlikte sonuna hangi dosyadan aldığını parantez içinde belirt. Örn: (uygulamali_egitimler.pdf)
    - Eğer cevabı bulamadıysan sadece "Belgelerde bu konu hakkında bilgi bulunmamaktadır." yaz.

    KURAL 4: DÜRÜSTLÜK
    - Bilgi yoksa uydurma, "Belgelerde bulunmamaktadır" de.

    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        
        # Basit negatif kontrolü
        negative_signals = ["bilgi bulunmamaktadır", "bilgiye rastlanmamıştır", "yer almamaktadır"]
        if any(signal in answer.lower() for signal in negative_signals):
            final_sources = []
        else:
            final_sources = sources[:5]

        return {"answer": answer, "sources": final_sources}

    except Exception as e:
        return {"answer": f"Cevap oluşturulurken hata: {str(e)}", "sources": []}
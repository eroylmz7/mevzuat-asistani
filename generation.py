import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

# --- 1. GEÇMİŞİ HATIRLAYAN SORU DÜZENLEYİCİ ---
def reformulate_question(question, chat_history, api_key):
    if not chat_history:
        return question

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )
    
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])

    prompt = f"""
    GÖREV: Sohbet geçmişine bakarak kullanıcının son sorusunu tek başına anlaşılır hale getir.
    
    SOHBET GEÇMİŞİ:
    {history_text}
    
    SON SORU: "{question}"
    
    KURALLAR:
    - Soru "Süresi ne kadar?", "Kaç kredi?" gibi eksikse, geçmişten özneyi (Örn: Lisans Mezuniyeti) bul ve tamamla.
    - Soru zaten netse aynen bırak.
    
    DÜZENLENMİŞ SORU:
    """
    
    try:
        return llm.invoke(prompt).content.strip()
    except:
        return question
    
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
    GÖREV: Aşağıdaki belge parçalarını kullanıcının sorusuna olan alaka düzeyine göre değerlendir.
    
    SORU: "{query}"
    
    ADAY BELGELER:
    {doc_text}
    
    KURALLAR:
    1. Soruda özellikle  Lisans ile ilgili mi Lisansüstü ile ilgili mi soru sorulmuş dikkat et.
    2. Soru "Staj" ise, "Yönerge" belgelerine öncelik ver.
    3. Soruya net cevap içeren belgeleri seç.
    
    ÇIKTI FORMATI (Sadece JSON):
    {{
        "selected_indices": [en iyi belgenin ID'si, ikinci en iyi ID, ...]
    }}
    En fazla 5 belge seç.
    """
    try:
        response = reranker_llm.invoke(rerank_prompt).content
        
        # --- JSON TEMİZLİK MEKANİZMASI (YENİ) ---
        
        cleaned_response = re.sub(r"```json|```", "", response).strip()
        
        selected_data = json.loads(cleaned_response)
        selected_indices = selected_data.get("selected_indices", [])
        
        # Seçilenleri döndür
        return [docs[i] for i in selected_indices if i < len(docs)]
    except:
        # Hata olursa en iyi ihtimalle ilk 5'i döndür 
        return docs[:5]

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. GÜVENLİK ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    refined_question = reformulate_question(question, chat_history, google_api_key)
    
    
    # --- 2. ANALİST AJAN (Sorgu Zenginleştirme) ---
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    translation_prompt = f"""
    Soru: "{refined_question}"

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
        hybrid_query = f"{refined_question} {official_terms}"
    except:
        hybrid_query = refined_question

    # --- 3. RETRIEVAL (KARARLI MOD) ---
    try:
        # Karmaşık if-else'i kaldırdık. Tek ve güçlü bir standart kullanacağız.
        initial_docs = vector_store.max_marginal_relevance_search(
            hybrid_query,
            k=25,             
            fetch_k=300,      
            lambda_mult=0.65  
        )
    except Exception as e:
        return {"answer": f"Veritabanı hatası: {str(e)}", "sources": []}
    
  
# --- 3. RE-RANKING (AKILLI ELEME) 🔥 ---
    # 25 belgeyi al, Gemini'ye ver, en iyi 5 tanesini seçtir.
    # Bu aşama "Lisans vs Yüksek Lisans" karışıklığını %100 çözer.
    final_docs = rerank_documents(refined_question, initial_docs, google_api_key)

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
    Elinizdeki belgeleri (context) kullanarak soruya (question) en doğru, resmi ve net cevabı ver.

    ELİNDEKİ BELGELER (context):
    {context_text}

    SORU: {refined_question}

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
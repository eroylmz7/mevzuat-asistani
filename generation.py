import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# chat_history uyarısını takma, kodun çalışmasını etkilemez.
def generate_answer(question, vector_store, chat_history):
    
    # --- 1. GÜVENLİK VE AYARLAR ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. HİBRİT ARAMA (TERİM ZENGİNLEŞTİRME) ---
    # SENİN İSTEDİĞİN MODEL: gemini-2.5-flash
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.2 
    )
    
    translation_prompt = f"""
    GÖREV: Aşağıdaki öğrenci sorusundaki "halk ağzı" kelimeleri, üniversite "resmi mevzuat" diline çevir.
    Sadece resmi terimleri çıktı olarak ver. Başka hiçbir kelime yazma.
    
    Örnekler:
    - "Staj defterini kime vericem?" -> "Uygulamalı Eğitim Dosyası Teslimi"
    - "Okulu dondurmak istiyorum" -> "Kayıt Dondurma Başvurusu"
    - "Dersten kaldım ne olacak?" -> "Ders Tekrarı Başarısızlık Durumu"
    
    Soru: "{question}"
    Resmi Karşılık:
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. BELGE GETİRME (Retrieval) ---
    # vector_store nesnesi app.py'den HuggingFace ayarıyla geldiği için 
    # burada embedding tanımlamaya gerek yok, kendi içinde halleder.
    try:
        # fetch_k değerini yüksek tutuyoruz ki alakasızları eleyip en iyileri seçsin.
        docs = vector_store.max_marginal_relevance_search(hybrid_query, k=8, fetch_k=30)
    except Exception as e:
        return {"answer": f"Arama sırasında hata oluştu (Veritabanı/Embedding uyumsuzluğu olabilir): {str(e)}", "sources": []}
    
    # --- 4. BAĞLAM OLUŞTURMA ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        clean_content = doc.page_content.replace("\n", " ").strip()
        context_text += f"\n--- BELGE PARÇASI {i+1} ---\n{clean_content}\n"
        
        # Kaynakça oluşturma
        source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page_num = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{source_name} (Sayfa {page_num})"
        
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. CEVAP ÜRETME (Generation) ---
    # SENİN İSTEDİĞİN MODEL: gemini-2.5-flash
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    final_template = f"""
    Sen üniversite mevzuat asistanısın. Görevin öğrencilerin sorularını SADECE aşağıdaki "RESMİ BELGELER"e dayanarak cevaplamaktır.
    
    RESMİ BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- ⚠️ KRİTİK KURALLAR (HARFİYEN UY) ---
    
    1. EKSİKSİZ LİSTELEME: 
       Kullanıcı "koşullar", "maddeler" gibi bir liste istiyorsa belgede geçen TÜM maddeleri yaz.
       
    2. CÜMLE BÜTÜNLÜĞÜ:
       Cümleleri asla yarım bırakma. Tamamla.
       
    3. FORMAT (ÖNEMLİ):
       - Eğer cevap bir liste gerektiriyorsa (örneğin: koşullar, belgeler, maddeler) o zaman **Maddeler (Bullet Points)** kullan.
       - Eğer cevap tek bir açıklama veya durum ise, **Normal Paragraf** olarak yaz. Gereksiz yere maddeleme yapma.
       
    4. OLUMSUZ DURUM:
       Eğer cevap verilen metinlerde KESİNLİKLE yoksa, sadece: "Verilen dokümanlarda bu bilgi yer almıyor." yaz.
       
    5. STAJ = UYGULAMALI EĞİTİM.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
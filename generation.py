import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vector_store, chat_history):
    # API Key Kontrolü
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- ADIM 1: ARAMA TERİMİ OLUŞTURMA (Sadece Bulmak İçin) ---
    # Burası cevabı etkilemez, sadece doğru PDF sayfasını bulmaya yarar.
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0 # Çeviri yaparken bile risk almıyoruz
    )
    
    translation_prompt = f"""
    GÖREV: Kullanıcının sorusunu, üniversite yönetmeliklerinde geçebilecek RESMİ TERİMLERE dönüştür.
    
    KURALLAR:
    1. Asla soruyu cevaplama.
    2. Sadece arama motoru için anahtar kelime üret.
    3. Eş anlamlıları düşün (Staj -> Uygulamalı Eğitim, Okul -> Üniversite vb.)
    
    Soru: "{question}"
    Arama Terimleri:
    """
    
    try:
        enhanced_query = llm_translator.invoke(translation_prompt).content
        # Loglara yazdırıp ne aradığını görebilirsin (İsteğe bağlı)
        print(f"🔍 Arama: {enhanced_query}") 
    except:
        enhanced_query = question 

    # --- ADIM 2: BELGE GETİRME ---
    # Pinecone'dan en alakalı 10 parçayı getiriyoruz
    docs = vector_store.max_marginal_relevance_search(enhanced_query, k=10, fetch_k=30)
    
    # --- ADIM 3: BAĞLAM OLUŞTURMA ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        # Metni temizle
        clean_content = doc.page_content.replace("\n", " ").strip()
        context_text += f"\n--- BELGE PARÇASI {i+1} ---\n{clean_content}\n"
        
        # Kaynakları topla
        source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page_num = int(doc.metadata.get("page", 0)) + 1
        src_str = f"{source_name} (Sayfa {page_num})"
        if src_str not in sources:
            sources.append(src_str)

    # --- ADIM 4: CEVAP ÜRETME (SIFIR YORUM MODU) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.0  # <--- KRİTİK AYAR: 0.0 demek "Robot Modu" demektir. Asla uyduramaz.
    )
    
    final_template = f"""
    Sen sadece verilen metinlere sadık kalan bir üniversite asistanısın.
    
    GÖREV: Aşağıdaki "RESMİ BELGELER" içindeki bilgileri kullanarak soruya cevap ver.
    
    RESMİ BELGELER:
    {context_text}
    
    SORU: {question}
    
    ÇOK KATI KURALLAR:
    1. Sadece ve sadece yukarıdaki "RESMİ BELGELER"de yazan bilgiyi kullan.
    2. Kendi yorumunu, dışarıdan bildiğin bilgileri ASLA ekleme.
    3. Belgede "Uygulamalı Eğitim" yazıyorsa ve öğrenci "Staj" dediyse, cevabında "Yönetmelikte Uygulamalı Eğitim olarak belirtildiği üzere..." diyerek düzelt ve cevabı ver.
    4. Eğer bilgi belgelerde YOKSA, "Verilen dokümanlarda bu sorunun cevabı bulunmamaktadır" de. Uydurma.
    5. Cevabın resmi ve net olsun.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
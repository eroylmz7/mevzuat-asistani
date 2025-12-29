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

    # --- ADIM 1: HİBRİT ARAMA TERİMİ OLUŞTURMA ---
    # Hem öğrencinin dediğini hem de resmi karşılığını aynı anda arayacağız.
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    translation_prompt = f"""
    GÖREV: Öğrencinin sorusundaki anahtar kelimelerin RESMİ MEVZUAT karşılıklarını bul.
    Sadece resmi terimleri yan yana yaz.
    
    Örnek:
    Soru: "Staj yerimi değiştirebilir miyim?"
    Cevap: Uygulamalı Eğitim İşletme Değişikliği
    
    Soru: "{question}"
    Cevap:
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content
        # SİHİRLİ DOKUNUŞ: İkisini birleştiriyoruz!
        # "Staj yerimi değiştirebilir miyim? Uygulamalı Eğitim İşletme Değişikliği"
        hybrid_query = f"{question} {official_terms}"
        
        # EKRANA YAZDIRALIM (Kullanıcı görsün ne arandığını)
        #with st.expander("🕵️‍♂️ Arka Plan İşlemleri (Debug)", expanded=False):
         #   st.write(f"**Orijinal Soru:** {question}")
          #  st.write(f"**Resmi Terimler:** {official_terms}")
           # st.write(f"**Veritabanında Aranan:** {hybrid_query}")
            
    except:
        hybrid_query = question 

    # --- ADIM 2: BELGE GETİRME (MMR ile Çeşitlilik) ---
    # fetch_k=40 yaptık ki havuz geniş olsun, ıskalamasın.
    docs = vector_store.max_marginal_relevance_search(hybrid_query, k=10, fetch_k=40)
    
    # --- ADIM 3: BAĞLAM OLUŞTURMA ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        clean_content = doc.page_content.replace("\n", " ").strip()
        context_text += f"\n--- BELGE PARÇASI {i+1} ---\n{clean_content}\n"
        
        source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page_num = int(doc.metadata.get("page", 0)) + 1
        src_str = f"{source_name} (Sayfa {page_num})"
        if src_str not in sources:
            sources.append(src_str)

    # --- ADIM 4: CEVAP ÜRETME (SIFIR TOLERANS) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.0 # Kesinlikle uydurmasın, sadece metni okusun.
    )
    
    final_template = f"""
    Sen üniversite mevzuat asistanısın.
    
    Aşağıdaki "RESMİ BELGELER"i oku ve soruya cevap ver.
    
    RESMİ BELGELER:
    {context_text}
    
    SORU: {question}
    
    KURALLAR:
    1. Belgede "Uygulamalı Eğitim" yazıyorsa ve öğrenci "Staj" dediyse bunları aynı şey kabul et.
    2. Cevabı belgelerin içinden bul ve net bir şekilde yaz.
    3. Eğer belgede YOKSA, "Verilen dokümanlarda bu bilgi yer almıyor" de.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
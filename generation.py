import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vector_store, chat_history):
    # --- 1. GÜVENLİK VE AYARLAR ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı (secrets.toml dosyasını kontrol et).", "sources": []}

    # --- 2. HİBRİT ARAMA (TERİM ZENGİNLEŞTİRME) ---
    # Gemini Flash çok ucuz ve hızlı olduğu için bu ön işlemi yapmak harika bir fikir.
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.2 # Yaratıcılık sıfır olsun, sadece çeviri yapsın.
    )
    
    # Prompt'u biraz daha "Emir kipi" ile yazdık ki sohbet etmeye çalışmasın.
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
        # Hem öğrencinin sorusunu hem de resmi terimi birleştirip arıyoruz.
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. BELGE GETİRME (MMR) ---
    # Fetch_k değerini yüksek tutuyoruz ki alakasızları eleyip en iyileri seçsin.
    docs = vector_store.max_marginal_relevance_search(hybrid_query, k=8, fetch_k=30)
    
    # --- 4. BAĞLAM OLUŞTURMA ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        clean_content = doc.page_content.replace("\n", " ").strip()
        context_text += f"\n--- BELGE PARÇASI {i+1} ---\n{clean_content}\n"
        
        # Kaynakça oluşturma
        source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page_num = int(doc.metadata.get("page", 0)) + 1
        src_str = f"{source_name} (Sayfa {page_num})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. CEVAP ÜRETME (GÜÇLENDİRİLMİŞ PROMPT) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 # Düşük sıcaklık = Daha kararlı ve tam cümleler
    )
    
    final_template = f"""
    Sen üniversite mevzuat asistanısın. Görevin öğrencilerin sorularını SADECE aşağıdaki "RESMİ BELGELER"e dayanarak cevaplamaktır.
    
    RESMİ BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- ⚠️ KRİTİK KURALLAR (HARFİYEN UY) ---
    
    1. EKSİKSİZ LİSTELEME: 
       Kullanıcı "koşullar", "maddeler", "belgeler" gibi bir liste istiyorsa (Örn: Mezuniyet koşulları), belgede geçen TÜM maddeleri yaz. Asla "vb." diyip kesme veya maddeleri atlama.
       
    2. CÜMLE BÜTÜNLÜĞÜ:
       Cümleleri asla yarım bırakma. Eğer bir cümle "bağlı olarak" veya "nedeniyle" gibi bitiyorsa, o cümlenin devamını da mutlaka yaz. Anlam bütünlüğü bozulmamalı.
       
    3. FORMAT:
       Cevabı okunabilirliği artırmak için **ALT ALTA MADDELER (Bullet Points)** halinde ver.
       
    4. OLUMSUZ DURUM:
       Eğer cevap verilen metinlerde KESİNLİKLE yoksa, sadece: "Verilen dokümanlarda bu bilgi yer almıyor." yaz. Başka hiçbir şey (kaynak, yorum vb.) ekleme.
       
    5. STAJ = UYGULAMALI EĞİTİM:
       "Staj" kelimesini "Uygulamalı Eğitim" ile eşdeğer tut.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
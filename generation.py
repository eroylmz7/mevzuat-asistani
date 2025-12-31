import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. AYARLAR ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. EVRENSEL ÇEVİRMEN (ARTIK KELİME EZBERLEMİYOR) ---
    # Gemini'ye diyoruz ki: "Sen Akademik Literatür Uzmanısın. Hangi kelimenin ne anlama geldiğini sen bul."
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    translation_prompt = f"""
    GÖREV: Kullanıcının sorusunu analiz et ve belge araması için "Resmi Literatür" formatına çevir.
    
    YÖNERGE:
    1. Kullanıcı "halk ağzı" veya "kampüs argosu" (Örn: Vize, Büt, Yaz Okulu, Dondurma) kullanmış olabilir.
    2. Bunları Türk Yükseköğretim Mevzuatında kullanılan GENEL GEÇER RESMİ TERİMLERE dönüştür. (Örn: "Ara Sınav", "Bütünleme", "Yaz Öğretimi", "Kayıt Dondurma").
    3. EĞER SORUDA BİR ŞART/KOŞUL VARSA: Arama terimine mutlaka "Koşulları", "Değerlendirme Kriterleri", "Sayısal Sınırları" gibi ifadeler ekle.
    4. Sadece oluşturduğun yeni arama cümlesini yaz.
    
    Soru: "{question}"
    Akademik Arama Cümlesi:
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        # Hem öğrencinin dediğini hem de resmi halini arıyoruz (Garantici yaklaşım)
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. ARAMA (Retrieval) ---
    try:
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query, 
            k=10, 
            fetch_k=40,    # Daha geniş havuz
            lambda_mult=0.7 # Çeşitliliği artırır
        )
    except Exception as e:
        return {"answer": f"Arama hatası: {str(e)}", "sources": []}
    
    # --- 4. BAĞLAM ---
    context_text = ""
    sources = []
    for i, doc in enumerate(docs):
        clean_content = doc.page_content.replace("\n", " ").strip()
        context_text += f"\n--- BELGE PARÇASI {i+1} ---\n{clean_content}\n"
        
        src = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{src} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. EVRENSEL CEVAPLAYICI (GENERATOR) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.2 
    )
    
    # BURASI ARTIK HER YÖNETMELİĞE UYAR
    final_template = f"""
    Sen bir Mevzuat Asistanısın. Görevin aşağıdaki "RESMİ BELGELER"i analiz ederek soruyu cevaplamaktır.
    
    RESMİ BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- ⚠️ CEVAPLAMA STRATEJİSİ ---
    
    1. TERMİNOLOJİ EŞLEŞTİRMESİ (ESNEKLİK):
       - Kullanıcı "Staj" diyebilir ama belgede "Mesleki Eğitim" veya "Uygulama" yazabilir. 
       - Kullanıcı "Vize" diyebilir ama belgede "Ara Sınav" yazabilir.
       - Bağlamdan bu eşleştirmeyi YAPAY ZEKA OLARAK SEN YAP ve belgedeki doğru terimi kullanarak cevap ver.
       
    2. SAYISAL VERİ HASSASİYETİ:
       - Eğer soru bir puan, not, süre veya kontenjan hakkındaysa; metindeki SADECE kelimelere değil, TABLOLARDAKİ SAYISAL DEĞERLERE odaklan.
       - Metin "belli bir ortalama" diyorsa ve tabloda "3.00" yazıyorsa, cevaba "3.00"ı ekle.
       
    3. FORMAT:
       - Eğer bir prosedür veya şartlar listesi varsa madde madde (Bullet Points) yaz.
       - Tekil bir bilgi soruluyorsa net bir cümle kur.
       
    4. SINIRLAR:
       - Sadece verilen metne sadık kal. Metinde yoksa "Dokümanlarda bu bilgi bulunmamaktadır" de.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Cevap üretme hatası: {str(e)}", "sources": []}
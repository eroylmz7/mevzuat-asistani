import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

# --- 1. RERANKER (HAKEM) --- 
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
        # 2500 karaktere çıkardık ki bağlam kopmasın
        doc_text += f"\n[ID: {i}] (Kaynak: {source}) -> {clean_content[:2500]}...\n"

    rerank_prompt = f"""
    GÖREV: Aşağıdaki belge parçalarını analiz et ve kullanıcının sorusuyla EN ALAKALI olanları seç.

    SORU: "{query}"

    ADAY BELGELER:
    {doc_text}

    SEÇİM STRATEJİSİ (GENEL KURALLAR):
    1. Belge, sorudaki ana konuyu (Staj, Kredi, AKTS, Puan) anlatıyor mu?
    2. Sorudaki detaylar belgede birebir geçmeyebilir. ANLAM olarak eşleşiyorsa SEÇ.
    3. Soru "Seviye" diyebilir, Belge "Düzey" diyebilir. Bunun gibi benzer anlamlıları aynı kabul et.
    3. (30 AKTS, %20, 65 puan) gibi sayısal veriler içeren belgeleri önceliklendir.
    
    ÇIKTI FORMATI (JSON):
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

# --- 2. ANA FONKSİYON ---
def generate_answer(question, vector_store, chat_history):
    
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # ---  SORGU TEMİZLEYİCİ VE ÇEVİRİCİ 
    try:
        cleaner_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=google_api_key,
            temperature=0.0 # Yaratıcılık yok, sadece temizlik
        )
        
        cleaning_prompt = f"""
        GÖREV: Kullanıcı sorusunu veritabanı araması için ZENGİNLEŞTİR ve RESMİLEŞTİR.
        
        YAPILACAKLAR:
        1. Gürültüyü sil ("lütfen", "acaba" vb.).
        2. EŞ ANLAMLILARI EKLE:
           - "Seviye" -> "Düzey / Puan / Skor"
           - "Şart" -> "Koşul / Kriter"
           - "Staj" -> "İşletmede Mesleki Eğitim / Uygulamalı Eğitim"
        3. Sorunun özünü koru.

        Orijinal Soru: "{question}"
        Optimize Edilmiş Sorgu:
        """
        optimized_query = cleaner_llm.invoke(cleaning_prompt).content.strip()
        
        # Arama yaparken optimize edilmiş sorguyu kullanacağız!
        # Ama cevap verirken orijinal soruyu (question) kullanacağız.
        
    except:
        optimized_query = question # Hata olursa orijinali kullan

    # --- ADIM 1: GENİŞ ARAMA (RETRIEVAL) ---
    try:
        # Arama A: Orijinal Soru (Belki parantez içi önemlidir?)
        docs_raw = vector_store.max_marginal_relevance_search(
            question, k=30, fetch_k=300, lambda_mult=0.6
        )
        
        # Arama B: Temiz Soru (Gürültüsüz)
        docs_clean = vector_store.max_marginal_relevance_search(
            optimized_query, k=30, fetch_k=300, lambda_mult=0.6
        )

        # --- DEDUPLICATION (TEKRAR ENGELLEME) ---
        seen_identifiers = set()
        initial_docs = []
        
        for doc in docs_clean + docs_raw:
            unique_id = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content[:500]
            )
            
            if unique_id not in seen_identifiers:
                initial_docs.append(doc)
                seen_identifiers.add(unique_id)

    except Exception as e:
        return {"answer": f"Veritabanı hatası: {str(e)}", "sources": []}
    
    # --- ADIM 2: RERANKING ---
    # Hakem'e ZENGİNLEŞTİRİLMİŞ SORUYU veriyoruz.
    final_docs = rerank_documents(optimized_query, initial_docs, google_api_key)

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
    # ==========================================
    # DEBUG (HATA AYIKLAMA) PENCERESİ
    # ==========================================
    # Bu kısım sayesinde Streamlit ekranında modelin okuduğu metni görebileceksin.
    # with st.expander("🔍 DEBUG: Modelin Okuduğu Ham Metin (Context)"):
    #     st.write(f"Toplam Karakter Sayısı: {len(context_text)}")
    #     st.write("Aşağıdaki metin, PDF'ten çekilip modele verilen ham veridir. Tabloların bozulup bozulmadığını buradan kontrol et:")
    #     st.code(context_text)
    # ==========================================

    # --- ADIM 4: CEVAPLAYICI ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.2 #  esneklik 
    )
    
    final_template = f"""
    Sen Bursa Uludağ Üniversitesi mevzuat asistanısın. 
    Elinizdeki belgeleri  kullanarak soruya en doğru, resmi ve net cevabı ver.

    ELİNDEKİ BELGELER:
    {context_text}

    SORU: {question}

    ---  CEVAPLAMA KURALLARI ---


    KURAL 1: SENTEZ VE BİRLEŞTİRME
    - Bilgiler parça parça olabilir (örn: Bir maddede süre, diğerinde AKTS yazar). Gerekirse bunları birleştirerek bütünlüklü cevap ver.
        Örnek: "lisans mezuniyet şartları nelerdir?" sorusu
    - PDF'ten gelen metinlerde TABLO yapıları bozulmuş ve satırlar birbirine girmiş olabilir.
    - Örnek: "Tezsiz Yüksek Lisans 10 30" gibi bir yazı görürsen, bunun "10 Ders" ve "30 Kredi" olduğunu bağlamdan çıkar.
    - Satır kaymalarına aldanma, kelimelerin ve sayıların yakınlığına bakarak mantıksal ilişki kur.
    
    KURAL 2: SAYISAL VERİLER
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
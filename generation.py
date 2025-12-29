import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vector_store, chat_history):
    # 1. API Key Kontrolü
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}
    
    # 2. Sohbet Geçmişini Hazırla
    history_text = ""
    if chat_history:
        # Son 4 mesajı alıp metne çeviriyoruz ki yapay zeka bağlamı anlasın
        for msg in chat_history[-4:]:
            role = "ÖĞRENCİ" if msg["role"] == "user" else "ASİSTAN"
            history_text += f"{role}: {msg['content']}\n"
    
    # 3. Ayarlar
    # Buluttan kaç parça belge getirsin? 
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Gemini Modelini Başlat
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Hız ve performans için ideal
        google_api_key=google_api_key,
        temperature=0.3
    )

    # 4. Prompt (Yapay Zekaya Talimat)
    template = f"""
    Sen üniversite mevzuatları konusunda uzman yardımsever bir asistansın.
    
    Önceki Konuşmalar:
    {history_text}
    
    Aşağıdaki MEVZUAT BİLGİSİ'ne göre soruyu cevapla.
    MEVZUAT BİLGİSİ:
    {{context}}
    
    SORU: {{question}}
    
    KURALLAR:
    1. Sadece verilen mevzuat bilgisine dayanarak cevap ver.
    2. Eğer bilgi dokümanlarda yoksa "Bu konuda dokümanlarda bilgi bulamadım" de, uydurma.
    3. Cevabın net, anlaşılır ve öğrenciye yardımcı olacak tonda olsun.
    
    CEVAP:
    """
    
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    # 5. Zinciri Kur (Otomatik İşlem)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True, # Kaynakları görmek istiyoruz
        chain_type_kwargs={"prompt": prompt_template}
    )

    # 6. Çalıştır ve Sonucu Döndür
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        
        # Kaynakları düzenle (Hangi dosya, hangi sayfa?)
        sources = []
        for doc in result["source_documents"]:
            source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
            # Sayfa numarası bazen 0'dan başlar, +1 ekliyoruz
            page_num = int(doc.metadata.get("page", 0)) + 1
            src_str = f"{source_name} (Sayfa {page_num})"
            if src_str not in sources:
                sources.append(src_str)
        
        return {"answer": answer, "sources": sources}
        
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
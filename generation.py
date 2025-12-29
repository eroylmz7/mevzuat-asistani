import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vectordb, chat_history):
    """
    Soruyu, belgeleri ve GEÇMİŞ KONUŞMALARI kullanarak cevap üretir.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # 1. Geçmiş Konuşmaları Metne Çevir (Son 3 mesajı al)
    # Bu kısım sayesinde "Ona başvurabilir miyim?" dediğinde "O"nun ne olduğunu anlar.
    history_text = ""
    if chat_history:
        # Son 4 mesajı al (Çok eskiye gidip kafasını karıştırmasın)
        for msg in chat_history[-4:]:
            role = "ÖĞRENCİ" if msg["role"] == "user" else "ASİSTAN"
            history_text += f"{role}: {msg['content']}\n"
    
    # 2. Retriever Ayarı (k=50 ile geniş tarama)
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})

    # 3. LLM Ayarı (Gemini 2.5 Flash - Hız ve Zeka)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1, 
        transport="rest"
    )

    # 4. GÜÇLENDİRİLMİŞ HAFIZALI PROMPT
    # history_text değişkenini f-string ile doğrudan prompt'un içine gömüyoruz.
    template = f"""
    Sen üniversite mevzuatları konusunda uzman bir asistansın.
    
    GÖREVİN:
    Aşağıdaki "Sohbet Geçmişi"ni ve "Doküman Bağlamı"nı kullanarak öğrencinin son sorusunu cevapla.
    
    ÖNEMLİ KURALLAR:
    1. HAFIZA KULLANIMI: Öğrenci "buna", "o zaman", "bu durumda" gibi atıflar yaparsa, neyi kastettiğini "Sohbet Geçmişi"nden anla.
    2. SADAKAT: Cevabı sadece verilen bağlama göre üret. Uydurma yapma.
    3. NETLİK: Cevabın kısa, net ve öğrenciye yardımcı olacak şekilde olsun.
    
    --- SOHBET GEÇMİŞİ (Dikkat et) ---
    {history_text}
    ----------------------------------

    --- DOKÜMAN BAĞLAMI ---
    {{context}}
    -----------------------

    SON SORU: {{question}}

    CEVAP:
    """
    
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    # 5. Zinciri Oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # 6. Yanıtı Üret
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        
        # Kaynakları düzenle
        sources = []
        for doc in result["source_documents"]:
            source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
            page_num = doc.metadata.get("page", 0) + 1
            source_info = f"{source_name} (Sayfa {page_num})"
            if source_info not in sources:
                sources.append(source_info)
        
        return {"answer": answer, "sources": sources[:5]}
        
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
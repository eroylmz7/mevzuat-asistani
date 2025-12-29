import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vectordb, chat_history):
    """
    Kullanıcı sorusunu alır, k=50 ile geniş tarama yapar ve 
    detaylı uzman promptu ile yanıt döner.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # 1. Geniş Tarama Ayarı (k=50)
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})

    # 2. LLM Ayarı (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        transport="rest"
    )

    # 3. GÜÇLENDİRİLMİŞ UZMAN PROMPT
    template = """
    Sen üniversite mevzuatları ve yönetmelikleri konusunda uzman bir hukuk danışmanısın.
    Aşağıdaki bağlamı (context) ve varsa önceki konuşmaları temel alarak soruyu cevapla.
    
    TEMEL KURALLAR:
    1. SADAKAT: Sadece sana verilen metne sadık kal. Metinde olmayan bir kuralı asla uydurma.
    2. BULAMAMA DURUMU: Eğer cevap metinde kesin olarak geçmiyorsa, "İlgili dökümanlarda bu konu hakkında bir bilgiye ulaşılamamıştır." de.
    3. HESAPLAMA: Eğer metinde sayılar veya süreler varsa (örn: 5 iş günü, 240 AKTS), bunları kullanıcıya net bir şekilde belirt.
    4. ÜSLUP: Resmi, yardımcı ve net bir dil kullan.

    Bağlam (Doküman Parçaları):
    {context}

    Soru: {question}

    Cevap: """
    
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    # 4. QA Chain Oluştur (Kaynakları döndürmesi için return_source_documents=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # 5. Yanıtı Al
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        
        # Kaynakları şık bir listeye dönüştür
        sources = []
        for doc in result["source_documents"]:
            source_name = os.path.basename(doc.metadata.get("source", "Bilinmeyen Belge"))
            page_num = doc.metadata.get("page", 0) + 1
            source_info = f"{source_name} (Sayfa {page_num})"
            if source_info not in sources:
                sources.append(source_info)
        
        # Hem yanıtı hem de benzersiz kaynak listesini dön
        return {"answer": answer, "sources": sources[:5]} # En alakalı 5 kaynağı göster
        
    except Exception as e:
        return {"answer": f"Bir hata oluştu: {str(e)}", "sources": []}
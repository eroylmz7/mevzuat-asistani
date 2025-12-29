import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vectordb, chat_history):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1, # Daha tutarlı olması için düşürdük
        transport="rest"
    )

    # DÜZELTİLMİŞ PROMPT
    template = """
    Sen üniversite mevzuat asistanısın. Verilen bağlamı kullanarak soruyu cevapla.
    
    KURALLAR:
    1. Sadece sorulan soruya odaklan. Konuyla ilgisi olmayan "Özel Durumlar" veya ekstra maddeleri listeleme.
    2. Cevabı mümkün olduğunca kısa ve net tut.
    3. Eğer sorunun cevabı metinde yoksa "Bilgi bulunamadı" de.
    4. Sayılar ve süreler varsa (gün, kredi vb.) net belirt.

    Bağlam: {context}
    Soru: {question}
    Cevap: """
    
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        
        # Kaynakları düzenle (Artık orijinal dosya adı gelecek)
        sources = []
        for doc in result["source_documents"]:
            source_name = doc.metadata.get("source", "Bilinmiyor")
            page_num = doc.metadata.get("page", 0) + 1
            source_info = f"{source_name} (Sayfa {page_num})"
            if source_info not in sources:
                sources.append(source_info)
        
        return {"answer": answer, "sources": sources[:4]} # Max 4 kaynak göster
    except Exception as e:
        return {"answer": f"Hata: {str(e)}", "sources": []}
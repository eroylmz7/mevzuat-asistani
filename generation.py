import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vectordb, chat_history):
    """
    Kullanıcı sorusunu alır, vektör tabanında arar ve Gemini ile yanıt döner.
    """
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # 1. Retriever Ayarı (Vektör tabanı üzerinden arama yap)
    retriever = vectordb.as_retriever(search_kwargs={"k": 60})

    # 2. LLM Ayarı
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Kota dostu model
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        transport="rest"
    )

    # 3. Prompt (Sistem İstemi)
    template = """
    Sen üniversite mevzuatları konusunda uzman bir asistansın. 
    Önceki konuşmaları ve aşağıdaki bağlamı kullanarak cevap ver.
    
    Bağlam: {context}
    Soru: {question}
    Cevap: """
    
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    # 4. QA Chain Oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # 5. Yanıtı Al
    result = qa_chain.invoke({"query": question})
    return result["result"]
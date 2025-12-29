import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def generate_answer(question, vector_store, chat_history):
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    
    # Geçmiş Sohbeti Hazırla
    history_text = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "ÖĞRENCİ" if msg["role"] == "user" else "ASİSTAN"
            history_text += f"{role}: {msg['content']}\n"
    
    # Retriever (Buluttan Arama)
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        transport="rest"
    )

    # Prompt
    template = f"""
    Sen üniversite mevzuatları konusunda uzman profesyonel bir asistansın.
    
    SOHBET GEÇMİŞİ:
    {history_text}
    
    MEVZUAT BİLGİSİ:
    {{context}}
    
    SORU: {{question}}
    
    KURALLAR:
    1. Sadece MEVZUAT BİLGİSİ'ne dayanarak cevap ver.
    2. Bilgi yoksa "Dokümanlarda bu bilgi bulunamadı" de.
    3. Sohbet geçmişini dikkate al.
    
    CEVAP:
    """
    
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
        
        sources = []
        for doc in result["source_documents"]:
            source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
            page_num = doc.metadata.get("page", 0) + 1
            source_info = f"{source_name} (Sayfa {page_num})"
            if source_info not in sources:
                sources.append(source_info)
        
        return {"answer": answer, "sources": sources[:5]}
        
    except Exception as e:
        return {"answer": f"Hata: {str(e)}", "sources": []}
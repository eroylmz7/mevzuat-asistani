import os
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from supabase import create_client
from pinecone import Pinecone
import io

# --- 1. GEMINI AYARLARI ---
def configure_gemini():
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("Google API Key bulunamadı!")

# --- 2. DEDEKTİF (İÇERİK ANALİZİ) ---
def analyze_pdf_complexity(file_path):
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        pages_to_check = min(len(doc), 3)
        for i in range(pages_to_check):
            page = doc[i]
            
            drawings = page.get_drawings()
            if len(drawings) > 20:
                return True, f"Sayfa {i+1}'de Yoğun Tablo ({len(drawings)} çizgi)"
            
            text = page.get_text().lower()
            if len(text) > 50:
                turkish_anchors = [" ve ", " bir ", " ile ", " için ", " bu ", " madde ", " üniversite "]
                match_count = sum(1 for word in turkish_anchors if word in text)
                if match_count == 0:
                    return True, f"Sayfa {i+1}'de Bozuk Metin/Encoding Hatası"
                    
        return False, "Düz Metin"
    except Exception as e:
        print(f"Analiz Hatası: {e}")
        return True, "Analiz Edilemedi (Güvenli Mod)"

# --- 3. VISION OKUMA (FİLTRELER KAPALI 🔥) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    target_model = 'gemini-2.5-flash'
    extracted_text = ""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    # 🔥 GÜVENLİK AYARLARI (BLOCK_NONE)
    # Bu ayarlar Gemini'nin "Bu içerik güvensiz olabilir" deyip susmasını engeller.
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    for page_num, page in enumerate(doc):
        if page_num == 0:
            st.toast(f"🚀 {target_model} ile tarama başladı... Sayfa 1/{total_pages}", icon="🤖")
            
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        try:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()

            model = genai.GenerativeModel(target_model)
            
            # safety_settings parametresini buraya ekledik!
            response = model.generate_content(
                [
                    """
                    GÖREV: Bu görseldeki belgeyi analiz et.
                    1. Tablo yapısını Markdown formatında koru.
                    2. Türkçe karakterleri düzelt.
                    3. Sadece metni ver.
                    """, 
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ],
                safety_settings=safety_settings  # 🔥 FİLTRELERİ KAPATAN KOD
            )
            
            if response.text:
                extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
            else:
                # Eğer yine de boş dönerse (çok nadir), hata verme, eski usül oku.
                extracted_text += page.get_text()
                
        except Exception as e:
            # Hata olsa bile kullanıcıyı korkutma, sessizce yedeğe geç.
            print(f"Gemini Vision Hatası (Sayfa {page_num+1}): {e}")
            extracted_text += page.get_text()
            
    return extracted_text

# --- 4. ANA İŞLEME FONKSİYONU ---
def process_pdfs(uploaded_files, use_vision_mode=False):
    try:
        supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception as e:
        st.error(f"Supabase hatası: {e}")
        return None
    
    all_documents = []
    
    if not os.path.exists("temp_pdfs"): os.makedirs("temp_pdfs")
        
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name, file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            except: pass

            is_complex, reason = analyze_pdf_complexity(file_path)
            force_vision = "tezyayin" in uploaded_file.name.lower()
            should_use_vision = use_vision_mode or is_complex or force_vision
            
            full_text = ""
            if should_use_vision:
                st.toast(f"Mod: Vision | Dosya: {uploaded_file.name}", icon="👁️")
                full_text = pdf_image_to_text_with_gemini(file_path)
            else:
                doc = fitz.open(file_path)
                for page in doc: full_text += page.get_text()

            # Boş içerik kontrolü (Hata durumunda)
            if not full_text.strip():
                 # Vision başarısız olduysa son çare olarak PyMuPDF ile tekrar dene
                 doc = fitz.open(file_path)
                 for page in doc: full_text += page.get_text()

            header_text = full_text[:300].replace("\n", " ").strip() if full_text else "Başlıksız"
            unified_doc = Document(
                page_content=f"BELGE KİMLİĞİ: {header_text}\nKAYNAK DOSYA: {uploaded_file.name}\n---\n{full_text}",
                metadata={"source": uploaded_file.name}
            )
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n|", "\nMADDE", "\n###", "\n\n", "\n", ". ", " "]
            )
            split_docs = text_splitter.split_documents([unified_doc])
            
            # Pinecone Boyut Kontrolü (Hata Önleyici)
            safe_docs = []
            for doc in split_docs:
                text_size = len(doc.page_content.encode('utf-8'))
                if text_size < 35000:
                    safe_docs.append(doc)
                else:
                    doc.page_content = doc.page_content[:15000] + "\n...(Kısaltıldı)"
                    safe_docs.append(doc)
            
            all_documents.extend(safe_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            try:
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except: pass
            
        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

    if all_documents:
        # 🔥 ÖNEMLİ: sentence-transformers sürümüne dikkat (CPU Modu)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            index_name="mevzuat-asistani"
        )
        return vector_store
    
    return None

# --- DİĞER FONKSİYONLAR ---
def delete_document_cloud(file_name):
    try:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        index_name = "mevzuat-asistani"
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        index.delete(filter={"source": file_name})
        try:
            supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
            supabase.table("dokumanlar").delete().eq("dosya_adi", file_name).execute()
            supabase.storage.from_("belgeler").remove([file_name])
        except Exception as e: print(f"Supabase silme hatası: {e}")
        return True, f"{file_name} başarıyla silindi."
    except Exception as e: return False, f"Hata: {str(e)}"

def connect_to_existing_index():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="mevzuat-asistani",
            embedding=embedding_model
        )
        return vector_store
    except Exception as e: return None
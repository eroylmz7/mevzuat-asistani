import os
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from supabase import create_client
from pinecone import Pinecone
import io # 🔥 EKLENDİ: Hafızada resim işlemi için gerekli

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
        
        # İlk 3 sayfaya bak
        pages_to_check = min(len(doc), 3)
        for i in range(pages_to_check):
            page = doc[i]
            
            # Tablo Çizgisi Kontrolü (Eşik: 20)
            drawings = page.get_drawings()
            if len(drawings) > 20:
                return True, f"Sayfa {i+1}'de Yoğun Tablo ({len(drawings)} çizgi)"
            
            # Dil/Encoding Kontrolü
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

# --- 3. VISION OKUMA (LIBRARY BUG FIX SÜRÜMÜ) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    
    # 🔥 SENİN İSTEDİĞİN MODEL
    target_model = 'gemini-2.5-flash'
    
    extracted_text = ""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    for page_num, page in enumerate(doc):
        if page_num == 0:
            st.toast(f"🚀 {target_model} ile tarama başladı... Sayfa 1/{total_pages}", icon="🤖")
            
        # Resmi al (Zoom=2)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        try:
            # 🔥 KRİTİK DÜZELTME: PIL Objesi yerine RAW BYTES gönderiyoruz.
            # Bu işlem 'PngImagePlugin' hatasını atlatır.
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()

            model = genai.GenerativeModel(target_model)
            
            response = model.generate_content([
                """
                GÖREV: Bu görseldeki belgeyi analiz et.
                1. Tablo yapısını Markdown olarak koru.
                2. Türkçe karakterleri düzelt.
                3. Sadece metni ver, yorum yapma.
                """, 
                {"mime_type": "image/jpeg", "data": image_bytes} # PIL yerine sözlük formatı
            ])
            
            if response.text:
                extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
            else:
                st.warning(f"⚠️ Sayfa {page_num + 1}: Model boş cevap döndü.")
                extracted_text += page.get_text()
                
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ GEMINI 2.5 HATASI (Sayfa {page_num + 1}): {error_msg}")
            # Hata durumunda yedeğe geç
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
            # 1. Kaydet
            uploaded_file.seek(0)
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Storage
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name, file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            except: pass

            # --- KARAR ANI ---
            is_complex, reason = analyze_pdf_complexity(file_path)
            
            # Zorunlu Vision
            force_vision = "tezyayin" in uploaded_file.name.lower()
            should_use_vision = use_vision_mode or is_complex or force_vision
            
            full_text = ""
            
            if should_use_vision:
                st.toast(f"Mod: Vision ({target_model}) | Dosya: {uploaded_file.name}\nSebep: {reason}", icon="👁️")
                full_text = pdf_image_to_text_with_gemini(file_path)
                
                # İçerik Kontrolü
                if len(full_text) < 100:
                    st.error(f"⚠️ UYARI: {uploaded_file.name} tarandı ama içerik çok kısa! (Hata oluşmuş olabilir)")
            else:
                doc = fitz.open(file_path)
                for page in doc: full_text += page.get_text()

            # --- BELGE OLUŞTURMA ---
            header_text = full_text[:300].replace("\n", " ").strip() if full_text else "Başlıksız"
            unified_doc = Document(
                page_content=f"BELGE KİMLİĞİ: {header_text}\nKAYNAK DOSYA: {uploaded_file.name}\n---\n{full_text}",
                metadata={"source": uploaded_file.name}
            )
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=300,
                separators=["\n|", "\nMADDE", "\n###", "\n\n", ". "]
            )
            split_docs = text_splitter.split_documents([unified_doc])
            all_documents.extend(split_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            # DB Güncelleme
            try:
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except: pass
            
        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

    # --- PINECONE ---
    if all_documents:
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

# --- DİĞER FONKSİYONLAR AYNI ---
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
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

# --- 2. GARANTİCİ DEDEKTİF 🕵️‍♂️ ---
def analyze_pdf_complexity(file_path):
    """
    1. Akademik kelime arar.
    2. Metin okunabiliyor mu diye bakar. Okunmuyorsa Vision açar.
    """
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        pages_to_check = min(len(doc), 3)
        total_text_len = 0
        all_text = ""

        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text().lower()
            total_text_len += len(text)
            all_text += text
            
            # --- KURAL 1: PARMAK İZİ KELİMELER ---
            academic_keywords = [
                "q1", "q2", "q3", "ssci", "sci-exp", "ahci", "scopus", 
                "yöksis", "doi", "çeyreklik", "quartile", "impact factor",
                "doktora", "yüksek lisans", "yayın şartı", "akts", "kredi"
            ]
            
            found = [kw for kw in academic_keywords if kw in text]
            if found:
                return True, f"Kritik Kelime Bulundu: '{found[0]}'"

        # --- KURAL 2: METİN YOKSA VEYA AZSA (TARANMIŞ PDF) ---
        # 3 sayfada toplam 200 karakterden az yazı varsa, bu dosya resimdir.
        if total_text_len < 200:
            return True, "Metin Okunamadı (Resim PDF)"

        # --- KURAL 3: BOZUK KARAKTER KONTROLÜ ---
        # Metin var ama 've', 'bir', 'ile' gibi kelimeler yoksa encoding bozuktur.
        turkish_anchors = [" ve ", " bir ", " ile ", " için ", " bu ", " veya "]
        hits = sum(1 for w in turkish_anchors if w in all_text)
        
        # Eğer yeterince yazı var ama hiç bağlaç yoksa, karakterler bozuktur.
        if total_text_len > 500 and hits == 0:
            return True, "Bozuk Metin (OCR Gerekli)"

        return False, "Standart Metin"
        
    except Exception as e:
        print(f"Analiz Hatası: {e}")
        return True, "Güvenli Mod (Hata)"

# --- 3. VISION OKUMA (HİYERARŞİ VE HATA KORUMALI) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    target_model = 'gemini-2.5-flash'
    extracted_text = ""
    doc = fitz.open(file_path)
    
    st.toast(f"👁️ Vision Aktif: {os.path.basename(file_path)}", icon="📸")
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        try:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()

            model = genai.GenerativeModel(target_model)
            
            # PROMPT
            prompt = """
            GÖREV: Bu belgeyi analiz et.
            1. Tablodaki her satıra "DOKTORA" veya "YÜKSEK LİSANS" başlığını ekle.
            2. Dipnotları ilgili maddeyle birleştir.
            3. Markdown tablosu olarak ver.
            """
            
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                safety_settings=safety_settings
            )
            
            try:
                if hasattr(response, 'text') and response.text:
                    extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
                else:
                    raise ValueError("Boş Cevap")
            except Exception:
                print(f"Sayfa {page_num+1} Vision okuyamadı, standart moda geçildi.")
                extracted_text += page.get_text()

        except Exception as e:
            print(f"Vision API Hatası: {e}")
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
            
            # --- DEDEKTİF KARARI ---
            is_complex, reason = analyze_pdf_complexity(file_path)
            
            if is_complex:
                st.warning(f"🟠 Vision Modu: {uploaded_file.name}\nSebep: {reason}")
            else:
                st.success(f"🟢 Hızlı Mod: {uploaded_file.name}\nSebep: Standart Metin")
            
            should_use_vision = use_vision_mode or is_complex
            
            full_text = ""
            if should_use_vision:
                full_text = pdf_image_to_text_with_gemini(file_path)
            else:
                doc = fitz.open(file_path)
                for page in doc: full_text += page.get_text()

            # Güvenlik Ağı
            if not full_text.strip():
                 doc = fitz.open(file_path)
                 for page in doc: full_text += page.get_text()

            header_text = full_text[:300].replace("\n", " ").strip() if full_text else "Başlıksız"
            unified_doc = Document(
                page_content=f"BELGE KİMLİĞİ: {header_text}\nKAYNAK DOSYA: {uploaded_file.name}\n---\n{full_text}",
                metadata={"source": uploaded_file.name}
            )
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, 
                chunk_overlap=300,
                separators=["\n|", "\nMADDE", "\n###", "\n\n", ". "]
            )
            split_docs = text_splitter.split_documents([unified_doc])
            
            safe_docs = []
            for doc in split_docs:
                text_size = len(doc.page_content.encode('utf-8'))
                if text_size < 38000:
                    safe_docs.append(doc)
                else:
                    doc.page_content = doc.page_content[:15000] + "\n...(Kısaltıldı)"
                    safe_docs.append(doc)
            
            all_documents.extend(safe_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name, file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except: pass
            
        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

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
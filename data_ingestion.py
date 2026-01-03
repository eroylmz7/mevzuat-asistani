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

# --- 2. YAPISAL MÜHENDİS DEDEKTİF (STRUKTÜREL ANALİZ) 🕵️‍♂️ ---
def analyze_pdf_complexity(file_path):
    """
    Kelimelere bakmaz. Belgenin 'İskelet Yapısını' analiz eder.
    1. Blok Sayısı (Tablolarda çok yüksektir).
    2. Yatay/Dikey Çizgi Sayısı (Tablolarda ızgara oluşturur).
    """
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        pages_to_check = min(len(doc), 3)
        
        for i in range(pages_to_check):
            page = doc[i]
            
            # --- KRİTER 1: METİN BLOK YOĞUNLUĞU ---
            # Standart metinlerde paragraflar birleşiktir (Az blok).
            # Tablolarda her hücre ayrı bir metin bloğudur (Çok blok).
            text_blocks = page.get_text("blocks")
            block_count = len(text_blocks)
            
            # Eşik Değer: Bir sayfada 40'tan fazla ayrı metin parçası varsa, 
            # bu %99 ihtimalle karmaşık bir tablodur. (Yönetmeliklerde genelde 10-15 olur).
            if block_count > 40:
                return True, f"Yüksek Parçalanma Tespit Edildi ({block_count} metin bloğu)"

            # --- KRİTER 2: IZGARA (GRID) ANALİZİ ---
            # Sadece çizgi saymak yetmez, yönlerine bakacağız.
            drawings = page.get_drawings()
            horizontal_lines = 0
            vertical_lines = 0
            
            for d in drawings:
                # 'rect' (kutu) veya 'line' (çizgi) olabilir.
                # Çizginin boyuna bakarak "süs" mü "yapı" mı olduğunu anlarız.
                rect = d['rect']
                width = rect.width
                height = rect.height
                
                # Yatay Çizgi: Genişliği yüksek, yüksekliği az
                if width > 100 and height < 5:
                    horizontal_lines += 1
                
                # Dikey Çizgi: Yüksekliği fazla, genişliği az
                if height > 50 and width < 5:
                    vertical_lines += 1
            
            # KARAR ANI:
            # Yönetmelik Çerçevesi: 2 Yatay + 2 Dikey çizgi olur.
            # Gerçek Tablo: Satır sayısı kadar Yatay (>5), Sütun sayısı kadar Dikey (>2) olur.
            if horizontal_lines > 5 and vertical_lines > 2:
                return True, f"Tablo Izgarası Tespit Edildi ({horizontal_lines} Yatay, {vertical_lines} Dikey Çizgi)"
        
        # Eğer yukarıdaki şartları sağlamıyorsa, çerçeveli bile olsa standart metindir.
        return False, "Standart Yapı (Izgara veya Parçalanma Yok)"
        
    except Exception as e:
        print(f"Analiz Hatası: {e}")
        return False, "Analiz Hatası -> Standart Mod"

# --- 3. VISION OKUMA (SESSİZ VE GÜÇLÜ) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    target_model = 'gemini-2.5-flash'
    extracted_text = ""
    doc = fitz.open(file_path)
    
    st.toast(f"👁️ VISION MODU: {os.path.basename(file_path)}", icon="📸")
    
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
            
            # Prompt: Yapısal bütünlüğü koru
            prompt = """
            GÖREV: Bu belgeyi analiz et.
            1. Eğer sayfada TABLO varsa, tabloyu bozmadan Markdown formatına çevir.
            2. Tablodaki her satırın başına, o satırın ait olduğu ana başlığı (Örn: "DOKTORA") ekle.
            3. Dipnotları metinle birleştir.
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
            print(f"API Hatası: {e}")
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
            
            # EKRAN BİLDİRİMLERİ (DEBUG)
            if is_complex:
                st.warning(f"🟠 Vision Modu: {uploaded_file.name}\nSebep: {reason}")
            else:
                st.success(f"🟢 Hızlı Mod: {uploaded_file.name}\nSebep: {reason}")
            
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
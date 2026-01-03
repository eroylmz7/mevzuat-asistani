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

# --- 1. GEMINI AYARLARI ---
def configure_gemini():
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("Google API Key bulunamadı!")

# --- 2. DEDEKTİF: TABLO YOĞUNLUĞU ANALİZİ ---
def is_pdf_table_heavy(file_path):
    """
    PDF'in içindeki vektör çizimlerini (çizgileri/kutuları) sayar.
    Eğer bir sayfada çok fazla çizgi varsa (Eşik: 15), orası yoğun bir tablodur.
    """
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False
        
        # İlk 3 sayfayı analiz etsek yeter (Genelde format bellidir)
        pages_to_check = min(len(doc), 3)
        
        for i in range(pages_to_check):
            page = doc[i]
            # Sayfadaki tüm çizim yollarını (border, line, rect) al
            drawings = page.get_drawings()
            
            # Eşik Değeri: 15 çizgi. Düz metinlerde genelde 1-2 çizgi olur.
            if len(drawings) > 15:
                print(f"Dedektif: {os.path.basename(file_path)} (Sayfa {i+1}) yoğun tablo yapısı içeriyor. ({len(drawings)} çizgi)")
                return True
                
        return False
    except Exception as e:
        print(f"Analiz hatası: {e}")
        return False 

# --- 3. VISION OKUMA (GEMINI 2.5 FLASH) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    # 🔥 GEMINI 2.5 FLASH KULLANIYORUZ
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    extracted_text = ""
    doc = fitz.open(file_path)
    
    for page_num, page in enumerate(doc):
        # Zoom=2 ile yüksek kalite resim al (OCR başarısı için önemli)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        try:
            response = model.generate_content([
                """
                GÖREV: Bu görseldeki belgeyi analiz et ve metne dönüştür.
                ÖNEMLİ KURALLAR:
                1. Bu belgede TABLOLAR var. Tablo yapısını Markdown formatında koruyarak aktar.
                2. Satır ve sütunların karışmasını engelle.
                3. Türkçe karakter hatalarını (varsa) düzelt.
                4. Sadece metni ver, yorum yapma.
                """, 
                img
            ])
            extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
        except Exception as e:
            print(f"Vision hatası (Sayfa {page_num}): {e}")
            extracted_text += page.get_text() # Hata olursa yedeğe dön
            
    return extracted_text

# --- 4. ANA İŞLEME FONKSİYONU ---
def process_pdfs(uploaded_files, use_vision_mode=False):
    try:
        supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception as e:
        st.error(f"Supabase bağlantı hatası: {e}")
        return None
    
    all_documents = []
    
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
        
    for uploaded_file in uploaded_files:
        try:
            # --- A. STORAGE YÜKLEME ---
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name,
                    file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            except Exception as e:
                print(f"Storage uyarısı: {e}")

            # --- B. GEÇİCİ DOSYA KAYDETME ---
            uploaded_file.seek(0)
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- C. KARAR ANI: NORMAL Mİ, VISION MI? ---
            # 1. Kullanıcı elle seçti mi? (use_vision_mode)
            # 2. Dedektif "Tablo var" dedi mi? (detected_table)
            detected_table = is_pdf_table_heavy(file_path)
            should_use_vision = use_vision_mode or detected_table
            
            full_text = ""
            
            if should_use_vision:
                reason = "Kullanıcı Seçimi" if use_vision_mode else "Yoğun Tablo Algılandı"
                st.toast(f"🤖 Yapay Zeka Gözü Devrede: {uploaded_file.name} ({reason})", icon="👁️")
                # Gemini 2.5 ile görerek oku
                full_text = pdf_image_to_text_with_gemini(file_path)
            else:
                # Standart Hızlı Okuma (PyMuPDF - fitz)
                doc = fitz.open(file_path)
                for page in doc: 
                    full_text += page.get_text()

            # --- D. BELGE OLUŞTURMA (BELGE KİMLİĞİ MANTIĞI) ---
            # İlk 300 karakteri başlık olarak al (Eski kodundaki mantık)
            header_text = full_text[:300].replace("\n", " ").strip() if full_text else "Başlıksız Belge"
            
            # Tek bir büyük belge oluşturuyoruz
            unified_doc = Document(
                page_content=f"BELGE KİMLİĞİ: {header_text}\n---\n{full_text}",
                metadata={"source": uploaded_file.name}
            )
            
            # --- E. PARÇALAMA (SPLITTING) ---
            # 1000/500 stratejisi + Markdown tablo ayracı (|) eklendi
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,      
                chunk_overlap=500,
                separators=["\n|", "\nMADDE ", "\nMadde ", "\nGEÇİCİ MADDE", "\n\n", "\n", ". ", " ", ""]
            )
            
            split_docs = text_splitter.split_documents([unified_doc])
            all_documents.extend(split_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            # Supabase Tablo Güncelleme
            try:
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except: pass
            
        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

    # --- 5. PINECONE ---
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

# --- SİLME VE BAĞLANTI (ESKİ KODUN AYNISI) ---
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
    except Exception as e:
        return False, f"Hata: {str(e)}"

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
    except Exception as e:
        st.error(f"Otomatik bağlantı hatası: {e}")
        return None
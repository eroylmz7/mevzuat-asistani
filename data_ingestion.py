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

# --- 2. AKILLI DEDEKTİF (KONUŞKAN VERSİYON 🗣️) ---
def analyze_pdf_complexity(file_path):
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        pages_to_check = min(len(doc), 3)
        complexity_score = 0
        reasons = []
        total_text_len = 0

        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text().lower()
            total_text_len += len(text)
            
            # 1. AKADEMİK KELİMELER (Yüksek Puan)
            high_priority = ["q1", "q2", "ssci", "sci", "doi", "yöksis", "çeyreklik", "enstitü", "anabilim", "tez", "yayın"]
            hits = [kw for kw in high_priority if kw in text]
            if hits:
                complexity_score += 5
                reasons.append(f"Kritik Kelimeler: {', '.join(hits[:3])}")

            # 2. YAPISAL KELİMELER
            medium_priority = ["tablo", "kriter", "şart", "koşul", "madde"]
            if any(kw in text for kw in medium_priority):
                complexity_score += 1

            # 3. ÇİZGİLER
            if len(page.get_drawings()) > 5:
                complexity_score += 1
                reasons.append("Tablo Çizgileri")

        # 🔥 KRİTİK KONTROL: Eğer hiç yazı yoksa (Scanned PDF), KESİN VISION AÇ!
        if total_text_len < 100:
            return True, "Metin Bulunamadı (Taranmış PDF)"

        # Skor 1 bile olsa aç (Çok hassas yaptık)
        if complexity_score >= 1:
            return True, f"Tespit Edildi (Skor: {complexity_score}, Sebepler: {reasons})"
            
        return False, "Düz Metin"
        
    except Exception as e:
        return True, f"Analiz Hatası: {str(e)}"

# --- 3. VISION OKUMA (HATA GÖSTEREN VERSİYON 🚨) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    target_model = 'gemini-2.5-flash' # Eğer bu hata verirse 1.5-flash dene
    extracted_text = ""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    # EKRANA BİLGİ BAS
    st.info(f"👀 Vision Modu Başladı! Model: {target_model} | Sayfa Sayısı: {total_pages}")
    
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
            
            prompt = """
            GÖREV: Bu akademik tabloyu VERİTABANI İÇİN oku.
            
            KURALLAR:
            1. Tablonun ANA BAŞLIĞINI (Örn: "DOKTORA") her satırın başına ekle.
               Örnek Çıktı: "DOKTORA MEZUNİYET ŞARTI: Q1 yayın gerekir."
            2. "VEYA" bağlaçlarını açıkla (Biri yeterlidir de).
            3. Dipnotları (yıldızlı yazılar) ilgili maddeye ekle.
            4. Markdown tablosu olarak ver.
            """
            
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                safety_settings=safety_settings
            )
            
            if response.text:
                extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
                # Başarılı olursa yeşil tik at
                # st.success(f"Sayfa {page_num+1} Okundu ✅") 
            else:
                st.warning(f"⚠️ Sayfa {page_num+1}: Model BOŞ cevap döndü!")
                extracted_text += page.get_text()
                
        except Exception as e:
            # 🔥 HATAYI GİZLEME, GÖSTER!
            st.error(f"❌ GEMINI HATASI (Sayfa {page_num+1}): {e}")
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
            
            # DEDEKTİF KARARI
            is_complex, reason = analyze_pdf_complexity(file_path)
            
            # EKRANA KARARI YAZDIR (DEBUG)
            if is_complex:
                st.warning(f"🕵️‍♂️ DEDEKTİF KARARI: Vision AÇIK\nDosya: {uploaded_file.name}\nSebep: {reason}")
            else:
                st.info(f"ℹ️ DEDEKTİF KARARI: Standart Mod\nDosya: {uploaded_file.name}\nSebep: Basit Metin")
            
            should_use_vision = use_vision_mode or is_complex
            
            full_text = ""
            if should_use_vision:
                full_text = pdf_image_to_text_with_gemini(file_path)
            else:
                doc = fitz.open(file_path)
                for page in doc: full_text += page.get_text()

            if not full_text.strip():
                 st.error("⚠️ UYARI: Belge içeriği boş çıkarıldı! (Yedek mod çalıştı)")
                 doc = fitz.open(file_path)
                 for page in doc: full_text += page.get_text()

            # Belge oluşturma
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
            
            # Pinecone Boyut Kontrolü
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
            
            # DB Temizliği
            try:
                # Storage yükleme (Daha önce yapılmadıysa)
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
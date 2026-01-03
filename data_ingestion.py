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

# --- 2. AKILLI DEDEKTİF (AYNI KALIYOR) ---
def analyze_pdf_complexity(file_path):
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        pages_to_check = min(len(doc), 3)
        complexity_score = 0
        reasons = []

        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text().lower()
            
            # PARMAK İZİ KELİMELER
            high_priority_keywords = [
                "q1", "q2", "q3", "ssci", "sci-exp", "ahci", "scopus", 
                "doi", "yöksis", "quartile", "çeyreklik", "impact factor"
            ]
            
            hit_academic = sum(1 for kw in high_priority_keywords if kw in text)
            if hit_academic > 0:
                return True, f"Akademik Tablo Terimleri Bulundu (Q1/DOI vb.)"

            # YAPISAL KELİMELER
            medium_priority_keywords = ["tablo", "kriter", "koşul", "şart", "sütun", "satır"]
            hit_structural = sum(1 for kw in medium_priority_keywords if kw in text)
            if hit_structural > 0: complexity_score += 1

            # ÇİZGİLER
            drawings = page.get_drawings()
            if len(drawings) > 10: complexity_score += 2

            # BOZUK METİN
            turkish_anchors = [" ve ", " bir ", " ile ", " için ", " bu "]
            if len(text) > 50 and sum(1 for w in turkish_anchors if w in text) == 0:
                return True, "Bozuk Metin / Encoding Hatası"

        if complexity_score >= 3:
            return True, f"Karmaşık Yapı (Skor: {complexity_score})"
            
        return False, "Standart Metin"
    except Exception as e:
        return True, "Güvenli Mod"

# --- 3. VISION OKUMA (HİYERARŞİ ODAKLI YENİ PROMPT 🔥) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    target_model = 'gemini-2.5-flash'
    extracted_text = ""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    # FİLTRELERİ KAPAT
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    for page_num, page in enumerate(doc):
        if page_num == 0:
            st.toast(f"🚀 {target_model} ile Hiyerarşik Tarama... (Sayfa 1/{total_pages})", icon="🧠")
            
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        try:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()

            model = genai.GenerativeModel(target_model)
            
            # 🔥 İŞTE SİHİRLİ PROMPT BURADA 🔥
            prompt = """
            GÖREV: Bu akademik belgeyi, özellikle tabloları, VERİTABANI İÇİN HAZIRLA.
            
            ÇOK ÖNEMLİ KURALLAR (Bunu uygulamazsan veri kaybolur):
            
            1. **HER SATIRA BAŞLIK EKLE:** Tablonun en başındaki ana başlığı (Örn: "DOKTORA" veya "YÜKSEK LİSANS") al ve tablonun içindeki HER BİR MADDENİN başına yaz.
               - Yanlış: "Q1 yayın gerekir."
               - Doğru: "DOKTORA MEZUNİYET ŞARTI: Q1 yayın gerekir."
               
            2. **"VEYA" MANTIĞINI AÇIKLA:** Eğer bir maddede "yayınlanmış VEYA DOI alınmış" diyorsa, bunu açıkça belirt:
               - "Makalenin basılmış olması ŞART DEĞİLDİR, sadece DOI numarası alınmış (yayına kabul edilmiş) olması da YETERLİDİR." şeklinde yorum ekle.
            
            3. **SANAT DALLARINI AYRIŞTIR:** Eğer "Resim", "Müzik" gibi alt dallar varsa, bunları mutlaka üst başlıkla birleştir:
               - "DOKTORA - SANATTA YETERLİK - RESİM ANASANAT DALI için sergi şartı şudur..."
               
            4. **DİPNOTLARI İLİŞKİLENDİR:** Tablo altındaki yıldızlı (*) notları ilgili maddenin altına ekle.
            
            5. Çıktıyı düzgün Türkçe cümleler ve Markdown maddeleri olarak ver.
            """
            
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                safety_settings=safety_settings
            )
            
            if response.text:
                extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
            else:
                extracted_text += page.get_text()
                
        except Exception as e:
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
            
            # Storage
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name, file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            except: pass

            # --- DEDEKTİF ---
            is_complex, reason = analyze_pdf_complexity(file_path)
            should_use_vision = use_vision_mode or is_complex
            
            full_text = ""
            if should_use_vision:
                st.toast(f"Mod: Vision | Dosya: {uploaded_file.name}\nTespit: {reason}", icon="👁️")
                full_text = pdf_image_to_text_with_gemini(file_path)
            else:
                doc = fitz.open(file_path)
                for page in doc: full_text += page.get_text()

            if not full_text.strip():
                 doc = fitz.open(file_path)
                 for page in doc: full_text += page.get_text()

            # Belge oluşturma
            header_text = full_text[:300].replace("\n", " ").strip() if full_text else "Başlıksız"
            unified_doc = Document(
                page_content=f"BELGE KİMLİĞİ: {header_text}\nKAYNAK DOSYA: {uploaded_file.name}\n---\n{full_text}",
                metadata={"source": uploaded_file.name}
            )
            
            # CHUNK SIZE ARTTIRMA (Bağlam kopmaması için)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # 1200'den 1500'e çıkardık, bütünlük bozulmasın.
                chunk_overlap=300,
                separators=["\n|", "\n###", "\n\n", ". "]
            )
            split_docs = text_splitter.split_documents([unified_doc])
            
            # Boyut Kontrolü
            safe_docs = []
            for doc in split_docs:
                text_size = len(doc.page_content.encode('utf-8'))
                if text_size < 38000: # Pinecone limiti 40k, güvenli sınır 38k
                    safe_docs.append(doc)
                else:
                    doc.page_content = doc.page_content[:15000] + "\n...(Kısaltıldı)"
                    safe_docs.append(doc)
            
            all_documents.extend(safe_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            # DB Temizliği
            try:
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except: pass
            
        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

    # Vector Store
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

# Diğerleri aynı...
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
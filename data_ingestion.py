import time
import os
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from supabase import create_client
from pinecone import Pinecone
import io
import collections
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. GEMINI AYARLARI ---
def configure_gemini():
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("Google API Key bulunamadı!")

# --- 2. SÜTUN HİZALAMA ANALİZİ (Aynen Kalıyor) ---
def analyze_pdf_complexity(file_path):
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        pages_to_check = min(len(doc), 3)
        for i in range(pages_to_check):
            page = doc[i]
            text_dict = page.get_text("dict")
            x_starts = []
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if len(span["text"].strip()) > 5:
                                x_starts.append(round(span["bbox"][0] / 20) * 20)
            if not x_starts: return True, "Metin Bulunamadı (Resim PDF)"
            counter = collections.Counter(x_starts)
            most_common = counter.most_common()
            significant_columns = 0
            active_cols = []
            for x_pos, count in most_common:
                if count >= 15:
                    significant_columns += 1
                    active_cols.append(f"X={x_pos}")
            if significant_columns >= 3:
                return True, f"Çoklu Sütun ({significant_columns} sütun)"
            text_plain = page.get_text().lower()
            if "q1" in text_plain and "çeyreklik" in text_plain:
                return True, "Akademik Terim (Q1)"
        return False, "Standart Metin"
    except Exception as e:
        print(f"Analiz Hatası: {e}")
        return False, "Hata -> Standart"

# --- 3. DOKÜMAN TÜRÜ TESPİTİ (Aynen Kalıyor) ---
def detect_document_title(text_preview, filename):
    try:
        if "GOOGLE_API_KEY" not in st.secrets: return filename
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.0
        )
        prompt = f"""
        GÖREV: Bu resmi belgenin RESMİ BAŞLIĞINI tespit et.
        DOSYA ADI: {filename}
        METİN ÖNİZLEME:
        {text_preview[:2000]}
        
        Sadece başlığı yaz, yorum yapma.
        """
        title = llm.invoke(prompt).content.strip()
        if len(title) > 150: return filename
        return title
    except: return filename

# --- 4. VISION MODU İLE  İŞLEME ---
def process_single_page_vision(page, page_num):
    """
    Tek bir sayfayı Gemini Vision ile okur ve metni döndürür.
    """
    configure_gemini()
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()

        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # PROMPT GÜNCELLENDİ: TABLO VE NOTLAR İÇİN DAHA SIKI
        prompt = """
        Bu sayfayı Markdown formatına çevir.
        1. TABLOLARI bozmadan |...| formatında yaz.
        2. Tablo içindeki sayıları ve başlıkları (Tezsiz, Kredi, AKTS) eksiksiz al.
        3. Sayfanın altındaki dipnotları "DİPNOT:" diye belirt.
        """
        
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )
        return response.text if response.text else page.get_text()
        
    except Exception as e:
        print(f"Vision Hatası (Sayfa {page_num}): {e}")
        return page.get_text() # Hata olursa normal oku

# --- 5. ANA İŞLEME FONKSİYONU ---
def process_pdfs(uploaded_files, use_vision_mode=False):
    try:
        supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except: return None
    
    # 1. Pinecone Index Bağlantısı 
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        vector_store = PineconeVectorStore(
            index_name="mevzuat-asistani",
            embedding=embedding_model,
            pinecone_api_key=st.secrets["PINECONE_API_KEY"]
        )
    except Exception as e:
        st.error(f"Pinecone Bağlantı Hatası: {e}")
        return None

    if not os.path.exists("temp_pdfs"): os.makedirs("temp_pdfs")
    
    total_docs_to_upload = []

    for uploaded_file in uploaded_files:
        try:
            # Dosyayı kaydet
            uploaded_file.seek(0)
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())

            # Karmaşıklık Analizi
            is_complex, reason = analyze_pdf_complexity(file_path)
            should_use_vision = use_vision_mode or is_complex
            
            if should_use_vision: st.warning(f"📸 Vision: {uploaded_file.name} ({reason})")
            else: st.success(f"⚡ Hızlı: {uploaded_file.name}")

            # --- SAYFA SAYFA İŞLEME ---
            doc = fitz.open(file_path)
            file_pages_docs = [] # Bu dosyanın sayfaları
            full_text_for_title = "" # Başlık tespiti için ilk sayfaları biriktir

            status_bar = st.progress(0)
            
            for i, page in enumerate(doc):
                # İlerleme çubuğu
                status_bar.progress((i + 1) / len(doc))
                
                # Metni Çıkar
                page_text = ""
                if should_use_vision:
                    page_text = process_single_page_vision(page, i+1)
                else:
                    page_text = page.get_text()
                
                # Başlık tespiti için ilk 2 sayfanın metnini sakla
                if i < 2: full_text_for_title += page_text + "\n"

                # DOKÜMAN OLUŞTUR 
                if page_text.strip():
                    new_doc = Document(
                        page_content=page_text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": i + 1, # <-- İŞTE ÇÖZÜM: Gerçek sayfa numarası
                            "complexity": "vision" if should_use_vision else "text"
                        }
                    )
                    file_pages_docs.append(new_doc)

            # Belge Başlığını Tespit Et
            detected_title = detect_document_title(full_text_for_title, uploaded_file.name)
            st.caption(f"🏷️ Tespit Edilen Başlık: **{detected_title}**")

            # Metadata'yı Güncelle (Başlığı ekle)
            for d in file_pages_docs:
                d.metadata["official_title"] = detected_title
                # İçeriğe de başlığı ekleyelim ki aramalarda çıksın
                d.page_content = f"BELGE: {detected_title}\nSAYFA: {d.metadata['page']}\n---\n{d.page_content}"

            # --- SPLITTER (Parçalama) ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, # Senaryoya göre değiştirilebilir değerler.
                chunk_overlap=300,
                separators=["\nMADDE", "\n\n", ". ", " ", ""]
            )
            
            # Listeyi split et
            chunks = text_splitter.split_documents(file_pages_docs)
            total_docs_to_upload.extend(chunks)

            # Temizlik (Dosyayı sil)
            doc.close()
            if os.path.exists(file_path): os.remove(file_path)

            # Supabase Yedekleme 
            try:
                uploaded_file.seek(0)
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name, file=uploaded_file.read(),
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
                supabase.table("dokumanlar").upsert({"dosya_adi": uploaded_file.name}).execute()
            except: pass

        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

    # --- TOPLU PINECONE YÜKLEMESİ ---
    if total_docs_to_upload:
        try:
            st.info(f"🚀 {len(total_docs_to_upload)} parça Pinecone'a yükleniyor...")
            
            batch_size = 20
            pbar = st.progress(0)
            for i in range(0, len(total_docs_to_upload), batch_size):
                batch = total_docs_to_upload[i : i + batch_size]
                vector_store.add_documents(batch)
                pbar.progress(min((i + batch_size) / len(total_docs_to_upload), 1.0))
                time.sleep(0.5) # Rate limit yememek için minik bekleme
            
            st.success("✅ Yükleme Tamamlandı! Veritabanı güncel.")
            return vector_store
        except Exception as e:
            st.error(f"Pinecone Yükleme Hatası: {e}")
            return None
    
    return None

# --- TEMİZLEME VE DİĞER FONKSİYONLAR ---
def delete_document_cloud(file_name):
    
    try:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        index_name = "mevzuat-asistani"
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        index.delete(filter={"source": file_name})
    except: pass
    
    try:
        supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        supabase.table("dokumanlar").delete().eq("dosya_adi", file_name).execute()
        supabase.storage.from_("belgeler").remove([file_name])
        return True, "Silindi"
    except Exception as e: return False, str(e)

def connect_to_existing_index():
    
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="mevzuat-asistani",
            embedding=embedding_model
        )
        return vector_store
    except: return None
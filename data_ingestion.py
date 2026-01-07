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

# --- 1. GEMINI AYARLARI ---
def configure_gemini():
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("Google API Key bulunamadı!")

# --- 2. SÜTUN HİZALAMA ANALİZİ  ---
def analyze_pdf_complexity(file_path):
    """
    Belgedeki metinlerin sol hizalamasına (X koordinatına) bakar.
    Yönetmelik girintilerini (indentation) tablo sütunu sanmaması için
    daha akıllı bir yoğunluk kontrolü yapar..
    """
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        # İlk 3 sayfayı tara
        pages_to_check = min(len(doc), 3)
        
        for i in range(pages_to_check):
            page = doc[i]
            
            # Kelimelerin koordinatlarını al
            text_dict = page.get_text("dict")
            x_starts = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Çok kısa yazıları (Madde no, a), b) gibi) ve boşlukları atla.
                            # Çünkü bunlar "Sütun" değil, "Madde İşaretidir".
                            if len(span["text"].strip()) > 5:
                                x_starts.append(round(span["bbox"][0] / 20) * 20)
            
            # Eğer sayfada hiç anlamlı yazı yoksa (Taranmış PDF), direkt Vision.
            if not x_starts:
                return True, "Metin Bulunamadı (Resim PDF)"

            # --- ANALİZ ---
            # X koordinatlarının frekansını say.
            counter = collections.Counter(x_starts)
            
            # En sık tekrar eden hizalamaları al
            most_common_alignments = counter.most_common()
            
            # Eşik Değer: Gerçek bir sütun olması için o hizada EN AZ 15 SATIR olmalı.
            # Yönetmelikteki a) b) c) şıkları genelde 3-5 satır sürer, bu yüzden elenirler.
            # Tablolar ise sayfa boyu sürdüğü için 20-30 satır olur.
            significant_columns = 0
            active_columns = [] # Debug için
            
            for x_pos, count in most_common_alignments:
                if count >= 15: # KRİTİK EŞİK: 15 Satır
                    significant_columns += 1
                    active_columns.append(f"X={x_pos} ({count} satır)")
            
            # KARAR: 
            # 3 veya daha fazla "YOĞUN" sütun varsa VISION AÇ.
            # (Yönetmeliklerde genelde sadece 1 yoğun sütun olur: Ana Metin)
            if significant_columns >= 3:
                return True, f"Çoklu Sütun Yapısı Tespit Edildi ({significant_columns} sütun: {active_columns})"
                
            # --- YEDEK KELİME KONTROLÜ  ---
            text_plain = page.get_text().lower()
            # Sadece 'Q1' ve 'Çeyreklik' kelimeleri bir aradaysa aç (Tez Tablosu için sigorta)
            if "q1" in text_plain and "çeyreklik" in text_plain:
                return True, "Akademik Terim (Q1) Tespit Edildi"

        return False, "Standart Akış Metni"
        
    except Exception as e:
        print(f"Analiz Hatası: {e}")
        return False, "Analiz Hatası -> Standart Mod"

# --- 3. VISION OKUMA (SESSİZ VE GÜVENLİ) ---
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
            
            prompt = """
            GÖREV: Bu akademik belgeyi analiz et.
            1. Eğer sayfada TABLO varsa, tabloyu bozmadan Markdown formatına çevir.
            2. Tablodaki her satırın başına, o satırın ait olduğu ana başlığı (Örn: "DOKTORA") ekle.
            3. **KRİTİK - TABLO ALTI NOTLAR:** Tablonun hemen altında veya sayfanın en altında yer alan cümlelere DİKKAT ET.
               - Özellikle **"...karar verir"**, **"...yetkilidir"**, **"...Kurulu"** gibi ifadeler içeren cümleleri ASLA ATLAMA.
               - Bu cümleleri **"GENEL HÜKÜM: [Cümle]"** formatında metnin en başına ekle.
               
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
                # Sessizce yedeğe geç
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
                
                separators=[
                    "\nMADDE",        # Önce Maddelere göre bölmeye çalışsın (En ideali)
                    "\nGEÇİCİ MADDE", # Geçici maddeleri de yakalayalım
                    "\n###",          # Başlıklar
                    "\n\n",           # Paragraflar
                    "\n",             # Satırlar
                    ". ",             # Cümleler
                    " ",              # Kelimeler
                    ""                # Harfler (son çare)
                ]
            )
            split_docs = text_splitter.split_documents([unified_doc])
            
            # Belgeleri ana listeye ekle (Burada uyumaya gerek yok)
            all_documents.extend(split_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            # Supabase işlemleri...
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
        try:
            st.info(f"🚀 Toplam {len(all_documents)} parça Google sunucularına parça parça işleniyor...")
            
            # 1. Önce Modeli ve Vektör Store'u Hazırla (Boş Olarak)
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
            
            # Pinecone bağlantısını kur
            vector_store = PineconeVectorStore(
                index_name="mevzuat-asistani",
                embedding=embedding_model,
                pinecone_api_key=st.secrets["PINECONE_API_KEY"]
            )
            
            # 2. BATCH UPLOAD (VAGON SİSTEMİ) 
            # 100 parçayı aynı anda atmak yerine 10'ar 10'ar atıp dinleniyoruz.
            batch_size = 10
            total_batches = len(all_documents) // batch_size + 1
            
            progress_bar = st.progress(0)
            
            for i in range(0, len(all_documents), batch_size):
                # 10 parçalık vagonu al
                batch = all_documents[i : i + batch_size]
                
                if batch:
                    # Vagonu Pinecone'a gönder
                    vector_store.add_documents(batch)
                    
                    # İlerleme çubuğunu güncelle
                    current_progress = min((i + batch_size) / len(all_documents), 1.0)
                    progress_bar.progress(current_progress)
                    
                    # Google Kotası İçin Fren: Her vagondan sonra 2 saniye bekle
                    time.sleep(2)
            
            st.success("✅ Tüm belgeler başarıyla vektörleştirildi!")
            return vector_store
            
        except Exception as e:
            st.error(f"Pinecone/Embedding Hatası: {str(e)}")
            return None
    
    return None

# --- DİĞERLERİ AYNI ---
def delete_document_cloud(file_name):
    # 1. Pinecone Temizliği (Hata Verirse Yutacağız)
    try:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        index_name = "mevzuat-asistani"
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        
        # Pinecone'dan silmeyi dene
        index.delete(filter={"source": file_name})
        
    except Exception as e:
        # Hata verirse (404 vs.) konsola yaz ama işlemi DURDURMA.
        # Çünkü amaç zaten dosyadan kurtulmak.
        print(f"Pinecone silme uyarısı (Önemsiz): {e}")

    # 2. Supabase ve Storage Temizliği (Asıl Kritik Kısım)
    try:
        supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
        
        # Veritabanı kaydını sil
        supabase.table("dokumanlar").delete().eq("dosya_adi", file_name).execute()
        
        # Dosyanın kendisini storage'dan sil
        supabase.storage.from_("belgeler").remove([file_name])
        
        return True, f"{file_name} başarıyla temizlendi."
        
    except Exception as e: 
        return False, f"Supabase silme hatası: {str(e)}"

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
    except Exception as e: return None
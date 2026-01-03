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

# --- 2. GERÇEK DEDEKTİF: İÇERİK VE YAPI ANALİZİ ---
def analyze_pdf_complexity(file_path):
    """
    Dosya adına ASLA bakmadan, sadece içeriği analiz eder.
    
    Döner: (bool, str) -> (Vision Gerekli mi?, Sebebi ne?)
    """
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        # Analiz için ilk 3 sayfaya bakmak yeterli ve hızlıdır
        pages_to_check = min(len(doc), 3)
        
        for i in range(pages_to_check):
            page = doc[i]
            
            # --- ANALİZ 1: GEOMETRİ (TABLO YOĞUNLUĞU) ---
            # Sayfadaki tüm vektör çizimlerini (çizgi, kutu, tablo kenarlığı) sayar.
            drawings = page.get_drawings()
            
            # Eşik Değeri: 20
            # Düz metinlerde (Yönetmelik vb.) genelde 0-5 arası çizgi olur (altbilgi/üstbilgi).
            # Tablolu belgelerde her hücre bir kutudur, sayı anında 50-100'e çıkar.
            if len(drawings) > 20:
                return True, f"Sayfa {i+1}'de Yoğun Tablo Yapısı ({len(drawings)} çizgi)"

            # --- ANALİZ 2: DİLBİLİM (KARAKTER BOZUKLUĞU / ENCODING) ---
            # Sayfadaki metni normal yolla çekip "Okunabilir Türkçe mi?" diye bakarız.
            text = page.get_text().lower()
            
            # Eğer sayfada yeterince yazı varsa (50 harften fazla) test et
            if len(text) > 50:
                # Bu kelimeler Türkçe metinlerde %99 ihtimalle geçer.
                # Eğer metin "sürdOrdÖğü" gibi bozuksa, bu kelimeler bulunamaz.
                turkish_anchors = [" ve ", " bir ", " ile ", " için ", " bu ", " madde ", " üniversite ", " olan ", " veya "]
                
                # Metnin içinde bu kelimelerden HİÇBİRİ yoksa, encoding bozuktur.
                match_count = sum(1 for word in turkish_anchors if word in text)
                
                if match_count == 0:
                    return True, f"Sayfa {i+1}'de Bozuk Metin/Encoding Hatası (Türkçe kelimeler bulunamadı)"

        # Her şey temizse, normal hızlı mod yeterlidir.
        return False, "Düz Metin"
        
    except Exception as e:
        print(f"Analiz hatası: {e}")
        return True, "Dosya Analiz Edilemedi (Güvenli Mod)" # Hata varsa risk alma, Vision aç

# --- 3. VISION OKUMA (GEMINI 2.5 FLASH) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    extracted_text = ""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    for page_num, page in enumerate(doc):
        # Kullanıcıya bilgi ver (Uzun sürerse panik yapmasın)
        if page_num == 0:
            st.toast(f"👁️ Yapay Zeka Gözü Devrede... (Sayfa 1/{total_pages})", icon="⏳")
            
        # Zoom=2 ile yüksek kalite resim al
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        try:
            response = model.generate_content([
                """
                GÖREV: Bu görseldeki belgeyi analiz et ve metne dönüştür.
                KURALLAR:
                1. Bu belgede TABLOLAR veya BOZUK KARAKTERLER var.
                2. Tablo yapısını Markdown formatında koruyarak aktar.
                3. Türkçe karakterleri düzelt (Örn: "sürdOrdÖğü" -> "sürdürdüğü").
                4. Sadece metni ver.
                """, 
                img
            ])
            extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
        except Exception as e:
            print(f"Vision hatası: {e}")
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
            except: pass 

            # --- B. GEÇİCİ DOSYA ---
            uploaded_file.seek(0)
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # --- C. KARAR ANI: İÇERİK ANALİZİ 🧠 ---
            # Dosya adına BAKMA, İçeriği TARA.
            is_complex, reason = analyze_pdf_complexity(file_path)
            
            # Vision Kullanılsın mı? (Kullanıcı istediyse VEYA İçerik karışık ise)
            should_use_vision = use_vision_mode or is_complex
            
            full_text = ""
            
            if should_use_vision:
                st.toast(f"🤖 Vision Modu: {uploaded_file.name}\nSebep: {reason}", icon="👁️")
                full_text = pdf_image_to_text_with_gemini(file_path)
            else:
                # Normal Hızlı Okuma
                doc = fitz.open(file_path)
                for page in doc: full_text += page.get_text()

            # --- D. BELGE OLUŞTURMA ---
            header_text = full_text[:300].replace("\n", " ").strip() if full_text else "Başlıksız Belge"
            
            unified_doc = Document(
                page_content=f"BELGE KİMLİĞİ: {header_text}\nKAYNAK DOSYA: {uploaded_file.name}\n---\n{full_text}",
                metadata={"source": uploaded_file.name}
            )
            
            # --- E. PARÇALAMA ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,      
                chunk_overlap=300,
                separators=["\n|", "\nMADDE", "\n###", "\n\n", ". "]
            )
            
            split_docs = text_splitter.split_documents([unified_doc])
            all_documents.extend(split_docs)
            
            if os.path.exists(file_path): os.remove(file_path)
            
            # Supabase Güncelleme
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
    except Exception as e: return None
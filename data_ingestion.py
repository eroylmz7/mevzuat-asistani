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

# --- 2. DEDEKTİF (İÇERİK ANALİZİ - YAPAY ZEKA KARAR MEKANİZMASI) ---
def analyze_pdf_complexity(file_path):
    """
    Bu fonksiyon dosyayı açar ve karmaşıklığını analiz eder.
    Dosya adına bakmaz, tamamen içeriğe odaklanır.
    """
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0: return False, "Boş Dosya"
        
        # Analiz için ilk 3 sayfaya bakmak performans/başarı dengesi için idealdir.
        pages_to_check = min(len(doc), 3)
        
        for i in range(pages_to_check):
            page = doc[i]
            
            # KRİTER 1: TABLO YOĞUNLUĞU (GEOMETRİK ANALİZ)
            # Sayfadaki vektör çizimlerini (tablo kenarlıkları, çizgiler) sayar.
            drawings = page.get_drawings()
            # Eşik Değeri: 15. Normal bir metin sayfasında 0-5 arası çizgi olur.
            # 15'ten fazla çizgi varsa, burası kesinlikle tablodur.
            if len(drawings) > 15:
                return True, f"Sayfa {i+1}'de Yoğun Tablo Yapısı Tespit Edildi ({len(drawings)} vektör çizimi)"
            
            # KRİTER 2: METİN KALİTESİ (SEMANTİK ANALİZ)
            # PyMuPDF ile metni çekip, Türkçe karakterlerin bozuk olup olmadığına bakar.
            text = page.get_text().lower()
            if len(text) > 50:
                # Bu kelimeler Türkçe metinlerde istatistiksel olarak en sık geçen bağlaçlardır.
                # Eğer metin "sürdOrdÖğü" gibi bozuksa, bu kelimeler bulunamaz.
                turkish_anchors = [" ve ", " bir ", " ile ", " için ", " bu ", " madde ", " üniversite ", " olan "]
                match_count = sum(1 for word in turkish_anchors if word in text)
                
                # Hiç bağlaç yoksa, metin encoding hatası (bozuk karakter) içeriyor demektir.
                if match_count == 0:
                    return True, f"Sayfa {i+1}'de Bozuk Metin/Encoding Hatası Tespit Edildi"
                    
        return False, "Standart Metin Yapısı"
        
    except Exception as e:
        # Analiz sırasında hata olursa, risk almayıp güvenli moda (Vision) geçmek en doğrusudur.
        print(f"Analiz Hatası: {e}")
        return True, "Otomatik Analiz Tamamlanamadı (Güvenli Mod)"

# --- 3. VISION OKUMA (AKILLI HİBRİT MOD) ---
def pdf_image_to_text_with_gemini(file_path):
    configure_gemini()
    target_model = 'gemini-2.5-flash'
    extracted_text = ""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    # Filtreleri kapatıyoruz ki resmi belgeleri engellemesin.
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    for page_num, page in enumerate(doc):
        # Kullanıcıya bilgi ver
        if page_num == 0:
            st.toast(f"🚀 {target_model} ile Derinlemesine Analiz... Sayfa 1/{total_pages}", icon="🧠")
            
        # Resmi yüksek çözünürlükte al (Zoom=2)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        try:
            # Resmi byte formatına çevir (Hata önleyici)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()

            model = genai.GenerativeModel(target_model)
            
            # HOCANIN SEVECEĞİ DETAYLI PROMPT
            response = model.generate_content(
                [
                    """
                    GÖREV: Bu akademik belgeyi analiz et ve yapılandırılmış veriye dönüştür.
                    
                    ADIMLAR:
                    1. **DİPNOT ANALİZİ:** Tabloların altında veya sayfa sonlarındaki küçük puntolu açıklamaları (örneğin (*) işaretli notlar) tespit et. Bu notlar genellikle yetki ve istisnaları belirtir, bunları ana metinle ilişkilendir.
                    
                    2. **SEMANTİK DÖNÜŞÜM:** Tablolardaki verileri sadece kopyalama; her satırı anlamlı bir cümleye dönüştür. 
                       Örn: "| Doktora | Q1 |" satırını -> "Doktora programı için Q1 yayın şartı aranır." şeklinde yaz.
                    
                    3. **FORMAT:** Tablo yapısını Markdown olarak koru ancak yukarıdaki açıklamaları da ekle.
                    
                    4. **DÜZELTME:** Türkçe karakter hatalarını onar.
                    """, 
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ],
                safety_settings=safety_settings
            )
            
            if response.text:
                extracted_text += f"\n--- Sayfa {page_num + 1} ---\n{response.text}\n"
            else:
                extracted_text += page.get_text()
                
        except Exception as e:
            # Hata durumunda sessizce standart metoda dön
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
            # 1. Dosyayı Geçici Olarak Kaydet
            uploaded_file.seek(0)
            file_path = os.path.join("temp_pdfs", uploaded_file.name)
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            # 2. Supabase Storage'a Yedekle
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                supabase.storage.from_("belgeler").upload(
                    path=uploaded_file.name, file=file_bytes,
                    file_options={"content-type": "application/pdf", "upsert": "true"}
                )
            except: pass

            # --- KARAR MEKANİZMASI (HİLE YOK, SAF ANALİZ) ---
            # Dosya adını kontrol eden kod bloğu KALDIRILDI.
            # Artık sadece matematiksel ve dilbilimsel analiz yapılıyor.
            
            is_complex, reason = analyze_pdf_complexity(file_path)
            
            # Vision kullanıp kullanmayacağımıza karar veriyoruz.
            should_use_vision = use_vision_mode or is_complex
            
            full_text = ""
            if should_use_vision:
                st.toast(f"Mod: Vision (Akıllı Tarama) | Dosya: {uploaded_file.name}\nTespit: {reason}", icon="👁️")
                full_text = pdf_image_to_text_with_gemini(file_path)
            else:
                # Basit dosyalarda hızlı okuma
                doc = fitz.open(file_path)
                for page in doc: full_text += page.get_text()

            # Güvenlik Kontrolü: Eğer Vision boş dönerse (API hatası vb.) yedeğe geç
            if not full_text.strip():
                 doc = fitz.open(file_path)
                 for page in doc: full_text += page.get_text()

            # 3. Belge Nesnesi Oluşturma
            header_text = full_text[:300].replace("\n", " ").strip() if full_text else "Başlıksız"
            unified_doc = Document(
                page_content=f"BELGE KİMLİĞİ: {header_text}\nKAYNAK DOSYA: {uploaded_file.name}\n---\n{full_text}",
                metadata={"source": uploaded_file.name}
            )
            
            # 4. Parçalama (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, 
                chunk_overlap=250,
                separators=["\n|", "\nMADDE", "\n###", "\n\n", "\n", ". ", " "]
            )
            split_docs = text_splitter.split_documents([unified_doc])
            
            # 5. Boyut Kontrolü (Pinecone Limit Aşımını Önleme)
            safe_docs = []
            for doc in split_docs:
                text_size = len(doc.page_content.encode('utf-8'))
                if text_size < 35000:
                    safe_docs.append(doc)
                else:
                    # Çok büyük parçayı güvenli sınıra çek
                    doc.page_content = doc.page_content[:15000] + "\n...(Sistem limiti nedeniyle kısaltıldı)"
                    safe_docs.append(doc)
            
            all_documents.extend(safe_docs)
            
            # Temizlik
            if os.path.exists(file_path): os.remove(file_path)
            
            # DB Kaydı
            try:
                supabase.table("dokumanlar").delete().eq("dosya_adi", uploaded_file.name).execute()
                supabase.table("dokumanlar").insert({"dosya_adi": uploaded_file.name}).execute()
            except: pass
            
        except Exception as e:
            st.error(f"Hata ({uploaded_file.name}): {e}")

    # 6. Vektör Veritabanına Yazma
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

# --- DİĞER STANDART FONKSİYONLAR (DEĞİŞİKLİK YOK) ---
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
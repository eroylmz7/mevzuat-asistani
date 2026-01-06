import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_answer(question, vector_store, chat_history):
    
    # --- 1. GÜVENLİK ---
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        return {"answer": "Hata: Google API Key bulunamadı.", "sources": []}

    # --- 2. ANALİST AJAN (Sorgu Zenginleştirme) ---
    llm_translator = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.1 
    )
    
    translation_prompt = f"""
    GÖREV: Kullanıcı sorusunu analiz et ve arama motoru için en kritik anahtar kelimeleri ekle.
    
    ANALİZ ADIMLARI:
    1. KONU TESPİTİ:
       - Akademik: "Tez", "Sınav", "Ders", "Jüri", "Yüksek Lisans" -> "LİSANSÜSTÜ EĞİTİM"
       - İdari: "Rektör", "Personel", "İzin", "Teşkilat", "Atama" -> "İDARİ MEVZUAT"
       - Disiplin: "Ceza", "Kopya", "Uzaklaştırma" -> "DİSİPLİN SUÇU"
       
    2. GÜNCELLİK VE DETAY:
       - Soru "Yayın şartı", "Mezuniyet kriteri" içeriyorsa -> "Senato Kararı", "Yayın Esasları", "Ek Madde" terimlerini ekle.
       - Soru bir tarih veya yürürlük soruyorsa -> "Yürürlük Tarihi", "Geçici Madde" ekle.
    
    Soru: "{question}"
    Geliştirilmiş Arama Sorgusu (Sadece terimler):
    """
    
    try:
        official_terms = llm_translator.invoke(translation_prompt).content.strip()
        hybrid_query = f"{question} {official_terms}"
    except:
        hybrid_query = question 

    # --- 3. RETRIEVAL (AYAR GÜNCELLEMESİ) ---
    try:
        # fetch_k=160 (Geniş tara) kalsın ama LLM'e gideni (k) düşürelim.
        # k=80 çok fazlaydı, 35-40 idealdir.
        docs = vector_store.max_marginal_relevance_search(
            hybrid_query, 
            k=25,            # DÜŞÜRÜLDÜ (Dikkati dağılmaması için)
            fetch_k=100,     # AYNI KALDI (Geniş tarasın)
            lambda_mult=0.7  # Çeşitliliği artırdık (Farklı belgelerden alsın)
        )
    except Exception as e:
        return {"answer": f"Veritabanı hatası: {str(e)}", "sources": []}
    
    # --- 4. AKILLI ETİKETLEME VE ÖNCELİKLENDİRME ---
    context_text = ""
    sources = []
    
    for doc in docs:
        content = doc.page_content.replace("\n", " ").strip()
        filename = os.path.basename(doc.metadata.get("source", "Bilinmiyor")).lower()
        
        # --- DOSYA ÖNCELİK ALGORİTMASI ---
        # Dosya ismine bakarak yapay zekaya "Bu belgeye ne kadar güvenmelisin?" sinyali veriyoruz.
        
        priority_tag = ""
        doc_category = "GENEL BELGE"
        
        # 1. EN YÜKSEK ÖNCELİK (Özel Esaslar, Ekler, Senato Kararları)
        if any(x in filename for x in ["tezyayın", "sart", "ek", "karar", "uygulama"]):
            priority_tag = "🔥 [YÜKSEK ÖNCELİK / ÖZEL HÜKÜM]"
            doc_category = "ÖZEL SENATO KARARI/YÖNERGESİ"
            
        # 2. ORTA ÖNCELİK (Yönetmelikler)
        elif "yonetmelik" in filename:
            doc_category = "GENEL YÖNETMELİK"
            
        # 3. KATEGORİ ETİKETLEME (Bağlam Karışıklığını Önlemek İçin)
        if "lisansustu" in filename:
            scope_tag = "(KAPSAM: LİSANSÜSTÜ)"
        elif "lisans" in filename and "lisansustu" not in filename:
            scope_tag = "(KAPSAM: LİSANS/ÖNLİSANS)"
        elif "teskilat" in filename or "personel" in filename:
            scope_tag = "(KAPSAM: İDARİ/PERSONEL)"
        else:
            scope_tag = "(KAPSAM: GENEL)"

        # Yapay Zekaya Gidecek Metin Bloğu
        context_text += f"\n--- DOSYA: {filename} {priority_tag} {scope_tag} ---\nİÇERİK: {content}\n"
        
        # Kaynak Listesi
        page = int(doc.metadata.get("page", 0)) + 1 if "page" in doc.metadata else 1
        src_str = f"{filename} (Sayfa {page})"
        if src_str not in sources:
            sources.append(src_str)

    # --- 5. CEVAPLAYICI (HUKUKÇU MODU) ---
    llm_answer = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=google_api_key,
        temperature=0.0 # Yaratıcılık sıfır, sadece kanıt.
    )
    
    final_template = f"""
    Sen, Üniversite Mevzuat Analistisin. Görevin, belgeleri hukuki hiyerarşi kurallarına göre analiz edip KESİN ve DOĞRU cevabı vermektir.
    
    ELİNDEKİ BELGELER (Context):
    {context_text}
    
    SORU: {question}
    
    --- 🧠 KARAR VERME MEKANİZMASI (BU KURALLARA UY) ---
    
    KURAL 1: BELGE TÜRÜNÜ TANI
    - Soru "Akademik" (Öğrenci, Sınav) ise -> Akademik belgelere bak.
    - Soru "İdari" (Rektör, Personel, Teşkilat) ise -> İdari belgelere bak (Öğrenci yönetmeliğini karıştırma).
    
    KURAL 2: HİYERARŞİ VE GÜNCELLİK (EN ÖNEMLİ KURAL) ⚖️
    - Eğer iki belge arasında çelişki varsa (Örn: Biri "X yapılabilir", diğeri "X yasaktır" diyorsa):
      A) Başlığında "🔥 [YÜKSEK ÖNCELİK]" yazan belgeye İTAAT ET. (O belge daha özel veya daha günceldir).
      B) Tarihi YENİ olan belgeye İTAAT ET (Metin içindeki tarihlere bak: 2025 > 2020).
      C) "Özel Hüküm" (Yönerge/Esaslar), "Genel Hüküm"den (Yönetmelik) üstündür.
    
    KURAL 3: KAPSAM İZOLASYONU
    - Soru "Yüksek Lisans" ise -> "Doktora" başlıklarını GÖRMEZDEN GEL.
    - Soru "Doktora" ise -> "Yüksek Lisans" başlıklarını GÖRMEZDEN GEL.
    - Soru "Personel/İdari" ise -> Akademik öğrenci kurallarını GÖRMEZDEN GEL.
    - Belgelerin bazıları TABLO formatındadır. Satır ve sütunların kaymış olabileceğini unutma.
    
    KURAL 4: HALÜSİNASYON ENGELLEME 
    - Belgede açıkça yazmıyorsa "Belgelerde bu bilgi bulunmamaktadır" de.
    - Tahmin yürütme, yorum yapma. Sadece metinde yazanı aktar.
    
    KURAL 5: REFERANS FORMATI
    - Cevap verirken, en son olarak bilgiyi hangi belgeden aldığını belirtmek için cümle sonuna formatını kullan.
    - Örnek: "Yüksek lisans için ALES puanı en az 55 olmalıdır."

    KURAL 6: TABLO OKUMA ŞÜPHECİLİĞİ
    - Metinler PDF tablolarından geldiği için satırlar birbirine karışmış olabilir.
    - Metinler seçilebilir olsa da (selectable text) bu pdf fotokopi çıktısı taranarak elde edilmiş olabilir, dikkat et.
    
    CEVAP:
    """
    
    try:
        answer = llm_answer.invoke(final_template).content
        return {"answer": answer, "sources": sources[:5]}
    except Exception as e:
        return {"answer": f"Cevap oluşturulurken hata: {str(e)}", "sources": []}
# ☁️ Kampüs Mevzuat Asistanı - Cloud Native Sürüm
Bu proje, Bursa Uludağ Üniversitesi mevzuatlarını analiz etmek, öğrencilerin sorularını doğal dilde yanıtlamak ve yönetmeliklerdeki karmaşık tabloları anlamlandırmak amacıyla geliştirilmiş, Bulut Tabanlı (Cloud Native) bir Yapay Zeka asistanıdır.

Proje, modern RAG (Retrieval-Augmented Generation) mimarisi üzerine kurulmuş olup, ölçeklenebilirlik için Pinecone, veri bütünlüğü için Supabase ve bilişsel zeka için Google Gemini 2.5 Flash modellerini kullanmaktadır.

---

## 🏗️ Sistem Mimarisi ve Teknik Detaylar
Proje üç ana teknik modülden oluşmaktadır. Her modül, belirli bir mikro-görevden sorumludur.

### 1. Veri İşleme ve Vektörleştirme (data_ingestion.py)
Bu modül, PDF belgelerinin "ham veri"den "anlamsal vektör"e dönüştürüldüğüü ETL (Extract-Transform-Load) hattıdır.

* **Hibrit PDF Okuma Stratejisi (Multimodal Parsing):**
    * Sistem, yüklenen her PDF'i önce analiz eder (`analyze_pdf_complexity`).
    * Eğer belge metin tabanlı ise, hızlı olması için **PyMuPDF (Fitz)** kullanılır.
    * Eğer belge taranmış resim ise veya karmaşık tablolar içeriyorsa, **Google Gemini 2.5 Flash Vision** modu devreye girer. Sayfanın fotoğrafı çekilerek LLM'den "Markdown" formatında tabloyu yeniden çizmesi istenir. Bu sayede tablo yapısı bozulmadan okunur.
* **Akıllı Bölümleme (Chunking):**
    * Belgeler `RecursiveCharacterTextSplitter` kullanılarak 2000 karakterlik parçalara bölünür.
    * **Chunk Overlap:** 300 olarak ayarlanmıştır. Bu, bir maddenin (Örn: Madde 5) iki parça arasında bölünse bile bağlamın kopmamasını sağlar.
* **Vektörleştirme (Embedding):**
    * Metin parçaları `google models/embedding-001` modeli ile sayısal vektörlere dönüştürülür ve **Pinecone** bulut veritabanına yüklenir.

### 2. Akıllı Cevap Üretimi ve Sıralama (`generation.py`)
Sistemin "Beyin" kısmıdır. Klasik arama yerine **"2 Aşamalı Erişim (2-Stage Retrieval)"** stratejisi kullanılmıştır.

* **Adım 1: Sorgu Zenginleştirme (Query Expansion):**
    * Kullanıcının ham sorusu (Örn: "Staj ne zaman?") bir LLM tarafından akademik literatüre uygun hale getirilir ve eş anlamlıları eklenir.(Örn: "Staj, İşletmede Mesleki Eğitim, Uygulamalı Eğitim tarihleri ve koşulları").
* **Adım 2: Geniş Arama (Retrieval):**
    * Optimize edilmiş sorgu ile Pinecone üzerinden **MMR (Maximal Marginal Relevance)** algoritması kullanılarak en alakalı 30 belge adayı getirilir. MMR, sadece benzerleri değil, konunun farklı yönlerini içeren çeşitli belgeleri seçer.
* **Adım 3: Yeniden Sıralama (Reranking - The Judge):** 🌟
    * Getirilen 30 belge, **Gemini 2.5 Flash** modeline "Hakem" rolüyle verilir. Sadece en alakalı ve kanıt niteliği taşıyan **Top 5** belge seçilir. Bu, halüsinasyon oranını düşürür.
* **Adım 4: Kanıtlı Cevaplama:**
    * Seçilen belgeler modele verilir ve cevap üretilir. Kaynaklar şeffaf bir şekilde HTML `<details>` yapısı ile "Kanıt Kutusu" olarak eklenir.

### 3. Kullanıcı Arayüzü ve Yönetim (`app.py`)
**Streamlit** arayüzü ile son kullanıcı ve yöneticiler sistemle etkileşime girer.

* **Supabase Entegrasyonu:**
    * **Kimlik Doğrulama (Auth):** Öğrenci ve Yönetici girişleri ayrıştırılmıştır.

    * **Loglama:** Soru-cevap geçmişi `sorgu_loglari` tablosuna kaydedilerek Admin panelinde analiz edilir.

* **Admin Paneli:** Yöneticiler yeni PDF yükleyebilir, mevcutları silebilir ve istatistikleri görebilir.

* **Asenkron Yapı:** Performans için `asyncio` döngüleri optimize edilmiş ve `st.rerun()` stratejisi ile anlık veritabanı güncelliği sağlanmıştır.



## 🛠️ Kurulum ve Dağıtım (Deployment)

Bu proje **Streamlit Cloud** üzerinde çalıştırılmak üzere tasarlanmıştır.

### 📋 Gereksinimler

1.  GitHub üzerindeki depoya tüm kodlar yüklenir.
2.  Streamlit Cloud panelinden `secrets.toml` ayarları yapılandırılır.

### 🔑 Ortam Değişkenleri (`secrets.toml`)

Projenin çalışması için aşağıdaki API anahtarlarının tanımlanması zorunludur:

```toml
GOOGLE_API_KEY = "AIzaSy..."       # Gemini Modelleri için
PINECONE_API_KEY = "pcsk_..."      # Vektör Veritabanı için
SUPABASE_URL = "https://..."       # Veri Tabanı URL
SUPABASE_KEY = "eyJ..."            # Veri Tabanı Key
```

### Kütüphaneler
 requirements.txt dosyasında aşağıdaki temel paketler bulunmalıdır:
```toml
streamlit, langchain-google-genai, langchain-pinecone, supabase, pymupdf
```

##  Neden Bu Mimari Seçildi?

| Özellik | Açıklama ve Avantajı |
| :--- | :--- |
| **Erişilebilirlik** | Herhangi bir cihazdan (Mobil, Tablet, PC) erişim sağlanır. |
| **Kullanıcı Yönetimi** | Supabase üzerinden kimlik doğrulama ve rol bazlı yetkilendirme (RBAC) sunar. |
| **Veri Kalıcılığı** | Uygulama yeniden başlatılsa bile Pinecone sayesinde veriler kaybolmaz. |
| **Gelişmiş Zeka** | Yerel donanıma bağlı kalmadan Google'ın en güçlü modelleri (Gemini Vision & Flash) kullanılır. |

##  Site Erişimi
<https://mevzuat-asistani-eren.streamlit.app/>  tıklayabilirsiniz.
Geliştirici: [Eren Yılmaz]
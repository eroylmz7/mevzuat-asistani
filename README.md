☁️ Kampüs Mevzuat Asistanı - Cloud Native Sürüm
Bu proje, Bursa Uludağ Üniversitesi mevzuatlarını analiz etmek, öğrencilerin sorularını doğal dilde yanıtlamak ve yönetmeliklerdeki karmaşık tabloları anlamlandırmak amacıyla geliştirilmiş, Bulut Tabanlı (Cloud Native) bir Yapay Zeka asistanıdır.

Proje, modern RAG (Retrieval-Augmented Generation) mimarisi üzerine kurulmuş olup, ölçeklenebilirlik için Pinecone, veri bütünlüğü için Supabase ve bilişsel zeka için Google Gemini 2.5 Flash modellerini kullanmaktadır.

🏗️ Sistem Mimarisi ve Teknik Detaylar
Proje üç ana teknik modülden oluşmaktadır. Her modül, belirli bir mikro-görevden sorumludur.

1. Veri İşleme ve Vektörleştirme (data_ingestion.py)
Bu modül, PDF belgelerinin "ham veri"den "anlamsal vektör"e dönüştürüldüğü ETL (Extract-Transform-Load) hattıdır.

Hibrit PDF Okuma Stratejisi (Multimodal Parsing):

Sistem, yüklenen her PDF'i önce analiz eder (analyze_pdf_complexity).

Eğer belge metin tabanlı ise, hızlı olması için PyMuPDF (Fitz) kullanılır.

Eğer belge taranmış resim ise veya karmaşık tablolar/sütunlar içeriyorsa, Google Gemini 2.5 Flash Vision modu devreye girer. Sayfanın fotoğrafı çekilir ve LLM'den "Markdown" formatında tabloyu yeniden çizmesi istenir. Bu sayede tablo yapısı bozulmadan okunur.

Akıllı Bölümleme (Chunking):

Belgeler RecursiveCharacterTextSplitter kullanılarak 2000 karakterlik parçalara bölünür.

Chunk Overlap: 300 olarak ayarlanmıştır. Bu, bir madde (Örn: Madde 5) iki parça arasında bölünse bile bağlamın kopmamasını sağlar.

Vektörleştirme (Embedding):

Metin parçaları Google models/embedding-001 modeli ile sayısal vektörlere dönüştürülür ve Pinecone bulut veritabanına yüklenir.

2. Akıllı Cevap Üretimi ve Sıralama (generation.py)
Burası, sistemin "Beyin" kısmıdır. Klasik arama yerine "2 Aşamalı Erişim (2-Stage Retrieval)" stratejisi kullanılmıştır.

Adım 1: Sorgu Zenginleştirme (Query Expansion):

Kullanıcının sorduğu ham soru (Örn: "Staj ne zaman?") doğrudan aranmaz.

Önce bir LLM, soruyu akademik literatüre uygun hale getirir ve eş anlamlıları ekler (Örn: "Staj, İşletmede Mesleki Eğitim, Uygulamalı Eğitim tarihleri ve koşulları").

Adım 2: Geniş Arama (Retrieval):

Optimize edilmiş sorgu ile Pinecone üzerinden MMR (Maximal Marginal Relevance) algoritması kullanılarak en alakalı 30 belge adayı getirilir. MMR, sadece benzerleri değil, konunun farklı yönlerini içeren çeşitli belgeleri seçer.

Adım 3: Yeniden Sıralama (Reranking - The Judge): 🌟 (Projenin en kritik özelliği)

Getirilen 30 belge, Gemini 2.5 Flash modeline "Hakem" rolüyle verilir.

Model, her belgeyi okur ve "Bu belge kullanıcının sorusuna gerçekten cevap veriyor mu?" diye analiz eder.

Sadece en alakalı ve kanıt niteliği taşıyan Top 5 belge seçilir. Bu, halüsinasyon oranını dramatik şekilde düşürür.

Adım 4: Kanıtlı Cevaplama:

Seçilen belgeler modele verilir ve cevap üretilir. Cevabın altına, kullanılan kaynaklar şeffaf bir şekilde HTML <details> yapısı ile "Açılır/Kapanır Kanıt Kutusu" olarak eklenir.

3. Kullanıcı Arayüzü ve Yönetim (app.py)
Son kullanıcı ve yöneticilerin sistemle etkileşime girdiği Streamlit arayüzüdür.

Supabase Entegrasyonu:

Kimlik Doğrulama (Auth): Öğrenci ve Yönetici (Admin) girişleri ayrıştırılmıştır.

Loglama: Kullanıcıların sorduğu sorular ve modelin verdiği cevaplar sorgu_loglari tablosuna kaydedilir. Bu veriler Admin panelinde analiz edilir.

Admin Paneli:

Yöneticiler sisteme yeni PDF yükleyebilir, mevcutları silebilir ve kullanım istatistiklerini görebilir.

Yükleme işlemi sırasında st.progress barları ile geri bildirim verilir.

Asenkron Yapı:

Streamlit Cloud ortamında performans sorunu yaşamamak için asyncio döngüleri optimize edilmiştir.

st.rerun() stratejisi ile, yeni yüklenen bir belge anında hafızaya alınır ve sorgulanabilir hale gelir.

🛠️ Kurulum ve Dağıtım (Deployment)
Bu proje Streamlit Cloud üzerinde çalıştırılmak üzere tasarlanmıştır.

Gereksinimler:

Github üzerindeki repoya kodlar yüklenir.

Streamlit Cloud panelinden secrets.toml ayarları yapılandırılır.

Ortam Değişkenleri (secrets.toml): Projenin çalışması için aşağıdaki API anahtarlarının tanımlanması zorunludur:
GOOGLE_API_KEY = "AIzaSy..."       # Gemini Modelleri için
PINECONE_API_KEY = "pcsk_..."      # Vektör Veritabanı için
SUPABASE_URL = "https://..."       # Veri Tabanı URL
SUPABASE_KEY = "eyJ..."            # Veri Tabanı Key

Kütüphaneler: requirements.txt dosyasında aşağıdaki temel paketler bulunmalıdır:

streamlit, langchain-google-genai, langchain-pinecone, supabase, pymupdf

Geliştirici: [Eren Yılmaz]
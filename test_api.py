import google.generativeai as genai
import sys

print("📡 Google Bağlantı Testi Başlıyor...")

# API Anahtarını buraya yapıştıracaksın (terminalden isteyecek)
api_key = input("Lütfen Google API Key'inizi yapıştırın: ").strip()

try:
    genai.configure(api_key=api_key)
    
    # Mevcut modelleri listele
    print("\n📋 Hesabınızda Erişilebilir Modeller:")
    found_flash = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
            if "gemini-1.5-flash" in m.name:
                found_flash = True

    if not found_flash:
        print("\n⚠️ UYARI: Listenizde 'gemini-1.5-flash' görünmüyor!")
    
    print("\n🧪 Deneme Mesajı Gönderiliyor...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Merhaba, bu bir test mesajıdır.")
    
    print("\n✅ BAŞARILI! Cevap:")
    print(response.text)

except Exception as e:
    print(f"\n❌ HATA OLUŞTU:\n{e}")
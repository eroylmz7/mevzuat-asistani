import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env dosyasını yükle
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Google'a bağlan
genai.configure(api_key=api_key)

print(f"🔑 Kullanılan Key: {api_key[:10]}... (Doğru mu kontrol et)")
print("\n📋 GOOGLE'IN KABUL ETTİĞİ MODELLER LİSTESİ:")
print("-" * 40)

try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ {m.name}")
            available_models.append(m.name)
            
    if not available_models:
        print("❌ HİÇBİR MODEL BULUNAMADI! (API Key veya Proje yetkisi sorunu)")
except Exception as e:
    print(f"❌ BAĞLANTI HATASI: {e}")

print("-" * 40)
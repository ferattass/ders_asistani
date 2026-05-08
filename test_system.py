"""Sistem Kontrol Scripti"""
import sys
print(f"Python: {sys.version}")
print("=" * 50)

hatalar = []

# 1. Temel kutuphaneler
print("\n[1] Temel Kutuphaneler:")
kutuphaneler = [
    ("streamlit", "Streamlit (Arayuz)"),
    ("fitz", "PyMuPDF (PDF Okuma)"),
    ("chromadb", "ChromaDB (Vektor DB)"),
    ("google.generativeai", "Google Gemini API"),
    ("spacy", "spaCy (NLP)"),
    ("sklearn", "Scikit-learn"),
    ("rank_bm25", "Rank-BM25"),
    ("plotly", "Plotly (Grafik)"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow (Gorsel)"),
    ("torch", "PyTorch"),
]
for modul, ad in kutuphaneler:
    try:
        __import__(modul)
        print(f"  ✅ {ad}")
    except ImportError as e:
        print(f"  ❌ {ad} - HATA: {e}")
        hatalar.append(ad)

# 2. LangChain modulleri
print("\n[2] LangChain Modulleri:")
langchain_modulleri = [
    ("langchain", "LangChain Core"),
    ("langchain_huggingface", "LangChain HuggingFace"),
    ("langchain_text_splitters", "LangChain Text Splitters"),
    ("langchain_community", "LangChain Community"),
]
for modul, ad in langchain_modulleri:
    try:
        __import__(modul)
        print(f"  ✅ {ad}")
    except ImportError as e:
        print(f"  ❌ {ad} - HATA: {e}")
        hatalar.append(ad)

# 3. spaCy modeli
print("\n[3] spaCy Dil Modeli:")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(f"  ✅ en_core_web_sm yuklu (v{nlp.meta['version']})")
except Exception as e:
    print(f"  ❌ en_core_web_sm yuklenemedi: {e}")
    hatalar.append("spaCy modeli")

# 4. Proje modulleri
print("\n[4] Proje Modulleri (src/):")
proje_modulleri = [
    ("src.veri_toplama", "veri_toplama.py"),
    ("src.parcalama", "parcalama.py"),
    ("src.vektor_deposu", "vektor_deposu.py"),
    ("src.arama_uretim", "arama_uretim.py"),
    ("src.geleneksel_model", "geleneksel_model.py"),
    ("src.degerlendirme", "degerlendirme.py"),
]
for modul, ad in proje_modulleri:
    try:
        __import__(modul)
        print(f"  ✅ {ad}")
    except Exception as e:
        print(f"  ❌ {ad} - HATA: {e}")
        hatalar.append(ad)

# 5. Config kontrolu
print("\n[5] Config Kontrolu:")
try:
    import config
    print(f"  ✅ PROJE_KLASORU: {config.PROJE_KLASORU}")
    print(f"  ✅ VERI_KLASORU: {config.VERI_KLASORU}")
    print(f"  ✅ VERITABANI_KLASORU: {config.VERITABANI_KLASORU}")
    print(f"  ✅ EMBEDDING_MODELI: {config.EMBEDDING_MODELI}")
    print(f"  ✅ PARCA_BOYUTU: {config.PARCA_BOYUTU}")
    print(f"  ✅ SPACY_MODELI: {config.SPACY_MODELI}")
    api_durum = "VAR (gizli)" if config.GEMINI_API_KEY else "YOK"
    print(f"  ✅ GEMINI_API_KEY: {api_durum}")
except Exception as e:
    print(f"  ❌ Config yuklenemedi: {e}")
    hatalar.append("config.py")

# 6. Test sorulari
print("\n[6] Test Sorulari:")
try:
    import json
    with open("data/test_questions.json", "r", encoding="utf-8") as f:
        sorular = json.load(f)
    print(f"  ✅ {len(sorular)} test sorusu mevcut")
    for i, s in enumerate(sorular[:3]):
        print(f"     Ornek {i+1}: {s['question'][:50]}...")
except Exception as e:
    print(f"  ❌ Test sorulari okunamadi: {e}")
    hatalar.append("test_questions.json")

# 7. ChromaDB durumu
print("\n[7] ChromaDB Durumu:")
try:
    import os
    db_path = config.VERITABANI_KLASORU
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"  ✅ Veritabani klasoru mevcut: {db_path}")
        from src.vektor_deposu import koleksiyonu_getir
        kol = koleksiyonu_getir()
        if kol:
            sayi = kol.count()
            print(f"  ✅ Koleksiyon '{config.KOLEKSIYON_ADI}' mevcut, {sayi} dokuman iceriyor")
        else:
            print(f"  ⚠️ Koleksiyon bulunamadi (henuz PDF yuklenmemis olabilir)")
    else:
        print(f"  ⚠️ Veritabani henuz olusturulmamis (normal - PDF yuklendikten sonra olusur)")
except Exception as e:
    print(f"  ❌ ChromaDB kontrol hatasi: {e}")

# 8. Gemini API testi
print("\n[8] Gemini API Testi:")
try:
    import google.generativeai as genai
    if config.GEMINI_API_KEY:
        genai.configure(api_key=config.GEMINI_API_KEY)
        modeller = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if modeller:
            print(f"  ✅ API baglantisi basarili! {len(modeller)} model mevcut")
            print(f"     Ornek modeller: {', '.join(modeller[:3])}")
        else:
            print(f"  ⚠️ API baglantisi var ama aktif model yok")
    else:
        print(f"  ⚠️ API key ayarlanmamis")
except Exception as e:
    print(f"  ❌ Gemini API hatasi: {e}")
    hatalar.append("Gemini API")

# SONUC
print("\n" + "=" * 50)
if hatalar:
    print(f"⚠️ SORUNLAR TESPIT EDILDI ({len(hatalar)} adet):")
    for h in hatalar:
        print(f"   - {h}")
else:
    print("✅ TUM KONTROLLER BASARILI! Sistem calismaya hazir.")
print("=" * 50)

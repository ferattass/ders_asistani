import os

# ============ DOSYA YOLLARI ============
PROJE_KLASORU = os.path.dirname(os.path.abspath(__file__))
VERI_KLASORU = os.path.join(PROJE_KLASORU, "data")
VERITABANI_KLASORU = os.path.join(PROJE_KLASORU, "chromadb_store")
TEST_SORULARI_DOSYASI = os.path.join(VERI_KLASORU, "test_questions.json")

# Dizinleri oluştur
os.makedirs(VERI_KLASORU, exist_ok=True)
os.makedirs(VERITABANI_KLASORU, exist_ok=True)

# ============ PARÇALAMA AYARLARI ============
PARCA_BOYUTU = 500        # Her bir chunk'ın karakter uzunluğu
PARCA_KESISIMI = 50       # Ardışık chunk'lar arasındaki örtüşme

# ============ MODEL AYARLARI ============
EMBEDDING_MODELI = "all-MiniLM-L6-v2"

# İngilizce Spacy modeli ile kavram (NER) çıkarımı yapacağız.
# Türkçe modeli olsaydı: spacy.load("tr_core_news_trf") gibi kullanacaktık.
SPACY_MODELI = "en_core_web_sm"

# ============ API AYARLARI ============
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ============ KOLEKSIYON ADI ============
KOLEKSIYON_ADI = "ders_notlari"

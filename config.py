import os

# Yollar (Paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chromadb_store")

# Dizinleri oluştur
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Chunking (Parçalama) Ayarları
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Modeller
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# İngilizce Spacy modeli ile kavram (NER) denemesi yapacağız. 
# Eğer Türkçe modeli olsaydı: spacy.load("tr_core_news_trf") gibi kullanacaktık.
SPACY_MODEL = "en_core_web_sm"

# Opsiyonel API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

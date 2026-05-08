<div align="center">
  <h1>🧠 Ders Asistanı & RAG Pipeline</h1>
  <p><strong>Bilgi Getirme (Information Retrieval) Sistemleri Karşılaştırma Platformu</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
  [![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
  [![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge)](https://www.trychroma.com/)
</div>

---

## 📖 Proje Hakkında
Bu proje, kapalı veri kümeleri (ders notları, şirket belgeleri, makaleler vb.) üzerinde çalışan, "Halüsinasyon" yapmayan ve güvenilir cevaplar üreten bir **RAG (Retrieval-Augmented Generation)** sistemidir.

Projenin en önemli özelliği, sadece bir sohbet botu olmakla kalmayıp; **TF-IDF, BM25, Semantik Arama ve Hibrit Arama** algoritmalarını bilimsel metriklerle (MRR, NDCG, Precision, BLEU) birbiriyle yarıştıran akademik bir laboratuvar ortamı sunmasıdır.

## ✨ Temel Özellikler
- 📄 **Akıllı Belge İşleme:** PyMuPDF ve Gemini Vision OCR ile taranmış PDF'lerden bile yüksek doğrulukla metin çıkarımı.
- 💬 **RAG Sohbet Asistanı:** Cevapların hangi belgenin hangi sayfasından çekildiğini şeffaflıkla gösteren (Benzerlik Skorları) arayüz.
- ⚔️ **Algoritma Çarpıştırma:** Girilen sorguyu 4 farklı arama motoruna gönderip, gecikme (ms) ve özgünlük bazında canlı performans analizi.
- 🧬 **Embedding Haritası (t-SNE):** Belgelerin 384 boyutlu vektör uzayındaki anlamsal kümelenmelerinin 2 boyutlu interaktif haritası.
- 📊 **Bilimsel Değerlendirme:** Yüklenen PDF'den otomatik test soruları üretme ve sistemi Global Başarı Metrikleriyle (NDCG, MAP, Faithfulness) test etme.
- 🎨 **Premium Arayüz:** Streamlit tabanlı "Aurora Dark" (Glassmorphism) modern UI tasarımı.

## 🛠️ Kullanılan Teknolojiler
- **Dil Modeli & OCR:** Google Gemini Pro & Vision
- **Embedding Modeli:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vektör Veritabanı:** ChromaDB
- **Doğal Dil İşleme (NLP):** spaCy, scikit-learn, rank-bm25
- **Görselleştirme:** Plotly

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükleyin
Proje dizininde bir terminal açın ve gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
# Veya manuel olarak:
pip install streamlit langchain langchain-huggingface chromadb google-generativeai pymupdf spacy rank-bm25 scikit-learn plotly
```

### 2. Dil Modelini İndirin
spaCy'nin İngilizce dil modelini (Varlık Tanıma için) indirin:
```bash
python -m spacy download en_core_web_sm
```

### 3. API Key Ayarı
`config.py` dosyasını açın ve kendi Google Gemini API anahtarınızı girin:
```python
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "BURAYA_API_KEYINIZI_YAZIN")
```

### 4. Uygulamayı Başlatın
```bash
streamlit run app.py
```

## 📁 Proje Yapısı (Mimari)
```text
📦 proje_dizini/
├── 📄 app.py                    # Ana Streamlit arayüzü ve sayfalar
├── 📄 config.py                 # Yollar, modeller ve global ayarlar
├── 📂 data/                     # Yüklenen PDF'ler ve üretilen JSON soruları
├── 📂 chromadb_store/           # Vektör veritabanı (otomatik oluşur)
└── 📂 src/
    ├── veri_toplama.py          # PDF Okuma ve Vision OCR işlemleri
    ├── parcalama.py             # Metinleri Chunk'lara bölme ve spaCy NER
    ├── vektor_deposu.py         # Embedding üretimi ve ChromaDB yönetimi
    ├── arama_uretim.py          # RAG Semantik Arama ve Gemini Promptlama
    ├── geleneksel_model.py      # TF-IDF, BM25, Hibrit arama ve t-SNE algoritmaları
    ├── degerlendirme.py         # MRR, NDCG, BLEU gibi bilimsel metrik hesaplamaları
    └── soru_uretici.py          # Gemini ile PDF'den otomatik test sorusu sentezleyici
```

## 🔬 Değerlendirme Metrikleri Ne İfade Eder?
*   **Precision (Hassasiyet):** Gelen belgelerdeki doğru anahtar kelime oranı.
*   **NDCG:** Arama motorunun doğru belgeleri "en üst sıraya" koyma başarısı.
*   **Faithfulness:** Dil modelinin PDF dışına çıkmama (Halüsinasyon yapmama) sadakati.
*   **MRR (Mean Reciprocal Rank):** Doğru cevabın kaçıncı denemede bulunduğu.

---
*Bu proje, Retrieval-Augmented Generation (RAG) mimarisini ve Bilgi Getirme (IR) sistemlerini derinlemesine incelemek amacıyla akademik bir perspektifle geliştirilmiştir.*

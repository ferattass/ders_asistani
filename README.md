# 🧠 Öğrenci Ders Asistanı (RAG Pipeline)

Bu proje; öğrenci ders notlarını (PDF), gelişmiş bir **RAG (Retrieval-Augmented Generation)** mimarisi ile analiz edip sorularınıza akıllıca cevaplar veren kişisel bir asistandır.

Aynı zamanda **TF-IDF** ve **BM25** gibi geleneksel yöntemlerle karşılaştırma yaparak, bilimsel metriklerle (Precision, Hit Rate, BLEU, ROUGE-L, Faithfulness) her yöntemi değerlendirir.

## 🚀 Proje Nasıl Çalıştırılır?

1. İlk kurulum için klasörün içindeki **`install.bat`** dosyasına iki kere tıklayın ve işlemlerin bitmesini bekleyin.
2. Proje klasöründe bir terminal açın ve aşağıdaki komutları yazın:
   ```bash
   venv\Scripts\activate
   streamlit run app.py
   ```
3. Açılan sayfada sol menüden bir **Google Gemini API Key** girin.
4. PDF dosyalarınızı yükleyip *"Veritabanını Oluştur"* butonuna basın.
5. Artık 4 farklı sayfadan projeyi kullanabilirsiniz!

---

## 🏗 Mimari (Modül Yapısı)

```
📦 muh/
├── 📄 app.py                    → Ana Streamlit Arayüzü (4 sayfa)
├── 📄 config.py                 → Tüm ayarlar ve yollar
├── 📂 src/
│   ├── veri_toplama.py          → PDF okuma + Gemini Vision OCR
│   ├── parcalama.py             → Metin parçalama + spaCy NER
│   ├── vektor_deposu.py         → HuggingFace Embedding + ChromaDB
│   ├── arama_uretim.py          → Semantic Retrieval + Gemini Generation
│   ├── geleneksel_model.py      → TF-IDF ve BM25 geleneksel arama
│   └── degerlendirme.py         → BLEU, ROUGE-L, Precision metrikleri
├── 📂 data/
│   ├── *.pdf                    → Ders notları (PDF)
│   └── test_questions.json      → Değerlendirme test soruları
├── 📂 chromadb_store/           → Kalıcı vektör veritabanı
└── 📂 .streamlit/config.toml    → Tema ayarları
```

### Adım 1: Veri Toplama (`src/veri_toplama.py`)
**PyMuPDF** ile PDF'lerden metin çıkarır. Opsiyonel olarak **Gemini Vision** ile görsel içindeki tablo/diyagramları da OCR ile okur.

### Adım 2: Parçalama (`src/parcalama.py`)
**Langchain RecursiveCharacterTextSplitter** ile metni 500 karakterlik akıllı parçalara böler. **spaCy** NER ile parçalardaki kişi, kurum, tarih gibi varlıkları tespit edip metadata'ya ekler.

### Adım 3: Vektörleştirme (`src/vektor_deposu.py`)
**HuggingFace (all-MiniLM-L6-v2)** ile parçaları sayısal vektörlere çevirir. **ChromaDB** ile diske kalıcı olarak kaydeder.

### Adım 4: Arama ve Üretim (`src/arama_uretim.py`)
Sorunuz vektöre çevrilir, ChromaDB'de **Kosinüs Benzerliği** ile en yakın parçalar bulunur. Bulunan bağlam + soru birlikte **Google Gemini API**'ye gönderilir ve doğal dilde cevap üretilir.

### Adım 5: Geleneksel Karşılaştırma (`src/geleneksel_model.py`)
Aynı parçalar üzerinde **TF-IDF** (Kelime Sıklığı) ve **BM25** (Gelişmiş Kelime Eşleşme) yöntemleriyle arama yapılır. RAG ile yan yana karşılaştırılır.

### Adım 6: Değerlendirme (`src/degerlendirme.py`)
15 adet test sorusu üzerinde tüm yöntemler çalıştırılır. **Precision@K**, **Hit Rate**, **BLEU**, **ROUGE-L** ve **Faithfulness** metrikleri hesaplanır, grafiklerle sunulur.

---

## 📊 Kullanılan Metrikler

| Metrik | Açıklama |
|--------|----------|
| Precision@K | Getirilen K belgeden kaçı doğru? |
| Hit Rate | En az 1 doğru belge bulundu mu? |
| BLEU | N-gram düzeyinde üretim kalitesi |
| ROUGE-L | En Uzun Ortak Alt Dizi (LCS) F1 |
| Faithfulness | Cevabın bağlama sadakati |

## ⚔️ Karşılaştırma

| Özellik | TF-IDF | BM25 | RAG (Embedding+LLM) |
|---------|--------|------|----------------------|
| Anlam Yakalama | ❌ | ❌ | ✅ |
| Hız | ⚡ ~ms | ⚡ ~ms | 🐢 ~2-5s |
| Maliyet | Ücretsiz | Ücretsiz | API maliyeti |
| Veri Gizliliği | ✅ Lokal | ✅ Lokal | ⚠️ API'ye gider |

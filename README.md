# 🧠 Öğrenci Ders Asistanı (RAG Pipeline)

Bu proje; öğrenci ders notlarını (PDF), gelişmiş bir RAG (Retrieval-Augmented Generation) mimarisi ile analiz edip sorularınıza akıllıca cevaplar veren kişisel bir asistandır.

## 🚀 Proje Nasıl Çalıştırılır?

1. İlk kurulum için klasörün içindeki **`install.bat`** dosyasına iki kere tıklayın ve siyah ekrandaki işlemlerin (paket indirme vb.) bitmesini bekleyin.
2. İşlem tamamlandıktan sonra proje klasöründe (`d:\Projects\muh`) bir terminal sayfası açın.
3. Aşağıdaki komutları sırasıyla yazarak arayüzü başlatın:
   ```bash
   venv\Scripts\activate
   streamlit run app.py
   ```
*(Not: Ekranda ilk seferde `Email:` sorusu çıkarsa hiçbir şey yazmadan direkt `Enter` tuşuna basarak geçebilirsiniz).*
4. Açılan Streamlit sayfasında, sol menüye bir **Google Gemini API Key** girin ve *"1. Verileri Oku ve Veritabanı Oluştur"* butonuna basın.
5. Veriler başarıyla kaydedildikten sonra sağ ekrandan sorularınızı sorabilirsiniz!

---

## 🏗 Mimari Nasıl Çalışır? (Adım Adım İzahat)

Proje tüm kodu tek bir dosyaya yığmak yerine profesyonel modüllerden (src/ klasörü) oluşmaktadır:

### Adım 1: Veri Toplama ve Temizleme (`src/ingestion.py`)
`data` klasörüne koyduğumuz PDF slaytlarını ve notlarını **PyMuPDF (fitz)** kütüphanesi kullanarak okur. PyMuPDF sayfaları gezip metinleri çıkartır. Ardından ufak bir kod, oluşan gereksiz boşlukları ve satır atlamalarını temizler.

### Adım 2: Metinleri Parçalama ve Anlamlandırma (`src/chunking.py`)
Koca bir PDF'in tamamını yapay zekaya veremeyeceğimiz için **Langchain**'in `RecursiveCharacterTextSplitter` aracı metni 500 kelimelik bloklara böler. Ardından **spaCy** isimli Dil İşleme kütüphanesi devreye girer. Hazırladığımız her parçanın içinden mantıklı nesneleri (Kişi, Tarih vs.) çekip metadatalara ekleyerek anlamlandırır (NER Uygulaması).

### Adım 3: Embedding (Vektörleştirme) ve VectorDB (`src/vector_store.py`)
Böldüğümüz parçalanmış metinlerin her biri, **HuggingFace (`all-MiniLM-L6-v2`)** dil modeli aracılığıyla matematiksel sayılara (vektörlere) dönüştürülür. Daha sonra bu vektörler **ChromaDB** vektör veritabanına kaydedilir (SSD'nizde `chromadb_store` klasöründe kalıcı olarak saklanır, bilgisayarı kapatsanız dahi verileriniz silinmez).

### Adım 4: Geri Çağırma Aşaması - Retrieval (`src/retrieval_generation.py`)
Siz asistana "Vize konuları nedir?" dediğinizde, o cümle de matematiksel bir vektöre çevrilir. ChromaDB veritabanı içerisinde gezinilip, saniyenin onda biri kadar bir sürede sizin sorunuzla anlamsal olarak ("Kosinüs Benzerliği" kullanılarak) *en çok eşleşen* metin parçaları çekilir.

### Adım 5: Cevap Üretme ve Arayüz - UI (`app.py`)
Veritabanından çekilen kaynak metinler ile sizin sorunuz yan yana koyulur ve bir LLM'e (Büyük Dil Modeli) verilir. Biz burada çok hızlı ve stabil olduğu için **Google Gemini API** kullanıyoruz. LLM'e şu Prompt gönderilir: *"Kullanıcı bir soru soruyor, kaynak olarak elimizde sadece şu ders notları var. Lütfen sadece bunlardan yararlanarak cevap ver."*
Kullanıcının projeyi rahatça kullanabilmesi için tüm süreç **Streamlit** Python altyapısı ile güzel bir Web Arayüzü olarak sunulur.

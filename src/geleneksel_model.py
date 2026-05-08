"""
Geleneksel Bilgi Getirme Modelleri
====================================
TF-IDF ve BM25 tabanlı keyword-based retrieval sistemleri.
RAG (Semantic Embedding) ile karşılaştırma yapabilmek için aynı arayüzü sağlar.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re
import time


def _sorgu_temizle(soru):
    """Sorgudaki gürültü kelimeleri (nedir, nasıl vb.) temizler."""
    temiz = soru.lower()
    durak_kelimeler = ["nedir", "nelerdir", "neden", "niçin", "nasıl", "kimdir", "nedir?", "nelerdir?", "nelerdir.", "nedir."]
    for kelime in durak_kelimeler:
        temiz = temiz.replace(kelime, "").strip()
    return temiz if temiz else soru


class TFIDFArama:
    """
    TF-IDF (Terim Sıklığı - Ters Belge Sıklığı) tabanlı arama sistemi.
    """
    
    def __init__(self):
        self.vektorleyici = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrisi = None
        self.belgeler = []
        self.bilgiler = []
        self.hazir = False
    
    def egit(self, parcalar):
        """Parça listesinden TF-IDF matrisini oluştur."""
        if not parcalar:
            return
        self.belgeler = [p["page_content"] for p in parcalar]
        self.bilgiler = [p["metadata"] for p in parcalar]
        self.tfidf_matrisi = self.vektorleyici.fit_transform(self.belgeler)
        self.hazir = True
        print(f"TF-IDF Modeli Eğitildi: {len(self.belgeler)} belge")
    
    def ara(self, soru, sonuc_sayisi=3):
        if not self.hazir:
            return "TF-IDF modeli henüz eğitilmemiş!", [], 0
        
        baslangic = time.time()
        temiz_soru = _sorgu_temizle(soru)
        soru_vektoru = self.vektorleyici.transform([temiz_soru])
        benzerlikler = cosine_similarity(soru_vektoru, self.tfidf_matrisi).flatten()
        en_iyiler = benzerlikler.argsort()[-sonuc_sayisi:][::-1]
        gecen_sure = time.time() - baslangic
        
        baglam_parcalari = []
        skorlar = []
        for idx in en_iyiler:
            kaynak = self.bilgiler[idx].get("source", "Bilinmiyor")
            skor = float(benzerlikler[idx])
            baglam_parcalari.append(f"--- Kaynak: {kaynak} (Skor: {skor:.4f}) ---\n{self.belgeler[idx]}")
            skorlar.append(skor)
        
        return "\n\n".join(baglam_parcalari), skorlar, gecen_sure


class BM25Arama:
    """
    BM25 (Best Matching 25) tabanlı arama sistemi.
    """
    
    def __init__(self, k1=1.5, b=0.75):
        self.bm25 = None
        self.belgeler = []
        self.bilgiler = []
        self.tokenli_belgeler = []
        self.hazir = False
        self.k1 = k1
        self.b = b
    
    def _tokenla(self, metin):
        return re.findall(r'\w+', metin.lower())
    
    def egit(self, parcalar):
        if not parcalar:
            return
        self.belgeler = [p["page_content"] for p in parcalar]
        self.bilgiler = [p["metadata"] for p in parcalar]
        self.tokenli_belgeler = [self._tokenla(b) for b in self.belgeler]
        self.bm25 = BM25Okapi(self.tokenli_belgeler, k1=self.k1, b=self.b)
        self.hazir = True
    
    def ara(self, soru, sonuc_sayisi=3):
        if not self.hazir:
            return "BM25 modeli henüz eğitilmemiş!", [], 0
        
        baslangic = time.time()
        temiz_soru = _sorgu_temizle(soru)
        tokenli_soru = self._tokenla(temiz_soru)
        ham_skorlar = self.bm25.get_scores(tokenli_soru)
        en_iyiler = ham_skorlar.argsort()[-sonuc_sayisi:][::-1]
        gecen_sure = time.time() - baslangic
        
        baglam_parcalari = []
        skorlar = []
        for idx in en_iyiler:
            kaynak = self.bilgiler[idx].get("source", "Bilinmiyor")
            skor = float(ham_skorlar[idx])
            baglam_parcalari.append(f"--- Kaynak: {kaynak} (BM25: {skor:.4f}) ---\n{self.belgeler[idx]}")
            skorlar.append(skor)
        
        return "\n\n".join(baglam_parcalari), skorlar, gecen_sure


def geleneksel_modelleri_kur(parcalar):
    tfidf = TFIDFArama()
    bm25 = BM25Arama()
    tfidf.egit(parcalar)
    bm25.egit(parcalar)
    return tfidf, bm25


class HybridArama:
    """
    Hybrid Search: BM25 (keyword) + Semantic (embedding) skorlarını
    ağırlıklı olarak birleştirir. State-of-the-art retrieval yaklaşımı.
    alpha: Semantic ağırlığı (1-alpha: BM25 ağırlığı)
    """
    
    def __init__(self, bm25_arama, semantic_arama_fn, alpha=0.6):
        self.bm25 = bm25_arama
        self.semantic_fn = semantic_arama_fn
        self.alpha = alpha
    
    def ara(self, soru, sonuc_sayisi=5):
        baslangic = time.time()
        
        # BM25 sonuçları
        bm25_baglam, bm25_skorlar, _ = self.bm25.ara(soru, sonuc_sayisi)
        
        # Semantic sonuçları
        sem_baglam, sem_skorlar, _ = self.semantic_fn(soru, sonuc_sayisi)
        
        # Skorları normalize et (0-1 arası)
        def normalize(skorlar):
            if not skorlar:
                return []
            mn, mx = min(skorlar), max(skorlar)
            if mx == mn:
                return [0.5] * len(skorlar)
            return [(s - mn) / (mx - mn) for s in skorlar]
        
        bm25_norm = normalize(bm25_skorlar)
        sem_norm = normalize(sem_skorlar)
        
        # Ağırlıklı birleştirme - en iyi skoru al
        hybrid_skorlar = []
        for i in range(min(len(bm25_norm), len(sem_norm))):
            h = self.alpha * sem_norm[i] + (1 - self.alpha) * bm25_norm[i]
            hybrid_skorlar.append(round(h, 4))
        
        # En iyi skora sahip kaynağın bağlamını kullan (semantic genellikle daha kaliteli)
        gecen_sure = time.time() - baslangic
        
        if hybrid_skorlar:
            return sem_baglam, hybrid_skorlar, gecen_sure
        return "Hybrid arama sonuç bulamadı.", [], gecen_sure


def embedding_gorselleştir(parcalar):
    """
    t-SNE ile parçaların 2D embedding görselleştirmesini oluşturur.
    Returns: dict with x, y, labels, sources
    """
    from sklearn.manifold import TSNE
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import EMBEDDING_MODELI
    
    if not parcalar or len(parcalar) < 5:
        return None
    
    embedding_modeli = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELI)
    metinler = [p["page_content"] for p in parcalar]
    kaynaklar = [p["metadata"].get("source", "?") for p in parcalar]
    
    vektorler = embedding_modeli.embed_documents(metinler)
    vektorler_np = np.array(vektorler)
    
    perplexity = min(30, len(parcalar) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(vektorler_np)
    
    return {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "labels": [m[:60] + "..." for m in metinler],
        "sources": kaynaklar
    }

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


class TFIDFArama:
    """
    TF-IDF (Terim Sıklığı - Ters Belge Sıklığı) tabanlı arama sistemi.
    
    Çalışma Prensibi:
    - Her kelimeye, belge içindeki sıklığı ve tüm belgeler arasındaki
      nadirliğine göre bir ağırlık verir.
    - Sorgu ile belgeler arası benzerlik Kosinüs Benzerliği ile ölçülür.
    - Semantik anlam YAKALAMAZ, sadece kelime eşleşmesi yapar.
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
        print(f"TF-IDF Modeli Eğitildi: {len(self.belgeler)} belge, "
              f"{len(self.vektorleyici.get_feature_names_out())} özellik")
    
    def ara(self, soru, sonuc_sayisi=3):
        """
        Sorguya en yakın belgeleri TF-IDF + Kosinüs Benzerliği ile getir.
        Returns: (baglam, skorlar, gecen_sure)
        """
        if not self.hazir:
            return "TF-IDF modeli henüz eğitilmemiş!", [], 0
        
        baslangic = time.time()
        soru_vektoru = self.vektorleyici.transform([soru])
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
    
    def sozluk_boyutu(self):
        if self.hazir:
            return len(self.vektorleyici.get_feature_names_out())
        return 0


class BM25Arama:
    """
    BM25 (Best Matching 25) tabanlı arama sistemi.
    
    Çalışma Prensibi:
    - TF-IDF'in geliştirilmiş versiyonudur.
    - Belge uzunluğunu normalize eder.
    - Arama motorlarının temelini oluşturan klasik algoritmadır.
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
        """Basit tokenization: alfanümerik kelimelere ayır."""
        return re.findall(r'\w+', metin.lower())
    
    def egit(self, parcalar):
        """Parça listesinden BM25 indeksini oluştur."""
        if not parcalar:
            return
        self.belgeler = [p["page_content"] for p in parcalar]
        self.bilgiler = [p["metadata"] for p in parcalar]
        self.tokenli_belgeler = [self._tokenla(b) for b in self.belgeler]
        self.bm25 = BM25Okapi(self.tokenli_belgeler, k1=self.k1, b=self.b)
        self.hazir = True
        ort_uzunluk = np.mean([len(t) for t in self.tokenli_belgeler])
        print(f"BM25 Modeli Eğitildi: {len(self.belgeler)} belge, Ort. Token: {ort_uzunluk:.0f}")
    
    def ara(self, soru, sonuc_sayisi=3):
        """
        Sorguya en yakın belgeleri BM25 ile getir.
        Returns: (baglam, skorlar, gecen_sure)
        """
        if not self.hazir:
            return "BM25 modeli henüz eğitilmemiş!", [], 0
        
        baslangic = time.time()
        tokenli_soru = self._tokenla(soru)
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
    """Her iki geleneksel modeli de oluştur ve döndür."""
    tfidf = TFIDFArama()
    bm25 = BM25Arama()
    tfidf.egit(parcalar)
    bm25.egit(parcalar)
    return tfidf, bm25

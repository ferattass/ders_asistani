"""
Değerlendirme ve Metrik Modülü
================================
RAG ve Geleneksel modelleri bilimsel metriklerle karşılaştırır.
Precision@K, Hit Rate, MRR, BLEU, ROUGE-L hesaplar.
"""

import json
import time
import math
import numpy as np
from collections import Counter


# ========== METRİK FONKSİYONLARI ==========

def bleu_hesapla(referans, uretilen, max_n=4):
    """
    BLEU skoru hesaplar.
    Üretilen metnin referans metne n-gram düzeyinde ne kadar benzediğini ölçer.
    """
    ref_kelimeler = referans.lower().split()
    urt_kelimeler = uretilen.lower().split()
    if not urt_kelimeler or not ref_kelimeler:
        return 0.0
    
    hassasiyetler = []
    for n in range(1, max_n + 1):
        ref_ngramlar = Counter(tuple(ref_kelimeler[i:i+n]) for i in range(len(ref_kelimeler) - n + 1))
        urt_ngramlar = Counter(tuple(urt_kelimeler[i:i+n]) for i in range(len(urt_kelimeler) - n + 1))
        if not urt_ngramlar:
            hassasiyetler.append(0)
            continue
        kesisen = sum(min(sayi, ref_ngramlar[ngram]) for ngram, sayi in urt_ngramlar.items())
        toplam = sum(urt_ngramlar.values())
        hassasiyetler.append(kesisen / toplam if toplam > 0 else 0)
    
    if any(h == 0 for h in hassasiyetler):
        return 0.0
    
    kisalik_cezasi = min(1.0, math.exp(1 - len(ref_kelimeler) / max(len(urt_kelimeler), 1)))
    log_ort = sum(math.log(h) for h in hassasiyetler) / max_n
    return kisalik_cezasi * math.exp(log_ort)


def rouge_l_hesapla(referans, uretilen):
    """
    ROUGE-L skoru hesaplar (En Uzun Ortak Alt Dizi tabanlı).
    """
    ref_kelimeler = referans.lower().split()
    urt_kelimeler = uretilen.lower().split()
    if not ref_kelimeler or not urt_kelimeler:
        return {"hassasiyet": 0, "duyarlilik": 0, "f1": 0}
    
    m, n = len(ref_kelimeler), len(urt_kelimeler)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_kelimeler[i-1] == urt_kelimeler[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    eod_uzunluk = dp[m][n]
    hassasiyet = eod_uzunluk / n if n > 0 else 0
    duyarlilik = eod_uzunluk / m if m > 0 else 0
    f1 = (2 * hassasiyet * duyarlilik / (hassasiyet + duyarlilik)) if (hassasiyet + duyarlilik) > 0 else 0
    return {"hassasiyet": round(hassasiyet, 4), "duyarlilik": round(duyarlilik, 4), "f1": round(f1, 4)}


def anahtar_kelime_hassasiyeti(bulunan_metin, beklenen_kelimeler):
    """Getirilen metinde beklenen anahtar kelimelerin ne kadarı var?"""
    if not beklenen_kelimeler:
        return 0.0
    metin_kucuk = bulunan_metin.lower()
    bulunan = sum(1 for kw in beklenen_kelimeler if kw.lower() in metin_kucuk)
    return bulunan / len(beklenen_kelimeler)


def sadakat_skoru(cevap, baglam):
    """Cevabın ne kadarı bağlama dayalı? (Faithfulness)"""
    if not cevap or not baglam:
        return 0.0
    cevap_kelimeleri = {k for k in cevap.lower().split() if len(k) > 2}
    baglam_kelimeleri = set(baglam.lower().split())
    if not cevap_kelimeleri:
        return 0.0
    ortak = cevap_kelimeleri.intersection(baglam_kelimeleri)
    return len(ortak) / len(cevap_kelimeleri)


# ========== DEĞERLENDİRİCİ SINIFI ==========

class Degerlendirici:
    """RAG ve Geleneksel modelleri kapsamlı metriklerle değerlendiren sınıf."""
    
    def __init__(self, test_dosyasi=None):
        self.test_sorulari = []
        if test_dosyasi:
            self.sorulari_yukle(test_dosyasi)
    
    def sorulari_yukle(self, dosya_yolu):
        """Test sorularını JSON dosyasından yükle."""
        try:
            with open(dosya_yolu, 'r', encoding='utf-8') as f:
                self.test_sorulari = json.load(f)
            print(f"✅ {len(self.test_sorulari)} test sorusu yüklendi.")
        except FileNotFoundError:
            print(f"⚠️ Test dosyası bulunamadı: {dosya_yolu}")
        except json.JSONDecodeError:
            print(f"⚠️ JSON parse hatası: {dosya_yolu}")
    
    def aramay_degerlendir(self, arama_fonksiyonu, yontem_adi="Model", sorular=None):
        """
        Bir arama (retrieval) fonksiyonunu test sorularıyla değerlendir.
        arama_fonksiyonu: soru alan → (baglam, skorlar, sure) döndüren fonksiyon
        """
        if sorular is None:
            sorular = self.test_sorulari
        if not sorular:
            return {"hata": "Test sorusu bulunamadı!"}
        
        sonuclar = {
            "yontem": yontem_adi,
            "hassasiyet_listesi": [],
            "isabet_listesi": [],
            "sure_listesi": [],
            "soru_detaylari": []
        }
        
        for s in sorular:
            baslangic = time.time()
            cikti = arama_fonksiyonu(s["question"])
            gecen = time.time() - baslangic
            
            if isinstance(cikti, tuple):
                baglam_metni = cikti[0]
                skorlar = cikti[1] if len(cikti) > 1 else []
                gecen = cikti[2] if len(cikti) > 2 else gecen
            else:
                baglam_metni = str(cikti)
                skorlar = []
            
            hassasiyet = anahtar_kelime_hassasiyeti(baglam_metni, s.get("relevant_keywords", []))
            isabet = 1.0 if hassasiyet > 0 else 0.0
            
            sonuclar["hassasiyet_listesi"].append(hassasiyet)
            sonuclar["isabet_listesi"].append(isabet)
            sonuclar["sure_listesi"].append(gecen)
            sonuclar["soru_detaylari"].append({
                "soru": s["question"],
                "hassasiyet": round(hassasiyet, 4),
                "isabet": isabet,
                "sure_ms": round(gecen * 1000, 2),
                "en_iyi_skor": round(skorlar[0], 4) if skorlar else 0,
                "onizleme": baglam_metni[:300] if isinstance(baglam_metni, str) else ""
            })
        
        sonuclar["ort_hassasiyet"] = round(float(np.mean(sonuclar["hassasiyet_listesi"])), 4)
        sonuclar["ort_isabet"] = round(float(np.mean(sonuclar["isabet_listesi"])), 4)
        sonuclar["ort_sure_ms"] = round(float(np.mean(sonuclar["sure_listesi"])) * 1000, 2)
        sonuclar["medyan_sure_ms"] = round(float(np.median(sonuclar["sure_listesi"])) * 1000, 2)
        sonuclar["toplam_soru"] = len(sorular)
        return sonuclar
    
    def uretimi_degerlendir(self, uretim_fonksiyonu, arama_fonksiyonu, sorular=None):
        """Cevap üretme kalitesini BLEU, ROUGE-L, Faithfulness ile değerlendir."""
        if sorular is None:
            sorular = self.test_sorulari
        
        sonuclar = {
            "bleu_listesi": [], "rouge_listesi": [],
            "sadakat_listesi": [], "sure_listesi": [],
            "soru_detaylari": []
        }
        
        for s in sorular:
            beklenen = s.get("expected_answer", "")
            if not beklenen:
                continue
            
            cikti = arama_fonksiyonu(s["question"])
            baglam = cikti[0] if isinstance(cikti, tuple) else str(cikti)
            
            baslangic = time.time()
            cevap = uretim_fonksiyonu(s["question"], baglam)
            gecen = time.time() - baslangic
            
            if not cevap:
                continue
            
            bleu = bleu_hesapla(beklenen, cevap)
            rouge = rouge_l_hesapla(beklenen, cevap)
            sadakat = sadakat_skoru(cevap, baglam)
            
            sonuclar["bleu_listesi"].append(bleu)
            sonuclar["rouge_listesi"].append(rouge["f1"])
            sonuclar["sadakat_listesi"].append(sadakat)
            sonuclar["sure_listesi"].append(gecen)
            sonuclar["soru_detaylari"].append({
                "soru": s["question"],
                "beklenen": beklenen[:100],
                "uretilen": cevap[:200],
                "bleu": round(bleu, 4),
                "rouge_l": round(rouge["f1"], 4),
                "sadakat": round(sadakat, 4),
                "sure_ms": round(gecen * 1000, 2)
            })
        
        sonuclar["ort_bleu"] = round(float(np.mean(sonuclar["bleu_listesi"])), 4) if sonuclar["bleu_listesi"] else 0
        sonuclar["ort_rouge_l"] = round(float(np.mean(sonuclar["rouge_listesi"])), 4) if sonuclar["rouge_listesi"] else 0
        sonuclar["ort_sadakat"] = round(float(np.mean(sonuclar["sadakat_listesi"])), 4) if sonuclar["sadakat_listesi"] else 0
        sonuclar["ort_sure_ms"] = round(float(np.mean(sonuclar["sure_listesi"])) * 1000, 2) if sonuclar["sure_listesi"] else 0
        sonuclar["degerlendirilen"] = len(sonuclar["soru_detaylari"])
        return sonuclar
    
    def karsilastirma_tablosu(self, yontem_sonuclari):
        """Birden fazla yöntemin sonuçlarını tablo formatında döndür."""
        satirlar = []
        for yontem, sonuc in yontem_sonuclari.items():
            satirlar.append({
                "Yöntem": yontem,
                "Ort. Hassasiyet": sonuc.get("ort_hassasiyet", 0),
                "İsabet Oranı": sonuc.get("ort_isabet", 0),
                "Ort. Süre (ms)": sonuc.get("ort_sure_ms", 0),
                "Medyan Süre (ms)": sonuc.get("medyan_sure_ms", 0),
            })
        return satirlar

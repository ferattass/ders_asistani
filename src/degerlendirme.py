"""
Değerlendirme ve Metrik Modülü
================================
RAG ve Geleneksel modelleri bilimsel metriklerle karşılaştırır.
Precision@K, Hit Rate, MRR, NDCG, BLEU, ROUGE-L hesaplar.
"""

import json
import time
import math
import re
import numpy as np
from collections import Counter


# ========== YARDIMCI: BAĞLAMI PARÇALARA ÇEVİR ==========

def baglami_parcala(baglam_metni):
    """
    '--- Kaynak: X ---\nmetin' formatındaki bağlamı parça listesine böler.
    Her parça dict: {metin, kaynak}
    """
    bolumler = re.split(r'---\s*Kaynak:.*?---', baglam_metni)
    parcalar = [p.strip() for p in bolumler if len(p.strip()) > 20]
    if not parcalar:
        return [{"metin": baglam_metni.strip(), "kaynak": "?"}]
    return [{"metin": p, "kaynak": f"Parça-{i+1}"} for i, p in enumerate(parcalar)]


# ========== METRİK FONKSİYONLARI ==========

def bleu_hesapla(referans, uretilen, max_n=4):
    """BLEU skoru — üretilen metnin referansa n-gram benzerliği."""
    ref_k = referans.lower().split()
    urt_k = uretilen.lower().split()
    if not urt_k or not ref_k:
        return 0.0
    hassasiyetler = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(tuple(ref_k[i:i+n]) for i in range(len(ref_k)-n+1))
        urt_ng = Counter(tuple(urt_k[i:i+n]) for i in range(len(urt_k)-n+1))
        if not urt_ng:
            hassasiyetler.append(0)
            continue
        kesisen = sum(min(c, ref_ng[ng]) for ng, c in urt_ng.items())
        toplam = sum(urt_ng.values())
        hassasiyetler.append(kesisen / toplam if toplam > 0 else 0)
    if any(h == 0 for h in hassasiyetler):
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_k) / max(len(urt_k), 1)))
    log_ort = sum(math.log(h) for h in hassasiyetler) / max_n
    return bp * math.exp(log_ort)


def rouge_l_hesapla(referans, uretilen):
    """ROUGE-L — En Uzun Ortak Alt Dizi (LCS) tabanlı F1 skoru."""
    ref_k = referans.lower().split()
    urt_k = uretilen.lower().split()
    if not ref_k or not urt_k:
        return {"hassasiyet": 0, "duyarlilik": 0, "f1": 0}
    m, n = len(ref_k), len(urt_k)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if ref_k[i-1]==urt_k[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs/n if n else 0
    r = lcs/m if m else 0
    f1 = (2*p*r/(p+r)) if (p+r) else 0
    return {"hassasiyet": round(p,4), "duyarlilik": round(r,4), "f1": round(f1,4)}


def anahtar_kelime_hassasiyeti(bulunan_metin, beklenen_kelimeler):
    """Getirilen metinde beklenen anahtar kelimelerin oranı (Precision)."""
    if not beklenen_kelimeler:
        return 0.0
    metin_k = bulunan_metin.lower()
    bulunan = sum(1 for kw in beklenen_kelimeler if kw.lower() in metin_k)
    return bulunan / len(beklenen_kelimeler)


def ortalama_hassasiyet_hesapla(baglam_metni, beklenen_kelimeler):
    """
    Average Precision (AP) hesaplar.
    Parçalar üzerinde gezerken hassasiyetin ortalamasını alır.
    """
    if not beklenen_kelimeler:
        return 0.0
    parcalar = baglami_parcala(baglam_metni)
    skorlar = []
    bulunan_sayisi = 0
    for i, p in enumerate(parcalar, 1):
        metin_k = p["metin"].lower()
        if any(kw.lower() in metin_k for kw in beklenen_kelimeler):
            bulunan_sayisi += 1
            skorlar.append(bulunan_sayisi / i)
    return np.mean(skorlar) if skorlar else 0.0


def mrr_hesapla(baglam_metni, beklenen_kelimeler):
    """
    Mean Reciprocal Rank — ilk alakalı parçanın sırasının tersi.
    TF-IDF/BM25/RAG'ın ilk chunk'larını ayrı ayrı değerlendirir.
    1.0 = ilk parça doğru, 0.5 = 2. parça doğru, 0.33 = 3. parça doğru.
    """
    if not beklenen_kelimeler:
        return 0.0
    parcalar = baglami_parcala(baglam_metni)
    for rank, parca in enumerate(parcalar, 1):
        metin_k = parca["metin"].lower()
        if any(kw.lower() in metin_k for kw in beklenen_kelimeler):
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_hesapla(baglam_metni, beklenen_kelimeler, k=3):
    """
    NDCG@K — sıralamayı dikkate alan ağırlıklı hassasiyet.
    İlk sıradaki doğru sonuç daha yüksek katkı sağlar.
    """
    if not beklenen_kelimeler:
        return 0.0
    parcalar = baglami_parcala(baglam_metni)[:k]
    dcg = 0.0
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(beklenen_kelimeler), k)))
    for i, parca in enumerate(parcalar):
        metin_k = parca["metin"].lower()
        ilgi = anahtar_kelime_hassasiyeti(parca["metin"], beklenen_kelimeler)
        dcg += ilgi / math.log2(i+2)
    return round(dcg/idcg, 4) if idcg > 0 else 0.0


def sadakat_skoru(cevap, baglam):
    """Cevabın bağlama sözcük örtüşme sadakati (Faithfulness)."""
    if not cevap or not baglam:
        return 0.0
    cevap_k = {k for k in cevap.lower().split() if len(k) > 2}
    baglam_k = set(baglam.lower().split())
    if not cevap_k:
        return 0.0
    return round(len(cevap_k & baglam_k) / len(cevap_k), 4)


def context_ozgunlugu(baglamlar_dict):
    """
    Her yöntemin bağlamındaki özgün token oranını hesaplar.
    baglamlar_dict: {"TF-IDF": metin, "BM25": metin, "RAG": metin}
    Returns: {"TF-IDF": 0.xx, ...}
    """
    def token_seti(metin):
        return set(w.lower() for w in re.findall(r'\w+', metin) if len(w) > 3)

    token_setleri = {y: token_seti(b) for y, b in baglamlar_dict.items()}
    yontemler = list(token_setleri.keys())
    ozgunlukler = {}
    for y in yontemler:
        diger = set().union(*[token_setleri[d] for d in yontemler if d != y])
        ozgun = token_setleri[y] - diger
        toplam = len(token_setleri[y])
        ozgunlukler[y] = round(len(ozgun)/toplam, 4) if toplam else 0.0
    return ozgunlukler


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
        Bir arama fonksiyonunu test sorularıyla değerlendir.
        Hesaplanan metrikler: Precision, Hit Rate, MRR, NDCG@3, Yanıt Süresi.
        """
        if sorular is None:
            sorular = self.test_sorulari
        if not sorular:
            return {"hata": "Test sorusu bulunamadı!"}

        sonuclar = {
            "yontem": yontem_adi,
            "hassasiyet_listesi": [],
            "isabet_listesi": [],
            "mrr_listesi": [],
            "ndcg_listesi": [],
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

            kw = s.get("relevant_keywords", [])
            hassasiyet = anahtar_kelime_hassasiyeti(baglam_metni, kw)
            isabet = 1.0 if hassasiyet > 0 else 0.0
            mrr = mrr_hesapla(baglam_metni, kw)
            ndcg = ndcg_hesapla(baglam_metni, kw)
            ap = ortalama_hassasiyet_hesapla(baglam_metni, kw)

            sonuclar["hassasiyet_listesi"].append(hassasiyet)
            sonuclar["isabet_listesi"].append(isabet)
            sonuclar["mrr_listesi"].append(mrr)
            sonuclar["ndcg_listesi"].append(ndcg)
            sonuclar["ap_listesi"] = sonuclar.get("ap_listesi", []) + [ap]
            sonuclar["sure_listesi"].append(gecen)
            
            # F1 hesapla (Hassasiyet + İsabet kombinasyonu olarak)
            f1 = (2 * hassasiyet * isabet) / (hassasiyet + isabet) if (hassasiyet + isabet) > 0 else 0.0

            sonuclar["soru_detaylari"].append({
                "soru": s["question"],
                "hassasiyet": round(hassasiyet, 4),
                "isabet": isabet,
                "f1": round(f1, 4),
                "mrr": mrr,
                "ndcg": ndcg,
                "map": round(ap, 4),
                "sure_ms": round(gecen * 1000, 2),
                "en_iyi_skor": round(float(skorlar[0]), 4) if skorlar else 0.0,
                "onizleme": baglam_metni[:300] if isinstance(baglam_metni, str) else ""
            })

        def _ort(lst): return round(float(np.mean(lst)), 4) if lst else 0.0

        sonuclar["ort_hassasiyet"] = _ort(sonuclar["hassasiyet_listesi"])
        sonuclar["ort_isabet"] = _ort(sonuclar["isabet_listesi"])
        sonuclar["ort_mrr"] = _ort(sonuclar["mrr_listesi"])
        sonuclar["ort_ndcg"] = _ort(sonuclar["ndcg_listesi"])
        sonuclar["ort_map"] = _ort(sonuclar.get("ap_listesi", []))
        sonuclar["ort_f1"] = _ort([d["f1"] for d in sonuclar["soru_detaylari"]])
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

        def _ort(lst): return round(float(np.mean(lst)), 4) if lst else 0
        sonuclar["ort_bleu"] = _ort(sonuclar["bleu_listesi"])
        sonuclar["ort_rouge_l"] = _ort(sonuclar["rouge_listesi"])
        sonuclar["ort_sadakat"] = _ort(sonuclar["sadakat_listesi"])
        sonuclar["ort_sure_ms"] = round(float(np.mean(sonuclar["sure_listesi"]))*1000, 2) if sonuclar["sure_listesi"] else 0
        sonuclar["degerlendirilen"] = len(sonuclar["soru_detaylari"])
        return sonuclar

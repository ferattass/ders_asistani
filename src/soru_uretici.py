"""
Otomatik Test Sorusu Üretici
===============================
PDF parçalarından Gemini API kullanarak
test soruları, beklenen cevaplar ve anahtar kelimeler üretir.
"""

import json
import google.generativeai as genai
import config


def test_sorulari_uret(parcalar, soru_sayisi=15):
    """
    Doküman parçalarından Gemini ile otomatik test soruları üretir.
    
    Args:
        parcalar: Veritabanındaki doküman parçaları listesi
        soru_sayisi: Üretilecek soru sayısı
    
    Returns:
        list: Test soruları listesi (her biri dict: question, relevant_keywords, expected_answer)
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        print("⚠️ Gemini API Key bulunamadı!")
        return []
    
    # Parçalardan temsili bir örneklem al (tüm içeriği göndermek çok uzun olabilir)
    toplam_metin = ""
    adim = max(1, len(parcalar) // 20)  # En fazla 20 parça al
    for i in range(0, len(parcalar), adim):
        metin = parcalar[i]["page_content"]
        toplam_metin += metin[:500] + "\n---\n"
    
    # Metni makul uzunlukta tut (Gemini token limiti)
    toplam_metin = toplam_metin[:8000]
    
    prompt = f"""Aşağıda bir ders notunun parçaları var. Bu içerikten TAM OLARAK {soru_sayisi} adet test sorusu üret.

HER SORU İÇİN:
1. "question": Türkçe, net ve cevaplanabilir bir soru
2. "relevant_keywords": Bu sorunun cevabında geçmesi gereken 5-7 anahtar kelime (küçük harf, Türkçe)
3. "expected_answer": 1-2 cümlelik beklenen cevap

KRİTİK KURALLAR:
- Sorular SADECE verilen metindeki bilgilere dayanmalı
- Genel bilgi soruları SORMA, sadece metinde geçen konuları sor
- Anahtar kelimeler metinde gerçekten geçen kelimeler olmalı
- Her soru farklı bir konuyu kapsamalı

ÇIKTI FORMATI: Sadece geçerli JSON array döndür, başka hiçbir şey yazma.
Markdown kod bloğu (```) KULLANMA. Düz JSON döndür.

[
  {{
    "question": "...",
    "relevant_keywords": ["...", "..."],
    "expected_answer": "..."
  }}
]

DERS NOTU İÇERİĞİ:
{toplam_metin}
"""
    
    try:
        genai.configure(api_key=api_key)
        mevcut_modeller = [
            m.name for m in genai.list_models()
            if 'generateContent' in m.supported_generation_methods
        ]
        if not mevcut_modeller:
            print("⚠️ Kullanılabilir Gemini modeli bulunamadı!")
            return []
        
        secilen = mevcut_modeller[-1] if 'gemini' in mevcut_modeller[-1] else mevcut_modeller[0]
        model = genai.GenerativeModel(secilen)
        yanit = model.generate_content(prompt)
        
        # JSON parse et
        metin = yanit.text.strip()
        # Markdown kod bloğu varsa temizle
        if metin.startswith("```"):
            metin = metin.split("\n", 1)[1]  # İlk satırı at
            if metin.endswith("```"):
                metin = metin[:-3]
            metin = metin.strip()
        
        sorular = json.loads(metin)
        
        # Validasyon
        gecerli = []
        for s in sorular:
            if all(k in s for k in ["question", "relevant_keywords", "expected_answer"]):
                if isinstance(s["relevant_keywords"], list) and len(s["relevant_keywords"]) >= 2:
                    gecerli.append(s)
        
        return gecerli
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse hatası: {e}")
        print(f"Ham yanıt: {metin[:500]}")
        return []
    except Exception as e:
        print(f"⚠️ Soru üretme hatası: {e}")
        return []


def sorulari_kaydet(sorular, dosya_yolu=None):
    """Üretilen soruları JSON dosyasına kaydeder."""
    if dosya_yolu is None:
        dosya_yolu = config.TEST_SORULARI_DOSYASI
    
    with open(dosya_yolu, 'w', encoding='utf-8') as f:
        json.dump(sorular, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {len(sorular)} soru kaydedildi: {dosya_yolu}")
    return dosya_yolu

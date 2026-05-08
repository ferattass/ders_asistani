"""
Arama ve Cevap Üretme Modülü
===============================
Soruyu vektörleştirip ChromaDB'den en yakın parçaları bulur (Retrieval).
Bulunan bağlam ile LLM'den cevap üretir (Generation).
"""

import time
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from config import EMBEDDING_MODELI
import config
from src.vektor_deposu import koleksiyonu_getir


def baglamlari_getir(soru, sonuc_sayisi=5):
    """
    Soruya anlamsal olarak en yakın metin parçalarını ChromaDB'den getirir.
    Returns: (baglam_metni, benzerlik_skorlari, gecen_sure)
    """
    koleksiyon = koleksiyonu_getir()
    if not koleksiyon:
        return "Veritabanı oluşturulmamış! (ChromaDB yok)", [], 0
    
    baslangic = time.time()
    
    # Sorgu Temizleme (Noise Reduction)
    temiz_soru = soru.lower()
    stop_words = ["nedir", "nelerdir", "neden", "niçin", "nasıl", "kimdir", "nedir?", "nelerdir?"]
    for word in stop_words:
        temiz_soru = temiz_soru.replace(word, "").strip()
    
    embedding_modeli = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELI)
    soru_vektoru = embedding_modeli.embed_query(temiz_soru if temiz_soru else soru)
    
    sonuclar = koleksiyon.query(
        query_embeddings=[soru_vektoru],
        n_results=sonuc_sayisi,
        include=["documents", "metadatas", "distances"]
    )
    gecen_sure = time.time() - baslangic
    
    if sonuclar and "documents" in sonuclar and sonuclar["documents"]:
        bulunan_metinler = sonuclar["documents"][0]
        bulunan_bilgiler = sonuclar["metadatas"][0] if "metadatas" in sonuclar else []
        mesafeler = sonuclar["distances"][0] if "distances" in sonuclar else []
        skorlar = [round(1 / (1 + d), 4) for d in mesafeler] if mesafeler else []
        
        baglam_parcalari = []
        for i, metin in enumerate(bulunan_metinler):
            kaynak = bulunan_bilgiler[i].get("source", "Bilinmiyor") if i < len(bulunan_bilgiler) else "Bilinmiyor"
            skor = skorlar[i] if i < len(skorlar) else 0
            baglam_parcalari.append(f"--- Kaynak: {kaynak} (Benzerlik: {skor}) ---\n{metin}")
        
        return "\n\n".join(baglam_parcalari), skorlar, gecen_sure
    
    return "Soruya uygun bir veri bulunamadı.", [], gecen_sure


def cevap_uret(soru, baglam):
    """Bulunan bağlam ile Google Gemini API kullanarak cevap üretir."""
    api_anahtari = config.GEMINI_API_KEY
    if not api_anahtari:
        return (
            f"🚨 **[Sistemde API Key Yok]**\n\n"
            f"Üretim Aşaması Çalışmadı, sadece bulunan bağlamı paylaşıyorum:\n\n{baglam}"
        )
    try:
        genai.configure(api_key=api_anahtari)
        mevcut_modeller = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        if not mevcut_modeller:
            return "Hata: API anahtarınız aktif bir modeli desteklemiyor."
        secilen = mevcut_modeller[-1] if 'gemini' in mevcut_modeller[-1] else mevcut_modeller[0]
        model = genai.GenerativeModel(secilen)
        
        istem = f"""
        Aşağıdaki ders notlarına dayanarak kullanıcının sorusunu cevapla. 
        Cevabı metinlerde yoksa 'Bu bilgi ders notlarında bulunmamaktadır' de. Kendinden bilgi ekleme.
        
        Bağlam (Ders Notları):
        {baglam}
        
        Soru: {soru}
        Cevap:
        """
        yanit = model.generate_content(istem)
        return yanit.text
    except Exception as e:
        return f"LLM Cevap Üretim Hatası: {str(e)}"

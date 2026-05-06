"""
Vektör Veritabanı Modülü
==========================
Metin parçalarını vektörlere dönüştürüp ChromaDB'ye kaydeder.
"""

from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from config import EMBEDDING_MODELI, VERITABANI_KLASORU, KOLEKSIYON_ADI


def veritabani_olustur(parcalar):
    """Parçalanmış metinleri vektörlere çevirip ChromaDB'ye kaydeder."""
    if not parcalar:
        print("Uyarı: Veritabanına kaydedilecek parça yok.")
        return None

    print(f"Embedding Başladı ({EMBEDDING_MODELI})... Lütfen Bekleyin.")
    embedding_modeli = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELI)
    istemci = chromadb.PersistentClient(path=VERITABANI_KLASORU)
    
    try:
        istemci.delete_collection(name=KOLEKSIYON_ADI)
    except Exception:
        pass
        
    koleksiyon = istemci.create_collection(name=KOLEKSIYON_ADI)
    
    metinler, bilgiler, kimlikler = [], [], []
    for i, parca in enumerate(parcalar):
        metinler.append(parca["page_content"])
        temiz_bilgi = {k: str(v) for k, v in parca["metadata"].items()}
        bilgiler.append(temiz_bilgi)
        kimlikler.append(f"parca_{i}")
    
    print(f"Toplam {len(metinler)} vektör oluşturuluyor...")
    vektorler = embedding_modeli.embed_documents(metinler)
    
    koleksiyon.add(
        documents=metinler, embeddings=vektorler,
        metadatas=bilgiler, ids=kimlikler
    )
    print("Vektör Veritabanı Başarıyla Oluşturuldu!")
    return koleksiyon


def koleksiyonu_getir():
    """Mevcut ChromaDB koleksiyonunu döndürür."""
    istemci = chromadb.PersistentClient(path=VERITABANI_KLASORU)
    try:
        return istemci.get_collection(name=KOLEKSIYON_ADI)
    except Exception:
        return None


def tum_dokumanlari_getir():
    """ChromaDB'deki tüm dökümanları çeker (geleneksel modellere beslemek için)."""
    koleksiyon = koleksiyonu_getir()
    if not koleksiyon:
        return []
    try:
        sonuclar = koleksiyon.get()
        return [
            {"page_content": metin, "metadata": sonuclar["metadatas"][i] if sonuclar["metadatas"] else {}}
            for i, metin in enumerate(sonuclar["documents"])
        ]
    except Exception:
        return []

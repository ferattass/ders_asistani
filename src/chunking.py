"""
Metin Parçalama ve Anlamlandırma Modülü
=========================================
Uzun metinleri küçük parçalara (chunk) böler.
spaCy ile NER (İsimli Varlık Tanıma) uygular.
"""

import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PARCA_BOYUTU, PARCA_KESISIMI, SPACY_MODELI


def dokumanlari_parcala(dokumanlar):
    """
    Belgeleri belirlenen boyutlarda küçük parçalara böler.
    RecursiveCharacterTextSplitter: Önce paragraflardan, sonra satırlardan,
    en son kelimelerden böler (akıllı parçalama).
    """
    if not dokumanlar:
        print("Uyarı: Bölünecek doküman yok.")
        return []

    # Langchain'in akıllı parçalayıcısı
    parcalayici = RecursiveCharacterTextSplitter(
        chunk_size=PARCA_BOYUTU,
        chunk_overlap=PARCA_KESISIMI,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    parcalar = []
    
    # SpaCy modelini yükle (NER - İsimli Varlık Tanıma için)
    try:
        nlp = spacy.load(SPACY_MODELI)
        print(f"spaCy Modeli yüklendi: {SPACY_MODELI}")
    except OSError:
        print(f"Uyarı: {SPACY_MODELI} modeli sistemde yok. Varlık Analizi atlanacak.")
        nlp = None

    for dokuman in dokumanlar:
        # Metni parçalara böl
        bolumler = parcalayici.split_text(dokuman["page_content"])
        
        for sira, bolum in enumerate(bolumler):
            parca_bilgisi = dokuman["metadata"].copy()
            parca_bilgisi["chunk_index"] = sira
            
            # NER: Parça içindeki varlıkları bul (Kişi, Kurum, Yer, Tarih)
            varliklar = []
            if nlp:
                nlp_sonucu = nlp(bolum[:1000])  # İlk 1000 karaktere bak (hız için)
                varliklar = [
                    varlik.text for varlik in nlp_sonucu.ents 
                    if varlik.label_ in ["PERSON", "ORG", "GPE", "DATE"]
                ]
                
                if varliklar:
                    parca_bilgisi["entities"] = ", ".join(list(set(varliklar)))
            
            parcalar.append({
                "page_content": bolum,
                "metadata": parca_bilgisi
            })
            
    print(f"Bölme İşlemi Bitti: {len(dokumanlar)} dosyadan {len(parcalar)} adet parça elde edildi.")
    return parcalar


if __name__ == "__main__":
    from src.ingestion import klasorden_oku
    ornek_dokumanlar = klasorden_oku()
    if ornek_dokumanlar:
        print(f"Örnek metin ilk 500 karakter:\n{ornek_dokumanlar[0]['page_content'][:500]}")
        parcalar = dokumanlari_parcala(ornek_dokumanlar)
        if parcalar:
            print("---")
            print(f"İlk Parça Bilgisi: {parcalar[0]['metadata']}")
            print(f"İlk Parça İçerik: {parcalar[0]['page_content'][:100]}...")

"""
Metin Parçalama ve Anlamlandırma Modülü
=========================================
Uzun metinleri küçük parçalara (chunk) böler.
spaCy ile NER (İsimli Varlık Tanıma) uygular.
"""

import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PARCA_BOYUTU, PARCA_KESISIMI, SPACY_MODELI


def dokumanlari_parcala(dokumanlar, ner_aktif=True, parca_boyutu=None):
    """
    Belgeleri belirlenen boyutlarda küçük parçalara böler.
    ner_aktif: spaCy NER varlık tanıma açık/kapalı
    parca_boyutu: Parça boyutu (None ise config'den alır)
    """
    if not dokumanlar:
        print("Uyarı: Bölünecek doküman yok.")
        return []

    if parca_boyutu is None:
        parca_boyutu = PARCA_BOYUTU

    parcalayici = RecursiveCharacterTextSplitter(
        chunk_size=parca_boyutu,
        chunk_overlap=PARCA_KESISIMI,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    parcalar = []
    
    nlp = None
    if ner_aktif:
        try:
            nlp = spacy.load(SPACY_MODELI)
            print(f"spaCy Modeli yüklendi: {SPACY_MODELI}")
        except OSError:
            print(f"Uyarı: {SPACY_MODELI} modeli sistemde yok. Varlık Analizi atlanacak.")

    for dokuman in dokumanlar:
        bolumler = parcalayici.split_text(dokuman["page_content"])
        for sira, bolum in enumerate(bolumler):
            parca_bilgisi = dokuman["metadata"].copy()
            parca_bilgisi["chunk_index"] = sira
            varliklar = []
            if nlp:
                nlp_sonucu = nlp(bolum[:1000])
                varliklar = [
                    v.text for v in nlp_sonucu.ents 
                    if v.label_ in ["PERSON", "ORG", "GPE", "DATE"]
                ]
                if varliklar:
                    parca_bilgisi["entities"] = ", ".join(list(set(varliklar)))
            parcalar.append({
                "page_content": bolum,
                "metadata": parca_bilgisi
            })
            
    print(f"Bölme Bitti: {len(dokumanlar)} dosyadan {len(parcalar)} parça elde edildi.")
    return parcalar

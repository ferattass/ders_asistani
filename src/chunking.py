import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, SPACY_MODEL

def split_documents(docs):
    """
    Belgeleri belirlenen boyutlarda küçük chunk'lara böler.
    Roadmap hedefine uygun olarak "Heading-aware" mantığına en yakın olan
    RecursiveCharacterTextSplitter kullanılarak, önce paragraflardan,
    sonra satırlardan, en son kelimelerden bölünür.
    """
    if not docs:
        print("Uyarı: Bölünecek doküman yok.")
        return []

    # Langchain'in akıllı parçalayıcısı
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Kendi raw verimizi Langchain metotuna uygun listeye çevirmiştik (page_content ve metadata)
    # Burada doğrudan string vererek kendi custom dict/object yapişimizi kullanabiliriz.
    
    chunks = []
    
    # SpaCy modelini yükle (NER - Named Entity Recognition için opsiyonel vizyon katar)
    try:
        nlp = spacy.load(SPACY_MODEL)
        print(f"Spacy Modeli yüklendi: {SPACY_MODEL}")
    except OSError:
        print(f"Uyarı: {SPACY_MODEL} modeli sistemde yok. Özel Konsept Analizi atlanacak.")
        nlp = None

    for doc in docs:
        # Metni böl
        splits = text_splitter.split_text(doc["page_content"])
        
        for i, split in enumerate(splits):
            chunk_metadata = doc["metadata"].copy()
            chunk_metadata["chunk_index"] = i
            
            # Opsiyonel NER: Chunk içindeki varlıkları bul (Kişi, Organizasyon, vb.)
            entities = []
            if nlp:
                # Sadece belli uzunluktaki metinleri analiz edelim hız için
                doc_nlp = nlp(split[:1000]) # İlk 1000 karaktere bak (örnek limit)
                entities = [ent.text for ent in doc_nlp.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]
                
                if entities:
                    # Virgülle ayırıp metadata'ya ekle ki arama esnasında faydası olsun
                    chunk_metadata["entities"] = ", ".join(list(set(entities)))
            
            chunks.append({
                "page_content": split,
                "metadata": chunk_metadata
            })
            
    print(f"Bölme İşlemi Bitti: {len(docs)} dosyadan {len(chunks)} adet chunk elde edildi.")
    return chunks

if __name__ == "__main__":
    # Test Modu
    from src.ingestion import load_documents_from_directory
    sample_docs = load_documents_from_directory()
    if sample_docs:
        print(f"Örnek metin ilk 500 karakter:\n{sample_docs[0]['page_content'][:500]}")
        chunks = split_documents(sample_docs)
        if chunks:
            print("---")
            print(f"İlk Chunk Metadata: {chunks[0]['metadata']}")
            print(f"İlk Chunk İçerik: {chunks[0]['page_content'][:100]}...")

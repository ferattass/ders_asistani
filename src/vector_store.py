import os
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from config import EMBEDDING_MODEL_NAME, DB_DIR

def build_vector_store(chunks):
    """
    Parçalanmış chunk'ları (vektörleri) oluşturup ChromaDB veritabanına kaydeder.
    """
    if not chunks:
        print("Uyarı: Veritabanına kaydedilecek chunk yok.")
        return None

    print(f"Embedding İşlemi Başladı ({EMBEDDING_MODEL_NAME})... Lütfen Bekleyin.")
    
    # Langchain HuggingFace Embeddings wrapper'ı
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Chroma veritabanı (Persistent -> SSD'de kalıcı)
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Koleksiyonu yarat veya sıfırla (üstüne yazmamak için siliyoruz, taze yapıyoruz)
    collection_name = "course_notes"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass # Yoksa hata verir, sorun değil
        
    collection = client.create_collection(name=collection_name)
    
    # Verileri ChromaDB'nin beklediği formata sok (Listeler vs.)
    documents = []
    metadatas = []
    ids = []
    
    for i, c in enumerate(chunks):
        documents.append(c["page_content"])
        
        # Chroma Database'de metadata dict içinde int veya string barınabilir
        # chunk_index gibi sayısal verileri tutabiliriz, emin olmak için stringe çevirebiliriz.
        cleaned_meta = {}
        for k, v in c["metadata"].items():
            cleaned_meta[k] = str(v)
            
        metadatas.append(cleaned_meta)
        ids.append(f"chunk_{i}")
    
    # Doğrudan Langchain üzerinden yapmak yerine, roadmap'teki gibi ChromaDB Core kütüphanesini kullanıyoruz:
    # collection.add kullanarak embed and store işlemi
    # HuggingFaceEmbeddings'i manuel çalıştırarak encode etmemiz gerek Chroma'ya özel yapıyorsak.
    # Ancak ChromaDB kendi yerleşik default ayarlarıyla (all-MiniLM-L6-v2) da text ekleyince otomatik yapar.
    # Yine de Langchain'in modelinden vektör çarkını kendimiz döndürebiliriz:
    
    print(f"Toplam {len(documents)} vektör oluşturuluyor ve diske yazılıyor...")
    embedding_vectors = embeddings.embed_documents(documents)
    
    collection.add(
        documents=documents,
        embeddings=embedding_vectors,
        metadatas=metadatas,
        ids=ids
    )
    
    print("Vektör DB Başarıyla Oluşturuldu!")
    return collection

def get_chroma_collection():
    """
    Daha sonra arama yapmak için mevcut koleksiyonu döndürür.
    """
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        return client.get_collection(name="course_notes")
    except Exception:
        return None

if __name__ == "__main__":
    from src.ingestion import load_documents_from_directory
    from src.chunking import split_documents    
    docs = load_documents_from_directory()
    chunks = split_documents(docs)
    build_vector_store(chunks)

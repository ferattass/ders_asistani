import os
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from config import GEMINI_API_KEY, EMBEDDING_MODEL_NAME
from src.vector_store import get_chroma_collection

def retrieve_answers(query, n_results=3):
    """
    Belirtilen soruya (query) en yakın olan metin parçalarını Getirir (Retrieval).
    """
    collection = get_chroma_collection()
    if not collection:
        return "Veritabanı oluşturulmamış! (ChromaDB yok)"
        
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    query_vector = embeddings.embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    
    # ChromaDB'nin sonuç formatından belgeleri çıkart
    if results and "documents" in results and results["documents"]:
        retrieved_texts = results["documents"][0] # En iyi eşleşen x chunk
        retrieved_metadatas = results["metadatas"][0] if "metadatas" in results else []
        
        context_parts = []
        for i, text in enumerate(retrieved_texts):
            source = retrieved_metadatas[i].get("source", "Bilinmiyor") if i < len(retrieved_metadatas) else "Bilinmiyor"
            context_parts.append(f"--- Kaynak: {source} ---\n{text}")
            
        full_context = "\n\n".join(context_parts)
        return full_context
    return "Soruya uygun bir veri bulunamadı."

def generate_answer(query, context):
    """
    Kullanıcıya mantıklı cevap vermek için (Generation) Google Gemini API kullanır.
    Eğer Anahtar (Key) yoksa, sadece ulaştığı metnin kendisini gösterir.
    """
    if not GEMINI_API_KEY:
        # API girilmemişse, bir LLM varmış gibi davran ama context'i ver!
        return f"🚨 [Sistemde API Key Yok] - Üretim Aşaması Çalışmadı, sadece bulunan bağlamı (Retrieval) aşağıda paylaşıyorum:\n\n{context}"
        
    # Gemini Bağlantısı
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Geminin uygun ve ücretsiz bir modelini seçelim
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Aşağıdaki ders notlarına (bağlam) dayanarak kullanıcının sorusunu cevapla. 
        Eğer sorunun cevabı metinlerde yoksa 'Bu bilgi ders notlarında bulunmamaktadır' de. Kendinden bilgi ekleme.
        
        Bağlam (Ders Notları):
        {context}
        
        Soru: {query}
        Cevap:
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"LLM Cevap Üretim Hatası (API'yi kontrol edin): {str(e)}"

if __name__ == "__main__":
    test_query = "RAG yapısı için hangi kütüphaneler kullanılır?"
    context = retrieve_answers(test_query)
    print("Bulunan Bağlam:\n", context)
    print("===================\nAI Cevabı:")
    print(generate_answer(test_query, context))

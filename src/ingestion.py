import os
import fitz  # PyMuPDF
from config import DATA_DIR
import re

def load_documents_from_directory(directory=DATA_DIR):
    """
    Belirtilen dizindeki tüm PDF dosyalarını okur.
    PyMuPDF (fitz) kullanılarak içerik metne dönüştürülür.
    """
    documents = []
    
    # Klasör yoksa uyarı ver
    if not os.path.exists(directory):
        print(f"Uyarı: {directory} bulunamadı.")
        return documents

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            try:
                # PDF Açma
                doc = fitz.open(file_path)
                full_text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Get_text ile basitçe çıkarıyoruz
                    full_text += page.get_text() + "\n"
                
                # Temizleme (Preprocess)
                cleaned_text = preprocess_text(full_text)
                
                documents.append({
                    "metadata": {"source": filename},
                    "page_content": cleaned_text
                })
                print(f"Başarıyla okundu: {filename} ({len(cleaned_text)} karakter)")
            except Exception as e:
                print(f"Hata: {filename} okunamadı. Detay: {str(e)}")
                
    return documents

def preprocess_text(text):
    """
    Gereksiz boşlukları, tekrar eden satır sonlarını temizler ve küçük harfe çevirebilir
    fakat anlam bütünlüğünü korumak adına sadece whitespace temizliği yapacağız.
    """
    if not text:
        return ""
    # Çoklu space/newline leri tek satıra toparlama
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == "__main__":
    docs = load_documents_from_directory()
    print(f"Toplam {len(docs)} doküman okunup temizlendi.")

"""
Veri Toplama ve Temizleme Modülü
==================================
PDF dosyalarını okur, metin çıkarır ve temizler.
Opsiyonel: Gemini Vision ile görsellerden OCR yapar.
"""

import os
import fitz  # PyMuPDF
import config
import re
import io
import google.generativeai as genai
from PIL import Image


def pdf_isle(doc, dosya_adi, ocr_aktif=False):
    """
    PDF dokümanından metin çıkar ve temizle.
    ocr_aktif=True ise Gemini Vision ile görsellerden de metin çıkarır.
    """
    tam_metin = ""
    api_var = bool(config.GEMINI_API_KEY) and ocr_aktif
    
    gorsel_modeli = None
    if api_var:
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            gorsel_modeli = genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            pass

    for sayfa_no in range(len(doc)):
        sayfa = doc.load_page(sayfa_no)
        tam_metin += sayfa.get_text() + "\n"
        
        if api_var and gorsel_modeli:
            resim_listesi = sayfa.get_images(full=True)
            for resim in resim_listesi:
                try:
                    xref = resim[0]
                    ham_resim = doc.extract_image(xref)
                    genislik = ham_resim.get("width", 0)
                    yukseklik = ham_resim.get("height", 0)
                    if genislik < 100 or yukseklik < 100:
                        continue
                    resim_bytes = ham_resim["image"]
                    pil_resim = Image.open(io.BytesIO(resim_bytes))
                    yanit = gorsel_modeli.generate_content([
                        "Lütfen bu görselin içindeki tüm anlamlı metinleri, tablo verilerini veya akış diyagramını "
                        "detaylıca yaz. Eğer içinde hiç yazı olmayan düz boş bir dekoratif resimse '[Dekoratif]' yaz ve bitir.",
                        pil_resim
                    ])
                    if yanit.text and "[Dekoratif]" not in yanit.text:
                        tam_metin += f"\n[Görselden (S. {sayfa_no+1}) Okunan OCR Metni]: {yanit.text}\n"
                except Exception:
                    pass
    
    temiz_metin = metni_temizle(tam_metin)
    return {
        "metadata": {"source": dosya_adi},
        "page_content": temiz_metin
    }


def yuklenen_dosyalari_oku(yuklenen_dosyalar, ocr_aktif=False):
    """Streamlit üzerinden yüklenen PDF dosyalarını bellekte okur."""
    dokumanlar = []
    for dosya in yuklenen_dosyalar:
        try:
            doc = fitz.open(stream=dosya.read(), filetype="pdf")
            dokuman_bilgisi = pdf_isle(doc, dosya.name, ocr_aktif=ocr_aktif)
            dokumanlar.append(dokuman_bilgisi)
        except Exception as e:
            print(f"Hata: {dosya.name} okunamadı. Detay: {str(e)}")
    return dokumanlar


def klasorden_oku(klasor=None):
    """Belirtilen dizindeki tüm PDF dosyalarını okur."""
    if klasor is None:
        klasor = config.VERI_KLASORU
    dokumanlar = []
    if not os.path.exists(klasor):
        print(f"Uyarı: {klasor} bulunamadı.")
        return dokumanlar
    for dosya_adi in os.listdir(klasor):
        if dosya_adi.lower().endswith(".pdf"):
            dosya_yolu = os.path.join(klasor, dosya_adi)
            try:
                doc = fitz.open(dosya_yolu)
                dokuman_bilgisi = pdf_isle(doc, dosya_adi, ocr_aktif=False)
                dokumanlar.append(dokuman_bilgisi)
                print(f"Başarıyla okundu: {dosya_adi} ({len(dokuman_bilgisi['page_content'])} karakter)")
            except Exception as e:
                print(f"Hata: {dosya_adi} okunamadı. Detay: {str(e)}")
    return dokumanlar


def metni_temizle(metin):
    """Gereksiz boşlukları ve tekrar eden satır sonlarını temizler."""
    if not metin:
        return ""
    metin = re.sub(r'\s+', ' ', metin)
    return metin.strip()

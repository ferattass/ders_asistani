import streamlit as st
import os

# Custom Modüllerimiz
from src.ingestion import load_documents_from_directory
from src.chunking import split_documents
from src.vector_store import build_vector_store
from src.retrieval_generation import retrieve_answers, generate_answer
import config

st.set_page_config(page_title="Öğrenci Ders Asistanı", page_icon="🧠", layout="wide")

st.title("🧠 Kişisel RAG Asistanı (Ders Notları)")
st.markdown("PDF slaytları ve okuma notlarından sana özel akıllı cevaplar üreten RAG altyapısı.")

# Sidebar (Sol Menü)
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Gemini API Key Dinamik Alımı
    api_key_input = st.text_input("Google Gemini API Key (Cevap Üretimi için)", type="password", value=config.GEMINI_API_KEY)
    if api_key_input:
        config.GEMINI_API_KEY = api_key_input
        
    st.markdown("---")
    st.header("🔄 Veritabanı İşlemleri")
    
    if st.button("1. Verileri Oku ve Veritabanı (Vector DB) Oluştur", use_container_width=True):
        with st.spinner("PDF'ler okunuyor, veriler chunk'lanıyor ve Vektör DB'ye atılıyor..."):
            docs = load_documents_from_directory()
            if not docs:
                st.error("DATA klasöründe hiç PDF bulunamadı!")
            else:
                st.info(f"{len(docs)} adet doküman okundu.")
                chunks = split_documents(docs)
                st.info(f"Metinler {len(chunks)} adet parçaya (chunk) bölündü. NER uygulandı.")
                
                collection = build_vector_store(chunks)
                if collection:
                    st.success("✅ Tüm veriler ChromaDB'ye kaydedildi! Artık soru sorabilirsiniz.")
                else:
                    st.error("Veritabanı oluşturulurken bir hata meydana geldi.")

# Ana Ekran (Chat)
st.header("Soru - Cevap")

# Mesaj geçmşini tutalım (Opsiyonel ama Streamlit için güzel görünür)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Yeni Soru
if query := st.chat_input("Ders notlarıyla ilgili sorunuz nedir?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Notlardaki bağlam aranıyor (Retrieval) ..."):
            # 1. Retrieval (Arama)
            context = retrieve_answers(query)
            
            # 2. Generation (Üretim)
            full_response = generate_answer(query, context)
            st.markdown(full_response)
            
            with st.expander("🔍 Geri Planda Okunan Not Bağlamı (Context)"):
                st.text(context)
                
    st.session_state.messages.append({"role": "assistant", "content": full_response})

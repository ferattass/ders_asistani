import streamlit as st
import os
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.veri_toplama import yuklenen_dosyalari_oku
from src.parcalama import dokumanlari_parcala
from src.vektor_deposu import veritabani_olustur, tum_dokumanlari_getir
from src.arama_uretim import baglamlari_getir, cevap_uret
from src.geleneksel_model import TFIDFArama, BM25Arama, geleneksel_modelleri_kur
from src.degerlendirme import Degerlendirici
import config

# ============ SAYFA AYARI ============
st.set_page_config(page_title="Ders Asistanı — RAG Pipeline", page_icon="🧠", layout="wide")

# ============ ÖZEL CSS ============
st.markdown("""
<style>
.comparison-header {
    background: linear-gradient(90deg, #00D4AA 0%, #0ea5e9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 32px; font-weight: 700;
}
.method-badge-tfidf { background: #ef4444; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; display: inline-block; }
.method-badge-bm25 { background: #f59e0b; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; display: inline-block; }
.method-badge-rag { background: #00D4AA; color: #0E1117; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
for anahtar in ["mesajlar", "parcalar", "tfidf", "bm25", "vt_hazir"]:
    if anahtar not in st.session_state:
        st.session_state[anahtar] = [] if anahtar in ["mesajlar", "parcalar"] else None

# ============ SOL MENÜ ============
with st.sidebar:
    st.markdown("## 🧠 Ders Asistanı")
    st.caption("RAG Pipeline ile Akıllı Soru-Cevap")
    st.markdown("---")

    api_key_girdi = st.text_input("🔑 Google Gemini API Key", type="password", value=config.GEMINI_API_KEY)
    if api_key_girdi:
        config.GEMINI_API_KEY = api_key_girdi

    st.markdown("---")
    st.subheader("⚙️ İşleme Ayarları")
    ocr_aktif = st.toggle("🖼️ Vision OCR (Görsel Okuma)", value=False, help="PDF içindeki resimlerden metin çıkarır. API kotası harcar!")
    ner_aktif = st.toggle("🏷️ NER (Varlık Tanıma)", value=True, help="spaCy ile kişi, kurum, tarih gibi varlıkları tespit eder.")
    parca_boyutu = st.slider("📏 Parça Boyutu (Chunk Size)", min_value=200, max_value=1500, value=500, step=50)

    st.markdown("---")
    yuklenen = st.file_uploader("📥 PDF Yükle", type=["pdf"], accept_multiple_files=True)

    if st.button("🚀 Veritabanını Oluştur", use_container_width=True):
        if not yuklenen:
            st.warning("Önce PDF yükleyin!")
        else:
            with st.spinner("PDF'ler işleniyor..."):
                dokumanlar = yuklenen_dosyalari_oku(yuklenen, ocr_aktif=ocr_aktif)
                if not dokumanlar:
                    st.error("Metin çıkarılamadı.")
                else:
                    st.info(f"📄 {len(dokumanlar)} doküman okundu.")
                    parcalar = dokumanlari_parcala(dokumanlar, ner_aktif=ner_aktif, parca_boyutu=parca_boyutu)
                    st.info(f"✂️ {len(parcalar)} parçaya bölündü.")
                    st.session_state.parcalar = parcalar

                    koleksiyon = veritabani_olustur(parcalar)
                    if koleksiyon:
                        st.session_state.vt_hazir = True
                        tfidf, bm25 = geleneksel_modelleri_kur(parcalar)
                        st.session_state.tfidf = tfidf
                        st.session_state.bm25 = bm25
                        st.success("✅ Tüm modeller hazır!")
                    else:
                        st.error("Veritabanı oluşturulamadı.")

    # Mevcut DB varsa geleneksel modelleri yükle
    if st.session_state.tfidf is None:
        try:
            mevcut = tum_dokumanlari_getir()
            if mevcut:
                st.session_state.parcalar = mevcut
                st.session_state.vt_hazir = True
                tfidf, bm25 = geleneksel_modelleri_kur(mevcut)
                st.session_state.tfidf = tfidf
                st.session_state.bm25 = bm25
        except Exception:
            pass  # DB henüz oluşturulmamış, sorun değil

    st.markdown("---")
    sayfa = st.radio("📌 Sayfa Seçin", [
        "💬 Akıllı Asistan",
        "⚔️ Karşılaştırmalı Test",
        "📊 Değerlendirme Paneli",
        "ℹ️ Sistem Hakkında"
    ])

# ============================================================
# SAYFA 1: AKILLI ASİSTAN (CHAT)
# ============================================================
if sayfa == "💬 Akıllı Asistan":
    st.markdown('<p class="comparison-header">💬 Akıllı Ders Asistanı</p>', unsafe_allow_html=True)
    st.caption("Ders notlarınızdan RAG altyapısıyla akıllı cevaplar üretir.")

    for mesaj in st.session_state.mesajlar:
        with st.chat_message(mesaj["role"]):
            st.markdown(mesaj["content"])

    if soru := st.chat_input("Ders notlarıyla ilgili sorunuzu yazın..."):
        st.session_state.mesajlar.append({"role": "user", "content": soru})
        with st.chat_message("user"):
            st.markdown(soru)

        with st.chat_message("assistant"):
            with st.status("⚙️ RAG Motoru Çalışıyor...", expanded=False) as durum:
                st.write("🔍 Soru vektörleştiriliyor...")
                baglam, skorlar, sure = baglamlari_getir(soru)
                st.write(f"📚 {len(skorlar)} ilgili parça bulundu ({sure:.3f}s)")
                st.write("✨ Gemini ile sentezleniyor...")
                cevap = cevap_uret(soru, baglam)
                durum.update(label="✅ Tamamlandı!", state="complete", expanded=False)

            st.markdown(cevap)

            if skorlar:
                with st.expander("📊 Benzerlik Skorları"):
                    for i, s in enumerate(skorlar):
                        st.progress(min(s, 1.0), text=f"Parça {i+1}: {s:.4f}")

            with st.expander("🔍 Kullanılan Bağlam (Context)"):
                st.text(baglam)

        st.session_state.mesajlar.append({"role": "assistant", "content": cevap})

# ============================================================
# SAYFA 2: KARŞILAŞTIRMALI TEST
# ============================================================
elif sayfa == "⚔️ Karşılaştırmalı Test":
    st.markdown('<p class="comparison-header">⚔️ Karşılaştırmalı Arama Testi</p>', unsafe_allow_html=True)
    st.caption("Aynı soruyu 3 farklı yöntemle arayın, sonuçları yan yana görün.")

    if not st.session_state.vt_hazir:
        st.warning("⚠️ Önce sol menüden PDF yükleyip veritabanı oluşturun.")
    else:
        test_sorusu = st.text_input("🔎 Bir soru yazın:", value="Fine-tuning nedir?")

        if st.button("🚀 3 Yöntemi Çalıştır", use_container_width=True):
            sonuclar = {}

            with st.spinner("TF-IDF çalışıyor..."):
                baglam1, skor1, sure1 = st.session_state.tfidf.ara(test_sorusu)
                sonuclar["TF-IDF"] = {"baglam": baglam1, "skorlar": skor1, "sure": sure1}

            with st.spinner("BM25 çalışıyor..."):
                baglam2, skor2, sure2 = st.session_state.bm25.ara(test_sorusu)
                sonuclar["BM25"] = {"baglam": baglam2, "skorlar": skor2, "sure": sure2}

            with st.spinner("RAG (Semantic) çalışıyor..."):
                baglam3, skor3, sure3 = baglamlari_getir(test_sorusu)
                sonuclar["RAG Semantic"] = {"baglam": baglam3, "skorlar": skor3, "sure": sure3}

            # Süre Karşılaştırma Grafiği
            st.markdown("### ⏱️ Yanıt Süreleri")
            sure_df = pd.DataFrame({
                "Yöntem": list(sonuclar.keys()),
                "Süre (ms)": [sonuclar[y]["sure"]*1000 for y in sonuclar]
            })
            renk_haritasi = {"TF-IDF": "#ef4444", "BM25": "#f59e0b", "RAG Semantic": "#00D4AA"}
            fig = px.bar(sure_df, x="Yöntem", y="Süre (ms)", color="Yöntem",
                         color_discrete_map=renk_haritasi,
                         template="plotly_dark")
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Yan Yana Sonuçlar
            st.markdown("### 📋 Bulunan Bağlamlar")
            sutunlar = st.columns(3)
            etiketler = [("TF-IDF", "method-badge-tfidf"), ("BM25", "method-badge-bm25"), ("RAG Semantic", "method-badge-rag")]

            for i, (yontem, css) in enumerate(etiketler):
                with sutunlar[i]:
                    st.markdown(f'<span class="{css}">{yontem}</span>', unsafe_allow_html=True)
                    st.markdown(f"**Süre:** {sonuclar[yontem]['sure']*1000:.1f}ms")
                    if sonuclar[yontem]["skorlar"]:
                        st.markdown(f"**En İyi Skor:** {sonuclar[yontem]['skorlar'][0]:.4f}")
                    with st.expander("Bağlam Göster"):
                        st.text(sonuclar[yontem]["baglam"][:500])

# ============================================================
# SAYFA 3: DEĞERLENDİRME PANELİ
# ============================================================
elif sayfa == "📊 Değerlendirme Paneli":
    st.markdown('<p class="comparison-header">📊 Bilimsel Değerlendirme Paneli</p>', unsafe_allow_html=True)
    st.caption("Test soruları üzerinde tüm yöntemleri çalıştırıp metrikleri raporlar.")

    if not st.session_state.vt_hazir:
        st.warning("⚠️ Önce sol menüden PDF yükleyip veritabanı oluşturun.")
    else:
        if st.button("🧪 Tüm Değerlendirmeyi Başlat", use_container_width=True, type="primary"):

            degerlendirici = Degerlendirici(config.TEST_SORULARI_DOSYASI)

            if not degerlendirici.test_sorulari:
                st.error("Test soruları yüklenemedi!")
            else:
                ilerleme = st.progress(0, text="Değerlendirme başlıyor...")

                # TF-IDF
                ilerleme.progress(10, text="TF-IDF değerlendiriliyor...")
                tfidf_sonuc = degerlendirici.aramay_degerlendir(
                    st.session_state.tfidf.ara, "TF-IDF"
                )
                # BM25
                ilerleme.progress(40, text="BM25 değerlendiriliyor...")
                bm25_sonuc = degerlendirici.aramay_degerlendir(
                    st.session_state.bm25.ara, "BM25"
                )
                # RAG
                ilerleme.progress(70, text="RAG Semantic değerlendiriliyor...")
                rag_sonuc = degerlendirici.aramay_degerlendir(
                    baglamlari_getir, "RAG Semantic"
                )
                ilerleme.progress(100, text="Tamamlandı!")

                # ========== METRİK KARTLARI ==========
                st.markdown("### 🎯 Özet Metrikler")
                tum_sonuclar = {"TF-IDF": tfidf_sonuc, "BM25": bm25_sonuc, "RAG Semantic": rag_sonuc}

                for yontem, sonuc in tum_sonuclar.items():
                    st.markdown(f"**{yontem}**")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Ort. Hassasiyet", f"{sonuc['ort_hassasiyet']:.2%}")
                    k2.metric("İsabet Oranı", f"{sonuc['ort_isabet']:.2%}")
                    k3.metric("Ort. Süre", f"{sonuc['ort_sure_ms']:.1f}ms")
                    k4.metric("Medyan Süre", f"{sonuc['medyan_sure_ms']:.1f}ms")

                # ========== KARŞILAŞTIRMA GRAFİĞİ ==========
                st.markdown("### 📈 Karşılaştırma Grafikleri")
                sekme1, sekme2, sekme3 = st.tabs(["Hassasiyet", "İsabet Oranı", "Yanıt Süreleri"])

                yontemler = list(tum_sonuclar.keys())
                renkler = ["#ef4444", "#f59e0b", "#00D4AA"]

                with sekme1:
                    fig = go.Figure(data=[go.Bar(
                        x=yontemler,
                        y=[tum_sonuclar[y]["ort_hassasiyet"] for y in yontemler],
                        marker_color=renkler, text=[f"{tum_sonuclar[y]['ort_hassasiyet']:.2%}" for y in yontemler],
                        textposition='auto'
                    )])
                    fig.update_layout(title="Ortalama Anahtar Kelime Hassasiyeti",
                                      yaxis_title="Hassasiyet", template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with sekme2:
                    fig = go.Figure(data=[go.Bar(
                        x=yontemler,
                        y=[tum_sonuclar[y]["ort_isabet"] for y in yontemler],
                        marker_color=renkler, text=[f"{tum_sonuclar[y]['ort_isabet']:.2%}" for y in yontemler],
                        textposition='auto'
                    )])
                    fig.update_layout(title="İsabet Oranı (Hit Rate)",
                                      yaxis_title="Oran", template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with sekme3:
                    fig = go.Figure(data=[go.Bar(
                        x=yontemler,
                        y=[tum_sonuclar[y]["ort_sure_ms"] for y in yontemler],
                        marker_color=renkler, text=[f"{tum_sonuclar[y]['ort_sure_ms']:.1f}ms" for y in yontemler],
                        textposition='auto'
                    )])
                    fig.update_layout(title="Ortalama Yanıt Süresi",
                                      yaxis_title="Süre (ms)", template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # ========== RADAR GRAFİĞİ ==========
                st.markdown("### 🕸️ Radar Karşılaştırması")
                kategoriler = ["Hassasiyet", "İsabet", "Hız (ters)"]

                def hiz_skoru(ms):
                    return max(0, min(1, 1 - ms / 5000))

                fig = go.Figure()
                for i, (yontem, sonuc) in enumerate(tum_sonuclar.items()):
                    degerler = [
                        sonuc["ort_hassasiyet"],
                        sonuc["ort_isabet"],
                        hiz_skoru(sonuc["ort_sure_ms"])
                    ]
                    fig.add_trace(go.Scatterpolar(
                        r=degerler + [degerler[0]], theta=kategoriler + [kategoriler[0]],
                        fill='toself', name=yontem,
                        line_color=renkler[i], opacity=0.7
                    ))
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)"),
                                  template="plotly_dark", height=450, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                # ========== DETAY TABLOSU ==========
                st.markdown("### 📋 Soru Bazlı Detaylar")
                for yontem, sonuc in tum_sonuclar.items():
                    with st.expander(f"📌 {yontem} — Soru Detayları"):
                        detay_df = pd.DataFrame(sonuc["soru_detaylari"])
                        detay_df = detay_df[["soru", "hassasiyet", "isabet", "sure_ms", "en_iyi_skor"]]
                        detay_df.columns = ["Soru", "Hassasiyet", "İsabet", "Süre (ms)", "En İyi Skor"]
                        st.dataframe(detay_df, use_container_width=True, hide_index=True)

                # ========== GENERATION DEĞERLENDİRME ==========
                if config.GEMINI_API_KEY:
                    st.markdown("### 🤖 Cevap Üretme Kalitesi (RAG + LLM)")
                    if st.button("📝 Üretim Değerlendirmesini Başlat"):
                        with st.spinner("Gemini API ile cevaplar üretiliyor..."):
                            gen_sonuc = degerlendirici.uretimi_degerlendir(cevap_uret, baglamlari_getir)
                        if gen_sonuc["degerlendirilen"] > 0:
                            g1, g2, g3, g4 = st.columns(4)
                            g1.metric("BLEU", f"{gen_sonuc['ort_bleu']:.4f}")
                            g2.metric("ROUGE-L", f"{gen_sonuc['ort_rouge_l']:.4f}")
                            g3.metric("Sadakat", f"{gen_sonuc['ort_sadakat']:.2%}")
                            g4.metric("Ort. Süre", f"{gen_sonuc['ort_sure_ms']:.0f}ms")

                            with st.expander("Üretilen Cevap Detayları"):
                                gen_df = pd.DataFrame(gen_sonuc["soru_detaylari"])
                                st.dataframe(gen_df, use_container_width=True, hide_index=True)

# ============================================================
# SAYFA 4: SİSTEM HAKKINDA
# ============================================================
elif sayfa == "ℹ️ Sistem Hakkında":
    st.markdown('<p class="comparison-header">🏗️ Sistem Mimarisi</p>', unsafe_allow_html=True)

    st.markdown("""
    ### RAG (Retrieval-Augmented Generation) Pipeline

    Bu proje, ders notlarınızı (PDF) **üç farklı yöntemle** arayıp karşılaştıran
    ve LLM ile akıllı cevaplar üreten bir sistemdir.
    """)

    st.markdown("### 📂 Modül Yapısı")
    modul_veri = {
        "Modül": ["veri_toplama.py", "parcalama.py", "vektor_deposu.py", "arama_uretim.py", "geleneksel_model.py", "degerlendirme.py"],
        "Görev": [
            "PDF okuma + Gemini Vision OCR",
            "Metin parçalama + spaCy NER",
            "HuggingFace Embedding + ChromaDB",
            "Semantic Retrieval + Gemini Generation",
            "TF-IDF ve BM25 geleneksel arama",
            "BLEU, ROUGE-L, Precision, Hit Rate metrikleri"
        ],
        "Kütüphane": ["PyMuPDF, Pillow, Gemini", "Langchain, spaCy", "HuggingFace, ChromaDB",
                       "HuggingFace, Gemini API", "scikit-learn, rank-bm25", "NumPy"]
    }
    st.dataframe(pd.DataFrame(modul_veri), use_container_width=True, hide_index=True)

    st.markdown("### ⚔️ Yöntem Karşılaştırma Tablosu")
    karsilastirma = {
        "Özellik": ["Arama Tipi", "Anlam Yakalama", "Hız", "Maliyet", "Veri Gizliliği", "Kurulum"],
        "TF-IDF": ["Kelime eşleşmesi", "❌ Hayır", "⚡ Çok hızlı (~ms)", "Ücretsiz", "✅ Tam lokal", "Kolay"],
        "BM25": ["Gelişmiş kelime eşleşmesi", "❌ Hayır", "⚡ Çok hızlı (~ms)", "Ücretsiz", "✅ Tam lokal", "Kolay"],
        "RAG (Embedding+LLM)": ["Anlamsal benzerlik", "✅ Evet", "🐢 Yavaş (~2-5s)", "API maliyeti", "⚠️ Veri API'ye gider", "Orta"]
    }
    st.dataframe(pd.DataFrame(karsilastirma), use_container_width=True, hide_index=True)

    st.markdown("### 🔬 Kullanılan Metrikler")
    st.markdown("""
    | Metrik | Açıklama | Kullanım |
    |--------|----------|----------|
    | **Precision@K** | K getirilen belgeden kaçı doğru? | Retrieval kalitesi |
    | **Hit Rate** | En az 1 doğru belge bulundu mu? | Retrieval kapsamı |
    | **BLEU** | N-gram hassasiyeti (üretim vs beklenen) | Generation kalitesi |
    | **ROUGE-L** | En uzun ortak alt dizi (LCS) F1 | Generation kalitesi |
    | **Faithfulness** | Cevabın bağlama sadakati | Halüsinasyon kontrolü |
    | **Yanıt Süresi** | Milisaniye cinsinden gecikme | Sistem performansı |
    """)

    # Sistem durumu
    st.markdown("### 💾 Sistem Durumu")
    d1, d2, d3 = st.columns(3)
    parca_sayisi = len(st.session_state.parcalar) if st.session_state.parcalar else 0
    d1.metric("Toplam Parça", parca_sayisi)
    d2.metric("VektörDB", "✅ Hazır" if st.session_state.vt_hazir else "❌ Yok")
    d3.metric("API Key", "✅ Var" if config.GEMINI_API_KEY else "❌ Yok")

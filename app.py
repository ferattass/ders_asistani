import streamlit as st
import os
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import textwrap

from src.veri_toplama import yuklenen_dosyalari_oku
from src.parcalama import dokumanlari_parcala
from src.vektor_deposu import veritabani_olustur, tum_dokumanlari_getir
from src.arama_uretim import baglamlari_getir, cevap_uret
from src.geleneksel_model import TFIDFArama, BM25Arama, HybridArama, geleneksel_modelleri_kur, embedding_gorselleştir
from src.degerlendirme import Degerlendirici
from src.soru_uretici import test_sorulari_uret, sorulari_kaydet
import config

# ============ SAYFA AYARI ============
st.set_page_config(page_title="Ders Asistanı — RAG Pipeline", page_icon="🧠", layout="wide")

# ============ ÖZEL CSS (MODERN & SEKSI) ============
st.markdown("""
<style>
    /* ═══════ AURORA DARK THEME ═══════ */

    /* Sidebar: Deep space gradient with subtle aurora */
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #070b14 0%, #0d1525 30%, #131a35 60%, #0f1629 100%) !important;
        border-right: 1px solid rgba(0,212,170,0.08);
        box-shadow: 4px 0 30px rgba(0,0,0,0.5);
    }

    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00D4AA, #0ea5e9, #a855f7, #00D4AA);
        background-size: 300% 100%;
        animation: shimmer 4s linear infinite;
    }

    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    /* Sidebar headings glow */
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00D4AA;
        text-shadow: 0 0 20px rgba(0,212,170,0.5);
        letter-spacing: 0.5px;
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: rgba(0,212,170,0.1) !important;
        margin: 18px 0 !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #00D4AA 0%, #0ea5e9 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,212,170,0.25) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,212,170,0.4) !important;
    }

    /* Sidebar radio pills */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 10px 14px !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0,212,170,0.08) !important;
    }

    /* Sidebar file uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(0,212,170,0.15) !important;
        border-radius: 14px !important;
        padding: 8px !important;
        transition: border-color 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,212,170,0.35) !important;
    }

    /* Sidebar slider */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: linear-gradient(90deg, #00D4AA, #0ea5e9) !important;
    }

    /* ═══════ MAIN CONTENT ═══════ */

    .comparison-header {
        background: linear-gradient(135deg, #00D4AA 0%, #0ea5e9 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 5px;
        filter: drop-shadow(0 0 20px rgba(0,212,170,0.15));
    }

    /* Glass cards with aurora border */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 28px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 22px;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .glass-card::after {
        content: '';
        position: absolute;
        inset: -1px;
        border-radius: 21px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(0,212,170,0), rgba(0,212,170,0.12), rgba(14,165,233,0.12), rgba(0,212,170,0));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }

    .glass-card:hover {
        transform: translateY(-6px) scale(1.01);
        background: rgba(255, 255, 255, 0.04);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4), 0 0 40px rgba(0,212,170,0.06);
    }

    /* Metric containers */
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 16px;
        background: rgba(0,0,0,0.35);
        border-radius: 12px;
        margin-top: 10px;
        border: 1px solid rgba(255,255,255,0.04);
        transition: background 0.3s ease;
    }

    .metric-container:hover {
        background: rgba(0,212,170,0.06);
    }

    /* Method badges */
    .method-badge {
        padding: 8px 22px;
        border-radius: 40px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        width: fit-content;
    }

    .method-badge-tfidf { background: linear-gradient(135deg, #f87171, #dc2626); color: white; box-shadow: 0 4px 20px rgba(239,68,68,0.3); }
    .method-badge-bm25 { background: linear-gradient(135deg, #fbbf24, #d97706); color: white; box-shadow: 0 4px 20px rgba(245,158,11,0.3); }
    .method-badge-rag { background: linear-gradient(135deg, #34d399, #059669); color: white; box-shadow: 0 4px 20px rgba(0,212,170,0.3); }

    /* Winner badge */
    .winner-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 900;
        position: absolute;
        top: -14px;
        right: 16px;
        z-index: 10;
        box-shadow: 0 4px 25px rgba(255,215,0,0.5);
        border: 2px solid rgba(0,0,0,0.3);
        letter-spacing: 0.5px;
    }

    /* Status widget */
    [data-testid="stStatusWidget"] {
        margin-bottom: 25px !important;
        border-radius: 14px;
    }

    /* Progress bar aurora gradient */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D4AA, #0ea5e9, #a855f7) !important;
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
    }

    /* Primary buttons on main area */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00D4AA 0%, #0ea5e9 100%) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,212,170,0.2) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,212,170,0.35) !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 12px !important;
        transition: background 0.3s ease !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 18px !important;
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        background: rgba(0,212,170,0.04);
        border-color: rgba(0,212,170,0.15);
    }

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(0,212,170,0.15) !important;
        transition: border-color 0.3s ease !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(0,212,170,0.4) !important;
        box-shadow: 0 0 20px rgba(0,212,170,0.08) !important;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        overflow: hidden;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.15); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,212,170,0.3); }
</style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
for anahtar in ["mesajlar", "parcalar", "tfidf", "bm25", "hybrid", "vt_hazir"]:
    if anahtar not in st.session_state:
        st.session_state[anahtar] = [] if anahtar in ["mesajlar", "parcalar"] else None

# ============ SOL MENÜ ============
with st.sidebar:
    st.markdown(textwrap.dedent("""
<div style="text-align:center; padding:20px 0 10px 0;">
<div style="font-size:48px; margin-bottom:8px;">🧠</div>
<div style="font-size:22px; font-weight:800; color:#00D4AA; letter-spacing:1px;">DERS ASİSTANI</div>
<div style="font-size:11px; color:#64748b; letter-spacing:3px; text-transform:uppercase; margin-top:4px;">RAG Pipeline v2.0</div>
<div style="height:2px; background:linear-gradient(90deg, transparent, #00D4AA, #0ea5e9, transparent); margin-top:14px;"></div>
</div>
    """).strip(), unsafe_allow_html=True)

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
                        st.session_state.hybrid = HybridArama(bm25, baglamlari_getir)
                        st.success("✅ Tüm modeller hazır (Hybrid dahil)!")
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
                st.session_state.hybrid = HybridArama(bm25, baglamlari_getir)
        except Exception:
            pass  # DB henüz oluşturulmamış, sorun değil

    st.markdown("---")
    sayfa = st.radio("📌 Sayfa Seçin", [
        "💬 Akıllı Asistan",
        "⚔️ Karşılaştırmalı Test",
        "📊 Değerlendirme Paneli",
        "🧬 Embedding Haritası",
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

# ============ SAYFA 2: KARŞILAŞTIRMALI TEST ============
elif sayfa == "⚔️ Karşılaştırmalı Test":
    st.markdown('<p class="comparison-header">⚔️ Karşılaştırmalı Arama</p>', unsafe_allow_html=True)
    st.markdown("##### Üç farklı arama algoritmasını gerçek zamanlı olarak çarpıştırın.")

    if not st.session_state.vt_hazir:
        st.info("👋 Başlamak için lütfen sol taraftan bir PDF yükleyin ve veritabanını oluşturun.")
        st.image("https://img.freepik.com/free-vector/modern-comparison-concept-with-flat-design_23-2147889397.jpg", width=400)
    else:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([4, 1])
            with col1:
                test_sorusu = st.text_input("🔍 Test etmek istediğiniz soruyu girin:", value="Fine-tuning nedir?", placeholder="Örn: Google Colab ne işe yarar?")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                calistir = st.button("🔥 Çarpıştır", use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

        if calistir:
            sonuclar = {}
            from src.degerlendirme import anahtar_kelime_hassasiyeti, context_ozgunlugu
            
            # 1. TF-IDF
            with st.spinner("TF-IDF taranıyor..."):
                baglam1, skor1, sure1 = st.session_state.tfidf.ara(test_sorusu)
                sonuclar["TF-IDF"] = {"baglam": baglam1, "skorlar": skor1, "sure": sure1}
            
            # 2. BM25
            with st.spinner("BM25 taranıyor..."):
                baglam2, skor2, sure2 = st.session_state.bm25.ara(test_sorusu)
                sonuclar["BM25"] = {"baglam": baglam2, "skorlar": skor2, "sure": sure2}
            
            # 3. RAG Semantic
            with st.spinner("RAG Semantic taranıyor..."):
                baglam3, skor3, sure3 = baglamlari_getir(test_sorusu)
                sonuclar["RAG Semantic"] = {"baglam": baglam3, "skorlar": skor3, "sure": sure3}
            
            # 4. Hybrid (BM25 + Semantic)
            with st.spinner("Hybrid Search taranıyor..."):
                baglam4, skor4, sure4 = st.session_state.hybrid.ara(test_sorusu)
                sonuclar["Hybrid"] = {"baglam": baglam4, "skorlar": skor4, "sure": sure4}

            # Özgünlük hesapla
            ozgunlukler = context_ozgunlugu({k: v["baglam"] for k,v in sonuclar.items()})

            # Üst Panel: Zaman ve Özgünlük
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ⏱️ Performans (ms)")
                df_sure = pd.DataFrame({"Model": list(sonuclar.keys()), "Süre": [s["sure"]*1000 for s in sonuclar.values()]})
                fig_sure = px.bar(df_sure, x="Model", y="Süre", color="Model", 
                                  color_discrete_map={"TF-IDF": "#ef4444", "BM25": "#f59e0b", "RAG Semantic": "#00D4AA", "Hybrid": "#a855f7"},
                                  template="plotly_dark")
                fig_sure.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_sure, use_container_width=True)
            
            with c2:
                st.markdown("### 🧬 Bağlam Özgünlüğü")
                df_ozgun = pd.DataFrame({"Model": list(ozgunlukler.keys()), "Özgünlük": [v*100 for v in ozgunlukler.values()]})
                fig_ozgun = px.pie(df_ozgun, values="Özgünlük", names="Model", hole=0.6,
                                   color="Model", color_discrete_map={"TF-IDF": "#ef4444", "BM25": "#f59e0b", "RAG Semantic": "#00D4AA", "Hybrid": "#a855f7"},
                                   template="plotly_dark")
                fig_ozgun.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_ozgun, use_container_width=True)

            # Yan Yana Detaylar
            st.markdown("### 📋 Algoritma Karşılaştırması")
            sutunlar = st.columns(4)
            detaylar = [
                ("TF-IDF", "method-badge-tfidf", "Kelime Frekansı"),
                ("BM25", "method-badge-bm25", "Probabilistik"),
                ("RAG Semantic", "method-badge-rag", "Vektör Uzayı"),
                ("Hybrid", "method-badge-rag", "BM25 + Semantic")
            ]

            # Basit kazanan belirleme (bu sorgu için en hızlı olanı işaretleyelim)
            en_hizli = min(sonuclar.keys(), key=lambda x: sonuclar[x]["sure"])

            for i, (yontem, css, aciklama) in enumerate(detaylar):
                is_winner = yontem == en_hizli
                
                with sutunlar[i]:
                    winner_html = '<div class="winner-badge">⚡ EN HIZLI</div>' if is_winner else ''
                    sure_ms = f"{sonuclar[yontem]['sure']*1000:.1f}"
                    ozgun_pct = f"{ozgunlukler[yontem]*100:.1f}"
                    onizleme = sonuclar[yontem]['baglam'][:250].replace('<', '&lt;').replace('>', '&gt;')
                    
                    kart = (
                        f'<div style="position:relative;padding:20px;background:rgba(255,255,255,0.04);border-radius:16px;border:1px solid rgba(255,255,255,0.08);min-height:220px;">'
                        f'{winner_html}'
                        f'<div class="method-badge {css}">{yontem}</div>'
                        f'<p style="font-size:12px;color:#94a3b8;margin-top:10px;">{aciklama}</p>'
                        f'<div class="metric-container"><span>Süre:</span> <b>{sure_ms}ms</b></div>'
                        f'<div class="metric-container"><span>Özgünlük:</span> <b>{ozgun_pct}%</b></div>'
                        f'<div style="margin-top:15px;font-size:13px;color:#cbd5e1;border-top:1px solid rgba(255,255,255,0.06);padding-top:10px;">'
                        f'{onizleme}...</div>'
                        f'</div>'
                    )
                    st.markdown(kart, unsafe_allow_html=True)
                    with st.expander("📄 Tam Metni Gör"):
                        st.write(sonuclar[yontem]['baglam'])

# ============ SAYFA 3: DEĞERLENDİRME PANELİ ============
elif sayfa == "📊 Değerlendirme Paneli":
    st.markdown('<p class="comparison-header">📊 Bilimsel Değerlendirme</p>', unsafe_allow_html=True)
    st.markdown("##### Test seti üzerindeki performansın derinlemesine analizi.")

    if not st.session_state.vt_hazir:
        st.warning("⚠️ Lütfen önce veritabanını oluşturun.")
    else:
        # Otomatik Soru Üretme
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('**📝 Test Soruları**')
        
        # Mevcut soru sayısını göster
        import json, os
        mevcut_soru_sayisi = 0
        try:
            with open(config.TEST_SORULARI_DOSYASI, 'r', encoding='utf-8') as f:
                mevcut_sorular = json.load(f)
                mevcut_soru_sayisi = len(mevcut_sorular)
        except Exception:
            pass
        
        st.caption(f"Mevcut test seti: **{mevcut_soru_sayisi}** soru")
        
        col_gen1, col_gen2 = st.columns([1, 1])
        with col_gen1:
            soru_adet = st.slider("Üretilecek soru sayısı", 5, 25, 15)
        with col_gen2:
            st.markdown('<br>', unsafe_allow_html=True)
            uret_btn = st.button("🤖 PDF'den Otomatik Soru Üret", use_container_width=True)
        
        if uret_btn:
            with st.spinner("Gemini ile PDF içeriğinden sorular üretiliyor..."):
                yeni_sorular = test_sorulari_uret(st.session_state.parcalar, soru_adet)
            if yeni_sorular:
                sorulari_kaydet(yeni_sorular)
                st.success(f"✅ {len(yeni_sorular)} soru üretildi ve kaydedildi!")
                with st.expander("📋 Üretilen Sorular"):
                    for i, s in enumerate(yeni_sorular, 1):
                        st.markdown(f"**{i}.** {s['question']}")
                        st.caption(f"Anahtar: {', '.join(s['relevant_keywords'])}")
            else:
                st.error("Soru üretilemedi. API key'i kontrol edin.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.info("Bu panel, test sorularını kullanarak Precision, Hit Rate, MRR, NDCG, MAP ve F1 metriklerini hesaplar.")
        baslat = st.button("🔬 Tüm Laboratuvar Testlerini Başlat", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if baslat:
            degerlendirici = Degerlendirici(config.TEST_SORULARI_DOSYASI)

            if not degerlendirici.test_sorulari:
                st.error("Test soruları yüklenemedi!")
            else:
                progress_bar = st.progress(0)
                
                with st.spinner("Modeller test ediliyor..."):
                    progress_bar.progress(10, text="TF-IDF test ediliyor...")
                    tfidf_s = degerlendirici.aramay_degerlendir(st.session_state.tfidf.ara, "TF-IDF")
                    
                    progress_bar.progress(30, text="BM25 test ediliyor...")
                    bm25_s = degerlendirici.aramay_degerlendir(st.session_state.bm25.ara, "BM25")
                    
                    progress_bar.progress(55, text="RAG Semantic test ediliyor...")
                    rag_s = degerlendirici.aramay_degerlendir(baglamlari_getir, "RAG Semantic")
                    
                    progress_bar.progress(80, text="Hybrid Search test ediliyor...")
                    hybrid_s = degerlendirici.aramay_degerlendir(st.session_state.hybrid.ara, "Hybrid")
                    
                    progress_bar.progress(100, text="Analiz tamamlandı!")

                tum_s = {"TF-IDF": tfidf_s, "BM25": bm25_s, "RAG Semantic": rag_s, "Hybrid": hybrid_s}

                # --- 🎯 Üst Metrikler ---
                st.markdown("### 🎯 Global Başarı Skorları")
                m1, m2, m3, m4 = st.columns(4)
                
                def get_winner(metric_key):
                    return max(tum_s.items(), key=lambda x: x[1][metric_key])[0]

                best_p = get_winner("ort_hassasiyet")
                best_n = get_winner("ort_ndcg")
                best_m = get_winner("ort_mrr")
                best_map = get_winner("ort_map")

                with m1:
                    badge_p = best_p.lower().replace(' ', '-')
                    st.markdown(f'<div class="glass-card"><small>En Hassas Arama</small><h2 style="color:#00D4AA">{tum_s[best_p]["ort_hassasiyet"]:.1%}</h2><span class="method-badge method-badge-{badge_p}">{best_p}</span></div>', unsafe_allow_html=True)
                with m2:
                    badge_n = best_n.lower().replace(' ', '-')
                    st.markdown(f'<div class="glass-card"><small>Sıralama Başarısı (NDCG)</small><h2 style="color:#0ea5e9">{tum_s[best_n]["ort_ndcg"]:.2f}</h2><span class="method-badge method-badge-{badge_n}">{best_n}</span></div>', unsafe_allow_html=True)
                with m3:
                    badge_m = best_m.lower().replace(' ', '-')
                    st.markdown(f'<div class="glass-card"><small>Hızlı Yanıt (MRR)</small><h2 style="color:#f59e0b">{tum_s[best_m]["ort_mrr"]:.2f}</h2><span class="method-badge method-badge-{badge_m}">{best_m}</span></div>', unsafe_allow_html=True)
                with m4:
                    badge_map = best_map.lower().replace(' ', '-')
                    st.markdown(f'<div class="glass-card"><small>Genel Kalite (MAP)</small><h2 style="color:#a855f7">{tum_s[best_map]["ort_map"]:.2f}</h2><span class="method-badge method-badge-{badge_map}">{best_map}</span></div>', unsafe_allow_html=True)

                # --- 📈 Grafik Analizleri ---
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Retrieval Analizi", "📉 Skor Dağılımı", "🕸️ Yetenek Radarı", "📋 Detaylı Veri"])
                
                yontemler = list(tum_s.keys())
                renkler = ["#ef4444", "#f59e0b", "#00D4AA", "#a855f7"]

                with tab1:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        # NDCG vs MRR vs MAP
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=yontemler, y=[tum_s[y]["ort_ndcg"] for y in yontemler], name="NDCG@3", marker_color="#0ea5e9"))
                        fig.add_trace(go.Bar(x=yontemler, y=[tum_s[y]["ort_mrr"] for y in yontemler], name="MRR", marker_color="#f59e0b"))
                        fig.add_trace(go.Bar(x=yontemler, y=[tum_s[y]["ort_map"] for y in yontemler], name="MAP", marker_color="#a855f7"))
                        fig.update_layout(title="Sıralama ve Bilgi Getirme Kalitesi", barmode='group', template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_b:
                        # F1 vs Precision
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=yontemler, y=[tum_s[y]["ort_hassasiyet"] for y in yontemler], name="Precision", mode='lines+markers', line_width=4))
                        fig2.add_trace(go.Scatter(x=yontemler, y=[tum_s[y]["ort_f1"] for y in yontemler], name="F1-Score", mode='lines+markers', line_width=4))
                        fig2.update_layout(title="Hassasiyet ve Denge (F1)", template="plotly_dark")
                        st.plotly_chart(fig2, use_container_width=True)

                with tab2:
                    st.markdown("#### Retrieval Benzerlik Skor Dağılımı")
                    # Tüm soru detaylarını birleştir
                    skor_data = []
                    for y in yontemler:
                        for s in tum_s[y]["soru_detaylari"]:
                            skor_data.append({"Model": y, "Benzerlik Skoru": s["en_iyi_skor"]})
                    
                    df_skor = pd.DataFrame(skor_data)
                    fig_hist = px.histogram(df_skor, x="Benzerlik Skoru", color="Model", barmode="overlay",
                                            color_discrete_map={"TF-IDF": "#ef4444", "BM25": "#f59e0b", "RAG Semantic": "#00D4AA", "Hybrid": "#a855f7"},
                                            marginal="box", template="plotly_dark")
                    fig_hist.update_layout(height=450)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with tab3:
                    st.markdown("#### Algoritma Karakteristiği")
                    kategoriler = ["Hassasiyet", "İsabet", "NDCG", "MRR", "MAP", "F1", "Hız (Ters)"]
                    def speed_score(ms): return max(0.1, 1 - (ms/2000))
                    
                    fig_radar = go.Figure()
                    for i, (y, s) in enumerate(tum_s.items()):
                        vals = [s["ort_hassasiyet"], s["ort_isabet"], s["ort_ndcg"], s["ort_mrr"], s["ort_map"], s["ort_f1"], speed_score(s["ort_sure_ms"])]
                        fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=kategoriler + [kategoriler[0]], fill='toself', name=y, line_color=renkler[i]))
                    
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, template="plotly_dark", height=500)
                    st.plotly_chart(fig_radar, use_container_width=True)

                with tab4:
                    st.markdown("#### Başarısızlık Analizi (Hard Questions)")
                    # Hangi sorularda tüm modeller düşük kaldı?
                    tum_detay = []
                    for y in yontemler:
                        df_y = pd.DataFrame(tum_s[y]["soru_detaylari"])
                        df_y["Model"] = y
                        tum_detay.append(df_y)
                    
                    df_full = pd.concat(tum_detay)
                    pivot_f1 = df_full.pivot(index="soru", columns="Model", values="f1")
                    pivot_f1["Ort. Zorluk"] = pivot_f1.mean(axis=1)
                    hard_questions = pivot_f1.sort_values("Ort. Zorluk").head(5)
                    
                    st.warning("Aşağıdaki sorular tüm modeller tarafından en düşük puanı aldı (Sistem için en zor sorular):")
                    st.dataframe(hard_questions, use_container_width=True)

                    for y, s in tum_s.items():
                        with st.expander(f"📌 {y} - Tüm Detaylar"):
                            df_detay = pd.DataFrame(s["soru_detaylari"])
                            st.dataframe(df_detay[["soru", "hassasiyet", "mrr", "ndcg", "map", "f1", "sure_ms"]], use_container_width=True, hide_index=True)

                # --- 🤖 Üretim (LLM) Değerlendirmesi ---
                st.markdown("---")
                st.markdown("### 🤖 RAG + LLM Üretim Kalitesi")
                if st.button("⚙️ Gemini Yanıtlarını Analiz Et"):
                    with st.spinner("Gemini API ile yanıtlar üretiliyor ve ölçülüyor..."):
                        gen_s = degerlendirici.uretimi_degerlendir(cevap_uret, baglamlari_getir)
                    
                    if gen_s["degerlendirilen"] > 0:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("BLEU (Dil Benzerliği)", f"{gen_s['ort_bleu']:.3f}")
                        c2.metric("ROUGE-L (Özetleme)", f"{gen_s['ort_rouge_l']:.3f}")
                        c3.metric("Faithfulness (Sadakat)", f"{gen_s['ort_sadakat']:.1%}")
                        c4.metric("Ort. Üretim Süresi", f"{gen_s['ort_sure_ms']:.0f}ms")
                        
                        st.markdown("#### Üretilen Yanıt Örnekleri")
                        st.dataframe(pd.DataFrame(gen_s["soru_detaylari"])[["soru", "bleu", "sadakat", "uretilen"]], use_container_width=True)

                # --- 📥 CSV Export ---
                st.markdown("---")
                st.markdown("### 📥 Sonuçları Dışa Aktar")
                export_rows = []
                for y, s in tum_s.items():
                    export_rows.append({
                        "Model": y,
                        "Precision": round(s["ort_hassasiyet"], 4),
                        "Hit Rate": round(s["ort_isabet"], 4),
                        "MRR": round(s["ort_mrr"], 4),
                        "NDCG": round(s["ort_ndcg"], 4),
                        "MAP": round(s["ort_map"], 4),
                        "F1": round(s["ort_f1"], 4),
                        "Süre (ms)": round(s["ort_sure_ms"], 1)
                    })
                df_export = pd.DataFrame(export_rows)
                st.dataframe(df_export, use_container_width=True, hide_index=True)
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button("📥 CSV Olarak İndir", csv_data, "rag_degerlendirme_sonuclari.csv", "text/csv", use_container_width=True)

# ============================================================
# SAYFA 4: EMBEDDING HARİTASI
# ============================================================
elif sayfa == "🧬 Embedding Haritası":
    st.markdown('<p class="comparison-header">🧬 Embedding Uzay Haritası</p>', unsafe_allow_html=True)
    st.markdown("##### Doküman parçalarının vektör uzayındaki 2D dağılımı (t-SNE)")

    if not st.session_state.vt_hazir or not st.session_state.parcalar:
        st.warning("⚠️ Lütfen önce veritabanını oluşturun.")
    else:
        if st.button("🔬 t-SNE Haritasını Oluştur", use_container_width=True, type="primary"):
            with st.spinner("Embedding vektörleri hesaplanıyor ve t-SNE uygulanıyor..."):
                tsne_data = embedding_gorselleştir(st.session_state.parcalar)
            
            if tsne_data:
                fig_tsne = px.scatter(
                    x=tsne_data["x"], y=tsne_data["y"],
                    color=tsne_data["sources"],
                    hover_name=tsne_data["labels"],
                    template="plotly_dark",
                    labels={"x": "t-SNE Boyut 1", "y": "t-SNE Boyut 2", "color": "Kaynak PDF"},
                    title="Doküman Parçalarının Vektör Uzayı Haritası"
                )
                fig_tsne.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)')))
                fig_tsne.update_layout(height=600, legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig_tsne, use_container_width=True)

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Toplam Parça", len(tsne_data["x"]))
                c2.metric("Kaynak Sayısı", len(set(tsne_data["sources"])))
                c3.metric("Embedding Boyutu", "384D → 2D")

                st.info("💡 **Yorumlama:** Birbirine yakın noktalar anlamsal olarak benzer parçaları temsil eder. Farklı renkteki (farklı PDF'lerden gelen) parçaların birbirine yakın olması, o konuların her iki belgede de ele alındığını gösterir.")
            else:
                st.error("t-SNE için yeterli parça yok (en az 5 gerekli).")

# ============================================================
# SAYFA 5: SİSTEM HAKKINDA
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

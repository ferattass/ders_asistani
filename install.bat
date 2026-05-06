@echo off
echo Sanal ortam olusturuluyor...
python -m venv venv
call venv\Scripts\activate.bat

echo Kutuphaneler indiriliyor (birkac dakika surebilir)...
pip install -r requirements.txt

echo spaCy dil modeli indiriliyor...
python -m spacy download en_core_web_sm

echo.
echo ==========================================
echo   Kurulum tamamlandi!
echo   Streamlit arayuzunu baslatmak icin:
echo   1. venv\Scripts\activate
echo   2. streamlit run app.py
echo ==========================================
pause

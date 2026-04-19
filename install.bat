@echo off
echo Sanal ortam olusturuluyor...
python -m venv venv
call venv\Scripts\activate.bat

echo Kutuphaneler indiriliyor (bu islem internet hizina bagli olarak birkac dakika surebilir)...
pip install -r requirements.txt

echo Spacy dil modeli indiriliyor...
python -m spacy download en_core_web_sm

echo Kurulum tamamlandi!
echo Streamlit arayuzunu baslatmak icin "venv\Scripts\activate" yapip "streamlit run app.py" yaziniz.
pause

# Kurum Asistan Chatbot

Flask tabanli bu proje, kurum ici bilgi tabani, destek talebi, hava durumu ve belge sorgulama senaryolarini tek ekranda toplayan bir asistandir. Uygulama LM Studio uyumlu bir OpenAI API sunucusu ile calisir ve yuklenen PDF/DOCX dosyalarini basit bir RAG akisiyla sorgulayabilir.

## Ozellikler

- Kurum ici bilgi tabanindan soru cevaplama
- Destek talebi olusturma ve dashboard uzerinden izleme
- Hava durumu sorgulama
- PDF ve DOCX yukleme, ozetleme ve belge ici arama
- Dashboard uzerinden gecmis, rapor ve talep goruntuleme
- Kullanici bazli model secimi

## Kurulum

1. Sanal ortam olusturun.

```bash
python -m venv .venv
```

2. Ortami aktif edin.

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

3. Bagimliliklari kurun.

```bash
pip install -r requirements.txt
```

4. Proje kokunde bir `.env` dosyasi olusturun.

```env
FLASK_SECRET_KEY=change-me
LLM_PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=google/gemma-3-12b
LM_STUDIO_API_KEY=
LM_STUDIO_MODELS=google/gemma-3-12b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_MODELS=qwen3.5:9b
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
OPENWEATHER_API_KEY=your_openweather_key
RESET_ON_STARTUP=0
```

5. Uygulamayi baslatin.

```bash
python app.py
```

Tarayicida `http://localhost:5000` adresini acin.

## Notlar

- `RESET_ON_STARTUP=1` yapilirsa sohbet gecmisi, destek talepleri, yuklenen raporlar ve vektor deposu uygulama acilisinda temizlenir.
- Varsayilan davranis olarak veri silme kapali tutulur.
- `LLM_PROVIDER=lmstudio` ya da `LLM_PROVIDER=ollama` ile aktif saglayiciyi secersin.
- Belge embedding akisi Ollama uzerinden `OLLAMA_EMBED_MODEL` kullanir.
- Testler `TEST_DATABASE_URL` ortam degiskeni ile ayri bir SQLite veritabani kullanir.

## Test

```bash
pytest -q
```

# Kurum İçi Akıllı Asistan Chatbotu

Bu proje, kurum içi kullanıma uygun, modern ve kullanıcı dostu bir sohbet botudur. Hem masaüstü hem mobilde şık bir arayüz sunar.

## Özellikler
- 💬 Doğal dilde sohbet edebilme
- 🌤️ Hava durumu sorgulama (örn: "İstanbul'da hava nasıl?")
- 📅 Tarih ve gün bilgisini sorma (örn: "Yarın ne gün?")
- 💼 Kurum içi destek talebi oluşturma (örn: "IT için destek kaydı aç")
- 🗂️ Tüm geçmiş sorguları ve destek taleplerini dashboard'da görüntüleme
- 🌙 Karanlık/Aydınlık tema seçeneği
- Modern, responsive ve kolay kullanımlı arayüz

## Kurulum
1. Depoyu klonlayın ve dizine girin.
2. Sanal ortam oluşturun ve aktif edin:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/Mac: source .venv/bin/activate
   ```
3. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
4. `.env` dosyasına OpenWeatherMap API anahtarınızı ekleyin:
   ```
   OPENWEATHER_API_KEY=API_ANAHTARINIZ
   ```
5. Ollama ile bir LLM modeli çalıştırın (örn: devstral-small veya llama3:8b).
6. Uygulamayı başlatın:
   ```bash
   python app.py
   ```
7. Tarayıcıda `http://localhost:5000` adresine gidin.

## Kullanım
- Sohbet ekranında mesajınızı yazıp gönderin.
- Hava durumu, tarih veya destek talebi gibi işlemleri doğal dilde sorabilirsiniz.
- Dashboard üzerinden geçmiş sorgularınızı ve destek taleplerinizi takip edebilirsiniz.
- Sağ üstteki tema butonuyla karanlık/aydınlık mod arasında geçiş yapabilirsiniz.

## Notlar
- Tüm veriler geçici olarak bellekte tutulur, kullanıcı kimliği desteği eklenebilir.
- Kendi kurumunuza göre departmanları ve özellikleri kolayca özelleştirebilirsiniz.

---
Geliştirici: emredeveloper

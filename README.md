# Kurum İçi Akıllı Asistan Chatbotu

 Kurumiçi süreçleri kolaylaştıran, LLM (yerel LM Studio/OpenAI uyumlu API) tabanlı, modern ve fonksiyonel bir sohbet asistanı. Hava durumu, kurum içi bilgi, destek talebi, belge yükleme ve daha fazlası tek ekranda!

## Özellikler
- 🤖 **LLM tabanlı doğal dilde sohbet** (LM Studio/OpenAI uyumlu API ile yerel model)
- 🔗 **Çoklu tool/fonksiyon zinciri**: Hava durumu, kurum içi bilgi, destek talebi, belge yükleme
- 🏢 **Kurum içi bilgi tabanı**: Sık sorulanlar, prosedürler, politikalar
- 🌤️ **Hava durumu sorgulama** (OpenWeatherMap API)
- 💼 **Destek talebi oluşturma** (departman, açıklama, aciliyet, kategori)
- 🗂️ **Dashboard**: Sorgu geçmişi, hava durumu geçmişi, destek talepleri ve yüklenen raporlar
- 📄 **Word/PDF rapor yükleme**: Hem sohbet ekranından hem dashboard'dan dosya ekleyip yönetme
- �� **Karanlık/Aydınlık tema** (localStorage ile kalıcı)
- 📱 **Modern, responsive ve mobil uyumlu arayüz**
- 🛡️ **API anahtarı .env ile güvenli**

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
<<<<<<< HEAD
5. LM Studio ile bir LLM modeli çalıştırın ve OpenAI uyumlu API'yi etkinleştirin:
   - LM Studio'yu açın ve tercih ettiğiniz modeli başlatın.
   - Server (OpenAI Compatible) özelliğini aktif edin. Varsayılan taban URL genelde `http://localhost:1234/v1` olur.
   - `.env` dosyanıza aşağıdaki değişkenleri ekleyin (örnek değerlerle):
     ```
     LM_STUDIO_BASE_URL=http://localhost:1234/v1
     LM_STUDIO_MODEL=Llama-3.1-8B-Instruct
     # LM_STUDIO_API_KEY=opsiyonel
     OPENWEATHER_API_KEY=API_ANAHTARINIZ
     ```
=======
5. Ollama ile bir LLM modeli çalıştırın https://ollama.com/search?c=tools  bu siteden tools destekli bir model olmalı :
   ```bash
   ollama run qwen3:8b
   ```
>>>>>>> 34a98a8b86e4f910e296f20344d84fb16bf1934d
6. Uygulamayı başlatın:
   ```bash
   python app.py
   ```
7. Tarayıcıda `http://localhost:5000` adresine gidin.

## Kullanım
- **Sohbet ekranı:** Mesajınızı yazıp gönderin, hava durumu, bilgi, destek, belge yükleme gibi işlemleri doğal dilde sorun.
- **Rapor yükleme:** Sohbet ekranından veya dashboard'dan Word/PDF dosyası ekleyin, dashboard'da yönetin ve indirin.
- **Dashboard:** Sorgu geçmişi, hava durumu, destek talepleri ve yüklenen raporları görüntüleyin.
- **Tema:** Sağ üstteki butonla karanlık/aydınlık mod arasında geçiş yapın.

## Notlar
- Tüm veriler geçici olarak bellekte tutulur, kullanıcı kimliği desteği eklenebilir.
- Departmanlar, bilgi tabanı ve tool zinciri kolayca özelleştirilebilir.
- Gelişmiş LLM entegrasyonu ile çoklu tool/fonksiyon zinciri desteklenir.

---
Geliştirici: emredeveloper

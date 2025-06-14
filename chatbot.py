import requests
import os
from dotenv import load_dotenv
import json
import datetime
from dateutil import parser

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "BURAYA_API_KEY_GIRIN")

class CitizenAssistantBot:
    DEPARTMANLAR = ["IT", "İnsan Kaynakları", "Muhasebe", "Teknik Servis"]

    KURUM_BILGI_TABANI = {
        "seyahat politikası": "Şirketimizde şehir dışı seyahatler için önceden onay alınmalı ve masraflar fatura ile belgelendirilmelidir.",
        "izin prosedürü": "Yıllık izinler için en az 3 gün önceden İK'ya başvurulmalıdır.",
        "mesai ücreti": "Fazla mesai ücretleri, ilgili ayın sonunda bordroya yansıtılır.",
        "yemekhane": "Yemekhane hafta içi 12:00-14:00 arası açıktır."
    }

    def __init__(self):
        self.history = []
        self.support_tickets = []
        self.waiting_for_department = False
        self.pending_ticket = None
        self.waiting_for_department_user = None
        self.waiting_for_description = False
        self.last_department = None

    def ollama_chat(self, prompt: str) -> str:
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt},
                stream=True
            )
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            full_response += data["response"]
                    except Exception:
                        continue
            return full_response if full_response else "LLM'den yanıt alınamadı."
        except Exception as e:
            return f"LLM hatası: {e}"

    def get_weather(self, city: str) -> str:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&lang=tr&units=metric"
            response = requests.get(url)
            data = response.json()
            if data.get("cod") != 200:
                if data.get("message", "") == "city not found":
                    return "Hangi şehir için hava durumunu öğrenmek istiyorsunuz?"
                return f"Hava durumu alınamadı: {data.get('message', '')}"
            desc = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"{city} için hava: {desc}, sıcaklık: {temp}°C"
        except Exception as e:
            return f"Hava durumu hatası: {e}"

    def normalize_dept(self, text):
        return text.strip().lower().replace("ı", "i").replace("ç", "c").replace("ş", "s").replace("ö", "o").replace("ü", "u").replace("ğ", "g")

    def get_kurum_bilgisi(self, soru):
        for anahtar, cevap in self.KURUM_BILGI_TABANI.items():
            if anahtar in soru.lower():
                return cevap
        return "Bu konuda bilgi bulunamadı."

    def process_message(self, message: str, user_id: str = "default") -> str:
        # Bağlam için son 3 mesajı al
        last_msgs = [h for h in self.history if h.get("user_id") == user_id][-3:]
        context = "\n".join([f"Kullanıcı: {h['kullanici']}\nBot: {h['cevap']}" for h in last_msgs])

        # Açıklama/departman state'leri (önceki kod aynı)
        if self.waiting_for_description and self.pending_ticket and self.waiting_for_department_user == user_id:
            self.pending_ticket['aciklama'] = message.strip()
            self.pending_ticket['okundu'] = False
            self.pending_ticket['olusturma_zamani'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            self.pending_ticket['user_id'] = user_id
            self.support_tickets.append(self.pending_ticket)
            self.waiting_for_description = False
            self.pending_ticket = None
            self.waiting_for_department_user = None
            return f"{self.last_department} departmanına destek talebiniz oluşturuldu. Talebinizin durumunu dashboard'dan takip edebilirsiniz."
        if self.waiting_for_department and self.pending_ticket and self.waiting_for_department_user == user_id:
            department_input = self.normalize_dept(message)
            matched_dept = None
            for dept in self.DEPARTMANLAR:
                norm_dept = self.normalize_dept(dept)
                if norm_dept in department_input or department_input in norm_dept:
                    matched_dept = dept
                    break
            if not matched_dept:
                return f"Geçersiz departman. Lütfen şu seçeneklerden birini yazın: {', '.join(self.DEPARTMANLAR)}"
            self.pending_ticket['departman'] = matched_dept
            self.last_department = matched_dept
            self.waiting_for_department = False
            self.waiting_for_description = True
            return "Lütfen destek talebinizin açıklamasını yazar mısınız?"

        # LLM ile tool chain/fonksiyon çağrısı
        llm_prompt = f'''
Aşağıda son konuşma geçmişi ve yeni kullanıcı mesajı var. Eğer bir veya birden fazla tool çağrısı gerekiyorsa bana şu formatta JSON döndür:
Tek tool için: {{"tool": "hava_durumu", "sehir": "İstanbul"}}
Çoklu tool için: [{{"tool": "hava_durumu", "sehir": "İstanbul"}}, {{"tool": "kurum_bilgisi", "soru": "seyahat politikası"}}]
Kurum içi bilgi için: {{"tool": "kurum_bilgisi", "soru": "seyahat politikası"}}
Destek talebi için: {{"tool": "destek_talebi", "departman": "IT", "aciklama": "Bilgisayarım bozuldu", "aciliyet": "acil", "kategori": "donanım"}}
Eğer tool çağrısı yoksa sadece normal cevabını ver.

Son konuşma geçmişi:
{context}

Kullanıcı mesajı: {message}
'''
        llm_response = self.ollama_chat(llm_prompt)
        try:
            data = json.loads(llm_response)
            # Çoklu tool chain desteği
            if isinstance(data, list):
                results = []
                for item in data:
                    results.append(self._handle_tool_call(item, message, user_id))
                return "\n\n".join(results)
            # Tek tool
            if isinstance(data, dict) and data.get('tool'):
                return self._handle_tool_call(data, message, user_id)
        except Exception:
            pass
        # Tool call yoksa, tarih ve diğer tool'lar için eski kodu kullan
        lower_msg = message.lower()
        if "bugün" in lower_msg:
            bugun = datetime.datetime.now()
            gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            cevap = f"Bugün {gunler[bugun.weekday()]}, {bugun.day} {bugun.strftime('%B %Y')}."
            self.history.append({
                "tip": "tarih",
                "kullanici": message,
                "cevap": cevap,
                "tarih": bugun.strftime("%Y-%m-%d"),
                "user_id": user_id
            })
            return cevap
        if "yarın" in lower_msg:
            yarin = datetime.datetime.now() + datetime.timedelta(days=1)
            gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            cevap = f"Yarın {gunler[yarin.weekday()]}, {yarin.day} {yarin.strftime('%B %Y')}."
            self.history.append({
                "tip": "tarih",
                "kullanici": message,
                "cevap": cevap,
                "tarih": yarin.strftime("%Y-%m-%d"),
                "user_id": user_id
            })
            return cevap
        if "dün" in lower_msg:
            dun = datetime.datetime.now() - datetime.timedelta(days=1)
            gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            cevap = f"Dün {gunler[dun.weekday()]}, {dun.day} {dun.strftime('%B %Y')}."
            self.history.append({
                "tip": "tarih",
                "kullanici": message,
                "cevap": cevap,
                "tarih": dun.strftime("%Y-%m-%d"),
                "user_id": user_id
            })
            return cevap
        # İngilizce tarih varsa Türkçe'ye çevir
        try:
            dt = parser.parse(message, fuzzy=True, dayfirst=True)
            gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
            gun_adi = gunler[dt.weekday()]
            ay_adi = aylar[dt.month-1]
            cevap = f"{dt.day} {ay_adi} {dt.year} {gun_adi}"
            self.history.append({
                "tip": "tarih",
                "kullanici": message,
                "cevap": cevap,
                "tarih": dt.strftime("%Y-%m-%d"),
                "user_id": user_id
            })
            return cevap
        except Exception:
            pass
        # LLM ile sohbet
        cevap = self.ollama_chat(message)
        self.history.append({
            "tip": "llm",
            "kullanici": message,
            "cevap": cevap,
            "user_id": user_id
        })
        return cevap

    def _handle_tool_call(self, data, message, user_id):
        tool = data.get('tool')
        if tool == 'hava_durumu':
            sehir = data.get('sehir', 'İstanbul')
            cevap = self.get_weather(sehir)
            self.history.append({
                "tip": "hava_durumu",
                "kullanici": message,
                "cevap": cevap,
                "sehir": sehir,
                "user_id": user_id
            })
            return cevap
        if tool == 'kurum_bilgisi':
            soru = data.get('soru', '')
            cevap = self.get_kurum_bilgisi(soru)
            self.history.append({
                "tip": "kurum_bilgisi",
                "kullanici": message,
                "cevap": cevap,
                "soru": soru,
                "user_id": user_id
            })
            return cevap
        if tool == 'destek_talebi':
            department_input = self.normalize_dept(data.get('departman', ''))
            matched_dept = None
            for dept in self.DEPARTMANLAR:
                norm_dept = self.normalize_dept(dept)
                if norm_dept in department_input or department_input in norm_dept:
                    matched_dept = dept
                    break
            aciklama = data.get('aciklama', None)
            aciliyet = data.get('aciliyet', 'normal')
            kategori = data.get('kategori', 'genel')
            if not matched_dept:
                self.waiting_for_department = True
                self.pending_ticket = {
                    "aciklama": aciklama,
                    "departman": None,
                    "okundu": False,
                    "olusturma_zamani": None,
                    "user_id": user_id,
                    "aciliyet": aciliyet,
                    "kategori": kategori
                }
                self.waiting_for_department_user = user_id
                return f"Geçersiz departman. Lütfen şu seçeneklerden birini yazın: {', '.join(self.DEPARTMANLAR)}"
            if not aciklama:
                self.pending_ticket = {
                    "aciklama": None,
                    "departman": matched_dept,
                    "okundu": False,
                    "olusturma_zamani": None,
                    "user_id": user_id,
                    "aciliyet": aciliyet,
                    "kategori": kategori
                }
                self.last_department = matched_dept
                self.waiting_for_description = True
                self.waiting_for_department_user = user_id
                self.waiting_for_department = False
                return "Lütfen destek talebinizin açıklamasını yazar mısınız?"
            ticket = {
                "aciklama": aciklama,
                "departman": matched_dept,
                "okundu": False,
                "olusturma_zamani": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "user_id": user_id,
                "aciliyet": aciliyet,
                "kategori": kategori
            }
            self.support_tickets.append(ticket)
            return f"{matched_dept} departmanına destek talebiniz oluşturuldu. (Aciliyet: {aciliyet}, Kategori: {kategori}) Dashboard'dan takip edebilirsiniz."
        return "Tool çağrısı tanınmadı."

    def get_history(self, user_id: str = "default"):
        return [h for h in self.history[-20:] if h.get("user_id") == user_id]

    def get_support_tickets(self, user_id: str = "default"):
        return [t for t in self.support_tickets[::-1] if t.get("user_id") == user_id]

    def mark_ticket_as_read(self, idx, user_id: str = "default"):
        user_tickets = [t for t in self.support_tickets if t.get("user_id") == user_id]
        if 0 <= idx < len(user_tickets):
            user_tickets[idx]['okundu'] = True
            return True
        return False

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

    def process_message(self, message: str, user_id: str = "default") -> str:
        # Eğer açıklama bekleniyorsa
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

        # Eğer departman bekleniyorsa
        if self.waiting_for_department and self.pending_ticket and self.waiting_for_department_user == user_id:
            department_input = self.normalize_dept(message)
            for dept in self.DEPARTMANLAR:
                if self.normalize_dept(dept) == department_input:
                    department = dept
                    break
            else:
                return f"Geçersiz departman. Lütfen şu seçeneklerden birini yazın: {', '.join(self.DEPARTMANLAR)}"
            self.pending_ticket['departman'] = department
            self.last_department = department
            self.waiting_for_department = False
            self.waiting_for_description = True
            return "Lütfen destek talebinizin açıklamasını yazar mısınız?"

        # LLM ile tool call/fonksiyon çağrısı
        llm_prompt = f'''
Aşağıdaki kullanıcı mesajını analiz et. 
Eğer bir tool çağrısı gerekiyorsa bana şu formatta JSON döndür: 
{{"tool": "hava_durumu", "sehir": "İstanbul"}} 
veya {{"tool": "destek_talebi", "departman": "IT", "aciklama": "Bilgisayarım bozuldu"}}. 
Eğer tool çağrısı yoksa sadece normal cevabını ver. 
Kullanıcı mesajı: {message}
'''
        llm_response = self.ollama_chat(llm_prompt)
        try:
            data = json.loads(llm_response)
            if data.get('tool') == 'hava_durumu':
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
            if data.get('tool') == 'destek_talebi':
                department = data.get('departman', '').capitalize()
                if department not in self.DEPARTMANLAR:
                    self.waiting_for_department = True
                    self.pending_ticket = {
                        "aciklama": None,
                        "departman": None,
                        "okundu": False,
                        "olusturma_zamani": None,
                        "user_id": user_id
                    }
                    self.waiting_for_department_user = user_id
                    return f"Geçersiz departman. Lütfen şu seçeneklerden birini yazın: {', '.join(self.DEPARTMANLAR)}"
                self.pending_ticket = {
                    "aciklama": data.get('aciklama', None),
                    "departman": department,
                    "okundu": False,
                    "olusturma_zamani": None,
                    "user_id": user_id
                }
                self.last_department = department
                if not self.pending_ticket['aciklama']:
                    self.waiting_for_description = True
                    self.waiting_for_department_user = user_id
                    self.waiting_for_department = False
                    return "Lütfen destek talebinizin açıklamasını yazar mısınız?"
                self.pending_ticket['olusturma_zamani'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                self.support_tickets.append(self.pending_ticket)
                self.pending_ticket = None
                return f"{department} departmanına destek talebiniz oluşturuldu. Talebinizin durumunu dashboard'dan takip edebilirsiniz."
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

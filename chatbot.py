import requests
import os
from dotenv import load_dotenv
import json
import datetime
import uuid # For generating ticket IDs
from dateutil import parser
import database # Import the new database module
from googletrans import Translator # For translation
from document_processor import processor as doc_processor # Import the document processor

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
        # self.histories and self.all_support_tickets are now removed
        # User states are kept in memory for the duration of a session's multi-turn interactions
        self.user_states = {}

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

    def normalize_dept(self, text: str) -> str:
        text = text.replace("İ", "i") # Convert uppercase dotted I to lowercase i
        text = text.replace("I", "i") # Convert uppercase dotless I to lowercase i (Turkish 'ı' becomes 'i')
        text = text.lower() # Convert to lowercase

        # Standard Turkish character normalization for search/matching
        replacements = {
            "ı": "i",
            "ç": "c",
            "ş": "s",
            "ö": "o",
            "ü": "u",
            "ğ": "g",
        }
        for char_tr, char_en in replacements.items():
            text = text.replace(char_tr, char_en)
        return text.strip()

    def get_kurum_bilgisi(self, soru):
        # Normalize the question for more robust matching
        normalized_soru = self.normalize_dept(soru) # Using normalize_dept for general text normalization
        for anahtar, cevap in self.KURUM_BILGI_TABANI.items():
            # Normalize keywords from the knowledge base as well for consistent matching
            normalized_anahtar = self.normalize_dept(anahtar)
            if normalized_anahtar in normalized_soru:
                return cevap
        return "Bu konuda bilgi bulunamadı."

    # The following def process_message was the start of the duplicated block.
    # The lines "if anahtar in soru.lower(): return cevap" and "return "Bu konuda..."
    # were remnants of the old get_kurum_bilgisi logic that got misplaced.
    # They are removed by this diff not including them before the correct process_message.

    def process_message(self, message: str, user_id: str) -> str:
        user_state = self.user_states.setdefault(user_id, {})
        bot_response = "" # Initialize bot_response

        # Bağlam için son 3 mesajı al from database
        db_history = database.get_chat_history(user_id, limit=3) # Returns newest first
        context_msgs = []
        for h_entry in reversed(db_history): # Reverse to get chronological order for context
            context_msgs.append(f"Kullanıcı: {h_entry['user_message']}\nBot: {h_entry['bot_response']}")
        context = "\n".join(context_msgs)

        # Açıklama/departman state'leri
        if user_state.get('waiting_for_description') and user_state.get('pending_ticket'):
            pending_ticket_info = user_state['pending_ticket']
            pending_ticket_info['aciklama'] = message.strip()

            ticket_id = uuid.uuid4().hex[:8]
            department = pending_ticket_info.get('departman', user_state.get('last_department'))
            description = pending_ticket_info['aciklama']
            priority = pending_ticket_info.get('aciliyet', 'normal')
            category = pending_ticket_info.get('kategori', 'genel')

            database.add_support_ticket(user_id, ticket_id, department, description, priority, category)

            bot_response = f"{department} departmanına {ticket_id} ID'li destek talebiniz oluşturuldu. Talebinizin durumunu dashboard'dan takip edebilirsiniz."
            database.add_chat_history(user_id, "destek_talebi_oluşturuldu", message, bot_response, json.dumps({
                "ticket_id": ticket_id, "department": department, "priority": priority, "category": category
            }))

            user_state['waiting_for_description'] = False
            user_state['pending_ticket'] = None
            user_state['last_department'] = None
            return bot_response

        if user_state.get('waiting_for_department') and user_state.get('pending_ticket'):
            department_input = self.normalize_dept(message)
            matched_dept = None
            # department_input is already normalized by process_message before this state is reached,
            # or by _handle_tool_call if LLM provides it.
            # Let's re-normalize here to be sure, or ensure it's always normalized before this check.
            # For now, assume department_input (user's typed message) needs normalization here.
            normalized_department_input = self.normalize_dept(message)

            for dept_option in self.DEPARTMANLAR:
                normalized_dept_option = self.normalize_dept(dept_option)
                if normalized_dept_option in normalized_department_input or normalized_department_input == normalized_dept_option:
                    matched_dept = dept_option # Store the original department name
                    break
            if not matched_dept:
                bot_response = f"Geçersiz departman. Lütfen şu seçeneklerden birini yazın: {', '.join(self.DEPARTMANLAR)}"
                database.add_chat_history(user_id, "destek_talebi_etkileşim", message, bot_response)
                return bot_response

            user_state['pending_ticket']['departman'] = matched_dept # Store original casing
            user_state['last_department'] = matched_dept
            user_state['waiting_for_department'] = False
            user_state['waiting_for_description'] = True
            bot_response = "Lütfen destek talebinizin açıklamasını yazar mısınız?"
            database.add_chat_history(user_id, "destek_talebi_etkileşim", message, bot_response)
            return bot_response

        # LLM ile tool chain/fonksiyon çağrısı
        llm_prompt = f'''
Aşağıda son konuşma geçmişi ve yeni kullanıcı mesajı var. Eğer bir veya birden fazla tool çağrısı gerekiyorsa bana şu formatta JSON döndür:
Tek tool için: {{"tool": "hava_durumu", "sehir": "İstanbul"}}
Çoklu tool için: [{{"tool": "hava_durumu", "sehir": "İstanbul"}}, {{"tool": "kurum_bilgisi", "soru": "seyahat politikası"}}]
Kurum içi bilgi için: {{"tool": "kurum_bilgisi", "soru": "seyahat politikası"}}
Destek talebi için: {{"tool": "destek_talebi", "departman": "IT", "aciklama": "Bilgisayarım bozuldu", "aciliyet": "acil", "kategori": "donanım"}}
Belge sorgulamak için: {{"tool": "belge_sorgulama", "sorgu": "Yıllık izin prosedürü nedir?"}}
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
            if isinstance(data, dict) and data.get('tool'): # Reverted to original condition
                return self._handle_tool_call(data, message, user_id)
        except Exception:
            pass
        # Tool call yoksa, tarih ve diğer tool'lar için eski kodu kullan
        lower_msg = message.lower()
        date_type = None
        date_details = None

        if "bugün" in lower_msg:
            bugun = datetime.datetime.now()
            gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            bot_response = f"Bugün {gunler[bugun.weekday()]}, {bugun.day} {bugun.strftime('%B %Y')}."
            date_type = "tarih_bugün"
            date_details = {"tarih_detay": bugun.strftime("%Y-%m-%d")}
        elif "yarın" in lower_msg:
            yarin = datetime.datetime.now() + datetime.timedelta(days=1)
            gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            bot_response = f"Yarın {gunler[yarin.weekday()]}, {yarin.day} {yarin.strftime('%B %Y')}."
            date_type = "tarih_yarın"
            date_details = {"tarih_detay": yarin.strftime("%Y-%m-%d")}
        elif "dün" in lower_msg:
            dun = datetime.datetime.now() - datetime.timedelta(days=1)
            gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            bot_response = f"Dün {gunler[dun.weekday()]}, {dun.day} {dun.strftime('%B %Y')}."
            date_type = "tarih_dün"
            date_details = {"tarih_detay": dun.strftime("%Y-%m-%d")}
        else:
            try:
                dt = parser.parse(message, fuzzy=True, dayfirst=True)
                gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
                aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
                gun_adi = gunler[dt.weekday()]
                ay_adi = aylar[dt.month-1]
                bot_response = f"{dt.day} {ay_adi} {dt.year} {gun_adi}"
                date_type = "tarih_parse"
                date_details = {"tarih_detay": dt.strftime("%Y-%m-%d")}
            except Exception:
                pass # Not a parsable date

        if date_type and bot_response:
            database.add_chat_history(user_id, date_type, message, bot_response, json.dumps(date_details) if date_details else None)
            return bot_response

        # LLM ile sohbet (eğer tool/tarih değilse)
        bot_response = self.ollama_chat(message) # Use original message for pure LLM chat
        database.add_chat_history(user_id, "llm_cevap", message, bot_response)
        return bot_response

    def _handle_tool_call(self, data, original_user_message: str, user_id: str):
        user_state = self.user_states.setdefault(user_id, {})
        tool = data.get('tool')
        bot_response = ""

        if tool == 'hava_durumu':
            sehir = data.get('sehir', 'İstanbul') # Default to İstanbul if not provided
            bot_response = self.get_weather(sehir)
            database.add_chat_history(user_id, "hava_durumu", original_user_message, bot_response, json.dumps({"sehir": sehir}))
            return bot_response

        if tool == 'kurum_bilgisi':
            soru = data.get('soru', '')
            bot_response = self.get_kurum_bilgisi(soru)
            database.add_chat_history(user_id, "kurum_bilgisi", original_user_message, bot_response, json.dumps({"soru": soru}))
            return bot_response

        if tool == 'belge_sorgulama':
            query = data.get('sorgu', '')
            if not query:
                return "Lütfen belge içinde ne aramak istediğinizi belirtin."

            # 1. Search for relevant document chunks
            search_results = doc_processor.search_in_documents(query, top_k=3)
            
            if not search_results:
                return "Yüklenmiş belgelerde bu konuyla ilgili bir bilgi bulamadım."

            # 2. Construct context for the LLM
            context_for_llm = "\n\n".join([result['text'] for result in search_results])
            
            # 3. Ask the LLM to generate a response based on the context
            rag_prompt = f'''
Kullanıcının sorusunu, aşağıda verilen belge içeriklerini kullanarak yanıtla. 
Cevabını sadece bu içeriklere dayandır. Cevabını maddeler halinde (markdown listesi kullanarak) düzenli ve okunaklı bir şekilde formatla.
Eğer cevap bu içeriklerde yoksa, 'Belgelerde bu konuda bilgi bulamadım' de.

Belge İçerikleri:
{context_for_llm}

Kullanıcı Sorusu: {query}

Yanıtın:
'''
            bot_response = self.ollama_chat(rag_prompt)
            database.add_chat_history(user_id, "belge_sorgulama", original_user_message, bot_response, json.dumps({"sorgu": query, "context": "..."})) # Context'i kaydetmekten kaçınarak veritabanı boyutunu küçült
            return bot_response

        if tool == 'destek_talebi':
            department_input_from_llm = data.get('departman', '')
            normalized_department_input_llm = self.normalize_dept(department_input_from_llm)
            matched_dept = None
            for dept_option in self.DEPARTMANLAR:
                normalized_dept_option = self.normalize_dept(dept_option)
                if normalized_dept_option in normalized_department_input_llm or normalized_department_input_llm == normalized_dept_option:
                    matched_dept = dept_option # Store original casing
                    break

            aciklama = data.get('aciklama', None)
            aciliyet = data.get('aciliyet', 'normal')
            kategori = data.get('kategori', 'genel')

            # This structure is for when LLM provides all details, or for setting up pending state
            current_ticket_data = {
                "aciklama": aciklama,
                "departman": matched_dept,
                "aciliyet": aciliyet,
                "kategori": kategori,
                "user_id": user_id # Store user_id with pending data
            }

            if not matched_dept:
                user_state['waiting_for_department'] = True
                user_state['pending_ticket'] = current_ticket_data # Store all gathered data
                bot_response = f"Anladım, destek talebi oluşturmak istiyorsunuz. Hangi departmana iletelim? Seçenekler: {', '.join(self.DEPARTMANLAR)}"
                database.add_chat_history(user_id, "destek_talebi_etkileşim", original_user_message, bot_response)
                return bot_response

            if not aciklama:
                user_state['pending_ticket'] = current_ticket_data # Store all gathered data
                user_state['last_department'] = matched_dept
                user_state['waiting_for_description'] = True
                user_state['waiting_for_department'] = False
                bot_response = f"{matched_dept} departmanı için destek talebi oluşturuyorum. Lütfen talebinizin açıklamasını yazar mısınız?"
                database.add_chat_history(user_id, "destek_talebi_etkileşim", original_user_message, bot_response)
                return bot_response

            # Both department and description are available directly from LLM tool call
            ticket_id = uuid.uuid4().hex[:8]
            database.add_support_ticket(user_id, ticket_id, matched_dept, aciklama, aciliyet, kategori)

            bot_response = f"{matched_dept} departmanına {ticket_id} ID'li destek talebiniz oluşturuldu. (Aciliyet: {aciliyet}, Kategori: {kategori}) Dashboard'dan takip edebilirsiniz."
            database.add_chat_history(user_id, "destek_talebi_oluşturuldu", original_user_message, bot_response, json.dumps({
                "ticket_id": ticket_id, "department": matched_dept, "aciliyet": aciliyet, "kategori": kategori
            }))

            # Clear states related to this ticket as it's now fully created
            user_state['pending_ticket'] = None
            user_state['waiting_for_department'] = False
            user_state['waiting_for_description'] = False
            user_state['last_department'] = None
            return bot_response

        bot_response = "Tool çağrısı tanınmadı veya işlenemedi."
        database.add_chat_history(user_id, "tool_hata", original_user_message, bot_response, json.dumps(data))
        return bot_response

    def get_history(self, user_id: str):
        # Returns list of dicts, already ordered by timestamp desc, limit 20 by default in db func
        return database.get_chat_history(user_id)

    def get_support_tickets(self, user_id: str):
        # Returns list of dicts, ordered by created_at desc, limit 50 by default in db func
        return database.get_support_tickets(user_id)

    def mark_ticket_as_read(self, idx: int, user_id: str):
        # get_support_tickets returns tickets ordered by created_at DESC (newest first)
        user_tickets = database.get_support_tickets(user_id)

        if 0 <= idx < len(user_tickets):
            ticket_to_mark = user_tickets[idx] # idx directly applies to the newest-first list
            ticket_id = ticket_to_mark['ticket_id']
            # The 'status' in DB for a read ticket could be 'read' or 'viewed'
            # The previous in-memory version just set an 'okundu' boolean.
            # We'll use 'read' as the status.
            return database.update_support_ticket_status(user_id, ticket_id, 'read')
        return False

    def translate_message(self, message: str, target_lang: str = 'en') -> str:
        """Translates a message to the target language using Google Translate."""
        if not message.strip():
            return "" # Or handle as an error: "No message provided for translation."

        try:
            translator = Translator()
            # Detect source language automatically, translate to target_lang
            translated_text = translator.translate(message, dest=target_lang).text
            return translated_text
        except Exception as e:
            # Log the error e if logging is set up
            print(f"Translation error: {e}") # Basic print for now
            return "Çeviri hizmetinde bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

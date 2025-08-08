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

# LM Studio (OpenAI uyumlu) yapılandırması
# Örn: LM_STUDIO_BASE_URL=http://localhost:1234/v1
#      LM_STUDIO_MODEL=YourModelName
#      LM_STUDIO_API_KEY=lm-studio (gerekmeyebilir)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "openai/gpt-oss-20b")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "")
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
        self.user_models = {}

    def set_user_model(self, user_id: str, model_name: str | None):
        if not user_id:
            return
        if model_name and isinstance(model_name, str) and model_name.strip():
            self.user_models[user_id] = model_name.strip()
        else:
            # Reset to default if empty
            if user_id in self.user_models:
                del self.user_models[user_id]

    def ollama_chat(self, prompt: str, model: str | None = None) -> str:
        """LM Studio'nun OpenAI uyumlu Chat Completions API'sini kullanır.

        İsim geriye dönük uyumluluk için korunmuştur (testler bu ismi mock'luyor).
        """
        try:
            url = f"{LM_STUDIO_BASE_URL}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if LM_STUDIO_API_KEY:
                headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

            payload = {
                "model": (model or LM_STUDIO_MODEL),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "stream": False
            }
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            # OpenAI uyumlu yanıttan içerik çıkarma
            choices = data.get("choices", [])
            if choices and "message" in choices[0] and choices[0]["message"].get("content"):
                return choices[0]["message"]["content"].strip()

            # Eski /v1/completions uyumluluğu için fallback (bazı LM Studio sürümleri text döndürebilir)
            if choices and choices[0].get("text"):
                return choices[0]["text"].strip()

            return "LLM'den yanıt alınamadı."
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
        if not isinstance(text, str):
            return ""
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
            bot_response = f"{matched_dept} departmanı için destek talebi oluşturuyorum. Lütfen talebinizin açıklamasını yazar mısınız?"
            database.add_chat_history(user_id, "destek_talebi_etkileşim", message, bot_response)
            return bot_response

        # 0) Hızlı bilgi tabanı kontrolü (LLM'e gitmeden önce)
        kb_answer = self.get_kurum_bilgisi(message)
        if kb_answer and kb_answer != "Bu konuda bilgi bulunamadı.":
            bot_response = kb_answer
            database.add_chat_history(user_id, "kurum_bilgisi_heuristic", message, bot_response, json.dumps({"matched": True}))
            return bot_response

        # 1) LLM ile tool chain/fonksiyon çağrısı
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
        llm_response = self.ollama_chat(llm_prompt, model=self.user_models.get(user_id))
        # LLM çıktısından JSON çıkarma (kod bloğu veya metin içinde gömülü olabilir)
        data = self._extract_json(llm_response)
        try:
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
        bot_response = self.ollama_chat(message, model=self.user_models.get(user_id)) # Use original message for pure LLM chat
        database.add_chat_history(user_id, "llm_cevap", message, bot_response)
        return bot_response

    def _extract_json(self, text: str):
        """Metin içinden JSON (dict veya list) güvenli şekilde çıkarmaya çalışır.

        - Doğrudan json.loads ile dener
        - Ardından ```json ... ``` veya ``` ... ``` bloklarını tarar
        - Son olarak ilk dengeli {..} veya [..] bloğunu kaba regex ile yakalamayı dener
        Başarısız olursa None döner.
        """
        if not text:
            return None
        # 1) Doğrudan JSON denemesi
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) Kod bloğu içinde `json` veya normal üç tırnaklı bloklar
        import re
        code_block_pattern = re.compile(r"```(json)?\s*([\s\S]*?)```", re.IGNORECASE)
        for match in code_block_pattern.finditer(text):
            candidate = match.group(2).strip()
            try:
                return json.loads(candidate)
            except Exception:
                continue

        # 3) İlk dengeli küme veya köşeli parantez içeriğini yakalama (kaba yaklaşım)
        # Not: Bu, iç içe parantezlerde başarısız olabilir, ancak pratikte çoğu LLM çıktısında yeterli olur.
        braces_match = re.search(r"\{[\s\S]*\}", text)
        if braces_match:
            candidate = braces_match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        brackets_match = re.search(r"\[[\s\S]*\]", text)
        if brackets_match:
            candidate = brackets_match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        return None

    def get_citizen_dashboard(self):
        """Dashboard özet verileri: destek talepleri, raporlar ve son etkileşimler.

        Dönen yapı, frontend için kolay tüketilebilir bir özet sağlar.
        """
        try:
            # Kullanıcı bağımsız genel özet
            reports = database.get_reports()
            total_reports = len(reports)
            processed_reports = sum(1 for r in reports if r.get('processed') == 1)
            unprocessed_reports = total_reports - processed_reports

            # Destek taleplerinde durum dağılımı
            # Tüm kullanıcılar için almak üzere direkt DB bağlantısıyla ufak sorgu
            conn = database.get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT status, COUNT(*) as cnt FROM support_tickets GROUP BY status")
            status_rows = cur.fetchall()
            conn.close()
            ticket_status_counts = {row[0]: row[1] for row in status_rows} if status_rows else {}

            # Son 10 sohbet girdisi (global) – tabloya kolayca doldurulabilmesi için
            conn = database.get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id, type, user_message, bot_response, timestamp FROM chat_history ORDER BY timestamp DESC LIMIT 10")
            last_interactions = [
                {
                    'user_id': row[0],
                    'type': row[1],
                    'user_message': row[2],
                    'bot_response': row[3],
                    'timestamp': row[4]
                }
                for row in cur.fetchall()
            ]
            conn.close()

            return {
                'reports': {
                    'total': total_reports,
                    'processed': processed_reports,
                    'unprocessed': unprocessed_reports
                },
                'tickets': {
                    'by_status': ticket_status_counts
                },
                'last_interactions': last_interactions
            }
        except Exception as e:
            return {
                'error': f'Dashboard verileri alınırken hata oluştu: {e}'
            }

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

            # Kullanıcının yüklediği raporları al
            user_reports = database.get_reports(user_id)
            if not user_reports:
                # Eğer belge yoksa ama kurum içi bilgi tabanında bulunuyorsa, onu yanıtla
                kb_fallback = self.get_kurum_bilgisi(query)
                if kb_fallback and kb_fallback != "Bu konuda bilgi bulunamadı.":
                    database.add_chat_history(user_id, "kurum_bilgisi_fallback", query, kb_fallback, json.dumps({"from": "belge_sorgulama"}))
                    return kb_fallback
                return "Herhangi bir belge yüklemediniz. Lütfen önce bir belge yükleyin."
            if len(user_reports) == 1:
                # Tek belge varsa, doğrudan o belgeyle sorgula
                report_id = user_reports[0]['id']
                return self._explain_report(report_id, query, original_user_message, user_id)
            else:
                # Çoklu belge varsa, kullanıcıya seçim sun
                report_options = [f"{r['id']}: {r['original_filename']}" for r in user_reports]
                options_text = '\n'.join(report_options)
                return f"Birden fazla belge bulundu. Lütfen açıklamak istediğiniz belgeyi seçin (ID ile):\n{options_text}\nÖrnek: belgeyi açıkla 12"

        # Kullanıcı belge seçip tekrar sorduğunda (ör: belgeyi açıkla 12)
        if tool == 'belge_sec_ve_acikla':
            report_id = data.get('report_id')
            query = data.get('sorgu', '')
            if not report_id or not query:
                return "Belge ID ve açıklama isteği gereklidir."
            return self._explain_report(report_id, query, original_user_message, user_id)

        if tool == 'destek_talebi':
            department_input_from_llm = data.get('departman') or ''
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

    def _explain_report(self, report_id, query, original_user_message, user_id):
        # Sadece seçili raporun chunk'larında arama yap
        search_results = doc_processor.search_in_documents(query, top_k=3)
        filtered_results = [r for r in search_results if str(r['report_id']) == str(report_id)]
        if not filtered_results:
            return "Seçili belgede bu konuyla ilgili bir bilgi bulamadım."
        context_for_llm = "\n\n".join([result['text'] for result in filtered_results])
        rag_prompt = f'''
Kullanıcının sorusunu, aşağıda verilen belge içeriklerini kullanarak yanıtla. 
Cevabını sadece bu içeriklere dayandır. Cevabını maddeler halinde (markdown listesi kullanarak) düzenli ve okunaklı bir şekilde formatla.
Eğer cevap bu içeriklerde yoksa, 'Belgelerde bu konuda bilgi bulamadım' de.

Belge İçerikleri:
{context_for_llm}

Kullanıcı Sorusu: {query}

Yanıtın:
'''
        bot_response = self.ollama_chat(rag_prompt, model=self.user_models.get(user_id))
        database.add_chat_history(user_id, "belge_sorgulama", original_user_message, bot_response, json.dumps({"sorgu": query, "report_id": report_id}))
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

import requests
import os
from dotenv import load_dotenv
import json
import datetime
from typing import Iterator
import uuid # For generating ticket IDs
from dateutil import parser
import database # Import the new database module
from googletrans import Translator # For translation
from document_processor import processor as doc_processor # Import the document processor

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio").strip().lower()

# LM Studio (OpenAI uyumlu) yapılandırması
# Örn: LM_STUDIO_BASE_URL=http://localhost:1234/v1
#      LM_STUDIO_MODEL=YourModelName
#      LM_STUDIO_API_KEY=lm-studio (gerekmeyebilir)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
# Zorunlu model
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-3-12b")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "BURAYA_API_KEY_GIRIN")


def get_default_model():
    if LLM_PROVIDER == "ollama":
        return OLLAMA_MODEL
    return LM_STUDIO_MODEL

class CitizenAssistantBot:

    def __init__(self):
        # self.histories and self.all_support_tickets are now removed
        # User states are kept in memory for the duration of a session's multi-turn interactions
        self.user_states = {}
        self.user_models = {}
        # Load departments dynamically at startup
        self.DEPARTMANLAR = database.get_departments()

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
            selected_model = model or get_default_model()

            if LLM_PROVIDER == "ollama":
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": selected_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2}
                    },
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                if data.get("response"):
                    return data["response"].strip()
                return "LLM'den yanÄ±t alÄ±namadÄ±."

            url = f"{LM_STUDIO_BASE_URL}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if LM_STUDIO_API_KEY:
                headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

            payload = {
                "model": selected_model,
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

    def ollama_chat_stream(self, prompt: str, model: str | None = None) -> Iterator[str]:
        selected_model = model or get_default_model()

        try:
            if LLM_PROVIDER == "ollama":
                with requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": selected_model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {"temperature": 0.2}
                    },
                    timeout=120,
                    stream=True
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                return

            headers = {"Content-Type": "application/json"}
            if LM_STUDIO_API_KEY:
                headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

            with requests.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                headers=headers,
                json={
                    "model": selected_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "stream": True
                },
                timeout=120,
                stream=True
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue
                    payload = raw_line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (((data.get("choices") or [{}])[0]).get("delta") or {}).get("content", "")
                    if delta:
                        yield delta
        except Exception as e:
            yield f"\nLLM hatasÄ±: {e}"

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
        """Kurum bilgisini DB'den bulup LLM ile kurumsal bir dille özetler.

        - DB'den uyan kayıtları toplar
        - Kayıtları bağlam olarak LLM'e verir
        - Uygun değilse nazik fallback yanıtı döner
        """
        try:
            # Test/Deterministik mod: LLM zenginleştirmesini devre dışı bırak
            if os.getenv("KB_LLM_MODE", "enrich").lower() == "direct":
                direct = database.search_kb_answer(soru)
                return direct or "Bu konuda bilgi bulunamadı."
            entries = database.search_kb_entries(soru)
            if not entries:
                # Tek eşleşme arayıp yine de dönmüyorsa None
                single = database.search_kb_answer(soru)
                if not single:
                    return "Bu konuda bilgi bulunamadı."
                entries = [{"keywords": "", "answer": single}]

            bullets = "\n".join([f"- {e['answer']}" for e in entries])
            prompt = f"""
Kurumsal bilgi notları aşağıda verilmiştir. Kullanıcının sorusunu bu notları temel alarak, açık ve güvenilir bir dille yanıtla. Bilinmeyen kısımlar için tahmin yürütme; bunun yerine "eldeki kayıtlarda yer almıyor" de. Gerekirse kısa madde işaretli özet kullan.

Soru: {soru}

Kurumsal notlar:
{bullets}

Yanıt:
"""
            llm_response = self.ollama_chat(prompt, model=None if not hasattr(self, 'user_models') else None)

            if not llm_response or llm_response.startswith("LLM hatası:") or "yanıt alınamadı" in llm_response.lower():
                fallback = database.search_kb_answer(soru)
                return fallback or "Bu konuda bilgi bulunamadı."

            return llm_response
        except Exception:
            return "Bu konuda bilgi bulunamadı."

    # The following def process_message was the start of the duplicated block.
    def _handle_support_ticket_interaction(self, message: str, user_id: str):
        """Handles the multi-turn conversation for creating a support ticket.
        Returns a response if the user is in a ticket creation flow, otherwise None.
        """
        user_state = self.user_states.setdefault(user_id, {})
        bot_response = None

        # State 1: User needs to provide a description for a pending ticket
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

            # Clear state
            user_state.pop('waiting_for_description', None)
            user_state.pop('pending_ticket', None)
            user_state.pop('last_department', None)
            return bot_response

        # State 2: User needs to provide a department for a pending ticket
        if user_state.get('waiting_for_department') and user_state.get('pending_ticket'):
            normalized_input = self.normalize_dept(message)
            matched_dept = next((dept for dept in self.DEPARTMANLAR if self.normalize_dept(dept) in normalized_input), None)

            if not matched_dept:
                bot_response = f"Geçersiz departman. Lütfen şu seçeneklerden birini yazın: {', '.join(self.DEPARTMANLAR)}"
                database.add_chat_history(user_id, "destek_talebi_etkileşim", message, bot_response)
                return bot_response

            user_state['pending_ticket']['departman'] = matched_dept
            user_state['last_department'] = matched_dept
            user_state['waiting_for_department'] = False
            user_state['waiting_for_description'] = True
            bot_response = f"{matched_dept} departmanı için destek talebi oluşturuyorum. Lütfen talebinizin açıklamasını yazar mısınız?"
            database.add_chat_history(user_id, "destek_talebi_etkileşim", message, bot_response)
            return bot_response

        return None # No state handled

    def _handle_quick_queries(self, message: str, user_id: str):
        """Tries to answer common queries without a full LLM tool-use prompt."""
        explicit_tool = self._extract_json((message or "").strip())
        if isinstance(explicit_tool, dict) and explicit_tool.get('tool'):
            return self._handle_tool_call(explicit_tool, message, user_id)

        # Quick knowledge base check
        kb_answer = self.get_kurum_bilgisi(message)
        if kb_answer and kb_answer != "Bu konuda bilgi bulunamadı.":
            database.add_chat_history(user_id, "kurum_bilgisi_llm", message, kb_answer, json.dumps({"matched": True}))
            return kb_answer

        # Quick summarization heuristic
        lower_msg = message.lower()
        if any(kw in lower_msg for kw in ["özet", "özetle", "özetini", "özet çıkar", "kısaca özet"]):
            user_reports = database.get_reports(user_id)
            if not user_reports:
                bot_response = "Herhangi bir belge yüklemediniz. Lütfen önce bir belge yükleyin."
                database.add_chat_history(user_id, "belge_ozet_hata", message, bot_response)
                return bot_response
            if len(user_reports) == 1:
                report_id = user_reports[0]['id']
                return self._summarize_report(report_id, message, user_id)
            else:
                report_options = [f"{r['id']}: {r['original_filename']}" for r in user_reports]
                options_text = '\n'.join(report_options)
                bot_response = (
                    "Birden fazla belge bulundu. Lütfen özetlemek istediğiniz belgeyi seçin (ID ile):\n"
                    f"{options_text}\n"
                    "Örnek: belgeyi özetle 12"
                )
                database.add_chat_history(user_id, "belge_ozet_secim", message, bot_response)
                return bot_response
        if any(kw in lower_msg for kw in ["açıkla", "acikla", "içerik", "icerik", "anlat"]):
            user_reports = database.get_reports(user_id)
            if len(user_reports) == 1:
                report_id = user_reports[0]['id']
                return self._explain_report(report_id, message, message, user_id)
            if len(user_reports) > 1:
                report_options = [f"{r['id']}: {r['original_filename']}" for r in user_reports]
                options_text = '\n'.join(report_options)
                bot_response = (
                    "Birden fazla belge bulundu. Lütfen açıklamak istediğiniz belgeyi seçin (ID ile):\n"
                    f"{options_text}\n"
                    "Örnek: belgeyi açıkla 12"
                )
                database.add_chat_history(user_id, "belge_aciklama_secim", message, bot_response)
                return bot_response
        return None

    def _decide_and_execute_tool(self, message: str, user_id: str):
        """Uses LLM to detect a tool call, then executes it."""
        db_history = database.get_chat_history(user_id, limit=3)
        context_msgs = [f"Kullanıcı: {h['user_message']}\nBot: {h['bot_response']}" for h in reversed(db_history)]
        context = "\n".join(context_msgs)

        llm_prompt = f'''
Aşağıda son konuşma geçmişi ve yeni kullanıcı mesajı var. Eğer bir veya birden fazla tool çağrısı gerekiyorsa bana şu formatta JSON döndür:
Tek tool için: {{"tool": "hava_durumu", "sehir": "İstanbul"}}
Çoklu tool için: [{{"tool": "hava_durumu", "sehir": "İstanbul"}}, {{"tool": "kurum_bilgisi", "soru": "seyahat politikası"}}]
Kurum içi bilgi için: {{"tool": "kurum_bilgisi", "soru": "seyahat politikası"}}
Destek talebi için: {{"tool": "destek_talebi", "departman": "IT", "aciklama": "Bilgisayarım bozuldu", "aciliyet": "acil", "kategori": "donanım"}}
Belge sorgulamak için: {{"tool": "belge_sorgulama", "sorgu": "Yıllık izin prosedürü nedir?"}}
Belge özetlemek için: {{"tool": "belge_ozetle"}}
Eğer tool çağrısı yoksa sadece normal cevabını ver.

Son konuşma geçmişi:
{context}

Kullanıcı mesajı: {message}
'''
        llm_response = self.ollama_chat(llm_prompt, model=self.user_models.get(user_id))
        data = self._extract_json(llm_response)

        try:
            if isinstance(data, list):
                results = [self._handle_tool_call(item, message, user_id) for item in data]
                return "\n\n".join(results)
            if isinstance(data, dict) and data.get('tool'):
                return self._handle_tool_call(data, message, user_id)
        except Exception as e:
            print(f"Tool execution error: {e}") # Log error
        return None

    def _handle_date_queries(self, message: str, user_id: str):
        """Handles simple, non-LLM date-related queries as a fallback."""
        lower_msg = message.lower()
        date_details = None
        date_type = None
        bot_response = None
        now = datetime.datetime.now()

        gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

        target_date = None
        if "bugün" in lower_msg:
            target_date, date_type = now, "tarih_bugün"
        elif "yarın" in lower_msg:
            target_date, date_type = now + datetime.timedelta(days=1), "tarih_yarın"
        elif "dün" in lower_msg:
            target_date, date_type = now - datetime.timedelta(days=1), "tarih_dün"
        else:
            try:
                target_date, date_type = parser.parse(message, fuzzy=True, dayfirst=True), "tarih_parse"
            except (parser.ParserError, TypeError):
                pass

        if target_date:
            day_name = gunler[target_date.weekday()]
            month_name = aylar[target_date.month - 1]
            bot_response = f"{target_date.day} {month_name} {target_date.year} {day_name}"
            date_details = {"tarih_detay": target_date.strftime("%Y-%m-%d")}

        if bot_response:
            database.add_chat_history(user_id, date_type, message, bot_response, json.dumps(date_details))
            return bot_response
        return None

    def process_message(self, message: str, user_id: str) -> str:
        """Processes a user message by routing it through different handlers."""
        # Step 1: Handle multi-turn conversation states (e.g., support ticket)
        state_response = self._handle_support_ticket_interaction(message, user_id)
        if state_response:
            return state_response

        # Step 2: Try quick handlers for KB and summarization before complex LLM calls
        quick_response = self._handle_quick_queries(message, user_id)
        if quick_response:
            return quick_response

        # Step 3: Use LLM to decide and execute a tool
        tool_response = self._decide_and_execute_tool(message, user_id)
        if tool_response:
            return tool_response

        # Step 4: Fallback for simple date queries
        date_response = self._handle_date_queries(message, user_id)
        if date_response:
            return date_response

        # Step 5: If no other handler caught the message, use LLM for a general chat response
        bot_response = self.ollama_chat(message, model=self.user_models.get(user_id))
        database.add_chat_history(user_id, "llm_cevap", message, bot_response)
        return bot_response

    def process_message_stream(self, message: str, user_id: str) -> Iterator[str]:
        state_response = self._handle_support_ticket_interaction(message, user_id)
        if state_response:
            yield state_response
            return

        quick_response = self._handle_quick_queries(message, user_id)
        if quick_response:
            yield quick_response
            return

        tool_response = self._decide_and_execute_tool(message, user_id)
        if tool_response:
            yield tool_response
            return

        date_response = self._handle_date_queries(message, user_id)
        if date_response:
            yield date_response
            return

        chunks = []
        for chunk in self.ollama_chat_stream(message, model=self.user_models.get(user_id)):
            chunks.append(chunk)
            yield chunk

        full_response = "".join(chunks).strip()
        if full_response:
            database.add_chat_history(user_id, "llm_cevap", message, full_response)

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

        if tool == 'belge_ozetle':
            # Kullanıcının yüklediği raporları alıp tekse özetle, çoksa seçim iste
            user_reports = database.get_reports(user_id)
            if not user_reports:
                bot_response = "Herhangi bir belge yüklemediniz. Lütfen önce bir belge yükleyin."
                database.add_chat_history(user_id, "belge_ozet_hata", original_user_message, bot_response)
                return bot_response
            if len(user_reports) == 1:
                report_id = user_reports[0]['id']
                return self._summarize_report(report_id, original_user_message, user_id)
            else:
                report_options = [f"{r['id']}: {r['original_filename']}" for r in user_reports]
                options_text = '\n'.join(report_options)
                return f"Birden fazla belge bulundu. Lütfen açıklamak istediğiniz belgeyi seçin (ID ile):\n{options_text}\nÖrnek: belgeyi açıkla 12"

        # Kullanıcı belge seçip tekrar sorduğunda (ör: belgeyi açıkla 12)
        if tool in {'belge_sec_ve_acikla', 'choose_file_and_explain'}:
            report_id = data.get('report_id')
            query = data.get('sorgu', '')
            if not report_id or not query:
                return "Belge ID ve açıklama isteği gereklidir."
            # Eğer kullanıcı mesajı bir özet talebi ise, açıklama yerine özet üret
            lower_q = str(query).lower()
            if any(kw in lower_q for kw in ["özet", "özetle", "özetini", "özet çıkar", "kısaca özet"]):
                return self._summarize_report(report_id, original_user_message, user_id)
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

    def _summarize_report(self, report_id, original_user_message, user_id: str):
        # İlgili belgeden en alakalı içerikleri çekip LLM ile kısa bir özet ürettir
        # Tüm chunk'ları almak yerine arama bazlı en iyi birkaç parçayı kullanıyoruz
        top_chunks = doc_processor.search_in_documents("", top_k=5)
        filtered = [r for r in top_chunks if str(r['report_id']) == str(report_id)]
        # Eğer boş sorgu ile sonuç zayıfsa, belgenin genel bağlamını almak için tekrar dene
        if not filtered:
            # Özeti çıkarmak için geniş bir anahtar kelime ile arama deneyelim
            guess_query = "genel özet giriş amaç sonuç bulgular conclusion abstract overview"
            filtered = [r for r in doc_processor.search_in_documents(guess_query, top_k=8) if str(r['report_id']) == str(report_id)]
        if not filtered:
            bot_response = "Seçili belgeden özet üretmek için yeterli içerik bulunamadı."
            database.add_chat_history(user_id, "belge_ozet", original_user_message, bot_response, json.dumps({"report_id": report_id}))
            return bot_response

        context_for_llm = "\n\n".join([res['text'] for res in filtered])
        prompt = f'''
Belgenin aşağıdaki parçalarına dayanarak kısa, net ve maddeler halinde bir özet üret.
- En fazla 6 madde yaz.
- Varsa amaç, yöntem, temel bulgular ve sonuçları vurgula.
- Metinde olmayan bilgi uydurma.

Belge Parçaları:
{context_for_llm}

Çıktı:
'''
        bot_response = self.ollama_chat(prompt, model=self.user_models.get(user_id))
        database.add_chat_history(user_id, "belge_ozet", original_user_message, bot_response, json.dumps({"report_id": report_id}))
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

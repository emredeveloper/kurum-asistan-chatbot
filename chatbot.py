import datetime
import json
import os
import re
import uuid
from typing import Iterator

import requests
from dateutil import parser
from dotenv import load_dotenv
from googletrans import Translator

import database
from document_processor import processor as doc_processor

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio").strip().lower()

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-e4b")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")

GENERAL_ASSISTANT_SYSTEM = """You are a helpful workplace assistant. Answer clearly and accurately.
Use markdown when it helps (headings, short lists, code blocks). If you do not know something, say so.
Reply in the same language the user writes in (e.g. Turkish → Turkish) unless they ask otherwise.
Stay on topic; use prior turns in this chat only when they are relevant."""

# Wire markers consumed by templates/index.html (collapsible "thinking" UI)
WIRE_THINK_START = "[[[THINK]]]"
WIRE_THINK_END = "[[[/THINK]]]"


def _coerce_reasoning_to_str(value) -> str:
    """Normalize LM Studio / OpenAI reasoning fields (str, list, or nested dict)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_coerce_reasoning_to_str(x) for x in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "reasoning_content", "value", "delta"):
            if key in value:
                inner = _coerce_reasoning_to_str(value.get(key))
                if inner.strip():
                    return inner
        parts = []
        for _k, v in value.items():
            if isinstance(v, (str, list, dict)):
                s = _coerce_reasoning_to_str(v)
                if s.strip() and len(s) < 100000:
                    parts.append(s.strip())
        return "\n\n".join(parts) if parts else ""
    return ""


def _reasoning_from_mapping(obj: dict | None) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in (
        "reasoning_content",
        "reasoning",
        "thinking",
        "thought",
    ):
        s = _coerce_reasoning_to_str(obj.get(key))
        if s.strip():
            return s.strip()
    return ""


def _lm_reply_with_thinking(reason: str, content: str) -> str:
    reason = (reason or "").strip()
    content = (content or "").strip()
    if reason:
        return f"{WIRE_THINK_START}{reason}{WIRE_THINK_END}{content}"
    return content


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
        self.DEPARTMENTS = database.get_departments()

    def set_user_model(self, user_id: str, model_name: str | None):
        if not user_id:
            return
        if model_name and isinstance(model_name, str) and model_name.strip():
            self.user_models[user_id] = model_name.strip()
        else:
            # Reset to default if empty
            if user_id in self.user_models:
                del self.user_models[user_id]

    @staticmethod
    def _reports_for_document_actions(user_id: str) -> list:
        """One row per original filename (latest upload wins). Avoids duplicate uploads of the same file."""
        rows = database.get_reports(user_id)
        if not rows:
            return []
        by_name: dict[str, dict] = {}
        for r in rows:
            name = (r.get("original_filename") or "").strip() or f"__report_{r.get('id')}"
            rid = int(r.get("id") or 0)
            prev = by_name.get(name)
            if prev is None or rid > int(prev.get("id") or 0):
                by_name[name] = r
        out = list(by_name.values())
        out.sort(key=lambda x: int(x.get("id") or 0), reverse=True)
        return out

    @staticmethod
    def _multi_doc_prompt_lines(reports: list, example_verb: str = "summarize") -> tuple[str, str]:
        """Build option lines and an example line using a real report id."""
        lines = [f"{r['id']}: {r['original_filename']}" for r in reports]
        options_text = "\n".join(lines)
        first_id = reports[0]["id"]
        example = f"Example: {example_verb} document {first_id}"
        return options_text, example

    def _messages_for_general_chat(self, user_prompt: str, user_id: str) -> list:
        """OpenAI-style messages: system + chronological history + latest user message."""
        messages: list = [{"role": "system", "content": GENERAL_ASSISTANT_SYSTEM}]
        hist = database.get_chat_history(user_id, limit=32)
        pairs: list[tuple[str, str]] = []
        for h in reversed(hist):
            um = (h.get("user_message") or "").strip()
            br = (h.get("bot_response") or "").strip()
            if not um or not br:
                continue
            if len(um) > 8000:
                um = um[:8000] + "…"
            if len(br) > 12000:
                br = br[:12000] + "…"
            pairs.append((um, br))
        max_turns = 10
        pairs = pairs[-max_turns:]
        for um, br in pairs:
            messages.append({"role": "user", "content": um})
            messages.append({"role": "assistant", "content": br})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _ollama_prompt_with_conversation(self, user_prompt: str, user_id: str) -> str:
        """Prefix recent turns for /api/generate (no native multi-turn)."""
        hist = database.get_chat_history(user_id, limit=32)
        blocks: list[str] = []
        for h in reversed(hist):
            um = (h.get("user_message") or "").strip()
            br = (h.get("bot_response") or "").strip()
            if not um or not br:
                continue
            if len(um) > 6000:
                um = um[:6000] + "…"
            if len(br) > 8000:
                br = br[:8000] + "…"
            blocks.append(f"User: {um}\nAssistant: {br}")
        blocks = blocks[-10:]
        if not blocks:
            return user_prompt
        prior = "\n\n".join(blocks)
        return (
            f"{GENERAL_ASSISTANT_SYSTEM}\n\n"
            f"Conversation so far:\n{prior}\n\n"
            f"User: {user_prompt}\nAssistant:"
        )

    def ollama_chat(self, prompt: str, model: str | None = None, conversation_user_id: str | None = None) -> str:
        """Send a chat completion request via the configured LLM provider.

        Name kept for backward compatibility (tests mock this method).
        """
        try:
            selected_model = model or get_default_model()

            if LLM_PROVIDER == "ollama":
                full_prompt = (
                    self._ollama_prompt_with_conversation(prompt, conversation_user_id)
                    if conversation_user_id
                    else prompt
                )
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": selected_model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {"temperature": 0.2}
                    },
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                if data.get("response"):
                    return data["response"].strip()
                return "No response received from LLM."

            url = f"{LM_STUDIO_BASE_URL}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if LM_STUDIO_API_KEY:
                headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

            if conversation_user_id:
                msg_list = self._messages_for_general_chat(prompt, conversation_user_id)
            else:
                msg_list = [{"role": "user", "content": prompt}]
            payload = {
                "model": selected_model,
                "messages": msg_list,
                "temperature": 0.2,
                "stream": False
            }
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                msg = choices[0]["message"] or {}
                content = (msg.get("content") or "").strip()
                reason = _reasoning_from_mapping(msg)
                merged = _lm_reply_with_thinking(reason, content)
                if merged:
                    return merged.strip()

            # Fallback for older /v1/completions style responses
            if choices and choices[0].get("text"):
                return choices[0]["text"].strip()

            return "No response received from LLM."
        except Exception as e:
            return f"LLM error: {e}"

    def ollama_chat_stream(self, prompt: str, model: str | None = None, conversation_user_id: str | None = None) -> Iterator[str]:
        selected_model = model or get_default_model()

        try:
            if LLM_PROVIDER == "ollama":
                full_prompt = (
                    self._ollama_prompt_with_conversation(prompt, conversation_user_id)
                    if conversation_user_id
                    else prompt
                )
                with requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": selected_model,
                        "prompt": full_prompt,
                        "stream": True,
                        "options": {"temperature": 0.2}
                    },
                    timeout=120,
                    stream=True
                ) as response:
                    response.raise_for_status()
                    response.encoding = "utf-8"
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = data.get("response", "") or ""
                        thinking = _coerce_reasoning_to_str(
                            data.get("thinking") or data.get("thought")
                        )
                        if thinking.strip():
                            yield f"{WIRE_THINK_START}{thinking}{WIRE_THINK_END}"
                        if chunk:
                            yield chunk
                return

            headers = {"Content-Type": "application/json"}
            if LM_STUDIO_API_KEY:
                headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

            if conversation_user_id:
                stream_messages = self._messages_for_general_chat(prompt, conversation_user_id)
            else:
                stream_messages = [{"role": "user", "content": prompt}]

            with requests.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                headers=headers,
                json={
                    "model": selected_model,
                    "messages": stream_messages,
                    "temperature": 0.2,
                    "stream": True
                },
                timeout=120,
                stream=True
            ) as response:
                response.raise_for_status()
                response.encoding = "utf-8"
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
                    choice0 = (data.get("choices") or [{}])[0]
                    d = choice0.get("delta") or {}
                    think = _reasoning_from_mapping(d)
                    if not think:
                        think = _reasoning_from_mapping(choice0.get("message"))
                    if not think:
                        think = _reasoning_from_mapping(data)
                    content = d.get("content") or ""
                    if think:
                        yield f"{WIRE_THINK_START}{think}{WIRE_THINK_END}"
                    if content:
                        yield content
        except Exception as e:
            yield f"\nLLM error: {e}"

    def get_weather(self, city: str) -> str:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&lang=en&units=metric"
            response = requests.get(url)
            data = response.json()
            if data.get("cod") != 200:
                if data.get("message", "") == "city not found":
                    return "Which city would you like the weather for?"
                return f"Could not retrieve weather: {data.get('message', '')}"
            desc = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"Weather in {city}: {desc}, temperature: {temp}°C"
        except Exception as e:
            return f"Weather error: {e}"

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

    def get_knowledge_base_info(self, question):
        """Look up institutional knowledge from the DB and summarise via LLM."""
        try:
            if os.getenv("KB_LLM_MODE", "enrich").lower() == "direct":
                direct = database.search_kb_answer(question)
                return direct or "No information was found on this topic."
            entries = database.search_kb_entries(question)
            if not entries:
                single = database.search_kb_answer(question)
                if not single:
                    return "No information was found on this topic."
                entries = [{"keywords": "", "answer": single}]

            bullets = "\n".join([f"- {e['answer']}" for e in entries])
            prompt = f"""
The following are institutional knowledge notes. Answer the user's question based on these notes in a clear and reliable manner. Do not speculate on unknown parts; instead say "this is not covered in our records." Use short bullet-point summaries if needed.

Question: {question}

Institutional notes:
{bullets}

Answer:
"""
            llm_response = self.ollama_chat(prompt)

            if not llm_response or llm_response.startswith("LLM error:") or "no response" in llm_response.lower():
                fallback = database.search_kb_answer(question)
                return fallback or "No information was found on this topic."

            return llm_response
        except Exception:
            return "No information was found on this topic."

    def _handle_support_ticket_interaction(self, message: str, user_id: str):
        """Handles the multi-turn conversation for creating a support ticket.
        Returns a response if the user is in a ticket creation flow, otherwise None.
        """
        user_state = self.user_states.setdefault(user_id, {})
        bot_response = None

        if user_state.get('waiting_for_description') and user_state.get('pending_ticket'):
            pending_ticket_info = user_state['pending_ticket']
            pending_ticket_info['description'] = message.strip()

            ticket_id = uuid.uuid4().hex[:8]
            department = pending_ticket_info.get('department', user_state.get('last_department'))
            description = pending_ticket_info['description']
            priority = pending_ticket_info.get('priority', 'normal')
            category = pending_ticket_info.get('category', 'general')

            database.add_support_ticket(user_id, ticket_id, department, description, priority, category)

            bot_response = f"Your support ticket with ID {ticket_id} has been created for the {department} department. You can track its status from the dashboard."
            database.add_chat_history(user_id, "support_ticket_created", message, bot_response, json.dumps({
                "ticket_id": ticket_id, "department": department, "priority": priority, "category": category
            }))

            user_state.pop('waiting_for_description', None)
            user_state.pop('pending_ticket', None)
            user_state.pop('last_department', None)
            return bot_response

        if user_state.get('waiting_for_department') and user_state.get('pending_ticket'):
            normalized_input = self.normalize_dept(message)
            matched_dept = next((dept for dept in self.DEPARTMENTS if self.normalize_dept(dept) in normalized_input), None)

            if not matched_dept:
                bot_response = f"Invalid department. Please choose one of the following: {', '.join(self.DEPARTMENTS)}"
                database.add_chat_history(user_id, "support_ticket_interaction", message, bot_response)
                return bot_response

            user_state['pending_ticket']['department'] = matched_dept
            user_state['last_department'] = matched_dept
            user_state['waiting_for_department'] = False
            user_state['waiting_for_description'] = True
            bot_response = f"Creating a support ticket for the {matched_dept} department. Please describe your request."
            database.add_chat_history(user_id, "support_ticket_interaction", message, bot_response)
            return bot_response

        return None

    def _handle_quick_queries(self, message: str, user_id: str):
        """Tries to answer common queries without a full LLM tool-use prompt."""
        explicit_tool = self._extract_json((message or "").strip())
        if isinstance(explicit_tool, dict) and explicit_tool.get('tool'):
            return self._handle_tool_call(explicit_tool, message, user_id)

        kb_answer = self.get_knowledge_base_info(message)
        if kb_answer and kb_answer != "No information was found on this topic.":
            database.add_chat_history(user_id, "knowledge_base_llm", message, kb_answer, json.dumps({"matched": True}))
            return kb_answer

        lower_msg = message.lower()
        summary_keywords = [
            "summary", "summarize", "summarise", "brief summary", "give me a summary",
            "özet", "ozet", "özetle", "ozetle", "özetini", "ozetini", "özet çıkar", "ozet cikar",
            "kısaca özet", "kisaca ozet", "belgeyi özetle", "belgeyi ozetle", "dosyayı özetle", "dosyayi ozetle",
        ]
        read_keywords = [
            "explain", "content", "describe", "tell me about", "what does the file",
            "içerik", "icerik", "içeriği", "icerigi", "açıkla", "acikla", "anlat", "anlat bana",
            "dosya", "belge", "oku", "okuyun", "okuman", "görüyorsun", "goruyorsun",
            "ne görüyorsun", "ne goruyorsun", "neler görüyorsun", "neler goruyorsun",
            "ne var", "neler var", "neler yazıyor", "neler yaziyor",
        ]
        if any(kw in lower_msg for kw in summary_keywords):
            user_reports = self._reports_for_document_actions(user_id)
            if not user_reports:
                bot_response = "You have not uploaded any documents. Please upload a document first."
                database.add_chat_history(user_id, "doc_summary_error", message, bot_response)
                return bot_response
            if len(user_reports) == 1:
                report_id = user_reports[0]['id']
                return self._summarize_report(report_id, message, user_id)
            else:
                options_text, example = self._multi_doc_prompt_lines(user_reports)
                bot_response = f"Multiple documents found. Choose one by ID:\n{options_text}\n{example}"
                database.add_chat_history(user_id, "doc_summary_selection", message, bot_response)
                return bot_response
        if any(kw in lower_msg for kw in read_keywords):
            user_reports = self._reports_for_document_actions(user_id)
            if len(user_reports) == 1:
                report_id = user_reports[0]['id']
                return self._explain_report(report_id, message, message, user_id)
            if len(user_reports) > 1:
                options_text, example = self._multi_doc_prompt_lines(user_reports)
                bot_response = f"Multiple documents found. Choose one by ID:\n{options_text}\n{example}"
                database.add_chat_history(user_id, "doc_explain_selection", message, bot_response)
                return bot_response
        return None

    def _decide_and_execute_tool(self, message: str, user_id: str):
        """Uses LLM to detect a tool call, then executes it."""
        db_history = database.get_chat_history(user_id, limit=3)
        context_msgs = [f"User: {h['user_message']}\nBot: {h['bot_response']}" for h in reversed(db_history)]
        context = "\n".join(context_msgs)

        llm_prompt = f'''
Below is the recent conversation history and a new user message. If one or more tool calls are needed, return JSON in the following format:
Single tool: {{"tool": "weather", "city": "Istanbul"}}
Multiple tools: [{{"tool": "weather", "city": "Istanbul"}}, {{"tool": "knowledge_base", "question": "travel policy"}}]
Internal knowledge: {{"tool": "knowledge_base", "question": "travel policy"}}
Support ticket: {{"tool": "support_ticket", "department": "IT", "description": "My computer is broken", "priority": "urgent", "category": "hardware"}}
Document query: {{"tool": "document_query", "query": "What is the annual leave procedure?"}}
Summarize document: {{"tool": "document_summarize"}}
If the user asks to read, explain, summarize, or describe an uploaded file in any language (e.g. Turkish), use document_query or document_summarize as appropriate.
If no tool call is needed, just provide a normal answer.

Recent conversation history:
{context}

User message: {message}
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
            import logging
            logging.getLogger(__name__).exception("Tool execution error")
        return None

    def _handle_date_queries(self, message: str, user_id: str):
        """Handles simple, non-LLM date-related queries as a fallback."""
        lower_msg = message.lower()
        date_details = None
        date_type = None
        bot_response = None
        now = datetime.datetime.now()

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]

        target_date = None
        if "today" in lower_msg:
            target_date, date_type = now, "date_today"
        elif "tomorrow" in lower_msg:
            target_date, date_type = now + datetime.timedelta(days=1), "date_tomorrow"
        elif "yesterday" in lower_msg:
            target_date, date_type = now - datetime.timedelta(days=1), "date_yesterday"
        else:
            try:
                target_date, date_type = parser.parse(message, fuzzy=True, dayfirst=True), "date_parse"
            except (parser.ParserError, TypeError):
                pass

        if target_date:
            day_name = days[target_date.weekday()]
            month_name = months[target_date.month - 1]
            bot_response = f"{day_name}, {month_name} {target_date.day}, {target_date.year}"
            date_details = {"date_detail": target_date.strftime("%Y-%m-%d")}

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
        bot_response = self.ollama_chat(
            message, model=self.user_models.get(user_id), conversation_user_id=user_id
        )
        database.add_chat_history(user_id, "llm_response", message, bot_response)
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
        for chunk in self.ollama_chat_stream(
            message, model=self.user_models.get(user_id), conversation_user_id=user_id
        ):
            chunks.append(chunk)
            yield chunk

        full_response = "".join(chunks).strip()
        if full_response:
            database.add_chat_history(user_id, "llm_response", message, full_response)

    def _extract_json(self, text: str):
        """Attempt to safely extract a JSON dict or list from text.

        Tries direct json.loads, then looks for ```json code blocks,
        and finally falls back to a greedy brace/bracket regex match.
        Returns None on failure.
        """
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) Code blocks: ```json ... ``` or ``` ... ```
        code_block_pattern = re.compile(r"```(json)?\s*([\s\S]*?)```", re.IGNORECASE)
        for match in code_block_pattern.finditer(text):
            candidate = match.group(2).strip()
            try:
                return json.loads(candidate)
            except Exception:
                continue

        # 3) Greedy match for first balanced braces or brackets
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
        """Return dashboard summary: support tickets, reports, and recent interactions."""
        try:
            reports = database.get_reports()
            total_reports = len(reports)
            processed_reports = sum(1 for r in reports if r.get('processed') == 1)
            unprocessed_reports = total_reports - processed_reports

            conn = database.get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT status, COUNT(*) as cnt FROM support_tickets GROUP BY status")
            status_rows = cur.fetchall()
            conn.close()
            ticket_status_counts = {row[0]: row[1] for row in status_rows} if status_rows else {}

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
                'error': f'Error retrieving dashboard data: {e}'
            }

    def _handle_tool_call(self, data, original_user_message: str, user_id: str):
        user_state = self.user_states.setdefault(user_id, {})
        tool = data.get('tool')
        bot_response = ""

        if tool == 'weather':
            city = data.get('city', 'Istanbul')
            bot_response = self.get_weather(city)
            database.add_chat_history(user_id, "weather", original_user_message, bot_response, json.dumps({"city": city}))
            return bot_response

        if tool == 'knowledge_base':
            question = data.get('question', '')
            bot_response = self.get_knowledge_base_info(question)
            database.add_chat_history(user_id, "knowledge_base", original_user_message, bot_response, json.dumps({"question": question}))
            return bot_response

        if tool == 'document_query':
            query = data.get('query', '')
            if not query:
                return "Please specify what you would like to search for in the document."

            user_reports = self._reports_for_document_actions(user_id)
            if not user_reports:
                kb_fallback = self.get_knowledge_base_info(query)
                if kb_fallback and kb_fallback != "No information was found on this topic.":
                    database.add_chat_history(user_id, "knowledge_base_fallback", query, kb_fallback, json.dumps({"from": "document_query"}))
                    return kb_fallback
                return "You have not uploaded any documents. Please upload a document first."
            if len(user_reports) == 1:
                report_id = user_reports[0]['id']
                return self._explain_report(report_id, query, original_user_message, user_id)
            else:
                options_text, example = self._multi_doc_prompt_lines(user_reports, example_verb="explain")
                return f"Multiple documents found. Choose one by ID:\n{options_text}\n{example}"

        if tool == 'document_summarize':
            user_reports = self._reports_for_document_actions(user_id)
            if not user_reports:
                bot_response = "You have not uploaded any documents. Please upload a document first."
                database.add_chat_history(user_id, "doc_summary_error", original_user_message, bot_response)
                return bot_response
            if len(user_reports) == 1:
                report_id = user_reports[0]['id']
                return self._summarize_report(report_id, original_user_message, user_id)
            else:
                options_text, example = self._multi_doc_prompt_lines(user_reports)
                return f"Multiple documents found. Choose one by ID:\n{options_text}\n{example}"

        if tool in {'choose_file_and_explain'}:
            report_id = data.get('report_id')
            query = data.get('query', '') or data.get('sorgu', '')
            if not report_id or not query:
                return "Document ID and a query are required."
            lower_q = str(query).lower()
            if any(kw in lower_q for kw in ["summary", "summarize", "summarise", "brief summary"]):
                return self._summarize_report(report_id, original_user_message, user_id)
            return self._explain_report(report_id, query, original_user_message, user_id)

        if tool == 'support_ticket':
            department_input_from_llm = data.get('department') or ''
            normalized_department_input_llm = self.normalize_dept(department_input_from_llm)
            matched_dept = None
            for dept_option in self.DEPARTMENTS:
                normalized_dept_option = self.normalize_dept(dept_option)
                if normalized_dept_option in normalized_department_input_llm or normalized_department_input_llm == normalized_dept_option:
                    matched_dept = dept_option
                    break

            description = data.get('description', None)
            priority = data.get('priority', 'normal')
            category = data.get('category', 'general')

            current_ticket_data = {
                "description": description,
                "department": matched_dept,
                "priority": priority,
                "category": category,
                "user_id": user_id
            }

            if not matched_dept:
                user_state['waiting_for_department'] = True
                user_state['pending_ticket'] = current_ticket_data
                bot_response = f"Understood, you want to create a support ticket. Which department should we forward it to? Options: {', '.join(self.DEPARTMENTS)}"
                database.add_chat_history(user_id, "support_ticket_interaction", original_user_message, bot_response)
                return bot_response

            if not description:
                user_state['pending_ticket'] = current_ticket_data
                user_state['last_department'] = matched_dept
                user_state['waiting_for_description'] = True
                user_state['waiting_for_department'] = False
                bot_response = f"Creating a support ticket for the {matched_dept} department. Please describe your request."
                database.add_chat_history(user_id, "support_ticket_interaction", original_user_message, bot_response)
                return bot_response

            ticket_id = uuid.uuid4().hex[:8]
            database.add_support_ticket(user_id, ticket_id, matched_dept, description, priority, category)

            bot_response = f"Your support ticket with ID {ticket_id} has been created for the {matched_dept} department. (Priority: {priority}, Category: {category}) You can track it from the dashboard."
            database.add_chat_history(user_id, "support_ticket_created", original_user_message, bot_response, json.dumps({
                "ticket_id": ticket_id, "department": matched_dept, "priority": priority, "category": category
            }))

            user_state['pending_ticket'] = None
            user_state['waiting_for_department'] = False
            user_state['waiting_for_description'] = False
            user_state['last_department'] = None
            return bot_response

        bot_response = "Unrecognized or unprocessable tool call."
        database.add_chat_history(user_id, "tool_error", original_user_message, bot_response, json.dumps(data))
        return bot_response

    def _explain_report(self, report_id, query, original_user_message, user_id):
        search_results = doc_processor.search_in_documents(query, top_k=6)
        filtered_results = [r for r in search_results if str(r['report_id']) == str(report_id)]
        if not filtered_results:
            return "No relevant information was found in the selected document."
        context_for_llm = "\n\n---\n\n".join([result['text'] for result in filtered_results])
        rag_prompt = f'''
You are answering from retrieved excerpts of ONE document. Rules:
- Use ONLY the excerpts below. Do not invent facts, citations, or sections not present.
- If the excerpts do not contain enough information, say clearly that it is not in the provided text.
- Answer in the same language as the user's question when possible.
- Prefer a short markdown structure: brief intro if needed, then bullet points with the key facts.

Document excerpts (may be out of order):
{context_for_llm}

User question: {query}

Your answer:
'''
        bot_response = self.ollama_chat(rag_prompt, model=self.user_models.get(user_id))
        database.add_chat_history(user_id, "document_query", original_user_message, bot_response, json.dumps({"query": query, "report_id": report_id}))
        return bot_response

    def _summarize_report(self, report_id, original_user_message, user_id: str):
        top_chunks = doc_processor.search_in_documents("", top_k=8)
        filtered = [r for r in top_chunks if str(r['report_id']) == str(report_id)]
        if not filtered:
            guess_query = "general summary introduction purpose conclusion findings abstract overview methods results"
            filtered = [r for r in doc_processor.search_in_documents(guess_query, top_k=10) if str(r['report_id']) == str(report_id)]
        if not filtered:
            bot_response = "Not enough content was found in the selected document to generate a summary."
            database.add_chat_history(user_id, "doc_summary", original_user_message, bot_response, json.dumps({"report_id": report_id}))
            return bot_response

        context_for_llm = "\n\n---\n\n".join([res['text'] for res in filtered])
        prompt = f'''
Summarize the document using ONLY the excerpts below.
- At most 7 bullet points; keep each bullet one or two sentences.
- Cover purpose/topic, main methods or setup, key results or claims, and conclusions or limitations if present.
- Do not add information that is not supported by the excerpts.
- Match the document's primary language (e.g. English paper → English summary; Turkish → Turkish).

Document excerpts (may be incomplete or out of order):
{context_for_llm}

Summary:
'''
        bot_response = self.ollama_chat(prompt, model=self.user_models.get(user_id))
        database.add_chat_history(user_id, "doc_summary", original_user_message, bot_response, json.dumps({"report_id": report_id}))
        return bot_response

    def get_history(self, user_id: str):
        # Returns list of dicts, already ordered by timestamp desc, limit 20 by default in db func
        return database.get_chat_history(user_id)

    def get_support_tickets(self, user_id: str):
        # Returns list of dicts, ordered by created_at desc, limit 50 by default in db func
        return database.get_support_tickets(user_id)

    def mark_ticket_as_read(self, idx: int, user_id: str):
        user_tickets = database.get_support_tickets(user_id)
        if 0 <= idx < len(user_tickets):
            ticket_id = user_tickets[idx]['ticket_id']
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
            import logging
            logging.getLogger(__name__).exception("Translation error")
            return "An error occurred with the translation service. Please try again later."

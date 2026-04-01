# Company Assistant Chatbot

This Flask project brings together internal knowledge base lookups, support ticket creation, weather queries, and document question-answering in a single interface. The app works with an OpenAI-compatible API such as LM Studio and can also use Ollama for both chat and document embeddings.

## Features

- Answer questions from the internal knowledge base
- Create support tickets and track them on the dashboard
- Fetch weather information
- Upload, summarize, and search PDF and DOCX files
- Review chat history, reports, and tickets from the dashboard
- Choose the active model per user session

## Setup

1. Create a virtual environment.

```bash
python -m venv .venv
```

2. Activate it.

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root.

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

5. Start the app.

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## Notes

- If `RESET_ON_STARTUP=1`, chat history, support tickets, uploaded reports, and the vector store are cleared on startup.
- By default, startup data deletion is disabled.
- Set `LLM_PROVIDER=lmstudio` or `LLM_PROVIDER=ollama` to choose the active provider.
- The document embedding flow uses Ollama through `OLLAMA_EMBED_MODEL`.
- Tests use a separate SQLite database via the `TEST_DATABASE_URL` environment variable.

## Test

```bash
pytest -q
```

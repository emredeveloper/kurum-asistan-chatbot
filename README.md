# Internal Smart Assistant Chatbot

A modern and functional LLM-based (local LM Studio/OpenAI compatible API) chatbot that simplifies internal processes. Weather updates, internal knowledge, support tickets, document uploads, and more — all in one screen!

## Features

* 🤖 **LLM-powered natural language chat** (local model via LM Studio/OpenAI compatible API)
* 🔗 **Multi-tool/function chaining**: weather, internal knowledge, support tickets, document upload
* 🏢 **Internal knowledge base**: FAQs, procedures, policies
* 🌤️ **Weather query** (OpenWeatherMap API)
* 💼 **Support ticket creation** (department, description, urgency, category)
* 🗂️ **Dashboard**: query history, weather history, support tickets, uploaded reports
* 📄 **Word/PDF report upload**: attach and manage files from both the chat screen and dashboard
* 🌗 **Dark/Light theme** (persistent with localStorage)
* 📱 **Modern, responsive, and mobile-friendly UI**
* 🛡️ **Secure API key management with .env**

## Installation

1. Clone the repository and navigate into the directory.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/Mac: source .venv/bin/activate
   ```
3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
4. Add your OpenWeatherMap API key to the `.env` file:

   ```
   OPENWEATHER_API_KEY=YOUR_API_KEY
   ```
5. Use LM Studio or Ollama (both can serve as an OpenAI-compatible server):

   * **LM Studio**:

     * Start a model and enable the OpenAI Compatible Server (e.g. `http://localhost:1234/v1`).
   * **Ollama (optional)**:

     * Run a model such as `ollama run qwen3:8b` and use an OpenAI-compatible proxy.
   * Example `.env`:

     ```
     LM_STUDIO_BASE_URL=http://localhost:1234/v1
     LM_STUDIO_MODEL=openai/gpt-oss-20b
     # LM_STUDIO_API_KEY=optional
     OPENWEATHER_API_KEY=YOUR_API_KEY
     ```
6. Start the application:

   ```bash
   python app.py
   ```
7. Open `http://localhost:5000` in your browser.

## Usage

* **Chat screen:** Type and send your message — ask about weather, internal knowledge, support, or document uploads naturally.
* **Report upload:** Add Word/PDF files from the chat screen or dashboard, manage and download them on the dashboard.
* **Dashboard:** View query history, weather history, support tickets, and uploaded reports.
* **Theme:** Switch between dark and light mode with the button in the top-right corner.

## Notes

* All data is temporarily stored in memory; user identity support can be added.
* Departments, knowledge base, and tool chains can be easily customized.
* Supports advanced multi-tool/function chaining with LLM integration.


Developer: emredeveloper


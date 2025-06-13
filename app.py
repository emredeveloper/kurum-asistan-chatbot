from flask import Flask, render_template, request, jsonify
from chatbot import CitizenAssistantBot
import json

app = Flask(__name__)
bot = CitizenAssistantBot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message.strip():
        return jsonify({'response': 'Lütfen bir mesaj girin.'})
    bot_response = bot.process_message(user_message)
    return jsonify({'response': bot_response})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/issues')
def get_issues():
    return jsonify(bot.get_citizen_dashboard())

@app.route('/translate', methods=['POST'])
def translate():
    message = request.json.get('message', '')
    target_lang = request.json.get('target_language', 'en')
    
    translated = bot.translate_message(message, target_lang)
    return jsonify({'translated_message': translated})

@app.route('/api/history')
def api_history():
    return jsonify(bot.get_history())

@app.route('/api/support_tickets')
def api_support_tickets():
    return jsonify(bot.get_support_tickets())

@app.route('/api/support_tickets/read/<int:idx>', methods=['POST'])
def api_support_tickets_read(idx):
    success = bot.mark_ticket_as_read(idx)
    return jsonify({'success': success})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

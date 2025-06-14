from flask import Flask, render_template, request, jsonify, send_from_directory
from chatbot import CitizenAssistantBot
import json
import os
from werkzeug.utils import secure_filename
import datetime

app = Flask(__name__)
bot = CitizenAssistantBot()

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads', 'reports')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Rapor meta verileri bellekte tutulacak (isteğe bağlı dosyaya da yazılabilir)
REPORTS = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/upload_report', methods=['POST'])
def upload_report():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'Dosya bulunamadı.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Dosya seçilmedi.'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_with_time = f"{timestamp}_{filename}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_with_time))
        uploader = request.form.get('uploader', 'Bilinmiyor')
        REPORTS.append({
            'filename': filename_with_time,
            'orjinal_ad': filename,
            'uploader': uploader,
            'upload_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
        return jsonify({'success': True, 'message': 'Rapor başarıyla yüklendi.'})
    else:
        return jsonify({'success': False, 'message': 'Sadece PDF veya Word dosyası yükleyebilirsiniz.'}), 400

@app.route('/reports', methods=['GET'])
def get_reports():
    return jsonify(REPORTS)

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

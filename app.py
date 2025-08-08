from flask import Flask, render_template, request, jsonify, send_from_directory, session
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
import logging
from logging.handlers import RotatingFileHandler
import time
from chatbot import CitizenAssistantBot
import json
import os
import uuid
from werkzeug.utils import secure_filename
import datetime
import database # Import the database module
from document_processor import processor as doc_processor # Import the document processor

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for session management
# Güvenlik ve yükleme sınırları
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB
bot = CitizenAssistantBot()

# Initialize the database - This should be done carefully.
# For testing, conftest.py will handle initializing the test DB.
# For production/development, it should be called explicitly at startup.
# database.init_db() # Removing from global scope.

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads', 'reports')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
ALLOWED_MIME_TYPES = {
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Basit rate limiting (dakikada 30 istek / kullanıcı oturumu)
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 30
_rate_limit_store = {}

def _rate_limit_key():
    # Oturuma göre sınırlama; yoksa IP'ye göre
    if 'user_id' in session:
        return f"uid:{session['user_id']}"
    return f"ip:{request.remote_addr}"

@app.before_request
def apply_rate_limiting():
    # Sadece belirli endpointler için uygulayalım
    if request.endpoint in {'chat', 'upload_report'}:
        key = _rate_limit_key()
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS
        timestamps = _rate_limit_store.get(key, [])
        # Eski kayıtları temizle
        timestamps = [t for t in timestamps if t >= window_start]
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            return jsonify({'success': False, 'message': 'Çok fazla istek. Lütfen biraz sonra tekrar deneyin.'}), 429
        timestamps.append(now)
        _rate_limit_store[key] = timestamps

# Loglama (Rotating)
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=1024*1024, backupCount=5)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

AVAILABLE_MODELS = [m.strip() for m in os.getenv('LM_STUDIO_MODELS', 'openai/gpt-oss-20b,openai/gpt-4o-mini,meta-llama/llama-3.1-8b-instruct').split(',') if m.strip()]

# Rapor meta verileri artık database'de tutulacak.
# REPORTS = [] # Bu satırı kaldırıyoruz.

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/welcome', methods=['GET'])
def welcome_message():
    if 'messages' not in session:
        session['messages'] = []
    return jsonify({'response': 'Merhaba! Ben Kurum Asistanı. Size aşağıdaki konularda yardımcı olabilirim:<br>- 🌤️ Güncel hava durumu bilgisi alabilirim.<br>- 💼 Destek talebi oluşturabilirim.<br>- 🏢 Kurum içi bilgi tabanımızdan sorularınızı yanıtlayabilirim.<br>- 📄 Belgelerinizi (Word/PDF) yükleyip, içerikleri hakkında sorular sorabilirsiniz.'})

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']

    user_message = request.json.get('message', '')
    if not user_message.strip():
        return jsonify({'response': 'Lütfen bir mesaj girin.'})
    bot_response = bot.process_message(user_message, user_id)
    return jsonify({'response': bot_response})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/api/issues')
def get_issues():
    return jsonify(bot.get_citizen_dashboard())

@app.route('/api/models', methods=['GET'])
def get_models():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    current = bot.user_models.get(user_id)
    return jsonify({'available': AVAILABLE_MODELS, 'current': current})

@app.route('/api/model', methods=['POST'])
def set_model():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    data = request.get_json(silent=True) or {}
    model = data.get('model')
    if model and model in AVAILABLE_MODELS:
        bot.set_user_model(user_id, model)
        return jsonify({'success': True, 'model': model})
    # Boş veya geçersiz model ismi gelirse varsayılanı kullan
    bot.set_user_model(user_id, None)
    return jsonify({'success': True, 'model': None})

@app.route('/translate', methods=['POST'])
def translate():
    message = request.json.get('message', '')
    target_lang = request.json.get('target_language', 'en')
    
    translated = bot.translate_message(message, target_lang)
    return jsonify({'translated_message': translated})

@app.route('/api/history')
def api_history():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4()) # Or return error/empty if no session expected
    user_id = session.get('user_id')
    if not user_id: # Should not happen if logic above is fine, but as a safeguard
        return jsonify([])
    return jsonify(bot.get_history(user_id))

@app.route('/api/support_tickets')
def api_support_tickets():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4()) # Or return error/empty
    user_id = session.get('user_id')
    if not user_id:
        return jsonify([])
    return jsonify(bot.get_support_tickets(user_id))

@app.route('/api/support_tickets/read/<int:idx>', methods=['POST'])
def api_support_tickets_read(idx):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'User session not found'}), 401
    user_id = session['user_id']
    success = bot.mark_ticket_as_read(idx, user_id)
    return jsonify({'success': success})

@app.route('/upload_report', methods=['POST'])
def upload_report():
    if 'user_id' not in session:
        # Create a user_id if one doesn't exist for the uploader
        # Alternatively, could deny if no active session, but this allows anonymous uploads to be tied to a session ID
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'Dosya bulunamadı.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Dosya seçilmedi.'}), 400

    # MIME türü kontrolü
    if file and allowed_file(file.filename):
        if file.mimetype not in ALLOWED_MIME_TYPES:
            return jsonify({'success': False, 'message': 'Geçersiz dosya türü.'}), 400
        original_filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        stored_filename_with_time = f"{timestamp}_{original_filename}"

        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename_with_time)
            file.save(file_path)
            uploader_name = request.form.get('uploader', 'Bilinmiyor') # Name of person uploading

            report_id = database.add_report(user_id, original_filename, stored_filename_with_time, uploader_name)

            # Process the document for RAG. In a real-world app, this should be
            # offloaded to a background worker (e.g., Celery, RQ).
            doc_processor.process_and_embed_document(file_path, report_id)

            return jsonify({'success': True, 'message': 'Rapor başarıyla yüklendi ve işlendi.'})
        except Exception as e:
            # Log the exception e
            return jsonify({'success': False, 'message': f'Rapor yüklenirken bir hata oluştu: {str(e)}'}), 500
    else:
        return jsonify({'success': False, 'message': 'Sadece PDF veya Word dosyası yükleyebilirsiniz.'}), 400

@app.route('/reports', methods=['GET'])
def get_reports():
    # Fetches all reports by default, as per database.get_reports() when user_id is None.
    # If user-specific reports were needed here, user_id from session would be passed.
    all_reports = database.get_reports()
    return jsonify(all_reports)

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/delete_report/<int:report_id>', methods=['DELETE'])
def delete_report_route(report_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Yetkisiz erişim.'}), 401

    try:
        # Get the filename before deleting the DB record to ensure we can delete the file
        report_details = database.get_report_by_id(report_id)
        if not report_details:
             return jsonify({'success': False, 'message': 'Rapor bulunamadı.'}), 404

        stored_filename = report_details['stored_filename']
        
        # Delete from database
        database.delete_report(report_id)

        # Delete the file from the filesystem
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Delete the document's vectors from the FAISS index
        doc_processor.delete_document(report_id)

        return jsonify({'success': True, 'message': 'Rapor başarıyla silindi.'})
    except Exception as e:
        print(f"Error deleting report {report_id}: {e}") # Log for debugging
        return jsonify({'success': False, 'message': f'Rapor silinirken bir hata oluştu: {str(e)}'}), 500

@app.route('/delete_all_reports', methods=['DELETE'])
def delete_all_reports():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Yetkisiz erişim.'}), 401
    try:
        # 1. Tüm raporları veritabanından çek
        all_reports = database.get_reports()
        # 2. Her bir raporun dosyasını sil
        for report in all_reports:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], report['stored_filename'])
            if os.path.exists(file_path):
                os.remove(file_path)
            # 3. FAISS vektörlerinden sil
            doc_processor.delete_document(report['id'])
        # 4. Veritabanından tüm raporları sil
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM uploaded_reports')
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Tüm raporlar ve ilişkili veriler silindi.'})
    except Exception as e:
        print(f"Error deleting all reports: {e}")
        return jsonify({'success': False, 'message': f'Tüm raporlar silinirken bir hata oluştu: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize the database when running the app directly
    database.init_db()
    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(e):
        return jsonify({'success': False, 'message': 'Dosya boyutu sınırı aşıldı (20MB).'}), 413

    @app.errorhandler(Exception)
    def handle_unexpected_error(e):
        if isinstance(e, HTTPException):
            return e
        app.logger.exception('Beklenmeyen sunucu hatası')
        return jsonify({'success': False, 'message': 'Sunucu hatası. Lütfen daha sonra tekrar deneyin.'}), 500

    app.run(debug=True, host='0.0.0.0', port=5000)

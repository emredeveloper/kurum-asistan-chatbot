import pytest
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import database


def unique_name(prefix):
    return f"{prefix}-{uuid.uuid4().hex}.pdf"


def test_welcome_route(client):
    """Test the /welcome route."""
    response = client.get('/welcome')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == {'response': 'Hello! I am the Company Assistant. I can help you with the following:<br>- Support ticket creation<br>- Questions about our internal knowledge base<br>- Upload Word/PDF documents and answer questions about their contents'}


def test_dashboard_route(client):
    """Test the /dashboard route."""
    response = client.get('/dashboard')
    assert response.status_code == 200
    response_text = response.data.decode('utf-8')
    assert "Query Dashboard" in response_text
    assert "Support Tickets" in response_text


def test_chat_route_basic(client, mocker):
    """Test the /chat route with a mocked bot response."""
    mocked_bot_response = "Hello, test user!"
    mock_process_message = mocker.patch('app.bot.process_message', return_value=mocked_bot_response)

    response = client.post('/chat', json={'message': 'Hello bot'})

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['response'] == mocked_bot_response

    mock_process_message.assert_called_once()
    call_args = mock_process_message.call_args[0]
    assert call_args[0] == 'Hello bot'
    assert isinstance(call_args[1], str)


def test_get_history_empty(client, test_db):
    """Test the /api/history route for a new user (should be empty)."""
    response = client.get('/api/history')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == []


def test_upload_report_requires_file(client):
    """Test the /upload_report route when no file is provided."""
    response = client.post('/upload_report', data={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['success'] == False
    assert json_data['message'] == 'File not found.'


def test_upload_report_empty_filename(client):
    """Test the /upload_report route when file is present but filename is empty."""
    from io import BytesIO
    data = {
        'file': (BytesIO(b"some dummy content"), '')
    }
    response = client.post('/upload_report', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['success'] == False
    assert json_data['message'] == 'No file selected.'


def test_upload_report_reports_processing_failure(client, mocker):
    from io import BytesIO
    import app as app_module

    mocker.patch("app.doc_processor.process_and_embed_document", return_value=False)
    filename = unique_name("bad")
    data = {
        'file': (BytesIO(b"%PDF-1.4 dummy"), filename),
        'uploader': 'Alice',
    }

    response = client.post('/upload_report', data=data, content_type='multipart/form-data')

    assert response.status_code == 502
    payload = response.get_json()
    assert payload["success"] is False
    assert "processing failed" in payload["message"]
    assert not any(r["original_filename"] == filename for r in database.get_reports(limit=10000))
    assert not any(filename in name for name in os.listdir(app_module.app.config["UPLOAD_FOLDER"]))


def test_reports_route_returns_only_current_user_reports(client, test_db):
    owner = f"user-a-{uuid.uuid4().hex}"
    outsider = f"user-b-{uuid.uuid4().hex}"
    own_file = unique_name("own")
    other_file = unique_name("other")
    own_id = database.add_report(owner, own_file, own_file, "Alice")
    database.add_report(outsider, other_file, other_file, "Bob")
    with client.session_transaction() as sess:
        sess["user_id"] = owner

    response = client.get('/reports')

    assert response.status_code == 200
    reports = response.get_json()
    assert [r["id"] for r in reports] == [own_id]


def test_download_report_requires_report_owner(client, test_db):
    owner = f"user-a-{uuid.uuid4().hex}"
    outsider = f"user-b-{uuid.uuid4().hex}"
    other_file = unique_name("other")
    database.add_report(outsider, other_file, other_file, "Bob")
    with client.session_transaction() as sess:
        sess["user_id"] = owner

    response = client.get(f'/download_report/{other_file}')

    assert response.status_code == 404


def test_delete_report_requires_report_owner(client, test_db):
    owner = f"user-a-{uuid.uuid4().hex}"
    outsider = f"user-b-{uuid.uuid4().hex}"
    other_file = unique_name("other")
    other_id = database.add_report(outsider, other_file, other_file, "Bob")
    with client.session_transaction() as sess:
        sess["user_id"] = owner

    response = client.delete(f'/delete_report/{other_id}')

    assert response.status_code == 404
    assert database.get_report_by_id(other_id) is not None


def test_delete_all_reports_only_deletes_current_user_reports(client, test_db, mocker):
    owner = f"user-a-{uuid.uuid4().hex}"
    outsider = f"user-b-{uuid.uuid4().hex}"
    own_file = unique_name("own")
    other_file = unique_name("other")
    own_id = database.add_report(owner, own_file, own_file, "Alice")
    other_id = database.add_report(outsider, other_file, other_file, "Bob")
    delete_doc = mocker.patch("app.doc_processor.delete_document")
    with client.session_transaction() as sess:
        sess["user_id"] = owner

    response = client.delete('/delete_all_reports')

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert database.get_report_by_id(own_id) is None
    assert database.get_report_by_id(other_id) is not None
    delete_doc.assert_called_once_with(own_id)

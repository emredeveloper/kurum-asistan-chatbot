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


def test_rate_limit_counts_under_one_key_from_the_first_request(client, mocker, monkeypatch):
    """The limiter runs in before_request; views used to mint user_id themselves, so
    a caller's first request was counted under its IP and the rest under a fresh uid,
    splitting the count and giving away part of the quota."""
    import app as app_module
    mocker.patch.object(app_module.bot, 'ollama_chat', return_value='sabit')
    monkeypatch.setattr(app_module, '_rate_limit_store', {})

    statuses = []
    for _ in range(app_module.RATE_LIMIT_MAX_REQUESTS + 1):
        statuses.append(client.post('/chat', json={'message': 'x'}).status_code)

    assert statuses[-1] == 429
    assert statuses[:-1] == [200] * app_module.RATE_LIMIT_MAX_REQUESTS
    assert len(app_module._rate_limit_store) == 1, "requests split across several keys"


def test_chat_stream_is_rate_limited(client, mocker, monkeypatch):
    """/chat_stream reaches the same LLM as /chat, so leaving it unlimited would
    hand out a free bypass of the quota."""
    import app as app_module
    monkeypatch.setattr(app_module, '_rate_limit_store', {})
    # Stub the model: 30 real generations would outrun the 60s window and the
    # limiter would legitimately never fire.
    mocker.patch.object(app_module.bot, 'process_message_stream',
                        side_effect=lambda *a, **k: iter(['ok']))
    assert 'chat_stream' in app_module.RATE_LIMITED_ENDPOINTS

    for _ in range(app_module.RATE_LIMIT_MAX_REQUESTS):
        response = client.post('/chat_stream', json={'message': 'x'})
        b"".join(response.response)
        response.close()

    blocked = client.post('/chat_stream', json={'message': 'x'})
    assert blocked.status_code == 429


def test_cross_user_endpoints_disabled_without_token(client, mocker):
    """These expose the staff directory and every ticket in the system, so with no
    token configured they must stay closed rather than default to open."""
    mocker.patch("app.ADMIN_API_TOKEN", "")

    for path in ('/api/users', '/api/support_tickets_all'):
        response = client.get(path)
        assert response.status_code == 404, path


def test_cross_user_endpoints_reject_wrong_token(client, mocker):
    mocker.patch("app.ADMIN_API_TOKEN", "gercek-token")

    for path in ('/api/users', '/api/support_tickets_all'):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers={'X-Admin-Token': 'yanlis'}).status_code == 401, path


def test_cross_user_endpoints_accept_valid_token(client, mocker):
    mocker.patch("app.ADMIN_API_TOKEN", "gercek-token")

    for path in ('/api/users', '/api/support_tickets_all'):
        response = client.get(path, headers={'X-Admin-Token': 'gercek-token'})
        assert response.status_code == 200, path
        assert isinstance(response.get_json(), list)


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

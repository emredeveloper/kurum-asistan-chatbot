import pytest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_welcome_route(client):
    """Test the /welcome route."""
    response = client.get('/welcome')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == {'response': 'Hello! I am the Company Assistant. I can help you with the following:<br>- Current weather information<br>- Support ticket creation<br>- Questions about our internal knowledge base<br>- Upload Word/PDF documents and answer questions about their contents'}


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

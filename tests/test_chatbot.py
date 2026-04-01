import pytest
import os
import sys
import json # Added for test_support_ticket_flow_no_llm
import importlib

# Add project root for imports if conftest.py isn't handling it fully for all test runners
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot import CitizenAssistantBot # Assuming chatbot.py is in the root

# No module-level bot instance. Each test will create its own if needed.

def test_normalize_dept():
    """Tests department name normalization."""
    bot = CitizenAssistantBot()
    assert bot.normalize_dept("IT") == "it"
    assert bot.normalize_dept("  İnsan Kaynakları  ") == "insan kaynaklari"
    assert bot.normalize_dept("Muhasebe") == "muhasebe"
    assert bot.normalize_dept("Teknik Servis") == "teknik servis"
    assert bot.normalize_dept("İK") == "ik" # Test with common abbreviation
    assert bot.normalize_dept("iNSAn kAYNaklaRI") == "insan kaynaklari" # Mixed case
    assert bot.normalize_dept("ÇÖĞÜŞçi") == "cogusci" # All Turkish chars

def test_get_kurum_bilgisi():
    """Tests fetching information from the knowledge base."""
    bot = CitizenAssistantBot()
    # Test with a known keyword
    seyahat_info = bot.get_kurum_bilgisi("seyahat politikası hakkında bilgi")
    assert "şehir dışı seyahatler için önceden onay alınmalı" in seyahat_info

    # Test with another known keyword
    izin_info = bot.get_kurum_bilgisi("izin prosedürü nedir?")
    assert "Yıllık izinler için en az 3 gün önceden İK'ya başvurulmalıdır" in izin_info

    # Test with a keyword not in the knowledge base
    unknown_info = bot.get_kurum_bilgisi("kantin menüsü")
    assert "No information was found on this topic." == unknown_info

    # Test case sensitivity (should be handled by "in soru.lower()")
    mesai_info_lower = bot.get_kurum_bilgisi("mesai ücreti")
    mesai_info_mixed = bot.get_kurum_bilgisi("MeSaİ ÜcReTi")
    assert "Fazla mesai ücretleri, ilgili ayın sonunda bordroya yansıtılır." in mesai_info_lower
    assert "Fazla mesai ücretleri, ilgili ayın sonunda bordroya yansıtılır." in mesai_info_mixed

    # Test with a partial match that should still work if logic is "keyword in question"
    yemekhane_info = bot.get_kurum_bilgisi("yemekhane saatleri")
    assert "Yemekhane hafta içi 12:00-14:00 arası açıktır." in yemekhane_info

def test_get_weather_city_not_found_handling(mocker):
    """
    Tests the bot's handling of a 'city not found' response from OpenWeatherMap API.
    Mocks requests.get to simulate the API returning a 404 for an unknown city.
    """
    bot = CitizenAssistantBot()
    # Mock the response from requests.get
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"cod": "404", "message": "city not found"}
    mock_response.status_code = 404 # Though the bot primarily checks data.get("cod")

    mocker.patch('requests.get', return_value=mock_response)

    # Call the get_weather method with a city that should trigger the mocked not found response
    weather_response = bot.get_weather("OlmayanBirSehir")

    # Assert that the bot returns the specific message for city not found
    assert weather_response == "Which city would you like the weather for?"

    # Example of testing a successful weather call (optional, but good practice)
    mock_success_response = mocker.Mock()
    mock_success_response.json.return_value = {
        "cod": 200,
        "weather": [{"description": "açık"}],
        "main": {"temp": 25}
    }
    mocker.patch('requests.get', return_value=mock_success_response)
    weather_success = bot.get_weather("Ankara")
    assert "Weather in Ankara: açık, temperature: 25°C" == weather_success

def test_support_ticket_flow_no_llm(mocker):
    """
    Tests the multi-turn support ticket creation flow without actual LLM calls.
    Mocks bot.ollama_chat to simulate LLM returning tool calls.
    Mocks database calls to prevent actual DB writes and focus on bot logic.
    """
    bot = CitizenAssistantBot() # Instantiate bot locally for this test
    test_user_id = "test_user_ticket_flow"

    # No need to clean bot.user_states if bot is created fresh for the test.
    # if test_user_id in bot.user_states:
    #     del bot.user_states[test_user_id]

    # Mock database calls
    mock_add_history = mocker.patch('database.add_chat_history')
    mock_add_ticket = mocker.patch('database.add_support_ticket')
    # Also mock get_chat_history as it's called by process_message to build context
    mock_get_history = mocker.patch('database.get_chat_history', return_value=[]) # Return empty history for context

    # --- Stage 1: User initiates a support ticket, LLM suggests tool but no details ---
    # Simulate user message like "I need help" which LLM interprets as a support request
    user_message_initiate = "Yardıma ihtiyacım var, bir sorunum var."

    # Mock ollama_chat to return a generic support ticket tool call (no department or description)
    mocker.patch.object(bot, 'ollama_chat', return_value=json.dumps({
        "tool": "destek_talebi",
        "departman": None,
        "aciklama": None,
        "aciliyet": "normal",
        "kategori": "genel"
    }))

    response1 = bot.process_message(user_message_initiate, test_user_id)

    # Bot should ask for the department
    expected_response1 = f"Understood, you want to create a support ticket. Which department should I send it to? Options: {', '.join(bot.DEPARTMANLAR)}"
    assert response1 == expected_response1
    assert bot.user_states[test_user_id]['waiting_for_department'] == True
    assert bot.user_states[test_user_id]['pending_ticket'] is not None
    mock_add_history.assert_called_with(test_user_id, "destek_talebi_etkileşim", user_message_initiate, expected_response1)

    # --- Stage 2: User provides the department ---
    user_message_department = "IT departmanı" # User specifies IT

    # No LLM call expected here, bot should use the direct user message
    response2 = bot.process_message(user_message_department, test_user_id)

    # Bot should now ask for the description
    expected_response2 = "I am creating a support ticket for the IT department. Please describe your request."
    assert response2 == expected_response2
    assert bot.user_states[test_user_id]['waiting_for_department'] == False
    assert bot.user_states[test_user_id]['waiting_for_description'] == True
    assert bot.user_states[test_user_id]['last_department'] == "IT" # Assuming IT is a valid DEPARTMANLAR
    assert bot.user_states[test_user_id]['pending_ticket']['departman'] == "IT"
    mock_add_history.assert_called_with(test_user_id, "destek_talebi_etkileşim", user_message_department, expected_response2)

    # --- Stage 3: User provides the description ---
    user_message_description = "Bilgisayarım açılmıyor."

    # No LLM call expected here
    response3 = bot.process_message(user_message_description, test_user_id)

    # Bot should confirm ticket creation and a ticket_id should be generated
    # We need to check if the response contains key phrases as ticket_id is random.
    assert "has been created for the IT department" in response3
    assert "support ticket with ID" in response3

    # Check that add_support_ticket was called
    mock_add_ticket.assert_called_once()
    args, _ = mock_add_ticket.call_args
    assert args[0] == test_user_id  # user_id
    assert args[2] == "IT"  # department
    assert args[3] == user_message_description  # description
    # args[1] is the ticket_id, which is random - we can check its type or length if needed
    assert isinstance(args[1], str) and len(args[1]) == 8

    # Check that chat history for final step was added
    # The actual bot_response for this call includes the ticket_id, so we check the call to mock_add_history
    final_history_call_args = mock_add_history.call_args_list[-1]
    call_args_final_history, _ = final_history_call_args
    assert call_args_final_history[0] == test_user_id
    assert call_args_final_history[1] == "destek_talebi_oluşturuldu"
    assert call_args_final_history[2] == user_message_description # user message
    assert "has been created for the IT department" in call_args_final_history[3]
    assert "support ticket with ID" in call_args_final_history[3]

    # User state should be cleared
    assert bot.user_states[test_user_id].get('pending_ticket') is None
    assert 'waiting_for_description' not in bot.user_states[test_user_id]
    assert bot.user_states[test_user_id].get('last_department') is None


def test_llm_studio_provider_request(mocker):
    os.environ["LLM_PROVIDER"] = "lmstudio"
    import chatbot as chatbot_module
    chatbot_module = importlib.reload(chatbot_module)
    bot = chatbot_module.CitizenAssistantBot()

    mock_response = mocker.Mock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "merhaba"}}]}
    mock_response.raise_for_status.return_value = None
    post_mock = mocker.patch('requests.post', return_value=mock_response)

    result = bot.ollama_chat("selam")

    assert result == "merhaba"
    assert post_mock.call_args.kwargs["json"]["model"] == chatbot_module.LM_STUDIO_MODEL
    assert "/chat/completions" in post_mock.call_args.args[0]


def test_ollama_provider_request(mocker):
    os.environ["LLM_PROVIDER"] = "ollama"
    import chatbot as chatbot_module
    chatbot_module = importlib.reload(chatbot_module)
    bot = chatbot_module.CitizenAssistantBot()

    mock_response = mocker.Mock()
    mock_response.json.return_value = {"response": "ollama yanit"}
    mock_response.raise_for_status.return_value = None
    post_mock = mocker.patch('requests.post', return_value=mock_response)

    result = bot.ollama_chat("selam")

    assert result == "ollama yanit"
    assert post_mock.call_args.kwargs["json"]["model"] == chatbot_module.OLLAMA_MODEL
    assert post_mock.call_args.args[0].endswith("/api/generate")


def test_explicit_document_tool_request_bypasses_general_llm(mocker):
    bot = CitizenAssistantBot()
    mocker.patch('database.get_reports', return_value=[{'id': 7, 'original_filename': 'proje_ozeti.docx'}])
    explain_mock = mocker.patch.object(bot, '_explain_report', return_value="Belge aciklamasi")

    response = bot.process_message(
        json.dumps({"tool": "choose_file_and_explain", "report_id": 7, "sorgu": "içeriği açıkla bana"}),
        "test_user_doc",
    )

    assert response == "Belge aciklamasi"
    explain_mock.assert_called_once()

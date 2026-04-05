import pytest
import os
import sys
import json
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot import CitizenAssistantBot


def test_normalize_dept():
    """Tests department name normalization."""
    bot = CitizenAssistantBot()
    assert bot.normalize_dept("IT") == "it"
    assert bot.normalize_dept("  İnsan Kaynakları  ") == "insan kaynaklari"
    assert bot.normalize_dept("Muhasebe") == "muhasebe"
    assert bot.normalize_dept("Teknik Servis") == "teknik servis"
    assert bot.normalize_dept("İK") == "ik"
    assert bot.normalize_dept("iNSAn kAYNaklaRI") == "insan kaynaklari"
    assert bot.normalize_dept("ÇÖĞÜŞçi") == "cogusci"


def test_get_knowledge_base_info():
    """Tests fetching information from the knowledge base."""
    bot = CitizenAssistantBot()
    travel_info = bot.get_knowledge_base_info("travel policy information")
    assert "prior approval" in travel_info.lower()

    leave_info = bot.get_knowledge_base_info("leave procedure?")
    assert "3 days" in leave_info

    unknown_info = bot.get_knowledge_base_info("quantum physics experiment results")
    assert unknown_info == "No information was found on this topic."

    overtime_info_lower = bot.get_knowledge_base_info("overtime pay")
    overtime_info_mixed = bot.get_knowledge_base_info("Overtime Pay")
    assert "payroll" in overtime_info_lower
    assert "payroll" in overtime_info_mixed

    cafeteria_info = bot.get_knowledge_base_info("cafeteria hours")
    assert "12:00" in cafeteria_info


def test_get_weather_city_not_found_handling(mocker):
    """Tests the bot's handling of a 'city not found' response from OpenWeatherMap API."""
    bot = CitizenAssistantBot()
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"cod": "404", "message": "city not found"}
    mock_response.status_code = 404

    mocker.patch('requests.get', return_value=mock_response)

    weather_response = bot.get_weather("NonExistentCity")
    assert weather_response == "Which city would you like the weather for?"

    mock_success_response = mocker.Mock()
    mock_success_response.json.return_value = {
        "cod": 200,
        "weather": [{"description": "clear sky"}],
        "main": {"temp": 25}
    }
    mocker.patch('requests.get', return_value=mock_success_response)
    weather_success = bot.get_weather("Ankara")
    assert "Weather in Ankara: clear sky, temperature: 25°C" == weather_success


def test_support_ticket_flow_no_llm(mocker):
    """Tests the multi-turn support ticket creation flow without actual LLM calls."""
    bot = CitizenAssistantBot()
    test_user_id = "test_user_ticket_flow"

    mock_add_history = mocker.patch('database.add_chat_history')
    mock_add_ticket = mocker.patch('database.add_support_ticket')
    mock_get_history = mocker.patch('database.get_chat_history', return_value=[])

    # Stage 1: User initiates a support ticket, LLM suggests tool but no details
    user_message_initiate = "I need help, I have a problem."

    mocker.patch.object(bot, 'ollama_chat', return_value=json.dumps({
        "tool": "support_ticket",
        "department": None,
        "description": None,
        "priority": "normal",
        "category": "general"
    }))

    response1 = bot.process_message(user_message_initiate, test_user_id)

    expected_response1 = f"Understood, you want to create a support ticket. Which department should we forward it to? Options: {', '.join(bot.DEPARTMENTS)}"
    assert response1 == expected_response1
    assert bot.user_states[test_user_id]['waiting_for_department'] == True
    assert bot.user_states[test_user_id]['pending_ticket'] is not None
    mock_add_history.assert_called_with(test_user_id, "support_ticket_interaction", user_message_initiate, expected_response1)

    # Stage 2: User provides the department
    user_message_department = "IT department"

    response2 = bot.process_message(user_message_department, test_user_id)

    expected_response2 = "Creating a support ticket for the IT department. Please describe your request."
    assert response2 == expected_response2
    assert bot.user_states[test_user_id]['waiting_for_department'] == False
    assert bot.user_states[test_user_id]['waiting_for_description'] == True
    assert bot.user_states[test_user_id]['last_department'] == "IT"
    assert bot.user_states[test_user_id]['pending_ticket']['department'] == "IT"
    mock_add_history.assert_called_with(test_user_id, "support_ticket_interaction", user_message_department, expected_response2)

    # Stage 3: User provides the description
    user_message_description = "My computer won't turn on."

    response3 = bot.process_message(user_message_description, test_user_id)

    assert "has been created for the IT department" in response3
    assert "support ticket with ID" in response3

    mock_add_ticket.assert_called_once()
    args, _ = mock_add_ticket.call_args
    assert args[0] == test_user_id
    assert args[2] == "IT"
    assert args[3] == user_message_description
    assert isinstance(args[1], str) and len(args[1]) == 8

    final_history_call_args = mock_add_history.call_args_list[-1]
    call_args_final_history, _ = final_history_call_args
    assert call_args_final_history[0] == test_user_id
    assert call_args_final_history[1] == "support_ticket_created"
    assert call_args_final_history[2] == user_message_description
    assert "has been created for the IT department" in call_args_final_history[3]
    assert "support ticket with ID" in call_args_final_history[3]

    assert bot.user_states[test_user_id].get('pending_ticket') is None
    assert 'waiting_for_description' not in bot.user_states[test_user_id]
    assert bot.user_states[test_user_id].get('last_department') is None


def test_llm_studio_provider_request(mocker):
    os.environ["LLM_PROVIDER"] = "lmstudio"
    import chatbot as chatbot_module
    chatbot_module = importlib.reload(chatbot_module)
    bot = chatbot_module.CitizenAssistantBot()

    mock_response = mocker.Mock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    mock_response.raise_for_status.return_value = None
    post_mock = mocker.patch('requests.post', return_value=mock_response)

    result = bot.ollama_chat("hi")

    assert result == "hello"
    assert post_mock.call_args.kwargs["json"]["model"] == chatbot_module.LM_STUDIO_MODEL
    assert "/chat/completions" in post_mock.call_args.args[0]


def test_ollama_provider_request(mocker):
    os.environ["LLM_PROVIDER"] = "ollama"
    import chatbot as chatbot_module
    chatbot_module = importlib.reload(chatbot_module)
    bot = chatbot_module.CitizenAssistantBot()

    mock_response = mocker.Mock()
    mock_response.json.return_value = {"response": "ollama reply"}
    mock_response.raise_for_status.return_value = None
    post_mock = mocker.patch('requests.post', return_value=mock_response)

    result = bot.ollama_chat("hi")

    assert result == "ollama reply"
    assert post_mock.call_args.kwargs["json"]["model"] == chatbot_module.OLLAMA_MODEL
    assert post_mock.call_args.args[0].endswith("/api/generate")


def test_explicit_document_tool_request_bypasses_general_llm(mocker):
    bot = CitizenAssistantBot()
    mocker.patch('database.get_reports', return_value=[{'id': 7, 'original_filename': 'project_summary.docx'}])
    explain_mock = mocker.patch.object(bot, '_explain_report', return_value="Document explanation")

    response = bot.process_message(
        json.dumps({"tool": "choose_file_and_explain", "report_id": 7, "query": "explain the content"}),
        "test_user_doc",
    )

    assert response == "Document explanation"
    explain_mock.assert_called_once()

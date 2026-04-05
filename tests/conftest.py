import pytest
import os
import sys

# Add the project root to the Python path to allow imports of app and chatbot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import database
from chatbot import CitizenAssistantBot
import app as app_module

TEST_DB_NAME = "test_chatbot_data.db"

@pytest.fixture(scope="session")
def app():
    """Create and configure a new app instance for each test session."""

    # Set the database URL for testing BEFORE database.init_db() is called
    os.environ['TEST_DATABASE_URL'] = TEST_DB_NAME

    # Clean up old test database if it exists from a previous failed run
    if os.path.exists(TEST_DB_NAME):
        os.remove(TEST_DB_NAME)

    # Initialize the test database.
    database.init_db(db_name_override=TEST_DB_NAME)

    # Now that the DB is initialized, create the bot and assign it to the app module's global variable
    app_module.bot = CitizenAssistantBot()
    flask_app = app_module.app


    flask_app.config.update({
        "TESTING": True,
        "SECRET_KEY": "testing_secret_key", # Test specific secret key
        # Add other test-specific configurations if needed
    })

    yield flask_app

    # Teardown: Clean up the test database file after the session
    if os.path.exists(TEST_DB_NAME):
        os.remove(TEST_DB_NAME)

    # Unset the environment variable
    if 'TEST_DATABASE_URL' in os.environ:
        del os.environ['TEST_DATABASE_URL']


@pytest.fixture()
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture()
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()


@pytest.fixture(scope="function")
def test_db(app):
    """Provides the database module after the test DB has been initialised."""
    yield database

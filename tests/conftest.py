import pytest
import os
import sys

# Add the project root to the Python path to allow imports of app and chatbot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app # Import the Flask app instance
import database # Import your database module

TEST_DB_NAME = "test_chatbot_data.db"

@pytest.fixture(scope="session")
def app():
    """Create and configure a new app instance for each test session."""
    flask_app.config.update({
        "TESTING": True,
        "SECRET_KEY": "testing_secret_key", # Test specific secret key
        # Add other test-specific configurations if needed
    })

    # Set the database URL for testing BEFORE database.init_db() is called by any other fixture or test
    os.environ['TEST_DATABASE_URL'] = TEST_DB_NAME

    # Clean up old test database if it exists from a previous failed run
    if os.path.exists(TEST_DB_NAME):
        os.remove(TEST_DB_NAME)

    # Initialize the test database (this will now use TEST_DB_NAME due to the env var)
    # Explicitly pass the test DB name to init_db for clarity and robustness during test setup
    database.init_db(db_name_override=TEST_DB_NAME)

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


@pytest.fixture(scope="function") # function scope for a clean db per test if needed, or session for shared
def test_db(app): # Depends on app fixture to ensure DB is initialized for testing
    """
    Provides the test database context.
    Ensures that init_db has been called using the TEST_DATABASE_URL.
    This fixture can be used by tests that need to directly interact with the DB setup.
    """
    # The 'app' fixture already handles setting the TEST_DATABASE_URL,
    # cleaning up old DBs, and initializing the DB.
    # This fixture is mostly a marker or can be used to yield a connection if needed.

    # For tests that might modify data and need a clean slate, ensure cleanup
    # or re-initialization if 'app' was session-scoped and this is function-scoped.
    # However, with app being session-scoped and test_db function-scoped,
    # the DB will persist across tests in a session.
    # If true isolation per test function for DB is needed, 'app' might need function scope too,
    # or this fixture needs to manage per-function DB state more actively.

    # For now, this relies on the session-scoped app fixture's DB setup.
    # If a test needs a truly empty DB, it should manage that itself or use a more specific fixture.
    yield database # Or yield database.get_db_connection() if tests need a connection object

    # No specific teardown here as the main app fixture handles DB file removal.
    # If transactions were used, they'd be rolled back here for function scope.

# Example of how you might provide a direct DB connection if needed by tests often:
# @pytest.fixture(scope="function")
# def db_conn(test_db): # Depends on test_db to ensure it's set up
#     conn = database.get_db_connection()
#     yield conn
#     conn.close()

# Note: The current 'test_db' fixture yields the 'database' module.
# Tests can then call database.get_db_connection() if they need a new connection.
# The `app` fixture being session-scoped means the test DB file (TEST_DB_NAME)
# will be created once per session and removed at the end.
# If tests modify the DB and expect clean states, they need to handle it or
# the `app` fixture scope might need to be `function`.
# For this project, session scope for app (and thus DB file) is likely fine for many tests,
# but tests for `/api/history` (like test_get_history_empty) will need careful handling
# or a more specific fixture to ensure the DB is empty for that test.

# Let's refine test_db to be more explicit about providing a clean state for relevant tests
# For now, the `app` fixture handles one-time setup and teardown per session.
# Tests needing pristine DB state per test function would require more complex fixture management
# or explicit cleanup within the test.
# The current `test_db` fixture is more of a dependency hook for tests that know they need DB interaction.
# A common pattern is to use a transaction and rollback for each test for SQL DBs,
# but for SQLite file manipulation, re-creating the file or tables per test is an option.

# For 'test_get_history_empty', it will run against the DB initialized by 'app'.
# It's important that no preceding tests in the session populate history for that user.
# Or, that test could specifically delete history for its test user before running.
# A simpler approach for now is that `test_get_history_empty` will use a unique user_id.

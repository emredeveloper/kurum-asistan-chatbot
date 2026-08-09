import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import database


@pytest.fixture()
def kb_db(tmp_path, monkeypatch):
    """A throwaway SQLite file seeded with the default knowledge base."""
    db_path = tmp_path / "kb_test.db"
    monkeypatch.setenv("TEST_DATABASE_URL", str(db_path))
    database.init_db(db_name_override=str(db_path))
    return database


@pytest.mark.parametrize("query,expected", [
    # The seed answers are English, but the UI and system prompt are Turkish, so
    # Turkish phrasings have to resolve too.
    ("yillik izin proseduru nedir?", "3 days"),
    ("yıllık izin prosedürü nedir?", "3 days"),
    ("fazla mesai ucreti nasil odenir?", "payroll"),
    ("fazla mesai ücreti", "payroll"),
    ("yemekhane saat kacta aciliyor?", "12:00"),
    ("yemekhane ne zaman açık?", "12:00"),
    ("seyahat politikasi nedir?", "prior approval"),
    # English keys must keep working.
    ("what is the leave procedure?", "3 days"),
    ("cafeteria hours", "12:00"),
])
def test_search_kb_answer_matches_turkish_and_english(kb_db, query, expected):
    answer = kb_db.search_kb_answer(query)
    assert answer is not None, f"no KB match for {query!r}"
    assert expected in answer


def test_search_kb_answer_returns_none_for_unrelated_query(kb_db):
    assert kb_db.search_kb_answer("kuantum fizigi deney sonuclari") is None


def test_search_kb_answer_ignores_blank_query(kb_db):
    assert kb_db.search_kb_answer("") is None
    assert kb_db.search_kb_answer("   ") is None


def test_reseeding_does_not_duplicate_rows(kb_db):
    """Seeding matches on the answer, not the keyword list, so extending keywords
    (e.g. adding Turkish synonyms) updates rows instead of inserting new ones."""
    conn = kb_db.get_db_connection()
    before = conn.cursor().execute("SELECT COUNT(*) FROM institution_knowledge").fetchone()[0]
    conn.close()

    kb_db.seed_default_knowledge()
    kb_db.seed_default_knowledge()

    conn = kb_db.get_db_connection()
    after = conn.cursor().execute("SELECT COUNT(*) FROM institution_knowledge").fetchone()[0]
    conn.close()
    assert after == before


def test_seeding_updates_keywords_on_existing_answer(kb_db):
    """A row seeded with only English keywords should gain the Turkish ones."""
    answer = "The cafeteria is open on weekdays between 12:00 and 14:00."
    conn = kb_db.get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE institution_knowledge SET keywords = ? WHERE answer = ?",
                ("cafeteria, lunch, meal", answer))
    conn.commit()
    conn.close()
    assert kb_db.search_kb_answer("yemekhane saat kacta") is None

    kb_db.seed_default_knowledge()

    assert kb_db.search_kb_answer("yemekhane saat kacta") == answer


def test_normalize_text_folds_turkish_diacritics():
    assert database._normalize_text("Prosedürü") == "proseduru"
    assert database._normalize_text("YILLIK İZİN") == "yillik izin"
    assert database._normalize_text("ÇÖĞÜŞ") == "cogus"
    assert database._normalize_text(None) == ""

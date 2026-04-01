import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from document_processor import DocumentProcessor


def test_search_in_documents_returns_metadata_for_blank_query():
    processor = DocumentProcessor()
    processor.metadata = {
        2: {"report_id": 20, "text": "ikinci parca", "embedding": [0.0, 1.0]},
        1: {"report_id": 10, "text": "ilk parca", "embedding": [1.0, 0.0]},
    }

    results = processor.search_in_documents("", top_k=2)

    assert results == [
        {"text": "ilk parca", "report_id": 10, "score": None},
        {"text": "ikinci parca", "report_id": 20, "score": None},
    ]


def test_search_in_documents_uses_embedding_similarity():
    processor = DocumentProcessor()
    processor.metadata = {
        1: {"report_id": 10, "text": "ilk parca", "embedding": [1.0, 0.0]},
        2: {"report_id": 20, "text": "ikinci parca", "embedding": [0.0, 1.0]},
    }
    processor._embed_many = lambda texts: [[1.0, 0.0]]

    results = processor.search_in_documents("bir sey ara", top_k=1)

    assert len(results) == 1
    assert results[0]["report_id"] == 10
    assert results[0]["text"] == "ilk parca"

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import document_processor
from document_processor import DocumentProcessor, LM_STUDIO_EMBED_MODEL, OLLAMA_EMBED_MODEL


def current_meta(report_id, text, embedding):
    processor = DocumentProcessor()
    sig = processor._embedding_signature(embedding)
    return {
        "report_id": report_id,
        "text": text,
        "embedding": embedding,
        "embedding_provider": sig["provider"],
        "embedding_model": sig["model"],
        "embedding_dimension": sig["dimension"],
    }


def test_blank_query_respects_report_id_filter():
    """A blank query walks chunks in id order; without the filter it always returned
    the oldest document, so summarising a newly uploaded report found nothing."""
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(4, "eski belge", [1.0, 0.0]),
        2: current_meta(4, "eski belge devam", [0.9, 0.1]),
        3: current_meta(1, "yeni belge", [0.0, 1.0]),
    }

    results = processor.search_in_documents("", top_k=8, report_id=1)

    assert results == [{"text": "yeni belge", "report_id": 1, "score": None}]


def test_search_enforces_allowed_report_ids():
    """The vector store is shared across users, so a search made on one user's
    behalf must never surface another user's document text."""
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(4, "A kullanicisinin gizli maas verisi", [1.0, 0.0]),
        2: current_meta(7, "B kullanicisinin belgesi", [1.0, 0.0]),
    }
    processor._embed_many = lambda texts: [[1.0, 0.0]]

    results = processor.search_in_documents("maas", top_k=5, allowed_report_ids=[7])

    assert [r["report_id"] for r in results] == [7]


def test_search_with_empty_allowlist_returns_nothing():
    processor = DocumentProcessor()
    processor.metadata = {1: current_meta(4, "gizli", [1.0, 0.0])}
    processor._embed_many = lambda texts: [[1.0, 0.0]]

    assert processor.search_in_documents("gizli", top_k=5, allowed_report_ids=[]) == []
    assert processor.search_in_documents("", top_k=5, allowed_report_ids=[]) == []


def test_blank_query_enforces_allowed_report_ids():
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(4, "baskasinin belgesi", [1.0, 0.0]),
        2: current_meta(7, "kendi belgem", [0.0, 1.0]),
    }

    results = processor.search_in_documents("", top_k=8, allowed_report_ids=[7])

    assert [r["report_id"] for r in results] == [7]


def test_similarity_search_respects_report_id_filter():
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(4, "eski belge", [1.0, 0.0]),
        2: current_meta(1, "yeni belge", [1.0, 0.0]),
    }
    processor._embed_many = lambda texts: [[1.0, 0.0]]

    results = processor.search_in_documents("ara", top_k=5, report_id=1)

    assert [r["report_id"] for r in results] == [1]


def test_reused_report_id_drops_stale_chunks(mocker, tmp_path):
    """SQLite restarts its counter on a fresh DB while the vector store persists,
    so a new document can inherit an old document's report_id."""
    processor = DocumentProcessor()
    processor.metadata = {1: current_meta(1, "onceki belgenin icerigi", [1.0, 0.0])}
    doc_path = tmp_path / "yeni.pdf"
    doc_path.write_bytes(b"%PDF-1.4 fake")
    mocker.patch.object(processor, "_extract_text_from_pdf", return_value="yeni icerik")
    mocker.patch.object(processor, "_embed_many", return_value=[[0.0, 1.0]])
    mocker.patch.object(processor, "_save")
    mocker.patch("document_processor.database.mark_report_as_processed")

    processor.process_and_embed_document(doc_path, report_id=1)

    texts = [m["text"] for m in processor.metadata.values()]
    assert "onceki belgenin icerigi" not in texts
    assert texts == ["yeni icerik"]


def test_prune_orphan_chunks_removes_only_unknown_reports(mocker):
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(4, "mevcut rapor", [1.0, 0.0]),
        2: current_meta(99, "silinmis rapor", [0.0, 1.0]),
    }
    mocker.patch.object(processor, "_save")
    mocker.patch("document_processor.database.get_reports", return_value=[{"id": 4}])

    result = processor.prune_orphan_chunks()

    assert result["removed_chunks"] == 1
    assert result["orphan_report_ids"] == ["99"]
    assert [m["text"] for m in processor.metadata.values()] == ["mevcut rapor"]


def test_prune_orphan_chunks_refuses_when_report_table_empty(mocker):
    """Pointing the app at a fresh/test database must not wipe real embeddings."""
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(4, "gercek belge", [1.0, 0.0]),
        2: current_meta(8, "gercek belge 2", [0.0, 1.0]),
    }
    save_mock = mocker.patch.object(processor, "_save")
    mocker.patch("document_processor.database.get_reports", return_value=[])

    result = processor.prune_orphan_chunks()

    assert result["removed_chunks"] == 0
    assert result["skipped"] == "empty_report_table"
    assert len(processor.metadata) == 2
    save_mock.assert_not_called()


def test_prune_orphan_chunks_refuses_bulk_removal(mocker):
    """Removing most of the store signals a misconfigured DB, not stale data."""
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(4, "a", [1.0, 0.0]),
        2: current_meta(8, "b", [0.0, 1.0]),
        3: current_meta(9, "c", [1.0, 1.0]),
    }
    mocker.patch.object(processor, "_save")
    mocker.patch("document_processor.database.get_reports", return_value=[{"id": 4}])

    result = processor.prune_orphan_chunks()

    assert result["removed_chunks"] == 0
    assert result["skipped"] == "bulk_removal"
    assert len(processor.metadata) == 3


def test_prune_orphan_chunks_keeps_everything_when_db_unreadable(mocker):
    processor = DocumentProcessor()
    processor.metadata = {1: current_meta(4, "mevcut rapor", [1.0, 0.0])}
    mocker.patch("document_processor.database.get_reports", side_effect=RuntimeError("db down"))

    result = processor.prune_orphan_chunks()

    assert result["removed_chunks"] == 0
    assert len(processor.metadata) == 1


def test_discard_document_matches_report_id_across_types():
    processor = DocumentProcessor()
    processor.metadata = {1: current_meta("7", "string id", [1.0, 0.0])}

    assert processor._discard_document(7) == 1


def test_search_in_documents_returns_metadata_for_blank_query():
    processor = DocumentProcessor()
    processor.metadata = {
        2: current_meta(20, "ikinci parca", [0.0, 1.0]),
        1: current_meta(10, "ilk parca", [1.0, 0.0]),
    }

    results = processor.search_in_documents("", top_k=2)

    assert results == [
        {"text": "ilk parca", "report_id": 10, "score": None},
        {"text": "ikinci parca", "report_id": 20, "score": None},
    ]


def test_search_in_documents_uses_embedding_similarity():
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(10, "ilk parca", [1.0, 0.0]),
        2: current_meta(20, "ikinci parca", [0.0, 1.0]),
    }
    processor._embed_many = lambda texts: [[1.0, 0.0]]

    results = processor.search_in_documents("bir sey ara", top_k=1)

    assert len(results) == 1
    assert results[0]["report_id"] == 10
    assert results[0]["text"] == "ilk parca"


def test_search_in_documents_skips_incompatible_embedding_dimensions():
    processor = DocumentProcessor()
    processor.metadata = {
        1: current_meta(10, "uyumlu parca", [1.0, 0.0]),
        2: current_meta(20, "bozuk parca", [1.0, 0.0, 0.0]),
    }
    processor._embed_many = lambda texts: [[1.0, 0.0]]

    results = processor.search_in_documents("bir sey ara", top_k=5)

    assert results == [{"text": "uyumlu parca", "report_id": 10, "score": 1.0}]


def test_search_in_documents_skips_legacy_embedding_metadata():
    processor = DocumentProcessor()
    processor.metadata = {
        1: {"report_id": 10, "text": "legacy parca", "embedding": [1.0, 0.0]},
        2: current_meta(20, "guncel parca", [1.0, 0.0]),
    }
    processor._embed_many = lambda texts: [[1.0, 0.0]]

    results = processor.search_in_documents("bir sey ara", top_k=5)

    assert results == [{"text": "guncel parca", "report_id": 20, "score": 1.0}]


def test_prune_incompatible_embeddings_removes_legacy_chunks(mocker):
    processor = DocumentProcessor()
    processor.metadata = {
        1: {"report_id": 10, "text": "legacy parca", "embedding": [1.0, 0.0]},
        2: current_meta(20, "guncel parca", [1.0, 0.0]),
    }
    save_mock = mocker.patch.object(processor, "_save")
    mark_mock = mocker.patch("document_processor.database.mark_report_as_unprocessed")

    result = processor.prune_incompatible_embeddings()

    assert result == {"removed_chunks": 1, "affected_report_ids": [10]}
    assert list(processor.metadata.keys()) == [2]
    save_mock.assert_called_once()
    mark_mock.assert_called_once_with(10)


def test_cosine_similarity_rejects_dimension_mismatch():
    processor = DocumentProcessor()

    try:
        processor._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
    except ValueError as exc:
        assert "Embedding dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected dimension mismatch to fail")


def test_process_document_stores_embedding_metadata(mocker, tmp_path):
    processor = DocumentProcessor()
    processor.metadata = {}
    doc_path = tmp_path / "sample.pdf"
    doc_path.write_bytes(b"%PDF-1.4 fake")
    mocker.patch.object(processor, "_extract_text_from_pdf", return_value="alpha beta gamma")
    mocker.patch.object(processor, "_embed_many", return_value=[[0.1, 0.2, 0.3]])
    save_mock = mocker.patch.object(processor, "_save")
    mark_mock = mocker.patch("document_processor.database.mark_report_as_processed")

    processor.process_and_embed_document(doc_path, report_id=42)

    stored = processor.metadata[1]
    assert stored["report_id"] == 42
    assert stored["embedding_provider"] == document_processor.LLM_PROVIDER
    assert stored["embedding_model"] == processor._embedding_model_name()
    assert stored["embedding_dimension"] == 3
    save_mock.assert_called_once()
    mark_mock.assert_called_once_with(42)


def test_embed_many_uses_lm_studio_openai_embeddings_api(mocker):
    mocker.patch.object(document_processor, "LLM_PROVIDER", "lmstudio")
    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }
    post = mocker.patch("document_processor.requests.post", return_value=mock_resp)
    processor = DocumentProcessor()
    out = processor._embed_many(["first", "second"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]
    assert post.call_count == 1
    url = post.call_args.args[0]
    assert url.endswith("/embeddings")
    assert post.call_args.kwargs["json"]["model"] == LM_STUDIO_EMBED_MODEL
    assert post.call_args.kwargs["json"]["input"] == ["first", "second"]


def test_embed_many_uses_ollama_when_llm_provider_ollama(mocker):
    mocker.patch.object(document_processor, "LLM_PROVIDER", "ollama")
    mock_resp = mocker.Mock()
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = {"embeddings": [[0.5], [0.25]]}
    post = mocker.patch("document_processor.requests.post", return_value=mock_resp)
    processor = DocumentProcessor()
    out = processor._embed_many(["x", "y"])
    assert out == [[0.5], [0.25]]
    assert post.call_count == 1
    assert "/api/embed" in post.call_args.args[0]
    assert post.call_args.kwargs["json"]["model"] == OLLAMA_EMBED_MODEL
    assert post.call_args.kwargs["json"]["input"] == ["x", "y"]

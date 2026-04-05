import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import document_processor
from document_processor import DocumentProcessor, LM_STUDIO_EMBED_MODEL, OLLAMA_EMBED_MODEL


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

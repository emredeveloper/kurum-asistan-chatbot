import json
import logging
import math
import os
import re
from pathlib import Path

import docx
import pymupdf
import pypdf
import requests

import database

logger = logging.getLogger(__name__)

# Same switch as chat: embeddings follow the active LLM provider.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio").strip().lower()

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "")
LM_STUDIO_EMBED_MODEL = os.getenv(
    "LM_STUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
)
try:
    _LM_STUDIO_EMBED_BATCH_SIZE = max(1, int(os.getenv("LM_STUDIO_EMBED_BATCH_SIZE", "64")))
except ValueError:
    _LM_STUDIO_EMBED_BATCH_SIZE = 64

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
VECTOR_STORE_PATH = os.path.join(os.getcwd(), "vector_store")
METADATA_FILE = os.path.join(VECTOR_STORE_PATH, "metadata.json")

if not os.path.exists(VECTOR_STORE_PATH):
    os.makedirs(VECTOR_STORE_PATH)


class DocumentProcessor:
    def __init__(self):
        self.metadata = {}
        self._load()

    def _load(self):
        if not os.path.exists(METADATA_FILE):
            self.metadata = {}
            return
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = {int(k): v for k, v in json.load(f).items()}
        except Exception as e:
            logger.warning("Could not load metadata: %s", e)
            self.metadata = {}

    def _save(self):
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        if text.strip():
            return text

        try:
            doc = pymupdf.open(file_path)
            return "".join(page.get_text() for page in doc)
        except Exception as e:
            logger.warning("PDF fallback extraction failed: %s", e)
            return ""

    def _extract_text_from_docx(self, file_path):
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _split_text(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks = []
        cursor = 0
        chunk_size = 900
        overlap = 120
        while cursor < len(text):
            end = min(len(text), cursor + chunk_size)
            piece = text[cursor:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            cursor = max(end - overlap, cursor + 1)
        return chunks

    def _embed_many(self, texts):
        """Route embeddings to Ollama or LM Studio based on `LLM_PROVIDER`."""
        if not texts:
            return []
        if LLM_PROVIDER == "ollama":
            return self._embed_many_ollama(texts)
        return self._embed_many_lm_studio(texts)

    def _embedding_model_name(self):
        if LLM_PROVIDER == "ollama":
            return OLLAMA_EMBED_MODEL
        return LM_STUDIO_EMBED_MODEL

    def _embedding_signature(self, vector):
        return {
            "provider": LLM_PROVIDER,
            "model": self._embedding_model_name(),
            "dimension": len(vector or []),
        }

    def _is_current_embedding(self, meta, expected_dimension: int | None = None):
        embedding = meta.get("embedding") or []
        dimension = int(meta.get("embedding_dimension") or 0)
        if not dimension:
            return False
        if expected_dimension is not None and (dimension != expected_dimension or len(embedding) != expected_dimension):
            return False
        return (
            meta.get("embedding_provider") == LLM_PROVIDER
            and meta.get("embedding_model") == self._embedding_model_name()
        )

    def prune_incompatible_embeddings(self):
        """Remove chunks produced by an old provider/model or without signature metadata."""
        removed_report_ids = set()
        ids_to_remove = []
        for chunk_id, meta in self.metadata.items():
            if not self._is_current_embedding(meta):
                ids_to_remove.append(chunk_id)
                if meta.get("report_id") is not None:
                    removed_report_ids.add(meta.get("report_id"))

        for chunk_id in ids_to_remove:
            self.metadata.pop(chunk_id, None)
        if ids_to_remove:
            self._save()
            for report_id in removed_report_ids:
                database.mark_report_as_unprocessed(report_id)
        return {
            "removed_chunks": len(ids_to_remove),
            "affected_report_ids": sorted(removed_report_ids),
        }

    def _embed_many_lm_studio(self, texts):
        """OpenAI-compatible `POST .../v1/embeddings` (LM Studio)."""
        url = f"{LM_STUDIO_BASE_URL}/embeddings"
        headers = {"Content-Type": "application/json"}
        if LM_STUDIO_API_KEY:
            headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), _LM_STUDIO_EMBED_BATCH_SIZE):
            batch = texts[start : start + _LM_STUDIO_EMBED_BATCH_SIZE]
            response = requests.post(
                url,
                headers=headers,
                json={"model": LM_STUDIO_EMBED_MODEL, "input": batch},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("data") or []
            items.sort(key=lambda x: int(x.get("index", 0)))
            for item in items:
                vector = item.get("embedding")
                if not vector:
                    raise RuntimeError("LM Studio returned an empty embedding vector.")
                all_vectors.append(vector)
            if len(items) != len(batch):
                raise RuntimeError(
                    f"Embedding batch size mismatch: sent {len(batch)}, got {len(items)} vectors."
                )

        if len(all_vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(all_vectors)}."
            )
        return all_vectors

    def _embed_many_ollama(self, texts):
        """Ollama `/api/embed` batch, then `/api/embeddings` per text if needed."""
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": OLLAMA_EMBED_MODEL, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings") or []
            if embeddings and len(embeddings) == len(texts):
                return embeddings
        except Exception as e:
            logger.debug("Ollama batch embed failed, falling back per text: %s", e)

        out: list[list[float]] = []
        for text in texts:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            vector = data.get("embedding")
            if not vector:
                raise RuntimeError("Ollama returned an empty embedding vector.")
            out.append(vector)
        return out

    def _cosine_similarity(self, left, right):
        if len(left or []) != len(right or []):
            raise ValueError(
                f"Embedding dimension mismatch: query={len(left or [])}, stored={len(right or [])}"
            )
        left_norm = math.sqrt(sum(x * x for x in left))
        right_norm = math.sqrt(sum(x * x for x in right))
        if left_norm == 0 or right_norm == 0:
            return -1.0
        dot = sum(x * y for x, y in zip(left, right))
        return dot / (left_norm * right_norm)

    def process_and_embed_document(self, file_path, report_id):
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.suffix.lower() == ".pdf":
                text = self._extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() in [".doc", ".docx"]:
                text = self._extract_text_from_docx(file_path)
            else:
                return False

            chunks = self._split_text(text)
            if not chunks:
                return False

            embeddings = self._embed_many(chunks)
            # A report_id is reused whenever the SQLite counter restarts (fresh DB,
            # manual delete). Drop any stale chunks carrying this id first, otherwise
            # the new document would be mixed with an unrelated older one.
            stale = self._discard_document(report_id)
            if stale:
                logger.info("Removed %s stale chunk(s) for reused report_id=%s", stale, report_id)
            start_id = max(self.metadata.keys()) + 1 if self.metadata else 1

            for index, chunk in enumerate(chunks):
                chunk_id = start_id + index
                signature = self._embedding_signature(embeddings[index])
                self.metadata[chunk_id] = {
                    "report_id": report_id,
                    "text": chunk,
                    "embedding": embeddings[index],
                    "embedding_provider": signature["provider"],
                    "embedding_model": signature["model"],
                    "embedding_dimension": signature["dimension"],
                }

            self._save()
            database.mark_report_as_processed(report_id)
            return True
        except Exception as e:
            logger.exception("Error processing document %s", file_path)
            return False

    def search_in_documents(self, query: str, top_k=5, report_id=None, allowed_report_ids=None):
        """Search stored chunks.

        `report_id` narrows the search to one document. `allowed_report_ids` is the
        tenancy boundary: the vector store is shared across users, so a caller acting
        on behalf of a user must pass that user's report ids or a query can surface
        another user's document text. An empty collection matches nothing.
        """
        if not self.metadata:
            return []

        allowed = None if allowed_report_ids is None else {str(r) for r in allowed_report_ids}

        def _belongs(meta):
            current = str(meta.get("report_id"))
            if allowed is not None and current not in allowed:
                return False
            return report_id is None or current == str(report_id)

        if not (query or "").strip():
            results = []
            for chunk_id in sorted(self.metadata.keys()):
                meta = self.metadata.get(chunk_id, {})
                if not self._is_current_embedding(meta) or not _belongs(meta):
                    continue
                results.append(
                    {
                        "text": meta.get("text", ""),
                        "report_id": meta.get("report_id"),
                        "score": None,
                    }
                )
                if len(results) >= top_k:
                    break
            return results

        query_vector = self._embed_many([query])[0]
        query_dimension = len(query_vector or [])
        scored = []
        for meta in self.metadata.values():
            if not _belongs(meta):
                continue
            embedding = meta.get("embedding")
            if not embedding:
                continue
            stored_dimension = int(meta.get("embedding_dimension") or len(embedding))
            if not self._is_current_embedding(meta, expected_dimension=query_dimension):
                logger.warning(
                    "Skipping document chunk with incompatible embedding signature: report_id=%s provider=%s model=%s stored_dim=%s query_dim=%s",
                    meta.get("report_id"),
                    meta.get("embedding_provider") or "legacy",
                    meta.get("embedding_model") or "legacy",
                    stored_dimension,
                    query_dimension,
                )
                continue
            scored.append(
                {
                    "text": meta["text"],
                    "report_id": meta["report_id"],
                    "score": self._cosine_similarity(query_vector, embedding),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def prune_orphan_chunks(self):
        """Drop chunks whose report no longer exists in the database.

        The vector store and SQLite keep their own state, so a report deleted
        outside the app (or a database that was reset) leaves chunks behind that
        can later collide with a reused report_id.
        """
        if not self.metadata:
            return {"removed_chunks": 0, "orphan_report_ids": []}

        try:
            known_ids = {str(r["id"]) for r in database.get_reports(limit=1000000)}
        except Exception as e:
            logger.warning("Could not read reports for orphan pruning: %s", e)
            return {"removed_chunks": 0, "orphan_report_ids": []}

        # An empty report table means the store and the database disagree wholesale:
        # a fresh/other database, or a pointer to the wrong file. Deleting every chunk
        # on that signal destroys embeddings that cost real time and money to rebuild,
        # so treat it as a misconfiguration and keep the data.
        if not known_ids:
            logger.warning(
                "Skipping orphan pruning: the database lists no reports but the vector "
                "store holds %s chunk(s). Refusing to clear it.", len(self.metadata)
            )
            return {"removed_chunks": 0, "orphan_report_ids": [], "skipped": "empty_report_table"}

        orphan_ids = set()
        ids_to_remove = []
        for chunk_id, meta in self.metadata.items():
            report_id = meta.get("report_id")
            if str(report_id) not in known_ids:
                ids_to_remove.append(chunk_id)
                orphan_ids.add(report_id)

        # Wiping most of the store at once is far more likely to be a misconfigured
        # database than a genuine cleanup. Report it instead of acting on it.
        if len(ids_to_remove) > len(self.metadata) * 0.5:
            logger.warning(
                "Skipping orphan pruning: it would remove %s of %s chunk(s), which "
                "suggests the wrong database is configured rather than stale data.",
                len(ids_to_remove), len(self.metadata),
            )
            return {"removed_chunks": 0, "orphan_report_ids": [], "skipped": "bulk_removal"}

        for chunk_id in ids_to_remove:
            self.metadata.pop(chunk_id, None)
        if ids_to_remove:
            self._save()
            logger.info(
                "Pruned %s orphan chunk(s) from report_id(s) %s",
                len(ids_to_remove), sorted(map(str, orphan_ids)),
            )
        return {
            "removed_chunks": len(ids_to_remove),
            "orphan_report_ids": sorted(map(str, orphan_ids)),
        }

    def _discard_document(self, report_id_to_delete) -> int:
        """Drop a document's chunks in memory. Returns how many were removed."""
        ids_to_remove = [
            chunk_id
            for chunk_id, meta in self.metadata.items()
            if str(meta.get("report_id")) == str(report_id_to_delete)
        ]
        for chunk_id in ids_to_remove:
            self.metadata.pop(chunk_id, None)
        return len(ids_to_remove)

    def delete_document(self, report_id_to_delete):
        self._discard_document(report_id_to_delete)
        self._save()


processor = DocumentProcessor()

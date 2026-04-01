import json
import math
import os
import re
from pathlib import Path

import docx
import fitz
import pypdf
import requests

import database

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
            print(f"Could not load metadata: {e}")
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
            doc = fitz.open(file_path)
            return "".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"PDF fallback extraction failed: {e}")
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
        if not texts:
            return []

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": OLLAMA_EMBED_MODEL, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings") or []
            if embeddings:
                return embeddings
        except Exception:
            pass

        embeddings = []
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
                raise RuntimeError("Embedding response is empty.")
            embeddings.append(vector)
        return embeddings

    def _cosine_similarity(self, left, right):
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
                return

            chunks = self._split_text(text)
            if not chunks:
                return

            embeddings = self._embed_many(chunks)
            start_id = max(self.metadata.keys()) + 1 if self.metadata else 1

            for index, chunk in enumerate(chunks):
                chunk_id = start_id + index
                self.metadata[chunk_id] = {
                    "report_id": report_id,
                    "text": chunk,
                    "embedding": embeddings[index],
                }

            self._save()
            database.mark_report_as_processed(report_id)
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")

    def search_in_documents(self, query: str, top_k=5):
        if not self.metadata:
            return []

        if not (query or "").strip():
            results = []
            for chunk_id in sorted(self.metadata.keys())[:top_k]:
                meta = self.metadata.get(chunk_id, {})
                results.append(
                    {
                        "text": meta.get("text", ""),
                        "report_id": meta.get("report_id"),
                        "score": None,
                    }
                )
            return results

        query_vector = self._embed_many([query])[0]
        scored = []
        for meta in self.metadata.values():
            embedding = meta.get("embedding")
            if not embedding:
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

    def delete_document(self, report_id_to_delete):
        ids_to_remove = [
            chunk_id
            for chunk_id, meta in self.metadata.items()
            if meta.get("report_id") == report_id_to_delete
        ]
        for chunk_id in ids_to_remove:
            self.metadata.pop(chunk_id, None)
        self._save()


processor = DocumentProcessor()

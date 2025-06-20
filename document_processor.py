import os
import re
from pathlib import Path
import numpy as np
import pypdf
import docx
from sentence_transformers import SentenceTransformer
import faiss
import json
import database
import fitz  # PyMuPDF

# Use a common, high-performance model suitable for multilingual or Turkish content if possible
# For this example, 'all-MiniLM-L6-v2' is a good starting point.
MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_STORE_PATH = os.path.join(os.getcwd(), 'vector_store')
INDEX_FILE = os.path.join(VECTOR_STORE_PATH, 'faiss_index.bin')
METADATA_FILE = os.path.join(VECTOR_STORE_PATH, 'metadata.json')

if not os.path.exists(VECTOR_STORE_PATH):
    os.makedirs(VECTOR_STORE_PATH)

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            try:
                self.index = faiss.read_index(INDEX_FILE)
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Could not load existing index or metadata: {e}")
                self.index = None
                self.metadata = []
        
        if self.index is None:
            # Get dimension from model
            d = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(d)  # Using L2 distance for similarity
            self.metadata = []


    def _save(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)

    def _extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    print(f"[RAG-WARN] PDF sayfa {page_num+1} boş veya metin çıkarılamadı.")
                text += page_text
        if not text.strip():
            print(f"[RAG-ERROR] PDF'den hiç metin çıkarılamadı: {file_path}")
            print(f"[RAG-INFO] PyMuPDF ile tekrar deneniyor...")
            try:
                doc = fitz.open(file_path)
                pymupdf_text = ""
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    if not page_text.strip():
                        print(f"[RAG-WARN] PyMuPDF ile de sayfa {page_num+1} boş veya metin çıkarılamadı.")
                    pymupdf_text += page_text
                text = pymupdf_text
                if text.strip():
                    print(f"[RAG-INFO] PyMuPDF ile metin çıkarıldı, uzunluk: {len(text)} karakter")
                else:
                    print(f"[RAG-ERROR] PyMuPDF ile de metin çıkarılamadı: {file_path}")
            except Exception as e:
                print(f"[RAG-ERROR] PyMuPDF ile metin çıkarılırken hata: {e}")
        return text

    def _extract_text_from_docx(self, file_path):
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    def _split_text(self, text):
        # A more robust splitting strategy
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        # Split by paragraphs, but also handle long lines.
        chunks = re.split(r'\n\n+', text) # Split by double newlines (paragraphs)
        final_chunks = []
        for chunk in chunks:
            # Further split chunks that are too long (e.g., > 512 tokens, rough estimate)
            if len(chunk) > 1000: # Character limit heuristic
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current_sub_chunk = ""
                for sentence in sentences:
                    if len(current_sub_chunk) + len(sentence) < 1000:
                        current_sub_chunk += sentence + " "
                    else:
                        final_chunks.append(current_sub_chunk.strip())
                        current_sub_chunk = sentence + " "
                if current_sub_chunk:
                    final_chunks.append(current_sub_chunk.strip())
            elif chunk.strip():
                final_chunks.append(chunk.strip())
        return final_chunks

    def process_and_embed_document(self, file_path, report_id):
        """
        Processes a single document, extracts text, creates embeddings, and adds to FAISS.
        `report_id` is the ID from the 'reports' table in the database.
        """
        try:
            print(f"[RAG-DEBUG] İşleme başlandı: {file_path}, report_id: {report_id}")
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"[RAG-ERROR] Dosya bulunamadı: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.suffix.lower() == '.pdf':
                text = self._extract_text_from_pdf(file_path)
                print(f"[RAG-DEBUG] PDF'den metin çıkarıldı, uzunluk: {len(text)} karakter")
            elif file_path.suffix.lower() in ['.doc', '.docx']:
                text = self._extract_text_from_docx(file_path)
                print(f"[RAG-DEBUG] DOCX'den metin çıkarıldı, uzunluk: {len(text)} karakter")
            else:
                print(f"[RAG-ERROR] Desteklenmeyen dosya tipi: {file_path.suffix}")
                return # Unsupported file type

            chunks = self._split_text(text)
            print(f"[RAG-DEBUG] Metin {len(chunks)} parçaya bölündü.")
            if not chunks:
                print(f"[RAG-ERROR] Metin parçalara ayrılamadı veya boş.")
                return

            embeddings = self.model.encode(chunks, convert_to_tensor=False)
            print(f"[RAG-DEBUG] Embeddingler oluşturuldu. Boyut: {len(embeddings)}")
            
            # Check if index exists and its dimension matches
            d = self.model.get_sentence_embedding_dimension()
            if self.index is None or self.index.d != d:
                self.index = faiss.IndexFlatL2(d)
                self.metadata = []


            self.index.add(np.array(embeddings).astype('float32'))
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'report_id': report_id,
                    'chunk_index': self.index.ntotal - len(chunks) + i,
                    'text': chunk
                }
                self.metadata.append(chunk_metadata)
            
            self._save()
            # Mark the report as processed in the database
            database.mark_report_as_processed(report_id)

        except Exception as e:
            print(f"Error processing document {file_path}: {e}")

    def search_in_documents(self, query: str, top_k=5):
        """
        Searches for a query across all processed documents.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    'text': meta['text'],
                    'report_id': meta['report_id'],
                    'score': distances[0][i]
                })
        return results

    def delete_document(self, report_id_to_delete):
        """
        Deletes all chunks associated with a report_id from the index and metadata.
        This is inefficient as it requires rebuilding parts of the index. A more robust
        solution would use an index that supports ID-based deletion like IndexIDMap.
        """
        if self.index is None or self.index.ntotal == 0:
            return

        # Find all vector indices that belong to the report to be deleted
        ids_to_remove = [
            i for i, meta in enumerate(self.metadata) 
            if meta.get('report_id') == report_id_to_delete
        ]

        if not ids_to_remove:
            return

        # Create a selector to remove the corresponding vectors
        # This is a bit complex for IndexFlatL2 which doesn't directly support removal.
        # A more direct approach is to rebuild the index without the deleted vectors.
        
        new_metadata = [
            meta for i, meta in enumerate(self.metadata) 
            if i not in ids_to_remove
        ]
        
        if not new_metadata: # If no vectors are left
            self.index.reset()
        else:
            # Reconstruct the index with the remaining vectors
            vectors_to_keep_indices = [
                i for i in range(self.index.ntotal) if i not in ids_to_remove
            ]
            
            # This operation can be slow with reconstruct_n on large indexes
            remaining_vectors = self.index.reconstruct_n(0, self.index.ntotal)[vectors_to_keep_indices]
            
            self.index.reset()
            self.index.add(remaining_vectors)

        self.metadata = new_metadata
        self._save()
        print(f"Removed report {report_id_to_delete} from FAISS index.")

# Singleton instance
processor = DocumentProcessor()
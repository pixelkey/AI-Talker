import os
import logging
from document_processing import load_documents_from_folder, load_single_document, create_document_entries
from faiss_utils import (
    add_vectors_to_faiss_index,
    save_faiss_index_metadata_and_docstore,
)
from config import CHUNK_SIZE_MAX, FAISS_INDEX_PATH, METADATA_PATH, DOCSTORE_PATH
from langchain.docstore.document import Document
import numpy as np

class VectorStoreClient:
    def __init__(self, vector_store, embeddings, normalize_text):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.normalize_text = normalize_text
        self._ensure_embedding_paths()
        
    def _ensure_embedding_paths(self):
        """Ensure all embedding-related directories exist"""
        for path in [FAISS_INDEX_PATH, METADATA_PATH, DOCSTORE_PATH]:
            abs_path = self._get_absolute_path(path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            logging.info(f"Ensured directory exists for: {abs_path}")
        
    def _get_absolute_path(self, path):
        """Convert relative path to absolute path"""
        if not os.path.isabs(path):
            # Get the project root directory (where scripts/ folder is)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # If path starts with ../, remove it since we're already at project root
            if path.startswith("../"):
                path = path[3:]
            return os.path.join(project_root, path)
        return path
        
    def update_embeddings_from_docstore(self, changed_chunks=None):
        """Update embeddings from the current docstore state."""
        try:
            docstore = self.vector_store.docstore
            index_to_docstore_id = self.vector_store.index_to_docstore_id
            
            # Convert docstore documents to chunks format
            chunks = []
            if changed_chunks:
                # Only process changed chunks
                for chunk_id in changed_chunks:
                    if chunk_id in docstore._dict:
                        doc = docstore._dict[chunk_id]
                        chunk = {
                            "id": chunk_id,
                            "content": doc.page_content,
                            "filepath": doc.metadata.get("filepath", ""),
                            "filename": doc.metadata.get("filename", ""),
                            "doc_id": doc.metadata.get("doc_id", "")
                        }
                        chunks.append(chunk)
            else:
                # Process all chunks (initial load)
                for doc_id, doc in docstore._dict.items():
                    chunk = {
                        "id": doc_id,
                        "content": doc.page_content,
                        "filepath": doc.metadata.get("filepath", ""),
                        "filename": doc.metadata.get("filename", ""),
                        "doc_id": doc.metadata.get("doc_id", "")
                    }
                    chunks.append(chunk)
            
            if not chunks:
                logging.info("No documents to update embeddings for")
                return
            
            if not changed_chunks:
                # Full reload - clear the index
                self.vector_store.index.reset()
                index_to_docstore_id.clear()
            else:
                # Remove old entries for changed chunks
                for chunk_id in changed_chunks:
                    if chunk_id in index_to_docstore_id:
                        idx = list(index_to_docstore_id.values()).index(chunk_id)
                        # Mark the vector as deleted in FAISS
                        self.vector_store.index.remove_ids(np.array([idx]))
                        del index_to_docstore_id[idx]
            
            # Add new embeddings
            add_vectors_to_faiss_index(
                chunks,
                self.vector_store,
                self.embeddings,
                self.normalize_text
            )
            logging.info(f"Updated embeddings for {len(chunks)} chunks")
            
            # Save updated index and metadata
            save_faiss_index_metadata_and_docstore(
                self.vector_store.index,
                self.vector_store.index_to_docstore_id,
                self.vector_store.docstore,
                self._get_absolute_path(FAISS_INDEX_PATH),
                self._get_absolute_path(METADATA_PATH),
                self._get_absolute_path(DOCSTORE_PATH)
            )
            
        except Exception as e:
            logging.error(f"Error updating embeddings from docstore: {str(e)}")
            raise

    def update_from_ingest_path(self, changed_files=None):
        """
        Update embeddings from files in the ingest path.
        
        Args:
            changed_files (list, optional): List of files that have changed.
                If None, updates all files (used for initial load).
        """
        try:
            # Get absolute ingest path relative to project root
            ingest_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "ingest"
            )
            
            if changed_files is None:
                # Initial load - process all files
                logging.info(f"Loading all documents from {ingest_path}")
                chunks = load_documents_from_folder(ingest_path, CHUNK_SIZE_MAX)
                # Get paths of all current chunks
                current_files = {
                    os.path.join(doc["filepath"], doc["filename"])
                    for doc in chunks
                }
            else:
                # Only process changed files
                logging.info(f"Processing changed files: {changed_files}")
                chunks = []
                # Get relative paths of changed files
                changed_relative_paths = set()
                for file_path in changed_files:
                    if not os.path.exists(file_path):
                        logging.warning(f"File no longer exists: {file_path}")
                        continue
                    # Get relative paths for the file
                    relative_path = os.path.relpath(os.path.dirname(file_path), ingest_path)
                    filename = os.path.basename(file_path)
                    relative_file_path = os.path.join(relative_path, filename)
                    changed_relative_paths.add(relative_file_path)
                    
                    # Load just this single file
                    file_chunks = load_single_document(file_path, CHUNK_SIZE_MAX)
                    if file_chunks:
                        new_chunks = create_document_entries(0, filename, relative_path, file_chunks)
                        chunks.extend(new_chunks)
                    else:
                        logging.warning(f"No chunks created for file: {file_path}")
            
            if not chunks:
                if changed_files:
                    logging.info("No valid chunks found in changed files")
                return
            
            # Track which chunks are being updated
            changed_chunk_ids = set()
            
            # Update docstore with new chunks
            docstore = self.vector_store.docstore
            for chunk in chunks:
                chunk_id = chunk["id"]
                changed_chunk_ids.add(chunk_id)
                doc = Document(
                    page_content=chunk["content"],
                    metadata=chunk
                )
                docstore._dict[chunk_id] = doc
            
            # Update embeddings only for changed chunks
            self.update_embeddings_from_docstore(changed_chunk_ids)
            
        except Exception as e:
            logging.error(f"Error updating embeddings: {str(e)}")
            raise

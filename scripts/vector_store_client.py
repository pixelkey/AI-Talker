import os
import logging
from document_processing import load_documents_from_folder, load_single_document, create_document_entries
from faiss_utils import (
    add_vectors_to_faiss_index,
    save_faiss_index_metadata_and_docstore,
)
from config import CHUNK_SIZE_MAX

class VectorStoreClient:
    def __init__(self, vector_store, embeddings, normalize_text):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.normalize_text = normalize_text
        self._ensure_embedding_paths()
        
    def _ensure_embedding_paths(self):
        """Ensure all embedding-related directories exist"""
        paths = [
            os.environ["FAISS_INDEX_PATH"],
            os.environ["METADATA_PATH"],
            os.environ["DOCSTORE_PATH"]
        ]
        
        for path in paths:
            if not os.path.isabs(path):
                # Convert relative path to absolute
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                abs_path = os.path.abspath(os.path.join(script_dir, path.lstrip("../")))
            else:
                abs_path = path
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(abs_path)
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    logging.info(f"Created directory: {directory}")
                except Exception as e:
                    logging.error(f"Error creating directory {directory}: {str(e)}")
                    raise
                
    def _get_absolute_path(self, path):
        """Convert relative path to absolute path"""
        if not os.path.isabs(path):
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.abspath(os.path.join(script_dir, path))
        return path
        
    def update_from_ingest_path(self, changed_files=None):
        """
        Update embeddings from files in the ingest path.
        
        Args:
            changed_files (list, optional): List of files that have changed.
                If None, updates all files (used for initial load).
        """
        try:
            # Get absolute ingest path
            ingest_path = self._get_absolute_path(os.environ["INGEST_PATH"])
            
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
                logging.info(f"Processing {len(changed_files)} changed files")
                chunks = []
                # Get relative paths of changed files
                changed_relative_paths = set()
                for file_path in changed_files:
                    if not os.path.exists(file_path):
                        continue
                    # Get relative paths for the file
                    relative_path = os.path.relpath(os.path.dirname(file_path), ingest_path)
                    filename = os.path.basename(file_path)
                    relative_file_path = os.path.join(relative_path, filename)
                    changed_relative_paths.add(relative_file_path)
                    
                    # Load just this single file
                    file_chunks = load_single_document(file_path, CHUNK_SIZE_MAX)
                    if file_chunks:
                        chunks.extend(create_document_entries(0, filename, relative_path, file_chunks))
                        logging.info(f"Loaded and chunked document {relative_file_path} into {len(file_chunks)} chunks")
            
            if not chunks:
                if changed_files:
                    logging.info("No valid chunks found in changed files")
                return
                
            # Remove embeddings for deleted/changed files
            docstore = self.vector_store.docstore
            index_to_docstore_id = self.vector_store.index_to_docstore_id
            
            # Find indices to remove
            indices_to_remove = []
            for idx, doc_id in index_to_docstore_id.items():
                doc = docstore._dict.get(doc_id)
                if doc:
                    filepath = os.path.join(
                        doc.metadata.get("filepath", ""),
                        doc.metadata.get("filename", "")
                    )
                    # Remove if file is deleted or changed, but never remove chat history
                    if doc.metadata.get("doc_id") != "chat_history" and (
                        (changed_files is None and filepath not in current_files) or
                        (changed_files is not None and filepath in changed_relative_paths)
                    ):
                        indices_to_remove.append(idx)
                        del docstore._dict[doc_id]
            
            # Remove vectors from FAISS index in reverse order
            if indices_to_remove:
                # Convert indices to numpy array for FAISS
                import numpy as np
                indices_array = np.array(sorted(indices_to_remove, reverse=True), dtype=np.int64)
                self.vector_store.index.remove_ids(indices_array)
                
                # Remove from docstore index
                for idx in indices_array:
                    del index_to_docstore_id[int(idx)]
                
                logging.info(f"Removed {len(indices_to_remove)} embeddings for deleted/changed files")
            
            # Add new embeddings
            add_vectors_to_faiss_index(
                chunks,
                self.vector_store,
                self.embeddings,
                self.normalize_text
            )
            logging.info(f"Added {len(chunks)} new chunks")
            
            # Save updated index and metadata
            save_faiss_index_metadata_and_docstore(
                self.vector_store.index,
                self.vector_store.index_to_docstore_id,
                self.vector_store.docstore,
                os.environ["FAISS_INDEX_PATH"],
                os.environ["METADATA_PATH"],
                os.environ["DOCSTORE_PATH"]
            )
            logging.info("Saved updated FAISS index and metadata")
            
        except Exception as e:
            logging.error(f"Error updating embeddings: {str(e)}")
            raise

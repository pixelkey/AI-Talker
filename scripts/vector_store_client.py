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
                path = os.path.join(script_dir, path.lstrip("../"))
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
                
    def _get_absolute_path(self, path):
        """Convert relative path to absolute path"""
        if not os.path.isabs(path):
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(script_dir, path.lstrip("../"))
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
            else:
                # Only process changed files
                logging.info(f"Processing {len(changed_files)} changed files")
                chunks = []
                for file_path in changed_files:
                    if not os.path.exists(file_path):
                        continue
                    # Get relative paths for the file
                    relative_path = os.path.relpath(os.path.dirname(file_path), ingest_path)
                    filename = os.path.basename(file_path)
                    
                    # Load just this single file
                    file_chunks = load_single_document(file_path, CHUNK_SIZE_MAX)
                    if file_chunks:
                        chunks.extend(create_document_entries(0, filename, relative_path, file_chunks))
                        logging.info(f"Loaded and chunked document {relative_path}/{filename} into {len(file_chunks)} chunks")
            
            if not chunks:
                if changed_files:
                    logging.info("No valid chunks found in changed files")
                return
                
            # Get paths of current chunks
            current_files = {
                os.path.join(doc["filepath"], doc["filename"])
                for doc in chunks
            }
            
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
                        (changed_files is not None and filepath in [os.path.relpath(f, ingest_path) for f in changed_files])
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

import os
import logging
from document_processing import load_documents_from_folder
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
        
    def update_from_ingest_path(self):
        """Update embeddings from all files in the ingest path"""
        try:
            # Get absolute ingest path
            ingest_path = self._get_absolute_path(os.environ["INGEST_PATH"])
            logging.info(f"Loading documents from {ingest_path}")
            
            # Load all current documents
            chunks = load_documents_from_folder(ingest_path, CHUNK_SIZE_MAX)
            
            # Get set of current file paths
            current_files = {
                os.path.join(doc["filepath"], doc["filename"])
                for doc in chunks
            }
            
            # Remove embeddings for files that no longer exist
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
                    if filepath not in current_files and doc.metadata.get("doc_id") != "chat_history":
                        indices_to_remove.append(idx)
                        del docstore._dict[doc_id]
            
            # Remove vectors from FAISS index in reverse order
            for idx in sorted(indices_to_remove, reverse=True):
                self.vector_store.index.remove_ids(idx)
                del index_to_docstore_id[idx]
            
            if indices_to_remove:
                logging.info(f"Removed {len(indices_to_remove)} embeddings for deleted files")
            
            # Add new embeddings
            if chunks:
                add_vectors_to_faiss_index(
                    chunks,
                    self.vector_store,
                    self.embeddings,
                    self.normalize_text
                )
                logging.info(f"Added {len(chunks)} new chunks to vector store")
            
            # Save updated index
            faiss_path = self._get_absolute_path(os.environ["FAISS_INDEX_PATH"])
            metadata_path = self._get_absolute_path(os.environ["METADATA_PATH"])
            docstore_path = self._get_absolute_path(os.environ["DOCSTORE_PATH"])
            
            save_faiss_index_metadata_and_docstore(
                self.vector_store.index,
                self.vector_store.index_to_docstore_id,
                self.vector_store.docstore,
                faiss_path,
                metadata_path,
                docstore_path
            )
            
            logging.info("Saved updated FAISS index and metadata")
            
        except Exception as e:
            logging.error(f"Error updating embeddings: {str(e)}")
            raise

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
        
    def update_from_ingest_path(self):
        """Update embeddings from all files in the ingest path"""
        try:
            # Load all documents from the ingest path
            ingest_path = os.environ["INGEST_PATH"]
            if not os.path.isabs(ingest_path):
                # Convert relative path to absolute
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ingest_path = os.path.join(script_dir, ingest_path.lstrip("../"))
                
            logging.info(f"Loading documents from {ingest_path}")
            chunks = load_documents_from_folder(ingest_path, CHUNK_SIZE_MAX)
            
            if not chunks:
                logging.info("No documents found to update")
                return
                
            # Add vectors to FAISS index
            add_vectors_to_faiss_index(
                chunks,
                self.vector_store,
                self.embeddings,
                self.normalize_text
            )
            
            # Get absolute paths for saving
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            faiss_path = os.path.join(script_dir, os.environ["FAISS_INDEX_PATH"].lstrip("../"))
            metadata_path = os.path.join(script_dir, os.environ["METADATA_PATH"].lstrip("../"))
            docstore_path = os.path.join(script_dir, os.environ["DOCSTORE_PATH"].lstrip("../"))
            
            # Save updated index
            save_faiss_index_metadata_and_docstore(
                self.vector_store.index,
                self.vector_store.index_to_docstore_id,
                self.vector_store.docstore,
                faiss_path,
                metadata_path,
                docstore_path
            )
            
            logging.info(f"Updated embeddings with {len(chunks)} chunks from {ingest_path}")
            
        except Exception as e:
            logging.error(f"Error updating embeddings: {str(e)}")
            raise

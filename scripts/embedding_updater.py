import os
import logging
import numpy as np
import faiss
from faiss_utils import save_faiss_index_metadata_and_docstore
from langchain.docstore.document import Document
import threading

logger = logging.getLogger(__name__)

class EmbeddingUpdater:
    def __init__(self, context):
        """
        Initialize the EmbeddingUpdater with necessary context.
        
        Args:
            context (dict): Contains loader, chunker, vector_store and other necessary components
        """
        self.context = context
        self.update_thread = None

    def update_embeddings(self, changed_files=None):
        """
        Update embeddings when files change.
        
        Args:
            changed_files (list, optional): List of file paths that have changed. 
                                          If None, processes all files in ingest directory.
        """
        try:
            # Get required components from context
            loader = self.context.get('loader')
            chunker = self.context.get('chunker')
            vector_store = self.context.get('vector_store')
            
            if not all([loader, chunker, vector_store]):
                logging.error("Missing required components for embedding update")
                return
            
            if changed_files:
                logging.info(f"Processing changed files: {changed_files}")
                for file_path in changed_files:
                    if os.path.exists(file_path):
                        # Load and chunk the changed file
                        documents = loader.load_documents([file_path])
                        if documents:
                            chunks = chunker.split_documents(documents)
                            # Update vector store with new chunks
                            vector_store.add_documents(chunks)
                            logging.info(f"Updated embeddings for {len(chunks)} chunks")
                            
                # Save updated index and metadata
                save_faiss_index_metadata_and_docstore(
                    vector_store.index,
                    vector_store.index_to_docstore_id,
                    vector_store.docstore,
                    os.environ["FAISS_INDEX_PATH"],
                    os.environ["METADATA_PATH"],
                    os.environ["DOCSTORE_PATH"]
                )
                logging.info(f"Updated embeddings for {len(changed_files)} files")
            else:
                # Load all documents from ingest directory
                logging.info("Loading all documents from /home/andrew/projects/app/python/talker/ingest")
                documents = loader.load()
                if documents:
                    chunks = chunker.split_documents(documents)
                    # Update vector store with all chunks
                    vector_store.add_documents(chunks)
                    logging.info(f"Updated embeddings for {len(chunks)} chunks")
                    # Save updated index and metadata
                    save_faiss_index_metadata_and_docstore(
                        vector_store.index,
                        vector_store.index_to_docstore_id,
                        vector_store.docstore,
                        os.environ["FAISS_INDEX_PATH"],
                        os.environ["METADATA_PATH"],
                        os.environ["DOCSTORE_PATH"]
                    )
                    logging.info("Updated embeddings from ingest directory")
            
        except Exception as e:
            logging.error(f"Error updating embeddings: {str(e)}")

    def update_chat_embeddings_async(self, history, state):
        """
        Update chat embeddings in a background thread.
        
        Args:
            history (list): Chat history to update embeddings with
            state (dict): State containing last_processed_index
        """
        def update_embeddings():
            try:
                logger.info("Starting chat history embedding update")
                # Removed the call to update_chat_embeddings as it was removed
                logger.info("Chat history embedding update complete")
            except Exception as e:
                logger.error(f"Error updating chat embeddings: {str(e)}", exc_info=True)

        self.update_thread = threading.Thread(target=update_embeddings)
        self.update_thread.daemon = True
        self.update_thread.start()

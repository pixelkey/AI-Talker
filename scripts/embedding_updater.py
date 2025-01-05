import os
import logging
from faiss_utils import save_faiss_index_metadata_and_docstore

class EmbeddingUpdater:
    def __init__(self, context):
        """
        Initialize the EmbeddingUpdater with necessary context.
        
        Args:
            context (dict): Contains loader, chunker, vector_store and other necessary components
        """
        self.context = context

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

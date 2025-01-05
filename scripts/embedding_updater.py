import os
import logging
import numpy as np
import faiss
from faiss_utils import save_faiss_index_metadata_and_docstore
from langchain.docstore.document import Document
import threading

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

    def update_chat_embeddings(self, history, state):
        """
        Update embeddings with new chat history.
        
        Args:
            history (list): Chat history to update embeddings with
            state (dict): State containing last_processed_index
        """
        try:
            print("\nUpdating embeddings...")
            chat_manager = self.context.get('chat_manager')
            vector_store = self.context.get('vector_store')
            embeddings = self.context.get('embeddings')
            
            if not all([chat_manager, vector_store, embeddings]):
                logging.error("Missing required components for chat embedding update")
                return
            
            # Get new messages from the current history
            new_messages = history[state["last_processed_index"]:]
            if new_messages:
                chat_text = chat_manager.format_for_embedding(new_messages)
                vectors = embeddings.embed_documents([chat_text])
                vectors = np.array(vectors, dtype="float32")
                faiss.normalize_L2(vectors)
                vector_store.index.add(vectors)
                
                chunk_id = str(len(vector_store.index_to_docstore_id))
                chunk = Document(
                    page_content=chat_text,
                    metadata={
                        "id": chunk_id,
                        "doc_id": "chat_history",
                        "filename": os.path.basename(chat_manager.current_file),
                        "filepath": "chat_history",
                        "chunk_size": len(chat_text),
                        "overlap_size": 0,
                    },
                )
                vector_store.docstore._dict[chunk_id] = chunk
                vector_store.index_to_docstore_id[len(vector_store.index_to_docstore_id)] = chunk_id
                state["last_processed_index"] = len(history)
                
                # Save the updated index
                save_faiss_index_metadata_and_docstore(
                    vector_store.index,
                    vector_store.index_to_docstore_id,
                    vector_store.docstore,
                    os.environ["FAISS_INDEX_PATH"],
                    os.environ["METADATA_PATH"],
                    os.environ["DOCSTORE_PATH"]
                )
                print("Embeddings update complete")
                
                # Process any pending file changes
                if hasattr(self.context['watcher'], 'pending_changes') and self.context['watcher'].pending_changes:
                    pending = self.context['watcher'].pending_changes.copy()
                    self.context['watcher'].pending_changes.clear()
                    self.update_embeddings(pending)
                    
        except Exception as e:
            print(f"Error updating embeddings: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_chat_embeddings_async(self, history, state):
        """
        Update chat embeddings in a background thread.
        
        Args:
            history (list): Chat history to update embeddings with
            state (dict): State containing last_processed_index
        """
        threading.Thread(
            target=self.update_chat_embeddings,
            args=(history, state),
            daemon=True
        ).start()

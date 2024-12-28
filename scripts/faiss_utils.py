# scripts/faiss_utils.py

import faiss
import pickle
import os
import logging
import numpy as np
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
import config


def save_faiss_index_metadata_and_docstore(
    faiss_index, metadata, docstore, faiss_index_path, metadata_path, docstore_path
):
    try:
        # Ensure parent directories exist
        for path in [faiss_index_path, metadata_path, docstore_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        # Save to temporary files first
        temp_faiss = faiss_index_path + ".tmp"
        temp_metadata = metadata_path + ".tmp"
        temp_docstore = docstore_path + ".tmp"
        
        logging.info(f"Saving to temporary files...")
        
        # Save FAISS index
        faiss.write_index(faiss_index, temp_faiss)
        logging.info(f"Saved FAISS index to temporary file: {os.path.abspath(temp_faiss)}")
        
        # Save metadata
        with open(temp_metadata, "wb") as f:
            pickle.dump(metadata, f)
        logging.info(f"Saved metadata to temporary file: {os.path.abspath(temp_metadata)}")
        
        # Save docstore
        with open(temp_docstore, "wb") as f:
            pickle.dump(docstore._dict, f)
        logging.info(f"Saved docstore to temporary file: {os.path.abspath(temp_docstore)}")
        
        logging.info(f"Moving temporary files to final locations...")
        
        # Atomically rename temporary files to final names
        os.rename(temp_faiss, faiss_index_path)
        logging.info(f"Moved {os.path.basename(temp_faiss)} to {os.path.basename(faiss_index_path)}")
        
        os.rename(temp_metadata, metadata_path)
        logging.info(f"Moved {os.path.basename(temp_metadata)} to {os.path.basename(metadata_path)}")
        
        os.rename(temp_docstore, docstore_path)
        logging.info(f"Moved {os.path.basename(temp_docstore)} to {os.path.basename(docstore_path)}")
        
        logging.info(f"Successfully saved all files atomically to {os.path.dirname(faiss_index_path)}")
    except Exception as e:
        # Clean up temporary files if they exist
        for temp_file in [temp_faiss, temp_metadata, temp_docstore]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logging.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up temporary file {temp_file}: {str(cleanup_error)}")
        logging.error(f"Error saving FAISS data: {str(e)}")
        raise


def load_faiss_index_metadata_and_docstore(
    faiss_index_path, metadata_path, docstore_path
):
    if (
        os.path.exists(faiss_index_path)
        and os.path.exists(metadata_path)
        and os.path.exists(docstore_path)
    ):
        faiss_index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        with open(docstore_path, "rb") as f:
            raw_docstore = pickle.load(f)
        
        # Convert raw dictionary data back into Document objects
        docstore = InMemoryDocstore({})
        for doc_id, doc_data in raw_docstore.items():
            if isinstance(doc_data, str):
                # If the stored data is just a string, create a Document object
                doc = Document(page_content=doc_data)
            else:
                # If it's already a Document object or has the expected structure
                doc = doc_data if isinstance(doc_data, Document) else Document(
                    page_content=doc_data.get('page_content', ''),
                    metadata=doc_data.get('metadata', {})
                )
            docstore._dict[doc_id] = doc

        logging.info(f"Loaded FAISS index, metadata, and docstore from disk: {os.path.abspath(faiss_index_path)}")
        return faiss_index, metadata, docstore
    return None, None, None


def train_faiss_index(vector_store, training_vectors, num_clusters):
    if not vector_store.index.is_trained:
        vector_store.index.train(training_vectors)
        logging.info(f"FAISS index trained with {num_clusters} clusters.")


def add_vectors_to_faiss_index(chunks, vector_store, embeddings, normalize_text):
    docstore = vector_store.docstore
    index_to_docstore_id = vector_store.index_to_docstore_id
    start_idx = max(map(int, index_to_docstore_id.keys())) + 1 if index_to_docstore_id else 0

    for idx, doc in enumerate(chunks):
        try:
            normalized_doc = normalize_text(doc["content"])
            # Use embed_documents for document embeddings
            vector = embeddings.embed_documents([normalized_doc])
            if not isinstance(vector, (list, np.ndarray)):
                raise ValueError(f"Expected list or ndarray from embed_documents, got {type(vector)}")
            
            # Ensure we have a proper numpy array
            if isinstance(vector, list):
                vector = vector[0] if isinstance(vector[0], (list, np.ndarray)) else vector
                vector = np.array(vector, dtype="float32")
            
            # Reshape to 2D if necessary
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            # Normalize the vector
            faiss.normalize_L2(vector)
            # Add the vector to the index
            vector_store.index.add(vector)

            # Use the document's ID as the docstore key
            chunk_id = doc["id"]  # Use the document's unique ID
            chunk = Document(
                page_content=normalized_doc,
                metadata=doc  # Include all original metadata
            )
            docstore._dict[chunk_id] = chunk
            index_to_docstore_id[start_idx + idx] = chunk_id
            logging.info(f"Added chunk {chunk_id} to vector store with normalized content: {normalized_doc[:100]}...")
        except Exception as e:
            logging.error(f"Error adding chunk {doc['id']} to vector store: {str(e)}")
            raise


def similarity_search_with_score(query, vector_store, embeddings, EMBEDDING_DIM, k=100):
    try:
        # Embed the query
        query_vector = embeddings.embed_query(query)
        if not isinstance(query_vector, (list, np.ndarray)):
            raise ValueError(f"Expected list or ndarray from embed_query, got {type(query_vector)}")
        
        # Convert to numpy array if needed
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype="float32")
        
        # Reshape to 2D if necessary
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        logging.info(f"Query embedding shape: {query_vector.shape}")
        logging.info(f"Query embedding first 5 values: {query_vector[0][:5]}")

        # Ensure query vector dimensionality matches the FAISS index dimensionality
        if query_vector.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[1]} does not match index dimension {EMBEDDING_DIM}"
            )

        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_vector)

        # Search the FAISS index
        D, I = vector_store.index.search(query_vector, k)

        results = []
        for i, score in zip(I[0], D[0]):
            if i != -1:  # Ensure valid index
                try:
                    chunk_id = vector_store.index_to_docstore_id.get(i, None)
                    if chunk_id is None:
                        raise KeyError(f"Chunk ID {i} not found in mapping.")
                    doc = vector_store.docstore.search(chunk_id)
                    if isinstance(doc, str):
                        # If doc is a string, create a Document object
                        doc = Document(page_content=doc)
                    results.append({
                        "id": chunk_id,
                        "content": doc.page_content,
                        "score": float(score),
                        "metadata": getattr(doc, 'metadata', {})
                    })
                    logging.info(f"Matched chunk {chunk_id} with score {score} and content: {doc.page_content[:200]}...")
                except KeyError as e:
                    logging.error(f"KeyError finding chunk id {i}: {e}")
                except Exception as e:
                    logging.error(f"Error processing search result {i}: {str(e)}")
        return results
    except Exception as e:
        logging.error(f"Error in similarity search: {str(e)}")
        return []

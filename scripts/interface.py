# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history, retrieve_and_format_references
from chat_history import ChatHistoryManager
from ingest_watcher import IngestWatcher
import speech_recognition as sr
import numpy as np
import io
import soundfile as sf
from gtts import gTTS
import tempfile
import os
import logging
import faiss
from langchain.docstore.document import Document
from vector_store_client import VectorStoreClient
from document_processing import normalize_text
import ollama

def setup_gradio_interface(context):
    """
    Sets up the Gradio interface.
    Args:
        context (dict): Context containing client, memory, and other settings.
    Returns:
        gr.Blocks: Gradio interface object.
    """
    # Initialize chat history manager and state
    chat_manager = ChatHistoryManager()
    state = {"last_processed_index": 0}

    # Initialize vector store client and LLM client
    vector_store_client = VectorStoreClient(
        context['vector_store'],
        context['embeddings'],
        normalize_text
    )
    context['vector_store_client'] = vector_store_client
    
    # Initialize Ollama client for local LLM
    if context.get('MODEL_SOURCE') == 'local':
        context['client'] = ollama

    # Initialize and start the ingest watcher
    def update_embeddings(changed_files=None):
        """Callback function to update embeddings when files change"""
        try:
            context['vector_store_client'].update_from_ingest_path(changed_files)
            if changed_files:
                logging.info(f"Updated embeddings for {len(changed_files)} files")
            else:
                logging.info("Updated embeddings from ingest directory")
            
        except Exception as e:
            logging.error(f"Error updating embeddings: {str(e)}")

    ingest_watcher = IngestWatcher(update_embeddings)
    ingest_watcher.start()

    with gr.Blocks(css=".separator { margin: 8px 0; border-bottom: 1px solid #ddd; }") as app:
        # Output fields
        chat_history = gr.Chatbot(label="Chat History", height=400)
        references = gr.Textbox(label="References", lines=2, interactive=False)
        audio_output = gr.Audio(
            label="Response Audio", 
            visible=True, 
            autoplay=True,
            show_download_button=False
        )

        # Input fields
        with gr.Row():
            input_text = gr.Textbox(label="Input Text", scale=4)
            audio_input = gr.Audio(
                label="Voice Input", 
                sources=["microphone"],
                type="filepath",
                streaming=False
            )

        with gr.Row():
            submit_button = gr.Button("Submit", visible=True)
            clear_button = gr.Button("Clear")

        # Initialize session state separately for each user
        session_state = gr.State(value=[])

        def text_to_speech(text):
            """Convert text to speech using gTTS"""
            if not text:
                return None
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                    temp_path = fp.name
                
                # Generate speech
                tts = gTTS(text=text, lang='en')
                tts.save(temp_path)
                
                return temp_path
            except Exception as e:
                print(f"Error in text-to-speech: {str(e)}")
                return None

        def transcribe_audio(audio_path):
            """Transcribe audio file to text"""
            if audio_path is None:
                return "", "", False, "No audio received"
            
            try:
                # Initialize recognizer
                recognizer = sr.Recognizer()
                
                # Use the audio file directly
                with sr.AudioFile(audio_path) as source:
                    audio = recognizer.record(source)
                
                # Perform the recognition
                text = recognizer.recognize_google(audio)
                # Return text and trigger submit
                return text, text, True, f"Transcribed: {text}"
            except Exception as e:
                return "", "", False, f"Error transcribing audio: {str(e)}"

        # Define a function to handle both reference retrieval and LLM response generation
        def handle_user_input(input_text, history):
            if not input_text.strip():
                return history, "", "", history, None, ""

            refs, filtered_docs, context_documents = retrieve_and_format_references(input_text, context)
            
            # Initialize history if needed
            if history is None:
                history = []
            
            # Add user message to history
            history.append((input_text, None))
            # Save history to file
            chat_manager.save_history(history)
            yield history, refs, "", history, None, ""

            # Generate the LLM response if references were found
            if filtered_docs:
                _, response, _ = chatbot_response(input_text, context_documents, context, history)
                # Add assistant response to history
                history[-1] = (input_text, response)
                # Save updated history
                chat_manager.save_history(history)
                
                # Generate speech for the response first
                audio_path = text_to_speech(response)
                yield history, refs, "", history, audio_path, "Response complete, starting recording..."
                
                # Update embeddings after response and audio are complete
                new_messages, new_last_index = chat_manager.get_new_messages(state["last_processed_index"])
                if new_messages:
                    chat_text = chat_manager.format_for_embedding(new_messages)
                    # Add chat history to vector store directly
                    vector_store = context['vector_store']
                    vectors = context['embeddings'].embed_documents([chat_text])
                    vectors = np.array(vectors, dtype="float32")
                    faiss.normalize_L2(vectors)
                    vector_store.index.add(vectors)
                    
                    # Add to docstore
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
                    
                    state["last_processed_index"] = new_last_index
                    
                    # Save the updated index
                    save_faiss_index_metadata_and_docstore(
                        vector_store.index,
                        vector_store.index_to_docstore_id,
                        vector_store.docstore,
                        os.environ["FAISS_INDEX_PATH"],
                        os.environ["METADATA_PATH"],
                        os.environ["DOCSTORE_PATH"]
                    )
                    logging.info("Updated embeddings with new chat history")
            else:
                response = "I don't have any relevant information to help with that query."
                history[-1] = (input_text, response)
                chat_manager.save_history(history)
                audio_path = text_to_speech(response)
                yield history, refs, "", history, audio_path, "Response complete, starting recording..."

        def clear_interface(history):
            cleared_history, cleared_refs, cleared_input = clear_history(context, history)
            return [], cleared_refs, cleared_input, [], None, ""

        # Setup event handlers with explicit state management
        submit_button.click(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output],
        ).success(
            lambda: None,
            None,
            audio_input
        )
        
        # Add audio completion handler to start recording
        audio_output.stop(
            lambda: None,
            None,
            audio_input
        )

        # Add audio input handler with auto-submit
        audio_input.change(
            transcribe_audio,
            inputs=[audio_input],
            outputs=[
                input_text,  # Update the input text
                gr.Textbox(visible=False),  # Temporary storage
                gr.Checkbox(visible=False),  # Trigger for submit
            ],
        ).then(
            handle_user_input,  # Chain to handle_user_input
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output],
        )

        # Add text input submission via Enter key
        input_text.submit(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output],
        )

    return app

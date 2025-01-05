# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history, retrieve_and_format_references
from chat_history import ChatHistoryManager
from ingest_watcher import IngestWatcher
from faiss_utils import save_faiss_index_metadata_and_docstore
import os
import logging
from vector_store_client import VectorStoreClient
from document_processing import normalize_text
import ollama
import time
from tts_utils import TTSManager
from embedding_updater import EmbeddingUpdater
from speech_recognition_utils import SpeechRecognizer

def setup_gradio_interface(context):
    """
    Sets up the Gradio interface.
    Args:
        context (dict): Context containing client, memory, and other settings.
    Returns:
        gr.Blocks: Gradio interface object.
    """
    print("\n=== Setting up Gradio interface ===")
    
    # Initialize TTS at startup
    tts_manager = TTSManager(context)
    if not context.get('tts'):
        print("Failed to initialize TTS")
        return None
        
    # Initialize speech recognizer
    speech_recognizer = SpeechRecognizer()
    
    # Initialize state
    state = {"last_processed_index": 0}

    # Initialize chat manager once at startup
    chat_manager = ChatHistoryManager()
    context['chat_manager'] = chat_manager

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

    # Initialize embedding updater
    embedding_updater = EmbeddingUpdater(context)
    
    # Create the watcher but don't start it yet - we'll manually trigger updates
    watcher = IngestWatcher(embedding_updater.update_embeddings)
    context['watcher'] = watcher

    def handle_user_input(input_text, history):
        """Handle user input and generate response with proper state management."""
        try:
            # Validate input
            if not input_text or not input_text.strip():
                return history, "", "", history, None, ""

            # Get chat manager from context
            chat_manager = context['chat_manager']

            # Get references and generate response
            refs, filtered_docs, context_documents = retrieve_and_format_references(input_text, context)
            
            # Generate the LLM response
            _, response, _ = chatbot_response(input_text, context_documents, context, history)
            
            # Update history with the new user and bot messages
            new_history = history + [(f"User: {input_text}", f"Bot: {response}")]
            chat_manager.save_history(new_history)
            
            # Generate speech for the response
            print("\nGenerating TTS response...")
            audio_path = tts_manager.text_to_speech(response)
            print("TTS generation complete")
            
            # Update embeddings in background
            embedding_updater.update_chat_embeddings_async(history, state)
            
            return history + [(f"User: {input_text}", f"Bot: {response}")], refs, "", history + [(f"User: {input_text}", f"Bot: {response}")], audio_path, "Processing complete"
            
        except Exception as e:
            print(f"Error in handle_user_input: {str(e)}")
            import traceback
            traceback.print_exc()
            return history, "", "", history, None, f"Error: {str(e)}"

    def clear_interface(history):
        cleared_history, cleared_refs, cleared_input = clear_history(context, history)
        return [], cleared_refs, cleared_input, [], None, ""

    # Setup event handlers with explicit state management
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

        submit_button.click(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[
                chat_history,     # updated chat history 
                references,       # references
                input_text,       # clear or reset user input
                session_state,    # updated session state
                audio_output,     # audio file path
                gr.Textbox(visible=False)  # status or debug
            ]
        ).success(
            lambda: None,
            None,
            audio_input
        )
        
        # Audio input with auto-submit
        audio_input.change(
            speech_recognizer.transcribe_audio,
            inputs=[audio_input],
            outputs=[
                input_text,           # recognized text
                gr.Textbox(visible=False),  # for debug message
                gr.Textbox(visible=False),  # status message
            ],
            queue=False
        ).success(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[
                chat_history,     # updated chat history 
                references,       # references
                input_text,       # clear or reset user input
                session_state,    # updated session state
                audio_output,     # audio file path
                gr.Textbox(visible=False)  # status or debug
            ]
        )
        
        # Add text input submission via Enter key with the same outputs as submit button
        input_text.submit(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[
                chat_history,     # updated chat history 
                references,       # references
                input_text,       # clear or reset user input
                session_state,    # updated session state
                audio_output,     # audio file path
                gr.Textbox(visible=False)  # status or debug
            ]
        )
        
        # Clear button
        clear_button.click(
            clear_interface,
            inputs=[session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, gr.Textbox(visible=False)]
        )

    return app

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
from self_reflection import SelfReflection

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
    
    # Initialize self reflection
    self_reflection = SelfReflection(context)
    
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
            # Notify self-reflection about user input
            self_reflection.notify_user_input()
            
            # Validate input
            if not input_text or not input_text.strip():
                return history, "", "", history, None, "", gr.update(value="")

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
            
            # Start self-reflection after response
            print("\nStarting self-reflection...")
            def update_ui_with_reflection(reflection_text):
                print(f"Updating reflection box with: {reflection_text[:100]}...")
                return gr.update(value=reflection_text)
            
            self_reflection.start_reflection(new_history, update_ui_with_reflection)
            
            return (
                new_history,  # chatbot
                refs,         # references
                "",          # input_text
                new_history, # session_state
                audio_path,  # audio_output
                "",         # status
                gr.update(value="Starting reflection...")  # reflection_box
            )
            
        except Exception as e:
            print(f"Error in handle_user_input: {str(e)}")
            import traceback
            traceback.print_exc()
            return history, "", "", history, None, f"Error: {str(e)}", gr.update(value="Error during reflection")

    def clear_interface(history):
        # Stop any ongoing reflection when clearing
        self_reflection.stop_reflection_loop()
        cleared_history, cleared_refs, cleared_input = clear_history(context, history)
        return [], cleared_refs, cleared_input, [], None, "", gr.update(value="")

    # Create Gradio interface
    with gr.Blocks() as interface:
        # Create interface components
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=400,
                    show_label=False,
                    layout="bubble"
                )
                references = gr.Textbox(
                    show_label=False,
                    placeholder="References will appear here...",
                    lines=4
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        input_text = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter, or use the microphone...",
                            lines=1
                        )
                    with gr.Column(scale=2, min_width=0):
                        submit_text = gr.Button("Submit")
                status = gr.Textbox(
                    show_label=False,
                    placeholder="Status will appear here...",
                    lines=1
                )
            with gr.Column(scale=2):
                reflection_box = gr.Textbox(
                    label="Self-Reflection",
                    placeholder="Agent's self-reflections will appear here...",
                    lines=10,
                    max_lines=20
                )
                audio_status = gr.Textbox(
                    show_label=False,
                    placeholder="Audio status will appear here...",
                    lines=1
                )
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speak",
                    show_label=True
                )
                audio_output = gr.Audio(
                    label="Response",
                    show_label=True
                )
                
        # Set up event handlers
        submit_text.click(
            handle_user_input,
            inputs=[input_text, chatbot],
            outputs=[
                chatbot,
                references,
                input_text,
                gr.State(value=[]),
                audio_output,
                status,
                reflection_box
            ],
            queue=False
        ).success(
            lambda: gr.update(interactive=True),
            None,
            [submit_text],
            queue=False
        )
        
        input_text.submit(
            handle_user_input,
            inputs=[input_text, chatbot],
            outputs=[
                chatbot,
                references,
                input_text,
                gr.State(value=[]),
                audio_output,
                status,
                reflection_box
            ],
            queue=False
        )
        
        # Audio input with auto-submit
        audio_input.change(
            speech_recognizer.transcribe_audio,
            inputs=[audio_input],
            outputs=[
                input_text,
                submit_text,
                audio_status
            ],
            queue=False
        ).success(
            handle_user_input,
            inputs=[input_text, chatbot],
            outputs=[
                chatbot,
                references,
                input_text,
                gr.State(value=[]),
                audio_output,
                status,
                reflection_box
            ],
            queue=False
        )
        
        clear_btn = gr.Button("Clear")
        clear_btn.click(
            clear_interface,
            inputs=[chatbot],
            outputs=[
                chatbot,
                references,
                input_text,
                gr.State(value=[]),
                audio_output,
                status,
                reflection_box
            ],
            queue=False
        )

    return interface

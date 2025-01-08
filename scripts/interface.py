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
from queue import Queue
from datetime import datetime

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
    context['embedding_updater'] = embedding_updater
    
    # Initialize self reflection
    self_reflection = SelfReflection(context)
    
    # Create the watcher but don't start it yet - we'll manually trigger updates
    watcher = IngestWatcher(embedding_updater.update_embeddings)
    context['watcher'] = watcher

    def parse_timestamp(timestamp):
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")

    def handle_user_input(input_text, history):
        """Handle user input and generate response with proper state management."""
        try:
            # Notify self-reflection about user input
            self_reflection.notify_user_input()
            
            # Validate input
            if not input_text or not input_text.strip():
                return history, "", "", history, None
                
            # Get chat manager from context
            chat_manager = context['chat_manager']

            # Set current time in context
            context['current_time'] = '2025-01-08T15:10:11+10:30'  # Using the provided time
            
            # Get references and generate response
            refs, filtered_docs, context_documents = retrieve_and_format_references(input_text, context)
            
            # Generate the LLM response and get updated references
            _, response, refs, _ = chatbot_response(input_text, context_documents, context, history)
            
            # Format messages with timestamp
            dt = parse_timestamp(context['current_time'])
            formatted_time = dt.strftime("%A, %Y-%m-%d %H:%M:%S %z")
            user_msg = f"[{formatted_time}]\nUser: {input_text}"
            bot_msg = f"[{formatted_time}]\nBot: {response}"
            
            # Update history with the new user and bot messages
            new_history = history + [(user_msg, bot_msg)]
            chat_manager.save_history(new_history)
            
            # Generate speech for the response
            print("\nGenerating TTS response...")
            audio_path = tts_manager.text_to_speech(response)
            print("TTS generation complete")
            
            # Update embeddings in background
            context['embedding_updater'].update_chat_embeddings_async(history, state)
            
            return (
                new_history,  # chatbot
                refs,        # references
                "",         # input_text (clear it)
                new_history, # gr.State
                audio_path   # audio_output
            )
            
        except Exception as e:
            print(f"Error in handle_user_input: {str(e)}")
            import traceback
            traceback.print_exc()
            return history, "", "", history, None

    def clear_interface(history):
        # Stop any ongoing reflection by setting the event
        if self_reflection and hasattr(self_reflection, 'stop_reflection'):
            self_reflection.stop_reflection.set()
            
        cleared_history, cleared_refs, cleared_input = clear_history(context, history)
        return [], cleared_refs, cleared_input, [], None

    # Create Gradio interface
    with gr.Blocks() as interface:
        # Create interface components
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    show_label=False,
                    layout="bubble"
                )
                audio_output = gr.Audio(
                    label="Response",
                    show_label=True,
                    autoplay=True
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=6):
                        input_text = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter, or use the microphone...",
                            lines=1,
                            max_lines=1,
                            container=False
                        )
                    with gr.Column(scale=4, min_width=0):
                        with gr.Group():
                            audio_input = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                label="Speak",
                                show_label=True,
                                scale=1
                            )
                with gr.Row():
                    # Create a container div for consistent button heights
                    gr.HTML("""
                        <style>
                        #submit-btn, #clear-btn {
                            height: 40px !important;
                            min-height: 40px !important;
                            max-height: 40px !important;
                            margin: 0 !important;
                            padding: 0 16px !important;
                            display: inline-flex !important;
                            align-items: center !important;
                            justify-content: center !important;
                            text-align: center !important;
                            line-height: 1 !important;
                        }
                        #submit-btn span, #clear-btn span {
                            display: inline-block !important;
                            visibility: visible !important;
                            opacity: 1 !important;
                        }
                        </style>
                    """)
                    with gr.Column(scale=3):
                        submit_text = gr.Button(
                            "Submit",
                            variant="primary",
                            elem_id="submit-btn"
                        )
                    with gr.Column(scale=1):
                        clear_btn = gr.Button(
                            "Clear",
                            variant="secondary",
                            elem_id="clear-btn"
                        )
            with gr.Column(scale=2):
                references = gr.Textbox(
                    show_label=False,
                    placeholder="References will appear here...",
                    lines=25
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
                audio_output
            ],
            queue=True
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
                audio_output
            ],
            queue=True
        )
        
        # Audio input with auto-submit
        audio_input.change(
            speech_recognizer.transcribe_audio,
            inputs=[audio_input],
            outputs=[
                input_text,
                submit_text
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
                audio_output
            ],
            queue=True
        )
        
        clear_btn.click(
            clear_interface,
            inputs=[chatbot],
            outputs=[
                chatbot,
                references,
                input_text,
                gr.State(value=[]),
                audio_output
            ],
            queue=False
        )

    return interface

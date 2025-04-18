# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history, retrieve_and_format_references
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
from continuous_listener import ContinuousListener
from self_reflection import SelfReflection
from queue import Queue
from datetime import datetime
from memory_cleanup import MemoryCleanupManager

# Configure logging
logger = logging.getLogger(__name__)

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
    context['tts_manager'] = tts_manager  # Add TTS manager to context
    if not context.get('tts'):
        print("Failed to initialize TTS")
        return None
    
    # Initialize speech recognizer
    speech_recognizer = SpeechRecognizer()
    
    # Initialize continuous listener if enabled
    continuous_listener = None
    if context.get('CONTINUOUS_LISTENING', True):
        activation_word = context.get('ACTIVATION_WORD', 'activate')
        deactivation_word = context.get('DEACTIVATION_WORD', 'stop')
        logger.info(f"Initializing continuous listener with activation word: '{activation_word}' and deactivation word: '{deactivation_word}'")
        continuous_listener = ContinuousListener(
            activation_word=activation_word,
            deactivation_word=deactivation_word
        )
        context['continuous_listener'] = continuous_listener
    
    # Initialize state
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

    # Initialize embedding updater
    embedding_updater = EmbeddingUpdater(context)
    context['embedding_updater'] = embedding_updater
    
    # Initialize self reflection
    self_reflection = SelfReflection(context)
    context['self_reflection'] = self_reflection
    self_reflection.start_reflection_thread()
    
    # Initialize memory cleanup manager
    memory_cleanup = MemoryCleanupManager(context)
    context['memory_cleanup'] = memory_cleanup
    memory_cleanup.start_cleanup_thread()
    if memory_cleanup.should_run_cleanup():
        logger.info("Memory cleanup needed - will run shortly")
    else:
        last_cleanup = memory_cleanup._get_last_cleanup_from_logs()
        if last_cleanup:
            logger.info(f"Last memory cleanup was at {last_cleanup}")
    
    # Create the watcher but don't start it yet - we'll manually trigger updates
    watcher = IngestWatcher(embedding_updater.update_embeddings)
    context['watcher'] = watcher

    def parse_timestamp(timestamp):
        if isinstance(timestamp, datetime):
            return timestamp
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")

    def handle_user_input(input_text, history):
        """Handle user input and generate response with proper state management."""
        try:
            # Notify self-reflection about user input
            self_reflection.notify_user_input()
            
            # Validate input
            if not input_text or not input_text.strip():
                return history, "", "", history, None
                
            # Set current time in context
            context['current_time'] = '2025-01-20T12:46:59+10:30'  # Using the actual current time with correct timezone
            
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
            
            # Generate speech for the response
            print("\nGenerating TTS response...")
            # Use the emotion-enhanced response if available
            tts_text = context.get('tts_response', response)
            print(f"\nDEBUG - Original response: {response[:100]}...")
            print(f"DEBUG - TTS text with emotion: {tts_text[:100]}...")
            audio_path = tts_manager.text_to_speech(tts_text)
            print("TTS generation complete")
            
            # Wait for TTS to fully complete before starting reflection
            while tts_manager.is_processing:
                time.sleep(0.1)  # Short sleep to prevent busy waiting
            
            # Start reflection after TTS is done
            self_reflection.start_reflection(new_history, lambda x: None)
            
            # Signal the continuous listener that response is complete
            if continuous_listener:
                continuous_listener.signal_response_complete()
                
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

    def handle_continuous_speech(text, history=None):
        """Handle speech from continuous listener and update UI elements"""
        if history is None:
            history = chat_history.value
            
        # Update the input text box with the recognized speech
        # This will be visible in the UI to show what was recognized
        return text, f"Continuous listener recognized: {text}"

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
                    autoplay=False  # Disabled auto-play since we're playing locally
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
                    speech_recognition_text = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        container=True,
                        scale=1,
                        elem_classes="full-width-textbox"
                    )
                    
                    gr.HTML("""
                        <style>
                        .gradio-container .full-width-textbox {
                            width: 100% !important;
                            max-width: 100% !important;
                        }
                        .gradio-container .gr-text-input {
                            border-radius: 8px !important;
                            padding: 8px 16px !important;
                            min-height: 40px !important;
                            background-color: var(--neutral-100);
                        }
                        </style>
                    """)
            with gr.Column(scale=2):
                references = gr.Textbox(
                    show_label=False,
                    placeholder="References will appear here...",
                    lines=25
                )
                
        # Set up event handlers
        chat_history = gr.State(value=[])
        
        input_text.submit(
            handle_user_input,
            inputs=[input_text, chat_history],
            outputs=[
                chatbot,
                references,
                input_text,
                chat_history,
                audio_output
            ],
            queue=True
        )
        
        # Audio input with auto-submit
        audio_input.stop_recording(
            speech_recognizer.transcribe_audio,
            inputs=[audio_input],
            outputs=[input_text, speech_recognition_text],
            queue=False
        ).success(
            handle_user_input,
            inputs=[input_text, chat_history],
            outputs=[
                chatbot,
                references,
                input_text,
                chat_history,
                audio_output
            ],
            queue=True
        )
        
        # Set up continuous listener if enabled
        if continuous_listener:
            # Set up callback for recognized speech
            def on_speech_recognized(text):
                # Update input box (just for visual feedback of what was recognized)
                input_text.update(text)
                speech_recognition_text.update(f"Continuous listener recognized: {text}")
                
                # Process the input just like a submitted text
                handle_user_input(text, chat_history.value)
                
            # Set the callback
            continuous_listener.callback = on_speech_recognized
            
            # Start the continuous listener
            continuous_listener.start()
            logger.info("Continuous listener started and ready")

    return interface

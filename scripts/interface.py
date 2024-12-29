# scripts/interface.py

import gradio as gr
from chatbot_functions import chatbot_response, clear_history, retrieve_and_format_references
from chat_history import ChatHistoryManager
from ingest_watcher import IngestWatcher
from faiss_utils import save_faiss_index_metadata_and_docstore
import speech_recognition as sr
import numpy as np
import io
import soundfile as sf
import torchaudio
import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
import tempfile
import os
import logging
import faiss
from langchain.docstore.document import Document
from vector_store_client import VectorStoreClient
from document_processing import normalize_text
import ollama

# Set CUDA_HOME if not set
if not os.environ.get('CUDA_HOME'):
    cuda_paths = [
        '/usr/lib/nvidia-cuda-toolkit',  # Debian/Ubuntu CUDA toolkit location
        '/usr/local/cuda',
        '/usr/cuda',
        '/opt/cuda'
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            os.environ['CUDA_HOME'] = path
            print(f"\nSet CUDA_HOME to {path}")
            break

# Test DeepSpeed availability
try:
    import deepspeed
    print(f"\nDeepSpeed version: {deepspeed.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA arch list: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'Not set')}")
except ImportError as e:
    print(f"\nDeepSpeed not available: {str(e)}")
except Exception as e:
    print(f"\nError testing DeepSpeed: {str(e)}")

def setup_gradio_interface(context):
    """
    Sets up the Gradio interface.
    Args:
        context (dict): Context containing client, memory, and other settings.
    Returns:
        gr.Blocks: Gradio interface object.
    """
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
            
    # Create and start the watcher
    watcher = IngestWatcher(update_embeddings)
    watcher.start()
    context['watcher'] = watcher  # Store in context to stop later if needed

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
            """Convert text to speech using Tortoise TTS with DeepSpeed optimization if available"""
            print("\n=== Starting text_to_speech ===")
            if not text:
                print("No text provided, returning None")
                return None
            try:
                print("Checking DeepSpeed availability...")
                # Check for CUDA and DeepSpeed availability
                use_deepspeed = False
                try:
                    import deepspeed
                    import os
                    print(f"CUDA available: {torch.cuda.is_available()}")
                    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
                    if torch.cuda.is_available():
                        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                    
                    if torch.cuda.is_available() and os.environ.get('CUDA_HOME'):
                        use_deepspeed = True
                        print("DeepSpeed optimization enabled")
                        print(f"DeepSpeed version: {deepspeed.__version__}")
                    else:
                        print("CUDA or CUDA_HOME not available, falling back to standard mode")
                except ImportError as e:
                    print(f"DeepSpeed import error: {str(e)}")
                    print("DeepSpeed not available, falling back to standard mode")
                except Exception as e:
                    print(f"Error checking DeepSpeed availability: {str(e)}")
                    print("Falling back to standard mode")

                print("Initializing TTS configuration...")
                # Initialize Tortoise TTS with basic configuration
                tts_config = {
                    "kv_cache": True,
                    "half": True,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "autoregressive_batch_size": 1,
                    "use_deepspeed": use_deepspeed  # Only pass the flag, not the full config
                }

                print(f"Initializing TTS with config: {tts_config}")
                tts = TextToSpeech(**tts_config)
                print("TTS initialized successfully")

                # Set a fixed random seed for consistent voice
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)
                
                print("Loading voice samples...")
                # Use a specific voice preset for consistency
                voice_samples, conditioning_latents = load_voice('train_dotrice', extra_voice_dirs=[])
                print("Voice samples loaded")
                
                # Split long text into smaller chunks at sentence boundaries
                print("Processing text chunks...")
                sentences = text.split('.')
                max_chunk_length = 100  # Maximum characters per chunk
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_length:
                        current_chunk += sentence + "."
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + "."
                if current_chunk:
                    chunks.append(current_chunk)

                print(f"Created {len(chunks)} chunks")

                # Process each chunk with minimal settings and DeepSpeed optimization if enabled
                all_audio = []
                for i, chunk in enumerate(chunks, 1):
                    print(f"Processing chunk {i}/{len(chunks)}")
                    if not chunk.strip():
                        print(f"Skipping empty chunk {i}")
                        continue
                        
                    gen = tts.tts_with_preset(
                        chunk,
                        voice_samples=voice_samples,
                        conditioning_latents=conditioning_latents,
                        preset='ultra_fast',
                        use_deterministic_seed=True,
                        num_autoregressive_samples=1,
                        diffusion_iterations=10,
                        cond_free=False,
                        temperature=0.8
                    )
                    print(f"Generated audio for chunk {i}")
                    
                    if isinstance(gen, tuple):
                        gen = gen[0]
                    if len(gen.shape) == 3:
                        gen = gen.squeeze(0)
                    
                    all_audio.append(gen)

                # Combine all audio chunks
                print("Combining audio chunks...")
                if all_audio:
                    combined_audio = torch.cat(all_audio, dim=1)
                    print("Audio chunks combined successfully")
                else:
                    print("No audio generated")
                    return None

                # Create a temporary file
                print("Saving audio to file...")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                    temp_path = fp.name
                    torchaudio.save(temp_path, combined_audio.cpu(), 24000)
                    print(f"Audio saved to {temp_path}")
                    
                return temp_path
            except Exception as e:
                print(f"Error in text-to-speech: {str(e)}")
                import traceback
                traceback.print_exc()
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
            # Create a new chat history manager for each session if history is empty
            if not history:
                chat_manager = ChatHistoryManager()
                context['chat_manager'] = chat_manager
            else:
                chat_manager = context.get('chat_manager')
                
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

            # Generate the LLM response
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


        def clear_interface(history):
            cleared_history, cleared_refs, cleared_input = clear_history(context, history)
            return [], cleared_refs, cleared_input, [], None, ""

        # Setup event handlers with explicit state management
        submit_button.click(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, gr.Textbox(visible=False)],  # Add status output
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
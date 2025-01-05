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
import time

def initialize_tts():
    """Initialize TTS and return the initialized objects"""
    print("\n=== Initializing TTS at startup ===")
    try:
        # Initialize TTS with optimal configuration
        tts_config = {
            "kv_cache": True,
            "half": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "autoregressive_batch_size": 1,  # larger GPU memory usage if set more than 1
            "use_deepspeed": True
        }
        print(f"Initializing TTS with config: {tts_config}")
        tts = TextToSpeech(**tts_config)
        print("TTS object created successfully")

        # Set fixed seed for consistent voice
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Get voice from config
        voice_name = os.getenv('TTS_VOICE', 'emma')
        print(f"Loading voice samples for {voice_name}...")
        voice_samples = load_voice(voice_name, extra_voice_dirs=[])[0]
        print(f"Voice samples loaded: {len(voice_samples)} samples")

        print("Computing conditioning latents...")
        gen_conditioning_latents = tts.get_conditioning_latents(voice_samples)
        print("Conditioning latents generated")

        return {
            'tts': tts,
            'voice_samples': voice_samples,
            'conditioning_latents': gen_conditioning_latents
        }
    except Exception as e:
        print(f"Error initializing TTS: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    print("\n=== Setting up Gradio interface ===")
    
    # Initialize TTS at startup
    tts_context = initialize_tts()
    if not tts_context:
        print("Failed to initialize TTS")
        return None
        
    # Store TTS objects in context
    context.update(tts_context)
    print("TTS objects stored in context")
    
    # Verify context contents
    print(f"Context contents after initialization:")
    print(f"- TTS object: {type(context.get('tts'))}")
    print(f"- Voice samples: {type(context.get('voice_samples'))}, length: {len(context.get('voice_samples'))}")
    print(f"- Conditioning latents: {type(context.get('conditioning_latents'))}")

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

    def update_embeddings(changed_files=None):
        """Callback function to update embeddings when files change"""
        try:
            if changed_files:
                logging.info(f"Processing changed files: {changed_files}")
                for file_path in changed_files:
                    if os.path.exists(file_path):
                        # Load and chunk the changed file
                        documents = context['loader'].load_documents([file_path])
                        if documents:
                            chunks = context['chunker'].split_documents(documents)
                            # Update vector store with new chunks
                            context['vector_store'].add_documents(chunks)
                            logging.info(f"Updated embeddings for {len(chunks)} chunks")
                            
                # Save updated index and metadata
                save_faiss_index_metadata_and_docstore(
                    context['vector_store'].index,
                    context['vector_store'].index_to_docstore_id,
                    context['vector_store'].docstore,
                    os.environ["FAISS_INDEX_PATH"],
                    os.environ["METADATA_PATH"],
                    os.environ["DOCSTORE_PATH"]
                )
                logging.info(f"Updated embeddings for {len(changed_files)} files")
            else:
                # Load all documents from ingest directory
                logging.info("Loading all documents from /home/andrew/projects/app/python/talker/ingest")
                documents = context['loader'].load()
                if documents:
                    chunks = context['chunker'].split_documents(documents)
                    # Update vector store with all chunks
                    context['vector_store'].add_documents(chunks)
                    logging.info(f"Updated embeddings for {len(chunks)} chunks")
                    # Save updated index and metadata
                    save_faiss_index_metadata_and_docstore(
                        context['vector_store'].index,
                        context['vector_store'].index_to_docstore_id,
                        context['vector_store'].docstore,
                        os.environ["FAISS_INDEX_PATH"],
                        os.environ["METADATA_PATH"],
                        os.environ["DOCSTORE_PATH"]
                    )
                    logging.info("Updated embeddings from ingest directory")
            
        except Exception as e:
            logging.error(f"Error updating embeddings: {str(e)}")

    # Create the watcher but don't start it yet - we'll manually trigger updates
    watcher = IngestWatcher(update_embeddings)
    context['watcher'] = watcher

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
            """Convert text to speech using Tortoise TTS"""
            print("\n=== Starting text_to_speech ===")
            
            if not text:
                print("No text provided, returning None")
                return None

            # Get TTS instance from context
            tts = context.get('tts')
            voice_samples = context.get('voice_samples')
            conditioning_latents = context.get('conditioning_latents')
            
            print(f"\nRetrieved from context:")
            print(f"- TTS object: {type(tts) if tts else None}")
            print(f"- Voice samples: {type(voice_samples) if voice_samples else None}, length: {len(voice_samples) if voice_samples else 0}")
            print(f"- Conditioning latents: {type(conditioning_latents) if conditioning_latents else None}")
            
            if not tts or not voice_samples or not conditioning_latents:
                print("TTS not properly initialized")
                return None
            
            try:
                print("Processing text chunks...")
                # Split long text into smaller chunks at sentence boundaries
                sentences = text.split('.')
                max_chunk_length = 100  # Maximum characters per chunk
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_length:
                        current_chunk += sentence + "."
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + "."
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                print(f"Created {len(chunks)} chunks: {chunks}")
                
                # Process each chunk
                all_audio = []
                for i, chunk in enumerate(chunks, 1):
                    print(f"\nProcessing chunk {i}/{len(chunks)}: '{chunk}'")
                    if not chunk.strip():
                        print(f"Skipping empty chunk {i}")
                        continue
                    
                    print("Generating autoregressive samples...")
                    try:
                        gen = tts.tts_with_preset(
                            chunk,
                            voice_samples=voice_samples,
                            conditioning_latents=conditioning_latents,
                            preset='ultra_fast',
                            use_deterministic_seed=True,
                            num_autoregressive_samples=1,
                            diffusion_iterations=10,
                            cond_free=True,
                            cond_free_k=2.0,
                            temperature=0.8
                        )
                        print(f"Generated audio for chunk {i}")
                    except RuntimeError as e:
                        print(f"Error generating audio for chunk {i}: {e}")
                        if "expected a non-empty list of Tensors" in str(e):
                            print("Retrying with different configuration...")
                            # Try again with modified settings
                            gen = tts.tts_with_preset(
                                chunk,
                                voice_samples=voice_samples,
                                conditioning_latents=conditioning_latents,
                                preset='ultra_fast',
                                use_deterministic_seed=True,
                                num_autoregressive_samples=2,
                                diffusion_iterations=10,
                                cond_free=False,
                                temperature=0.8
                            )
                            print("Retry successful")
                        else:
                            raise
                    
                    if isinstance(gen, tuple):
                        gen = gen[0]
                    if len(gen.shape) == 3:
                        gen = gen.squeeze(0)
                    
                    print(f"Audio shape for chunk {i}: {gen.shape}")
                    all_audio.append(gen)

                # Combine all audio chunks
                print("\nCombining audio chunks...")
                if all_audio:
                    combined_audio = torch.cat(all_audio, dim=1)
                    print(f"Audio chunks combined successfully, final shape: {combined_audio.shape}")
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
                return "", "", None
            
            try:
                # Initialize recognizer
                recognizer = sr.Recognizer()
                
                # Use the audio file directly
                with sr.AudioFile(audio_path) as source:
                    audio = recognizer.record(source)
                
                # Perform the recognition
                text = recognizer.recognize_google(audio)
                return text, text, f"Transcribed: {text}"
            except Exception as e:
                return "", "", f"Error transcribing audio: {str(e)}"

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
                audio_path = text_to_speech(response)
                print("TTS generation complete")
                
                # Update embeddings in background
                def update_chat_embeddings():
                    try:
                        print("\nUpdating embeddings...")
                        # Get new messages from the current history
                        new_messages = history[state["last_processed_index"]:]
                        if new_messages:
                            chat_text = chat_manager.format_for_embedding(new_messages)
                            vector_store = context['vector_store']
                            vectors = context['embeddings'].embed_documents([chat_text])
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
                            if hasattr(context['watcher'], 'pending_changes') and context['watcher'].pending_changes:
                                pending = context['watcher'].pending_changes.copy()
                                context['watcher'].pending_changes.clear()
                                update_embeddings(pending)
                    except Exception as e:
                        print(f"Error updating embeddings: {str(e)}")
                        import traceback
                        traceback.print_exc()

                # Start background thread for embedding updates
                import threading
                threading.Thread(target=update_chat_embeddings, daemon=True).start()
                
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
            transcribe_audio,
            inputs=[audio_input],
            outputs=[
                input_text,           # recognized text
                gr.Textbox(visible=False),  # for debug message
                gr.Textbox(visible=False),  # status message
            ]
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

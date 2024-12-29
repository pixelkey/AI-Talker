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
import subprocess
import time
import requests

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

def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
        memory_used, memory_total = map(int, output.decode('utf-8').strip().split(','))
        return memory_used, memory_total
    except Exception as e:
        print(f"Error getting GPU memory usage: {e}")
        return None, None

def log_gpu_memory(stage):
    memory_used, memory_total = get_gpu_memory_usage()
    if memory_used is not None:
        print(f"GPU Memory at {stage}: {memory_used}MB / {memory_total}MB ({(memory_used/memory_total)*100:.1f}%)")

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

    def stop_ollama_server():
        print("Stopping Ollama server to free GPU memory...")
        try:
            # Try graceful shutdown first via API
            try:
                requests.post("http://localhost:11434/api/shutdown")
                time.sleep(3)
            except:
                pass
            
            # Check if server is still running
            try:
                response = requests.get("http://localhost:11434/api/version")
                if response.status_code == 200:
                    print("Server still running, checking process")
                    # Get process info without requiring root
                    try:
                        result = subprocess.run(['pgrep', 'ollama'], capture_output=True, text=True)
                        if result.stdout.strip():
                            pid = result.stdout.strip()
                            print(f"Found Ollama process: {pid}")
                            # Try SIGTERM first
                            subprocess.run(['kill', '-TERM', pid], check=False)
                            time.sleep(2)
                            
                            # Check if still running
                            if subprocess.run(['kill', '-0', pid], check=False).returncode == 0:
                                print("Process still running, using SIGKILL")
                                subprocess.run(['kill', '-KILL', pid], check=False)
                                time.sleep(2)
                    except Exception as e:
                        print(f"Process management error: {e}")
            except requests.ConnectionError:
                print("Server already stopped")
            
            print("Ollama server stopped successfully")
            
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            print(f"Error stopping Ollama server: {e}")

    def start_ollama_server():
        print("Starting Ollama server...")
        try:
            # Check if server is already running
            try:
                response = requests.get("http://localhost:11434/api/version")
                if response.status_code == 200:
                    print("Ollama server already running")
                    return
            except:
                pass
                
            subprocess.Popen(['ollama', 'serve'])
            time.sleep(5)  # Wait for server to start
            print("Ollama server started successfully")
        except Exception as e:
            print(f"Error starting Ollama server: {e}")

    def verify_ollama_server():
        """Verify Ollama server is running and responsive"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Try to connect to Ollama server
                response = requests.get("http://localhost:11434/api/version")
                if response.status_code == 200:
                    print("Ollama server is responsive")
                    return True
                else:
                    print(f"Ollama server returned status code: {response.status_code}")
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries}: Ollama server not responsive: {str(e)}")
        
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
        print("Failed to verify Ollama server after all retries")
        return False

    def ensure_ollama_running():
        print("Ensuring Ollama server is running...")
        try:
            # First verify server is responsive
            if not verify_ollama_server():
                print("Warning: Ollama server not responsive")
                # Try to restart
                stop_ollama_server()
                start_ollama_server()
                if not verify_ollama_server():
                    print("Failed to start Ollama server")
                    return False
            else:
                print("Ollama server is responsive")

            # Reinitialize the client with the correct configuration
            if context.get('MODEL_SOURCE') == "local":
                from langchain_community.llms import Ollama
                context['client'] = Ollama(model=context.get('LLM_MODEL'))
                print("Ollama client reinitialized")
            return True
        except Exception as e:
            print(f"Error ensuring Ollama is running: {e}")
            return False

    def chatbot_response(input_text, context_documents, context, history):
        try:
            # Ensure Ollama is running before proceeding
            if context.get('MODEL_SOURCE') == "local" and not ensure_ollama_running():
                return None, "Error: Ollama server not available", None

            # Process embeddings and get response using original logic
            refs = []
            if context_documents:
                refs = [doc.page_content for doc in context_documents]
            
            # Format conversation history for context
            conversation_context = ""
            if history:
                for human, assistant in history[-3:]:  # Include last 3 exchanges
                    if human and assistant:
                        conversation_context += f"Human: {human}\nAssistant: {assistant}\n"
            
            # Construct the prompt with context
            context_str = "\n".join(refs) if refs else ""
            prompt = f"Context:\n{context_str}\n\nConversation History:\n{conversation_context}\nHuman: {input_text}\nAssistant:"
            
            try:
                if context.get('MODEL_SOURCE') == "local":
                    response = context['client'].invoke(prompt)
                else:
                    response = "Error: Invalid model source"
            except Exception as e:
                print(f"Error getting LLM response: {e}")
                response = str(e)
                
            return refs, response, context_str
        except Exception as e:
            print(f"Error in chatbot response: {e}")
            return None, str(e), None

    def chat(input_text, history, refs):
        try:
            # Initialize history if needed
            if history is None:
                history = []
            history.append((input_text, None))
            
            # First yield to update UI
            yield history, refs, "", history, None, ""

            # Generate the LLM response
            refs, response, context_str = chatbot_response(input_text, context_documents, context, history)
            
            # Update history with response
            if history:
                history[-1] = (input_text, response)
                
            # Save conversation
            save_conversation_history(history)
            
            yield history, refs, "", history, None, ""
            
        except Exception as e:
            print(f"Error in chat: {e}")
            if history:
                history[-1] = (input_text, f"Error: {str(e)}")
            yield history, refs, "", history, None, ""

    def get_llm_response(user_input, conversation_history=None, context_str=""):
        try:
            if context.get('MODEL_SOURCE') == "local":
                # Ensure Ollama is running before making request
                if not ensure_ollama_running():
                    return "Error: Ollama server not available"
                
                # Format the conversation history
                formatted_history = ""
                if conversation_history:
                    for msg in conversation_history:
                        if msg:  # Only add non-empty messages
                            formatted_history += f"Human: {msg}\n"
            
                # Construct the prompt
                prompt = f"{formatted_history}Human: {user_input}\nAssistant:"
                
                try:
                    response = context['client'].invoke(prompt)
                    return response
                except AttributeError:
                    # Fallback to completion if invoke not available
                    response = context['client'].complete(prompt)
                    return response.text
            else:
                return "Error: Invalid model source"
        except Exception as e:
            print(f"Error in local LLM response: {e}")
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

        # Get references and generate response
        refs, filtered_docs, context_documents = retrieve_and_format_references(input_text, context)
        
        # Initialize history if needed
        if history is None:
            history = []
        
        # Add user message to history
        history.append((input_text, None))
        
        # First yield to update UI with user message
        yield history, refs, "", history, None, ""

        # Generate the LLM response
        response = get_llm_response(input_text, history)
        
        # Add assistant response to history and save
        history[-1] = (input_text, response)
        chat_manager.save_history(history)
        
        # Generate speech for the response
        print("\nGenerating TTS response...")
        audio_path = text_to_speech(response)
        print("TTS generation complete")
        
        # Yield the response with audio
        yield history, refs, "", history, audio_path, "Response complete, updating embeddings..."
        
        print("\nUpdating embeddings...")
        # Update embeddings after TTS is complete
        new_messages, new_last_index = chat_manager.get_new_messages(state["last_processed_index"])
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
            print("Embeddings update complete")
            
            # Now process any pending file changes
            if hasattr(context['watcher'], 'pending_changes') and context['watcher'].pending_changes:
                pending = context['watcher'].pending_changes.copy()
                context['watcher'].pending_changes.clear()
                update_embeddings(pending)
        
        # Final yield after everything is complete
        yield history, refs, "", history, audio_path, "Processing complete"

    def clear_interface(history):
        cleared_history, cleared_refs, cleared_input = clear_history(context, history)
        return [], cleared_refs, cleared_input, [], None, ""

    # Initialize TTS once at startup
    print("=== Initializing TTS at startup ===")
    tts_instance = None
    try:
        log_gpu_memory("at start")
        # Initialize TTS with configuration
        tts_instance = init_tts()
        log_gpu_memory("after TTS init")
        print("TTS initialized successfully")
    except Exception as e:
        print(f"Error initializing TTS: {e}")

    def init_tts():
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

        # Initialize TTS with basic configuration
        tts_config = {
            "kv_cache": True,
            "half": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "autoregressive_batch_size": 2,
            "use_deepspeed": use_deepspeed
        }

        print(f"Initializing TTS with config: {tts_config}")
        tts = TextToSpeech(**tts_config)

        # Set a fixed random seed for consistent voice
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        print("Loading voice samples...")
        # Use a specific voice preset for consistency
        voice_samples, conditioning_latents = load_voice('train_dotrice', extra_voice_dirs=[])
        print("Voice samples loaded")

        # Store voice samples and latents with TTS instance
        tts.voice_samples = voice_samples
        tts.conditioning_latents = conditioning_latents

        return tts

    def tts_generate(tts, chunk):
        try:
            gen = tts.tts_with_preset(
                chunk,
                voice_samples=tts.voice_samples,
                conditioning_latents=tts.conditioning_latents,
                preset='ultra_fast',
                use_deterministic_seed=True,
                num_autoregressive_samples=1,
                diffusion_iterations=10,
                cond_free=False,
                temperature=0.8
            )
        except RuntimeError as e:
            if "expected a non-empty list of Tensors" in str(e):
                print("Retrying with different configuration...")
                # Try again with modified settings
                gen = tts.tts_with_preset(
                    chunk,
                    voice_samples=tts.voice_samples,
                    conditioning_latents=tts.conditioning_latents,
                    preset='ultra_fast',
                    use_deterministic_seed=True,
                    num_autoregressive_samples=2,
                    diffusion_iterations=10,
                    cond_free=False,
                    temperature=0.8
                )
            else:
                raise

        if isinstance(gen, tuple):
            gen = gen[0]
        if len(gen.shape) == 3:
            gen = gen.squeeze(0)
    
        return gen

    def text_to_speech(text):
        """Convert text to speech using Tortoise TTS with DeepSpeed optimization if available"""
        print("\n=== Starting text_to_speech ===")
        start_time = time.time()
        log_gpu_memory("start")
        
        # Record Ollama management time
        ollama_stop_start = time.time()
        stop_ollama_server()
        ollama_stop_time = time.time() - ollama_stop_start
        log_gpu_memory("after stopping Ollama")
        print(f"Time to stop Ollama: {ollama_stop_time:.2f}s")
        
        if not text:
            print("No text provided, returning None")
            return None
        try:
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
                "autoregressive_batch_size": 2,
                "use_deepspeed": use_deepspeed
            }

            print(f"Initializing TTS with config: {tts_config}")
            tts = TextToSpeech(**tts_config)
            log_gpu_memory("after TTS init")
            print("TTS initialized successfully")

            # Set a fixed random seed for consistent voice
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            print("Loading voice samples...")
            # Use a specific voice preset for consistency
            voice_samples, conditioning_latents = load_voice('train_dotrice', extra_voice_dirs=[])
            print("Voice samples loaded")
            
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
            
            print(f"Created {len(chunks)} chunks")
            log_gpu_memory("before processing chunks")
            
            # Record TTS processing time
            tts_start = time.time()
            all_audio = []
            for i, chunk in enumerate(chunks, 1):
                chunk_start = time.time()
                print(f"Processing chunk {i}/{len(chunks)}")
                log_gpu_memory(f"before chunk {i}")
                
                if not chunk.strip():
                    print(f"Skipping empty chunk {i}")
                    continue
                
                print("Generating autoregressive samples..")
                try:
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
                    chunk_time = time.time() - chunk_start
                    print(f"Chunk {i} processing time: {chunk_time:.2f}s")
                    log_gpu_memory(f"after chunk {i}")
                    print(f"Generated audio for chunk {i}")
                except RuntimeError as e:
                    if "expected a non-empty list of Tensors" in str(e):
                        print("Retrying with different configuration...")
                        retry_start = time.time()
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
                        retry_time = time.time() - retry_start
                        print(f"Retry processing time: {retry_time:.2f}s")
                    else:
                        raise
                if isinstance(gen, tuple):
                    gen = gen[0]
                if len(gen.shape) == 3:
                    gen = gen.squeeze(0)
                
                all_audio.append(gen)

            tts_processing_time = time.time() - tts_start
            print(f"Total TTS processing time: {tts_processing_time:.2f}s")

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
            
            # After TTS is complete, clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log_gpu_memory("after TTS cleanup")
                
            # Record Ollama restart time
            ollama_start_time = time.time()
            start_ollama_server()
            log_gpu_memory("after restarting Ollama")
            
            # Ensure Ollama is running and responsive
            if not ensure_ollama_running():
                print("Warning: Ollama server not properly started")
                # Try one more time with forced cleanup
                stop_ollama_server()
                time.sleep(2)
                start_ollama_server()
                if not ensure_ollama_running():
                    print("Failed to start Ollama server after retry")
            else:
                print("Ollama server verified and running")
                
            ollama_restart_time = time.time() - ollama_start_time
            print(f"Time to restart Ollama: {ollama_restart_time:.2f}s")
            
            total_time = time.time() - start_time
            print(f"\nTiming Summary:")
            print(f"Ollama stop time: {ollama_stop_time:.2f}s")
            print(f"TTS processing time: {tts_processing_time:.2f}s")
            print(f"Ollama restart time: {ollama_restart_time:.2f}s")
            print(f"Total time: {total_time:.2f}s")
            print(f"Ollama overhead: {(ollama_stop_time + ollama_restart_time):.2f}s ({((ollama_stop_time + ollama_restart_time)/total_time)*100:.1f}% of total)")
            
            return temp_path

        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def split_text_into_chunks(text):
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
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"Created {len(chunks)} chunks")
        return chunks

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
                gr.Textbox(visible=False),  # Status message
            ],
        ).then(
            handle_user_input,  # Chain to handle_user_input
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, gr.Textbox(visible=False)],
        )

        # Add text input submission via Enter key
        input_text.submit(
            handle_user_input,
            inputs=[input_text, session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, gr.Textbox(visible=False)],
        )

        # Add clear button handler
        clear_button.click(
            clear_interface,
            inputs=[session_state],
            outputs=[chat_history, references, input_text, session_state, audio_output, gr.Textbox(visible=False)],
        )

    return app  # Make sure we return the app object
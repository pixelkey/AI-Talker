# scripts/config.py

import os
from dotenv import load_dotenv
import logging

# Environment configuration setup
env_config = {
    "MODEL_SOURCE": {"default": "local", "type": str},  # Set to "local" or "openai"
    "OPENAI_API_KEY": {"default": "your-api-key-here", "type": str},
    "EMBEDDING_MODEL": {"default": "nomic-embed-text:latest", "type": str},
    "EMBEDDING_DIM": {"default": "768", "type": int},
    "FAISS_INDEX_PATH": {"default": "embeddings/faiss_index.bin", "type": str},
    "METADATA_PATH": {"default": "embeddings/metadata.pkl", "type": str},
    "DOCSTORE_PATH": {"default": "embeddings/docstore.pkl", "type": str},
    "INGEST_PATH": {"default": "ingest", "type": str},
    "SYSTEM_PROMPT": {
        "default": (
            "You are a helpful assistant with ideas and opinions about everything."
            "You natually assess and match the mood of the person speaking to you."
            "You may be supplied with memory references or knowledge from other sources, including the internet if relevant. If you don't know something, don't make up an answer. Just let the person know you don't know."
            "Your knowledge cutoff is not relevant to the conversation when the web results are relevant."
            "If web results are relevant, ensure you respond based on the web results. Summarize the web results if needed but ensure all relevant information is included."
            "If the references do not seem relevant to the conversation, just ignore the reference memory and respond based on the conversation."
            "Keep your sentences relatively short and concise. Use as many sentences as you need but keep them short."
            "Do not make up memories. Only use memories that are from the references or conversation history."
            "Try to keep the conversation interesting and engaging. Ask a question that will help the conversation move forward."
            "Don't include URLS in your answers. Convert any non-speakable text into natural language. For example, 'e.g.' should be displayed as 'for example'."
            "Change acronyms and abbreviations into their full form."
            "Do not use phrases like 'it sounds like' or 'it looks like'."
        ),
        "type": str
    },
    "SIMILARITY_THRESHOLD": {"default": "0.25", "type": float},
    "TOP_SIMILARITY_RESULTS": {"default": "10", "type": int},
    "LLM_MODEL": {"default": "mistral:7b", "type": str},
    "LLM_MAX_TOKENS": {"default": "128000", "type": int},
    "CHUNK_SIZE_MAX": {"default": "512", "type": int},
    "CHUNK_OVERLAP_PERCENTAGE": {"default": "20", "type": int},
    "TOKEN_ENCODING": {"default": "cl100k_base", "type": str},
    "TTS_VOICE": {"default": "emma", "type": str},  # Default TTS voice to use
}

# Reset environment variables before loading .env to ensure they are not reused
for key in env_config:
    if key in os.environ:
        del os.environ[key]

# Load environment variables from .env file
load_dotenv()

# Apply environment settings
for key, settings in env_config.items():
    value = os.getenv(key, settings["default"])
    converted_value = settings["type"](value)
    os.environ[key] = str(converted_value)  # Store as string in OS environment
    globals()[key] = converted_value  # Set globally in the script

# Configure logging
logging.basicConfig(level=logging.INFO)

# Log the settings for verification
for key in env_config.keys():
    logging.info(f"{key}: {globals()[key]}")

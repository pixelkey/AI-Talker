# scripts/document_processing.py

import os
import logging
import config
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize, blankline_tokenize
from nltk import download as nltk_download
import re
import json
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# Set NLTK data path to local directory and disable downloads
nltk_data_dir = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.insert(0, nltk_data_dir)

# Only download if punkt_tab is missing
def ensure_nltk_data():
    """Ensure NLTK data is available, downloading only if missing."""
    try:
        # Try to load the tokenizer
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        # If not found, download it
        print("Downloading required NLTK data...")
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        print("Download complete.")

# Check for NLTK data at startup
ensure_nltk_data()

CHUNK_OVERLAP_PERCENTAGE = int(config.CHUNK_OVERLAP_PERCENTAGE)

# Initialize tiktoken for token counting with cl100k_base encoding
token_encoding = tiktoken.get_encoding(config.TOKEN_ENCODING)

def normalize_text(text):
    """
    Normalize text by removing excessive whitespace (more than one line).
    """
    # Replace multiple line breaks with a single space
    text = re.sub(r'\n\s*\n+', ' ', text.strip())

    return text

def chunk_text_hybrid(text, chunk_size_max):
    """
    Split text into chunks based on paragraphs and sentences. 
    Retain paragraph breaks to preserve context.
    Chunks overlap by a percentage of the chunk size.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size_max (int): The maximum size of each chunk in tokens.
    
    Returns:
        list: List of text chunks with their token counts and overlap sizes.
    """
    paragraphs = blankline_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for paragraph in paragraphs:
        paragraph_tokens = token_encoding.encode(paragraph)
        paragraph_size = len(paragraph_tokens)

        if paragraph_size > chunk_size_max:
            process_large_paragraph(
                paragraph, chunk_size_max, chunks, current_chunk, current_chunk_size
            )
            current_chunk, current_chunk_size = finalize_chunk(current_chunk, chunks)
            continue

        if current_chunk_size + paragraph_size > chunk_size_max:
            current_chunk, current_chunk_size = finalize_chunk(current_chunk, chunks)

        current_chunk.append(paragraph)
        current_chunk_size += paragraph_size

    if current_chunk:
        chunks.append((" ".join(current_chunk), current_chunk_size))

    overlap_size_max = int((CHUNK_OVERLAP_PERCENTAGE / 100) * chunk_size_max)
    overlapped_chunks = add_chunk_overlap(chunks, chunk_size_max, overlap_size_max)

    return overlapped_chunks

def add_chunk_overlap(chunks, chunk_size_max, overlap_size_max):
    """
    Add overlap between chunks by a specified number of tokens, ensuring overlap
    occurs at the sentence level and includes meaningful content.
    
    Args:
        chunks (list): List of chunks with their token counts.
        chunk_size_max (int): The maximum size of each chunk in tokens.
        overlap_size_max (int): The number of tokens to overlap between chunks.
    
    Returns:
        list: List of overlapped chunks with updated token counts and overlap sizes.
    """
    overlapped_chunks = []
    previous_chunk_sentences = []

    for chunk, chunk_size in chunks:
        current_chunk_tokens = token_encoding.encode(chunk)

        if previous_chunk_sentences:
            # Create the overlap text by joining previous sentences
            overlap_text = " ".join(previous_chunk_sentences)
            overlap_tokens = token_encoding.encode(overlap_text)
            current_chunk_tokens = overlap_tokens + current_chunk_tokens

        # Ensure the chunk doesn't exceed the max size
        if len(current_chunk_tokens) > chunk_size_max:
            current_chunk_tokens = current_chunk_tokens[:chunk_size_max]

        # Decode the current chunk to text
        current_chunk_text = token_encoding.decode(current_chunk_tokens)

        # Tokenize the current chunk into sentences
        sentences = sent_tokenize(current_chunk_text)

        # Identify sentences for the next overlap
        overlap_text = ""
        overlap_tokens = []
        while sentences and len(overlap_tokens) < overlap_size_max:
            overlap_text = sentences.pop(-1) + " " + overlap_text
            overlap_tokens = token_encoding.encode(overlap_text.strip())

        overlap_size = len(overlap_tokens)  # Get the correct overlap size in tokens

        # If it is the first chunk, there is no overlap. So set it to 0
        if not overlapped_chunks:
            overlap_tokens = []
            overlap_size = 0

        # Save the current chunk and prepare the next overlap
        overlapped_chunks.append((current_chunk_text, chunk_size, overlap_size))
        previous_chunk_sentences = sent_tokenize(overlap_text.strip())

    return overlapped_chunks

def process_large_paragraph(
    paragraph, chunk_size_max, chunks, current_chunk, current_chunk_size
):
    """
    Process a large paragraph by splitting it into smaller chunks based on sentences.
    
    Args:
        paragraph (str): The paragraph to be processed.
        chunk_size_max (int): The maximum size of each chunk in tokens.
        chunks (list): The list to store the resulting chunks.
        current_chunk (list): The current chunk being constructed.
        current_chunk_size (int): The current size of the chunk being constructed.
    """
    sentences = sent_tokenize(paragraph)
    for sentence in sentences:
        sentence_tokens = token_encoding.encode(sentence)
        sentence_size = len(sentence_tokens)

        if current_chunk_size + sentence_size > chunk_size_max:
            current_chunk, current_chunk_size = finalize_chunk(current_chunk, chunks)

        current_chunk.append(sentence)
        current_chunk_size += sentence_size

def finalize_chunk(current_chunk, chunks):
    """
    Finalize the current chunk by joining its content and adding it to the list of chunks.
    
    Args:
        current_chunk (list): The current chunk being finalized.
        chunks (list): The list to store the resulting chunks.
    
    Returns:
        tuple: An empty list and a size of 0 to reset the current chunk.
    """
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_size = len(token_encoding.encode(chunk_text))
        chunks.append((chunk_text, chunk_size))
    return [], 0

def load_file_content(file_path):
    """
    Load the content of a text file.
    
    Args:
        file_path (str): The path to the text file.
    
    Returns:
        str: The content of the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_json_content(file_path):
    """
    Load and format the content of a JSON file.
    
    Args:
        file_path (str): The path to the JSON file.
    
    Returns:
        tuple: (formatted content string, timestamp if found in JSON)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not data:
            return "", None
            
        # Try to extract timestamp from JSON data
        timestamp = None
        if isinstance(data, dict):
            timestamp = data.get('timestamp')
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            timestamp = data[0].get('timestamp')
            
        # Format the content
        if isinstance(data, dict):
            content = json.dumps(data, indent=2)
        elif isinstance(data, list):
            content = "\n".join(json.dumps(item, indent=2) for item in data)
        else:
            content = str(data)
            
        return content, timestamp
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {e}")
        return "", None

def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO format timestamp string into a datetime object.
    Handles both basic ISO format and extended formats.
    """
    try:
        # Remove the 'T' separator if present
        timestamp_str = timestamp_str.replace('T', ' ')
        
        # Split into date, time, and timezone parts
        date_time = timestamp_str.rsplit('+', 1)[0].rsplit('-', 1)[0].strip()
        
        # Parse the main part
        dt = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
        
        # Handle timezone if present
        if '+' in timestamp_str:
            # Extract hours and minutes from timezone
            tz = timestamp_str.split('+')[1]
            if ':' in tz:
                h, m = map(int, tz.split(':'))
            else:
                h, m = int(tz[:2]), int(tz[2:]) if len(tz) > 2 else 0
            dt = dt.replace(tzinfo=timezone(timedelta(hours=h, minutes=m)))
        elif timestamp_str.count('-') > 2:  # Has negative timezone
            tz = timestamp_str.split('-')[-1]
            if ':' in tz:
                h, m = map(int, tz.split(':'))
            else:
                h, m = int(tz[:2]), int(tz[2:]) if len(tz) > 2 else 0
            dt = dt.replace(tzinfo=timezone(timedelta(hours=-h, minutes=-m)))
            
        return dt
    except Exception as e:
        logging.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return datetime.now().astimezone()

def load_single_document(file_path, chunk_size_max):
    """
    Load a single document and create chunks.
    
    Args:
        file_path (str): The path to the file.
        chunk_size_max (int): The maximum size of each text chunk.
        
    Returns:
        tuple: (List of chunks with metadata, timestamp)
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        # Get file modification time and format with weekday
        mtime = os.path.getmtime(file_path)
        dt = datetime.fromtimestamp(mtime).astimezone()
        file_timestamp = dt.strftime("%A, %Y-%m-%d %H:%M:%S %z")
        
        content = ""
        json_timestamp = None
        
        if ext == '.json':
            content, json_timestamp = load_json_content(file_path)
            # If JSON timestamp exists, format it with weekday
            if json_timestamp:
                try:
                    dt = parse_timestamp(json_timestamp)
                    json_timestamp = dt.strftime("%A, %Y-%m-%d %H:%M:%S %z")
                except (ValueError, TypeError):
                    pass
        else:
            content = load_file_content(file_path)
        
        if not content:
            return [], None
            
        # Normalize the content
        normalized_content = normalize_text(content)
        
        # Create chunks
        chunks = chunk_text_hybrid(normalized_content, chunk_size_max)
        
        # Use JSON timestamp if available, otherwise use file modification time
        timestamp = json_timestamp if json_timestamp else file_timestamp
        
        return chunks, timestamp
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return [], None

def create_document_entries(doc_id, filename, filepath, chunks, timestamp=None):
    """
    Create document entries with unique IDs for each chunk, including file path and timestamp.
    
    Args:
        doc_id (int): The document ID (unused, kept for backward compatibility).
        filename (str): The filename of the document.
        filepath (str): The relative path of the document.
        chunks (list): The chunks of text content and their token counts.
        timestamp (str): The timestamp for the document.
    
    Returns:
        list: List of document dictionaries with chunk ID, document ID, content, filename, filepath, timestamp, token count, and overlap metadata.
    """
    # Create document ID by combining filepath and filename
    full_doc_id = os.path.join(filepath, filename) if filepath != '.' else filename
    # Sanitize the document ID (only replaces spaces with underscores)
    safe_doc_id = sanitize_id(full_doc_id)
    
    return [
        {
            "id": f"{safe_doc_id}#chunk{chunk_idx}",  # Unique chunk ID that includes the document path
            "doc_id": safe_doc_id,  # Document ID is the full relative path with spaces replaced
            "content": chunk,
            "filename": filename,
            "filepath": filepath,
            "timestamp": timestamp,  # Add timestamp to metadata
            "chunk_size": chunk_size,  # The token count of the chunk
            "overlap_size": overlap_size  # The token count of the overlap
        }
        for chunk_idx, (chunk, chunk_size, overlap_size) in enumerate(chunks)
    ]

def load_documents_from_folder(folder_path, chunk_size_max):
    """
    Load all supported text files from the specified folder and its subdirectories.
    
    Args:
        folder_path (str): The root folder path to search for text files.
        chunk_size_max (int): The maximum size of each text chunk.
    
    Returns:
        list: List of document dictionaries containing the content and metadata.
    """
    documents = []
    supported_extensions = {'.txt', '.json'}  # Add more supported extensions as needed
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in supported_extensions:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, folder_path)
                
                # Load document and get chunks with timestamp
                chunks, timestamp = load_single_document(file_path, chunk_size_max)
                if chunks:
                    # Create document entries with timestamp
                    doc_entries = create_document_entries(
                        len(documents), filename, relative_path, chunks, timestamp
                    )
                    documents.extend(doc_entries)
    
    return documents

def sanitize_id(id_str):
    """
    Sanitize an ID string by replacing spaces with underscores.
    Keeps '.' and '/' intact for readability and hierarchical structure.
    
    Args:
        id_str (str): The ID string to sanitize
        
    Returns:
        str: Sanitized ID string
    """
    # Replace spaces with underscores, keep dots and slashes
    return id_str.replace(' ', '_')

# scripts/chatbot_functions.py

import os
import logging
from faiss_utils import similarity_search_with_score
from document_processing import normalize_text
from CognitiveProcessing import summarize_rag_results, determine_and_perform_web_search
import config
from typing import Tuple, Dict, List
from datetime import datetime, timezone, timedelta
from typing import Any

def retrieve_and_format_references(input_text: str, context: Dict) -> Tuple[str, List[Dict], str]:
    """
    Retrieve relevant documents and format references.
    Args:
        input_text (str): The user's input.
        context (dict): Context containing client, memory, and other settings.
    Returns:
        Tuple: references, filtered_docs, and context_documents.
    """
    # Skip if web search is disabled
    if context.get('skip_web_search', False):
        return "", [], ""
        
    # Normalize the user's input text
    normalized_input = normalize_text(input_text)
    logging.info(f"Normalized input: {normalized_input}")

    # 1. Retrieve relevant documents
    filtered_docs = retrieve_relevant_documents(normalized_input, context)
    logging.info(f"Retrieved {len(filtered_docs) if filtered_docs else 0} documents")
    if not filtered_docs:
        return "", [], "No relevant documents found."

    # 2. Format references
    references = build_references(filtered_docs, context)
    logging.info(f"Built references: {bool(references)}")
    if not references:
        return "", [], "No formatted references available."

    # 3. Summarize if over character limit
    logging.info(f"About to summarize {len(filtered_docs)} documents")
    summarized_context = summarize_rag_results(filtered_docs, context=context, query=normalized_input)
    if not summarized_context:
        # If summarization fails, use original references
        summarized_context = references
        logging.info("Using original references as fallback")
    
    logging.info(f"RAG Summary: {summarized_context[:200]}...")
    return summarized_context, filtered_docs, summarized_context

def chatbot_response(input_text, context_documents, context, history):
    """
    Handle user input, generate a response, and update the conversation history.
    """
    logger = logging.getLogger(__name__)
    # Get current time from context and parse it
    current_time = context.get('current_time', '')
    try:
        dt = parse_timestamp(current_time)
        formatted_time = dt.strftime("%A, %Y-%m-%d %H:%M:%S %z")
    except (ValueError, TypeError):
        formatted_time = current_time
    
    # Format chat history with timestamps
    formatted_history = format_chat_history(history, current_time)
    if formatted_history:
        formatted_history = f"Conversation History:\n{formatted_history}\n"
    
    # Check if web search is needed based on RAG results
    web_search_results = {"needs_web_search": False, "web_results": ""} if context.get('skip_web_search', False) else determine_and_perform_web_search(input_text, context_documents or "", context)
    
    # Initialize final references and context
    final_references = []
    final_context = []
    
    # Add formatted history to context first to prioritize conversation context
    if formatted_history:
        final_context.append(formatted_history)
    
    # Add web search results if they exist and haven't been included yet
    if web_search_results["needs_web_search"] and web_search_results["web_results"]:
        web_ref = "Web Search Results:\n" + web_search_results["web_results"]
        
        # Add to final context for LLM
        final_context.append(web_ref)
        
        # Format web search results for history display and saving
        timestamp_msg = f"[{formatted_time}]\nUser: {input_text}"
        history.append([timestamp_msg, f"[{formatted_time}]\nBot: {web_ref}"])
        
        # Save the updated history to persist web search results
        if context.get("chat_manager"):
            context["chat_manager"].save_history(history)
    
    # Add RAG results if they exist and aren't duplicates of what's in history
    if context_documents:
        # Check if these documents are already in the conversation history
        history_text = "\n".join(str(msg) for pair in history for msg in pair)
        if context_documents not in history_text:
            final_references.append(context_documents)
            final_context.append(context_documents)
    
    # If no results at all, add placeholder
    if not final_references and not final_context:
        no_results = "No search results."
        final_references.append(no_results)
        final_context.append(no_results)
    
    # Combine all references with consistent formatting
    final_references_str = "\n\n".join(final_references)
    context_documents = "\n\n".join(final_context)
    
    # Generate the response based on the model source
    response_text = generate_response(input_text, context_documents, context, history)
    if response_text is None:
        return history, "Error generating response.", final_references_str, ""

    # Analyze emotional content and get TTS cue
    logger.info("\n=== Emotion Analysis for Response ===")
    logger.info(f"User input: {input_text}")
    logger.info(f"Generated response: {response_text}")
    
    emotion_cue = analyze_emotion_and_tone(response_text, context)
    logger.info(f"Selected emotion cue: {emotion_cue}")
    
    # Prepend emotion cue to response for TTS
    tts_response = f"{emotion_cue} {response_text}"
    logger.info(f"Final TTS response (first 100 chars): {tts_response[:100]}...")
    
    # Store the TTS-formatted response in context for use by interface.py
    context['tts_response'] = tts_response
    
    # Return the history unchanged, the original response (without emotion cue), references, and a cleared input field
    return history, response_text, final_references_str, ""

def retrieve_relevant_documents(normalized_input, context):
    """
    Retrieve relevant documents using similarity search.
    """
    try:
        if not context.get("vector_store"):
            logging.warning("No vector store available")
            return []

        if not context.get("embeddings"):
            logging.warning("No embeddings model available")
            return []

        if not context.get("EMBEDDING_DIM"):
            logging.warning("No embedding dimension specified")
            return []

        search_results = similarity_search_with_score(
            normalized_input, context["vector_store"], context["embeddings"], context["EMBEDDING_DIM"]
        )
        logging.info(f"Retrieved {len(search_results) if search_results else 0} documents with scores.")
    except KeyError as e:
        logging.error(f"Error while retrieving documents: {e}")
        return []

    # Filter the results based on a similarity score threshold
    filtered_results = [
        result for result in search_results if result['score'] >= context["SIMILARITY_THRESHOLD"]
    ]
    logging.info(
        f"Filtered results by similarity threshold: {[result['score'] for result in filtered_results]}"
    )

    # Remove duplicates based on the content of the documents
    seen_contents = set()
    unique_filtered_results = []
    for result in filtered_results:
        content_hash = hash(result['content'])
        if content_hash not in seen_contents:
            unique_filtered_results.append(result)
            seen_contents.add(content_hash)

    # Sort the filtered results by similarity score in descending order
    unique_filtered_results.sort(key=lambda x: x['score'], reverse=True)
    filtered_docs = unique_filtered_results[:context["TOP_SIMILARITY_RESULTS"]]

    # Log top similarity results
    logging.info(
        f"Top similarity results: {[(res['id'], res['score']) for res in filtered_docs]}"
    )

    return filtered_docs

def format_references(filtered_docs: List[Dict], context: Dict = None) -> str:
    """
    Format references in a consistent way for both interface and LLM.
    """
    if not filtered_docs:
        return ""

    # Format each document with consistent numbering, metadata, and timestamps
    formatted_refs = []
    for idx, doc in enumerate(filtered_docs, 1):
        # Extract timestamp from metadata if available
        timestamp = doc['metadata'].get('timestamp', '')
        timestamp_str = f" | {timestamp}" if timestamp else ""
        
        ref = f"{idx}. Context Document {doc['metadata'].get('doc_id', '')} - Chunk {doc['id']} | Path: {doc['metadata'].get('filepath', '')}/{doc['metadata'].get('filename', '')}{timestamp_str}\n{doc['content']}"
        formatted_refs.append(ref)

    return "\n\n".join(formatted_refs)

def build_references(filtered_docs: List[Dict], context: Dict = None) -> str:
    """
    Construct the reference list from filtered documents.
    Uses the common format_references function.
    """
    refs = format_references(filtered_docs, context)
    return refs if refs else ""

def build_context_documents(filtered_docs: List[Dict], context: Dict = None) -> str:
    """
    Combine content from filtered documents to form the context documents.
    Uses the common format_references function.
    """
    return format_references(filtered_docs, context)

def generate_response(input_text, context_documents, context, history):
    """
    Generate the LLM response based on the model source.
    """
    try:
        if config.MODEL_SOURCE == "openai":
            return generate_openai_response(input_text, context_documents, context, history)
        elif config.MODEL_SOURCE == "local":
            return generate_local_response(input_text, context_documents, context, history)
        else:
            logging.error(f"Unsupported MODEL_SOURCE: {config.MODEL_SOURCE}")
            return None
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None

def generate_openai_response(input_text, context_documents, context, history):
    """
    Generate response using OpenAI API.
    """
    messages = [{"role": "system", "content": context['SYSTEM_PROMPT']}]

    # Add context documents as a system message
    messages.append({"role": "system", "content": f"Context Documents:\n{context_documents}"})

    # Add conversation history as user-assistant pairs
    if history:
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg.replace("User: ", "")})
            messages.append({"role": "assistant", "content": bot_msg.replace("Bot: ", "")})

    # Add the current user input
    messages.append({"role": "user", "content": input_text})

    # Log the messages being sent
    logging.info(f"Messages sent to OpenAI API: {messages}")

    # Generate the LLM response
    response = context["client"].chat.completions.create(
        model=context["LLM_MODEL"],
        messages=messages,
        max_tokens=min(context["LLM_MAX_TOKENS"] - len(context["encoding"].encode(str(messages))), 8000),
    )
    response_text = response.choices[0].message.content
    logging.info("Generated LLM response successfully.")

    return response_text

def generate_local_response(input_text, context_documents, context, history):
    """
    Generate response using local model.
    """
    # Get the appropriate system prompt
    system_prompt = context.get('system_prompt', context['SYSTEM_PROMPT'])
    
    # Build the prompt for the local model
    prompt = build_local_prompt(system_prompt, history, context_documents, input_text)

    # Log the final prompt sent to the local LLM
    logging.info(f"Final prompt sent to local LLM:\n{prompt}")

    # Calculate the max tokens for the model
    try:
        tokens_consumed = len(context["encoding"].encode(prompt))
        max_tokens = min(
            context["LLM_MAX_TOKENS"] - tokens_consumed, 8000
        )
    except Exception as e:
        logging.warning(f"Token encoding error: {e}")
        max_tokens = 8000  # Fallback to default max tokens

    # Generate the LLM response
    try:
        response = context["client"].chat(
            model=context["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response['message']['content']
        logging.info("Generated LLM response successfully.")
        return response_text
    except Exception as e:
        logging.error(f"Error in local LLM response: {e}")
        return None

def build_local_prompt(system_prompt, history, context_documents, input_text):
    """
    Build the prompt for the local model, including conversation history and context documents.
    """
    prompt = f"{system_prompt}\n\n"

    if history:
        prompt += "Conversation History:\n"
        for user_msg, bot_msg in history:
            prompt += f"{user_msg}\n{bot_msg}\n"
        prompt += "\n"

    prompt += f"Context Documents:\n{context_documents}\n\nUser Prompt:\n{input_text}"
    return prompt

def evaluate_rag_results(references: str, query: str, context: dict) -> str:
    """
    Evaluate RAG results for relevance to the query.
    Returns filtered references if any are relevant.
    """
    if not references:
        return ""
        
    prompt = f"""Evaluate if these references from the local knowledge base are relevant to answering the query.

Query: "{query}"

References:
{references}

Evaluate if these references provide relevant information to answer the query.
A reference is relevant if it:
1. Directly addresses the query or provides useful context
2. Contains accurate and applicable information
3. Helps in forming a complete answer

A reference is NOT relevant if it:
1. Is completely unrelated to the query
2. Contains only tangential information
3. Provides no value in answering the query

Respond in this format:
KEEP_REFERENCES: [true/false]
REASON: [Brief explanation of why references should be kept or removed]
FILTERED_REFERENCES: [If keeping references, include only the relevant ones here. Otherwise write 'none']
"""

    if config.MODEL_SOURCE == "openai":
        response = context["client"].chat.completions.create(
            model=context["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        eval_text = response.choices[0].message.content.strip()
    else:
        response = context["client"].chat(
            model=context["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
        )
        eval_text = response['message']['content'].strip()
    
    # Parse evaluation
    eval_lines = eval_text.split('\n')
    keep_refs = eval_lines[0].lower().endswith('true')
    filtered_refs = ""
    
    if keep_refs:
        # Extract the filtered references section
        for i, line in enumerate(eval_lines):
            if line.startswith('FILTERED_REFERENCES:'):
                filtered_refs = '\n'.join(eval_lines[i+1:]).strip()
                break
    
    return filtered_refs if keep_refs else ""

def clear_history(context, history):
    """
    Clear the chat history and reset the session state.
    Args:
        context (dict): Context containing memory and other settings.
        history (list): Session state to be cleared.
    Returns:
        Tuple: Cleared chat history, cleared references, cleared input field, and session state.
    """
    try:
        # Clear memory
        if context.get("memory"):
            context["memory"].clear()

        # Clear chat history
        cleared_history = []
        cleared_refs = ""
        cleared_input = ""

        return cleared_history, cleared_refs, cleared_input

    except Exception as e:
        logging.error(f"Error clearing history: {e}")
        return history, "", ""

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

def format_chat_history(history: List[str], current_time: str) -> str:
    """
    Format chat history with timestamps including weekday.
    Args:
        history: List of chat messages
        current_time: Current timestamp from the user's machine
    Returns:
        Formatted chat history string
    """
    if not history:
        return ""
    
    formatted_history = []
    for i, msg in enumerate(history):
        # Get timestamp from message metadata if available, otherwise use current time
        timestamp = msg.get('timestamp', current_time) if isinstance(msg, dict) else current_time
        try:
            # Parse the timestamp and format with weekday
            dt = parse_timestamp(timestamp)
            formatted_time = dt.strftime("%A, %Y-%m-%d %H:%M:%S %z")
            role = "User" if i % 2 == 0 else "Bot"
            content = msg['content'] if isinstance(msg, dict) else msg
            formatted_history.append(f"[{formatted_time}] {role}: {content}")
        except (ValueError, TypeError) as e:
            # Fallback if timestamp parsing fails
            role = "User" if i % 2 == 0 else "Bot"
            content = msg['content'] if isinstance(msg, dict) else msg
            formatted_history.append(f"[{timestamp}] {role}: {content}")
    
    return "\n".join(formatted_history)

def analyze_emotion_and_tone(text: str, context: Dict[str, Any]) -> str:
    """
    Analyze the emotional content and appropriate tone for a response.
    Returns an emotional cue string suitable for Tortoise TTS.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n=== Starting Emotion Analysis ===")
    logger.info(f"Analyzing text (first 100 chars): {text[:100]}...")
    
    try:
        prompt = """Analyze this text and determine the most appropriate emotional tone for text-to-speech synthesis.
Consider:
1. Emotional content (happy, sad, concerned, excited, etc.)
2. Speaking style (formal, casual, gentle, firm)
3. Pacing (slow, moderate, energetic)

Return ONLY the emotional cue in brackets, like: [happy and energetic] or [concerned and gentle]
Keep it simple - use 2-3 descriptive words maximum.

Text to analyze:
{text}

Emotional Cue:"""

        logger.info("Sending prompt to LLM for emotion analysis")
        if context["MODEL_SOURCE"] == "openai":
            response = context["client"].chat.completions.create(
                model=context["LLM_MODEL"],
                messages=[{"role": "user", "content": prompt.format(text=text)}],
                max_tokens=20,
                temperature=0.7
            )
            emotion_cue = response.choices[0].message.content.strip()
            logger.info(f"OpenAI emotion analysis response: {emotion_cue}")
        else:
            response = context["client"].chat(
                model=context["LLM_MODEL"],
                messages=[{"role": "user", "content": prompt.format(text=text)}]
            )
            emotion_cue = response['message']['content'].strip()
            logger.info(f"Local LLM emotion analysis response: {emotion_cue}")
            
        # Ensure the response is properly formatted
        if not emotion_cue.startswith('[') or not emotion_cue.endswith(']'):
            logger.info(f"Reformatting emotion cue: {emotion_cue} -> [{emotion_cue}]")
            emotion_cue = f"[{emotion_cue}]"
            
        logger.info(f"Final emotion cue: {emotion_cue}")
        return emotion_cue
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}", exc_info=True)
        fallback_cue = "[neutral and clear]"
        logger.info(f"Using fallback emotion cue: {fallback_cue}")
        return fallback_cue

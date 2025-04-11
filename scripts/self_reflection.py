import threading
import time
import logging
import json
import os
from queue import Queue, Empty
from chatbot_functions import chatbot_response
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime, timedelta
import pytz
import re
from langchain.docstore.document import Document  # Fix import path for Document class
from faiss_utils import save_faiss_index_metadata_and_docstore  # Add import for save function
from config import FAISS_INDEX_PATH, METADATA_PATH, DOCSTORE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfReflection:
    def __init__(self, context):
        """Initialize self reflection manager"""
        self.context = context
        self.reflection_thread = None
        self.stop_reflection = threading.Event()
        self.pause_event = threading.Event()
        self.user_input_queue = Queue()
        self.is_reflecting = False
        
        # Initialize history manager
        from self_reflection_history import SelfReflectionHistoryManager
        self.history_manager = SelfReflectionHistoryManager()
        context['history_manager'] = self.history_manager  # Add to context for other components
        
        self.embedding_updater = context['embedding_updater']

        # Memory expiry settings (in days)
        self.memory_expiry = {
            'long_term': None,  # Never expires
            'mid_term': 30,     # 30 days
            'short_term': 7     # 7 days
        }
        
        # Memory processing prompts
        self.memory_prompts = {
            'long_term': """Extract meaningful information from this conversation that would be valuable for future interactions.
Focus on:
- Personal preferences and interests
- Important facts or knowledge shared
- Significant opinions or views expressed

Create natural, contextual statements that capture the essence of what was shared.""",
            
            'mid_term': """Extract information from this conversation that affects future interactions.
Focus on:
- Preferences that influence our conversation
- Context that shapes understanding
- Important details that guide interactions

Create natural statements that preserve the meaning and relevance.""",
            
            'short_term': """Extract the most significant piece of information from this conversation.
Focus on what matters most for maintaining context and understanding.
Create a natural statement that captures the key insight."""
        }

        # Simple prompt focused on getting a single numerical score
        self.surprise_score_prompt = """Rate how significant or surprising the information in this conversation is on a scale from 0.0 to 1.0.

IMPORTANT: Learning someone's name or identity information is HIGHLY significant (should score 0.8-1.0) unless this information was already known.

Consider these factors for high scores (0.8-1.0):
- First time learning someone's name or identity (this is crucial for building relationship context)
- Major personal preferences or strong opinions
- Important facts about the person
- Significant life events or experiences
- Key decisions or commitments made

Consider these factors for medium scores (0.5-0.7):
- Updates to previously known information
- General interests or casual preferences
- Day-to-day activities or plans

Consider these factors for low scores (0.0-0.4):
- Basic greetings or chitchat ("hello", "how are you", weather talk) should always get 0.0
- Small talk without new information
- Repetition of known information
- If the information is already known in the context
- If a web search failed or the information was not found

Remember: Names and identity information are especially important for maintaining conversation context.
Respond with only a number between 0.0 and 1.0."""

        # Separate prompt for reasoning (optional, only if score is high)
        self.reasoning_prompt = (
            "Explain in one short sentence why this conversation received a score of {score}."
        )

        # Surprise thresholds
        self.surprise_thresholds = {
            'high': 0.8,   # Score >= 0.8 goes to long-term
            'medium': 0.5,  # Score >= 0.5 goes to mid-term, below goes to short-term
            'low': 0.2     # Score >= 0.2 is considered for memory processing
        }

    def _extract_score(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        try:
            # Look for any number between 0.0 and 1.0 in the response
            matches = re.findall(r'0?\.[0-9]+', response)
            if matches:
                score = float(matches[0])
                return score
            return 0.0
        except Exception as e:
            logger.error(f"Error extracting score: {e}")
            return 0.0

    def _determine_memory_type(self, surprise_score: float) -> Tuple[str, Optional[datetime]]:
        """Determine memory type and expiry based on surprise score"""
        # Parse current time if it's a string
        current_time = self.context.get('current_time')
        if isinstance(current_time, str):
            current_time = self._parse_timestamp(current_time)
        
        if surprise_score >= self.surprise_thresholds['high']:
            return 'long_term', None
        elif surprise_score >= self.surprise_thresholds['medium']:
            expiry = current_time + timedelta(days=self.memory_expiry['mid_term'])
            return 'mid_term', expiry
        else:
            expiry = current_time + timedelta(days=self.memory_expiry['short_term'])
            return 'short_term', expiry

    def _get_llm_response(self, prompt: str, conversation_text: str) -> str:
        """Get a direct response from the LLM without emotion processing"""
        try:
            # Create a simplified context without TTS/emotion
            temp_context = {
                'client': self.context['client'],
                'MODEL_SOURCE': self.context.get('MODEL_SOURCE', 'local'),
                'skip_web_search': True,
                'skip_emotion': True,  # Skip emotion processing
                'system_prompt': prompt,
                'LLM_MODEL': self.context.get('LLM_MODEL', 'mistral')  # Use the configured model
            }
            
            # Get direct response from LLM
            response = temp_context['client'].chat(
                model=temp_context['LLM_MODEL'],
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': conversation_text}
                ]
            )
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise

    def _evaluate_web_search_need(self, conversation_text: str) -> Tuple[bool, Optional[str]]:
        """Evaluate if web search is needed and generate search query"""
        try:
            # Prompt to evaluate web search need
            evaluation_prompt = """Based on this conversation, determine if a web search would be valuable. 
            Consider these factors:
            - Are there questions about current events, facts, or information?
            - Is there a need to verify or expand on mentioned topics?
            - Would additional context from the web enhance understanding?
            - Is real-time or up-to-date information needed?
            
            If a search is needed, provide a clear, focused search query WITHOUT quotes.
            If no search is needed, explain why briefly.
            
            Respond in this format:
            SEARCH_NEEDED: [YES/NO]
            QUERY: [search query if needed, or "none" if not needed]
            REASON: [brief explanation]"""

            # Get evaluation from LLM
            response = self._get_llm_response(evaluation_prompt, conversation_text)
            
            # Parse response
            search_needed = 'SEARCH_NEEDED: YES' in response.upper()
            query_match = re.search(r'QUERY:\s*(.+?)(?:\n|$)', response)
            query = query_match.group(1).strip() if query_match else None
            
            # Clean up the query
            if query:
                # Remove any quotes from the query
                query = query.strip('"\'')
                # Return None if query is "none"
                if query.lower() == 'none':
                    query = None
                
            return search_needed, query
            
        except Exception as e:
            logger.error(f"Error in web search evaluation: {e}")
            return False, None
            
    def _process_web_search_results(self, query: str, content: str) -> Tuple[str, Optional[str]]:
        """Process web search results into memory"""
        try:
            # Prompt for processing web search results
            processing_prompt = f"""Analyze this web search content about "{query}" and create a concise, informative summary.
            Focus on:
            1. Key facts and information
            2. Relevance to the original query
            3. Any time-sensitive information
            
            Format the summary as a natural, contextual statement that would be useful for future reference.
            If the content is not relevant or useful, respond with "No relevant information found."
            """
            
            # Get summary from LLM
            summary = self._get_llm_response(processing_prompt, content)
            
            # If no relevant info, return None
            if "No relevant information found" in summary:
                return 'short_term', None
                
            # Determine memory type based on content
            memory_type = 'mid_term'  # Default to mid-term for web search results
            if any(term in content.lower() for term in ['today', 'current', 'latest', 'breaking']):
                memory_type = 'short_term'  # Use short-term for very time-sensitive info
            elif any(term in content.lower() for term in ['history', 'fundamental', 'principle', 'definition']):
                memory_type = 'long_term'  # Use long-term for foundational knowledge
                
            return memory_type, summary
            
        except Exception as e:
            logger.error(f"Error processing web search results: {e}")
            return 'short_term', None

    def _calculate_expiry(self, memory_type: str) -> Optional[datetime]:
        """Calculate expiry date for a memory type"""
        try:
            # Parse current time if it's a string
            current_time = self.context.get('current_time')
            if isinstance(current_time, str):
                current_time = self._parse_timestamp(current_time)
            
            if memory_type == 'long_term':
                return None  # Long-term memories don't expire
            elif memory_type == 'mid_term':
                return current_time + timedelta(days=self.memory_expiry['mid_term'])
            else:  # short_term
                return current_time + timedelta(days=self.memory_expiry['short_term'])
        except Exception as e:
            logging.error(f"Error calculating expiry: {str(e)}")
            return None

    def process_conversation(self, current_exchange: List[Tuple[str, str]]) -> None:
        """Process a conversation exchange and generate reflection"""
        try:
            logger.info("=== SELF REFLECTION: Starting conversation processing ===")
            
            # Set context flag to indicate we're in reflection mode
            # This prevents reflection output from being treated as a new input
            self.context['is_reflection'] = True
            logger.info("SELF REFLECTION: Set is_reflection flag to True")
            
            # Format conversation history
            conversation_text = self._format_history(current_exchange)
            logger.info(f"SELF REFLECTION: Formatted history for reflection with {len(current_exchange)} exchanges")
            
            # Step 1: Get surprise score
            score_response = self._get_llm_response(self.surprise_score_prompt, conversation_text)
            surprise_score = self._extract_score(score_response)
            logger.info(f"SELF REFLECTION: Extracted surprise score: {surprise_score}")
            
            # Step 2: Check if web search is needed - BUT DON'T ACTUALLY PERFORM IT NOW
            # This was causing a feedback loop - disable automatic web search during reflection
            """
            search_needed, search_query = self._evaluate_web_search_need(conversation_text)
            web_memory = None
            """
            # Disable automatic web search during reflection to prevent feedback loop
            search_needed = False
            search_query = None
            web_memory = None
            logger.info("SELF REFLECTION: Web search explicitly disabled to prevent feedback loops")
            
            # Step 3: Determine memory type and expiry for conversation
            memory_type, expiry = self._determine_memory_type(surprise_score)
            logger.info(f"SELF REFLECTION: Determined memory type: {memory_type}, expiry: {expiry}")
            
            # Step 4: Process memory if score warrants retention
            memory_data = None
            
            if surprise_score >= self.surprise_thresholds['low']:
                logger.info(f"SELF REFLECTION: Score {surprise_score} >= threshold {self.surprise_thresholds['low']}, processing memory")
                memory_prompt = self.memory_prompts[memory_type]
                logger.info(f"SELF REFLECTION: Using {memory_type} memory prompt")
                memory_data = self._get_llm_response(memory_prompt, conversation_text)
                logger.info(f"SELF REFLECTION: Generated memory: {memory_data[:200]}...")
                
                # Save initial memory first
                if memory_data:
                    current_time = self.context.get('current_time')
                    if isinstance(current_time, str):
                        current_time = self._parse_timestamp(current_time)
                        
                    metadata = {
                        'memory_type': memory_type,
                        'expiry_date': expiry.isoformat() if expiry else None,
                        'surprise_score': surprise_score,
                        'timestamp': current_time.isoformat()
                    }
                    self.history_manager.add_reflection(memory_data, context=metadata)
                    
                    # Add to vector store
                    memory_doc = Document(
                        page_content=memory_data,
                        metadata={
                            'source': 'conversation',
                            'type': memory_type,
                            'expiry_date': metadata['expiry_date'],
                            'surprise_score': surprise_score,
                            'timestamp': metadata['timestamp'],
                            'content_type': 'memory'
                        }
                    )
                    vector_store = self.context.get('vector_store')
                    if vector_store:
                        logger.info(f"SELF REFLECTION: Vector store before adding memory - docstore size: {len(vector_store.docstore._dict)}")
                        logger.info(f"SELF REFLECTION: Paths - FAISS: {FAISS_INDEX_PATH}, META: {METADATA_PATH}, DOCSTORE: {DOCSTORE_PATH}")
                        
                        # First, add the document to the vector store
                        logger.info(f"SELF REFLECTION: Adding memory doc with content: {memory_doc.page_content[:100]}...")
                        vector_store.add_documents([memory_doc])
                        
                        # Verify document was added
                        logger.info(f"SELF REFLECTION: Vector store after adding memory - docstore size: {len(vector_store.docstore._dict)}")
                        logger.info(f"SELF REFLECTION: Added conversation memory to embeddings")
                        
                        # Convert relative paths to absolute paths
                        from vector_store_utils import calculate_file_paths
                        import os  # Ensure os is imported in this scope
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        abs_faiss_path, abs_metadata_path, abs_docstore_path = calculate_file_paths(
                            script_dir, FAISS_INDEX_PATH, METADATA_PATH, DOCSTORE_PATH
                        )
                        
                        logger.info(f"SELF REFLECTION: Using absolute paths - FAISS: {abs_faiss_path}, META: {abs_metadata_path}, DOCSTORE: {abs_docstore_path}")
                        
                        # Now save the updated vector store to disk
                        save_faiss_index_metadata_and_docstore(
                            vector_store.index,
                            vector_store.index_to_docstore_id,
                            vector_store.docstore,
                            abs_faiss_path,
                            abs_metadata_path, 
                            abs_docstore_path
                        )
                        
                        # Verify file was saved and has content
                        docstore_size = os.path.getsize(abs_docstore_path) if os.path.exists(abs_docstore_path) else 0
                        logger.info(f"SELF REFLECTION: Saved updated vector store to disk - docstore file size: {docstore_size} bytes")
                        
                        # Process web search if needed and score warrants it
                        if search_needed and search_query and surprise_score >= self.surprise_thresholds['low']:
                            logger.info("SELF REFLECTION: Performing web search for: " + search_query)
                            
                            # Get web search results
                            from CognitiveProcessing import perform_web_search, get_detailed_web_content
                            search_results = perform_web_search(search_query, self.context.get('ddgs'))
                            
                            if search_results:
                                # Get detailed content
                                web_results = get_detailed_web_content(search_results, search_query, self.context)
                                if web_results:
                                    # Process web search results
                                    web_memory_type, web_memory_content = self._process_web_search_results(search_query, web_results)
                                    if web_memory_content:
                                        current_time = self.context.get('current_time')
                                        if isinstance(current_time, str):
                                            current_time = self._parse_timestamp(current_time)
                                            
                                        web_memory = {
                                            'content': web_memory_content,
                                            'type': web_memory_type,
                                            'query': search_query,
                                            'timestamp': current_time.isoformat()
                                        }
            else:
                logger.info(f"SELF REFLECTION: Score {surprise_score} below threshold {self.surprise_thresholds['low']}, skipping memory processing")

            # Save web memory if it exists
            if web_memory:
                web_metadata = {
                    'memory_type': web_memory['type'],
                    'expiry_date': self._calculate_expiry(web_memory['type']).isoformat() if self._calculate_expiry(web_memory['type']) else None,
                    'timestamp': web_memory['timestamp'],
                    'query': web_memory['query']
                }
                self.history_manager.add_reflection(web_memory['content'], context=web_metadata)
                
                # Add to vector store
                web_doc = Document(
                    page_content=web_memory['content'],
                    metadata={
                        'source': 'web_search',
                        'type': web_memory['type'],
                        'expiry_date': web_metadata['expiry_date'],
                        'query': web_memory['query'],
                        'timestamp': web_memory['timestamp'],
                        'content_type': 'web_memory'
                    }
                )
                if vector_store:
                    vector_store.add_documents([web_doc])
                    logger.info(f"SELF REFLECTION: Added web search memory to embeddings")
                
                # Save updated vector store
                if vector_store:
                    logger.info(f"SELF REFLECTION: About to save vector store to disk - docstore size: {len(vector_store.docstore._dict)}")
                    logger.info(f"SELF REFLECTION: Paths - FAISS: {FAISS_INDEX_PATH}, META: {METADATA_PATH}, DOCSTORE: {DOCSTORE_PATH}")
                    
                    # Convert relative paths to absolute paths
                    from vector_store_utils import calculate_file_paths
                    import os  # Ensure os is imported in this scope
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    abs_faiss_path, abs_metadata_path, abs_docstore_path = calculate_file_paths(
                        script_dir, FAISS_INDEX_PATH, METADATA_PATH, DOCSTORE_PATH
                    )
                    
                    logger.info(f"SELF REFLECTION: Using absolute paths - FAISS: {abs_faiss_path}, META: {abs_metadata_path}, DOCSTORE: {abs_docstore_path}")
                    
                    save_faiss_index_metadata_and_docstore(
                        vector_store.index,
                        vector_store.index_to_docstore_id,
                        vector_store.docstore,
                        abs_faiss_path,
                        abs_metadata_path, 
                        abs_docstore_path
                    )
                    
                    # Verify file was saved and has content
                    import os
                    docstore_size = os.path.getsize(abs_docstore_path) if os.path.exists(abs_docstore_path) else 0
                    logger.info(f"SELF REFLECTION: Saved updated vector store to disk - docstore file size: {docstore_size} bytes")
                    
            logger.info(f"SELF REFLECTION: Processed conversation: memory_type={memory_type}, score={surprise_score}, "
                       f"memory_length={len(memory_data) if memory_data else 0}, "
                       f"web_memory={'yes' if web_memory else 'no'}")

        except Exception as e:
            logger.error(f"SELF REFLECTION ERROR: Error in conversation processing: {e}")
            raise
        finally:
            # Reset the reflection flag when done, regardless of success or failure
            self.context['is_reflection'] = False
            logger.info("SELF REFLECTION: Reset is_reflection flag to False")

    def _clean_message(self, msg: str) -> str:
        """Clean a message by removing timestamps and extra formatting"""
        # Remove timestamp pattern [Day, YYYY-MM-DD HH:MM:SS +ZZZZ]
        msg = re.sub(r'\[[^]]*\]', '', msg)
        # Remove User/Bot prefix if present
        msg = re.sub(r'^(User|Bot):\s*', '', msg)
        return msg.strip()

    def _format_history(self, history: List[Tuple[str, str]]) -> str:
        """Format conversation history into a readable string."""
        if not history:  # Handle None or empty history
            return ""
            
        formatted = []
        for msg in history:
            # Handle nested list format from interface
            if isinstance(msg, list):
                user_msg, bot_msg = msg
                formatted.extend([
                    f"User: {self._clean_message(user_msg)}",
                    f"Bot: {self._clean_message(bot_msg)}"
                ])
            # Handle tuple format
            elif isinstance(msg, tuple):
                user_msg, bot_msg = msg
                if not user_msg.startswith("User: "):
                    user_msg = f"User: {user_msg}"
                if not bot_msg.startswith("Bot: "):
                    bot_msg = f"Bot: {bot_msg}"
                formatted.append(user_msg)
                formatted.append(bot_msg)
            # Handle string format
            else:
                formatted.append(msg)
        
        return "\n".join(formatted)

    def start_reflection_thread(self) -> None:
        """Start the reflection thread"""
        if not self.reflection_thread or not self.reflection_thread.is_alive():
            self.stop_reflection.clear()
            self.reflection_thread = threading.Thread(target=self._reflection_loop)
            self.reflection_thread.start()

    def stop_reflection_thread(self) -> None:
        """Stop the reflection thread"""
        if self.reflection_thread and self.reflection_thread.is_alive():
            self.stop_reflection.set()
            self.reflection_thread.join()

    def pause_reflection(self):
        """Pause the reflection process"""
        logger.info("SELF REFLECTION: Self reflection paused")
        self.pause_event.set()
        
    def resume_reflection(self):
        """Resume the reflection process"""
        logger.info("SELF REFLECTION: Self reflection resumed")
        self.pause_event.clear()

    def _reflection_loop(self) -> None:
        """Main reflection loop"""
        while not self.stop_reflection.is_set():
            try:
                # Check if we're paused
                if self.pause_event.is_set():
                    logger.debug("SELF REFLECTION: Self reflection is paused, sleeping...")
                    time.sleep(1)
                    continue
                    
                # Process any queued conversations
                try:
                    history = self.user_input_queue.get(timeout=1)
                    self.is_reflecting = True
                    self.process_conversation(history)
                    self.is_reflecting = False
                except Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"SELF REFLECTION ERROR: Error in reflection loop: {str(e)}")
                logger.error(traceback.format_exc())
                self.is_reflecting = False
                time.sleep(5)  # Wait a bit before retrying

    def queue_conversation(self, history: List[Tuple[str, str]]) -> None:
        """Queue a conversation for reflection"""
        if history:
            self.user_input_queue.put(history)

    def notify_user_input(self):
        """Notify that user input has been received"""
        logger.info("SELF REFLECTION: Notifying about user input")
        self.stop_reflection_thread()  # Stop current reflection if any
        self.user_input_queue.put(None)  # Signal to stop current processing

    def get_current_reflections(self):
        """Get current reflections from history manager"""
        return self.history_manager.get_current_reflections()

    def store_insight(self, category, insight, context=None):
        """Store a significant insight from the conversation.
        
        Args:
            category (str): Type of insight (e.g., 'knowledge', 'pattern', 'preference')
            insight (str): The actual insight to store
            context (dict, optional): Additional context about the insight
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        insight_data = {
            'timestamp': timestamp,
            'category': category,
            'insight': insight,
            'context': context or {}
        }
        self.history_manager.add_insight(insight_data)
        logger.info(f"SELF REFLECTION: Stored new insight in category '{category}': {insight}")

    def _should_continue_reflecting(self, messages, reflection_history):
        """Determine if additional reflection is valuable"""
        # Check system resources first
        if is_gpu_too_hot():
            logger.warning("SELF REFLECTION: GPU temperature too high, stopping reflection")
            return False
            
        # Stop after max reflections
        if len(reflection_history) >= 3:
            return False
            
        # Always do at least one reflection
        if not reflection_history:
            return True
            
        # Check previous reflections for diminishing returns
        if len(reflection_history) >= 2:
            last_two = [r[1] for r in reflection_history[-2:]]
            # Check for similarity in themes
            if self._are_reflections_similar(last_two[0], last_two[1]):
                return False
                
        return True

    def _are_reflections_similar(self, ref1, ref2):
        """Check if two reflections cover similar themes"""
        # Extract aspects
        aspect1 = ref1.split('**')[1] if '**' in ref1 else ''
        aspect2 = ref2.split('**')[1] if '**' in ref2 else ''
        
        # Check for theme similarity
        similar_themes = [
            ('knowledge', 'learning', 'understanding'),
            ('interaction', 'communication', 'response'),
            ('improvement', 'enhancement', 'adaptation'),
            ('pattern', 'behavior', 'preference')
        ]
        
        for theme_group in similar_themes:
            if (any(t in aspect1.lower() for t in theme_group) and
                any(t in aspect2.lower() for t in theme_group)):
                return True
                
        return False

    def _format_history(self, history):
        """Format chat history into a readable string."""
        if not history:  # Handle None or empty history
            return ""
            
        formatted = []
        for msg in history:
            # Handle nested list format from interface
            if isinstance(msg, list):
                user_msg, bot_msg = msg
                formatted.append(f"User: {user_msg}")
                formatted.append(f"Bot: {bot_msg}")
            # Handle tuple format
            elif isinstance(msg, tuple):
                user_msg, bot_msg = msg
                if not user_msg.startswith("User: "):
                    user_msg = f"User: {user_msg}"
                if not bot_msg.startswith("Bot: "):
                    bot_msg = f"Bot: {bot_msg}"
                formatted.append(user_msg)
                formatted.append(bot_msg)
            # Handle string format
            else:
                formatted.append(msg)
        
        return "\n".join(formatted)

    def _get_message_content(self, message):
        """Extract actual message content without User:/Bot: prefix"""
        if isinstance(message, list):
            return message[1]  # Get bot message from list format
        if isinstance(message, tuple):
            return message[1].replace("Bot: ", "")
        return message.replace("User: ", "").replace("Bot: ", "")

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp with flexible format handling"""
        try:
            # Try different timestamp formats
            formats = [
                '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone
                '%Y-%m-%d %H:%M:%S%z',  # Standard format with timezone
                '%Y-%m-%d %H:%M:%S',     # Without timezone
            ]
            
            # Clean up the timestamp string
            timestamp_str = timestamp_str.strip()
            
            # Handle timezone separately if it exists
            if '+' in timestamp_str:
                main_part, tz_part = timestamp_str.rsplit('+', 1)
                if ':' in tz_part:  # Handle +HH:MM format
                    tz_part = tz_part.replace(':', '')
                timestamp_str = f"{main_part}+{tz_part}"
            
            # Try each format
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If none of the formats work, try parsing with dateutil
            from dateutil import parser
            return parser.parse(timestamp_str)
            
        except Exception as e:
            logging.error(f"SELF REFLECTION ERROR: Error parsing timestamp {timestamp_str}: {str(e)}")
            # Return current time as fallback
            return datetime.now()

    def start_reflection(self, history: List[Tuple[str, str]], callback: Callable[[str], None] = None) -> None:
        """Start an asynchronous reflection on the conversation history"""
        try:
            # Prevent multiple reflections from starting at once
            if self.is_reflecting:
                logger.info("SELF REFLECTION: Reflection already in progress, skipping")
                return
                
            # Set reflecting flag
            self.is_reflecting = True
            logger.info("=== SELF REFLECTION: Starting new reflection process ===")
            
            # Play self-reflection sound if continuous_listener is available
            try:
                continuous_listener = self.context.get('continuous_listener')
                if continuous_listener and hasattr(continuous_listener, 'play_self_reflection_sound'):
                    logger.info("SELF REFLECTION: Playing self-reflection sound")
                    continuous_listener.play_self_reflection_sound()
            except Exception as e:
                logger.error(f"SELF REFLECTION: Error playing self-reflection sound: {e}")
            
            # Copy history to prevent modification
            history_copy = list(history)
            
            # Skip if empty history or paused
            if not history_copy or self.pause_event.is_set():
                logger.info("SELF REFLECTION: No history or reflection paused, skipping reflection")
                self.is_reflecting = False
                return
                
            # Get the latest exchange only - just the last turn
            if len(history_copy) > 0:
                latest_exchange = [history_copy[-1]]
                logger.info(f"SELF REFLECTION: Processing only the latest exchange: {latest_exchange[0][0][:50]}...")
            else:
                latest_exchange = []
                logger.info("SELF REFLECTION: No exchanges to process")
                
            # Process in a thread to not block main operation
            def process_thread():
                try:
                    logger.info("SELF REFLECTION: Thread started for processing reflection")
                    # Process the single latest exchange only
                    self.process_conversation(latest_exchange)
                    logger.info("SELF REFLECTION: Thread completed processing reflection")
                    # IMPORTANT: Set callback to None to prevent the reflection from affecting the main flow
                    # callback(result)  # DISABLED - Don't call back with reflection results
                    
                finally:
                    self.is_reflecting = False
                    logger.info("SELF REFLECTION: Reflection process completed")
                    
            # Start thread
            self.reflection_thread = threading.Thread(target=process_thread)
            self.reflection_thread.daemon = True
            self.reflection_thread.start()
            logger.info("SELF REFLECTION: Reflection thread started")
            
        except Exception as e:
            logger.error(f"SELF REFLECTION ERROR: Error starting reflection: {e}")
            self.is_reflecting = False

import threading
import time
import logging
import json
from queue import Queue, Empty
from chatbot_functions import chatbot_response
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import pytz
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfReflection:
    def __init__(self, context):
        """Initialize self reflection manager"""
        self.context = context
        self.reflection_thread = None
        self.stop_reflection = threading.Event()
        self.user_input_queue = Queue()
        self.is_reflecting = False
        
        # Initialize history manager
        from self_reflection_history import SelfReflectionHistoryManager
        self.history_manager = SelfReflectionHistoryManager()
        self.history_manager.start_new_session()
        context['history_manager'] = self.history_manager  # Add to context for other components
        
        self.embedding_updater = context['embedding_updater']

        # Memory expiry settings (in days)
        self.memory_expiry = {
            'long_term': None,  # Never expires
            'mid_term': 30,     # 30 days
            'short_term': 7     # 7 days
        }

        self.surprise_thresholds = {
            'high': 0.8,   # Score >= 0.8 goes to long-term
            'medium': 0.5  # Score >= 0.5 goes to mid-term, below goes to short-term
        }

        # Simple prompt focused on getting a single numerical score
        self.surprise_score_prompt = (
            "Rate how surprising and memorable this conversation is on a scale from 0.0 to 1.0.\n"
            "Consider:\n"
            "- How unexpected or novel was the interaction?\n"
            "- Was important or unique information shared?\n"
            "- Was there significant emotional content?\n\n"
            "Respond with ONLY a number between 0.0 and 1.0. For example: 0.7"
        )

        # Separate prompt for reasoning (optional, only if score is high)
        self.reasoning_prompt = (
            "Explain in one short sentence why this conversation received a score of {score}."
        )

    def _extract_score(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        try:
            # Clean the response and extract the first number
            response = response.strip()
            # Look for a decimal number between 0 and 1
            import re
            matches = re.findall(r'0?\.[0-9]+', response)
            if matches:
                score = float(matches[0])
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            return 0.0
        except Exception as e:
            logger.error(f"Error extracting score: {e}")
            return 0.0

    def _determine_memory_type(self, surprise_score: float) -> Tuple[str, Optional[datetime]]:
        """Determine memory type and expiry based on surprise score"""
        current_time = datetime.now(pytz.timezone('Australia/Adelaide'))
        
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

    def process_conversation(self, current_exchange: List[Tuple[str, str]]) -> None:
        """Process a conversation exchange and generate reflection"""
        try:
            logger.info("Starting conversation processing")
            
            # Format conversation history
            conversation_text = self._format_history(current_exchange)
            
            # Step 1: Get surprise score
            score_response = self._get_llm_response(self.surprise_score_prompt, conversation_text)
            surprise_score = self._extract_score(score_response)
            
            # Step 2: Get reasoning if score is high enough
            reasoning = ""
            if surprise_score >= self.surprise_thresholds['medium']:
                reasoning_prompt = self.reasoning_prompt.format(score=surprise_score)
                reasoning = self._get_llm_response(reasoning_prompt, conversation_text)

            # Determine memory type and expiry
            memory_type, expiry = self._determine_memory_type(surprise_score)

            # Save reflection with metadata
            self.history_manager.add_reflection(
                conversation_text,  # Store the actual conversation
                context={
                    'memory_type': memory_type,
                    'expiry_date': expiry.isoformat() if expiry else None,
                    'surprise_score': surprise_score,
                    'reasoning': reasoning,
                    'timestamp': datetime.now(pytz.timezone('Australia/Adelaide')).isoformat(),
                    'original_text': self._format_history(current_exchange)  # Store original for reference
                }
            )

            logger.info(f"Processed conversation: memory_type={memory_type}, score={surprise_score}")

        except Exception as e:
            logger.error(f"Error in conversation processing: {e}")
            raise

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
                formatted.extend([
                    f"User: {self._clean_message(user_msg)}",
                    f"Bot: {self._clean_message(bot_msg)}"
                ])
            # Handle string format
            else:
                formatted.append(self._clean_message(msg))
        
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

    def _reflection_loop(self) -> None:
        """Main reflection loop"""
        while not self.stop_reflection.is_set():
            try:
                # Get next conversation to process
                current_exchange = self.user_input_queue.get(timeout=1)
                
                # Check for stop signal
                if current_exchange is None:
                    continue
                    
                self.is_reflecting = True
                
                # Process the conversation
                self.process_conversation(current_exchange)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in reflection loop: {e}")
            finally:
                self.is_reflecting = False

    def queue_conversation(self, history: List[Tuple[str, str]]) -> None:
        """Queue a conversation for reflection"""
        if history:
            self.user_input_queue.put(history)

    def notify_user_input(self):
        """Notify that user input has been received"""
        logger.info("Notifying about user input")
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
        logger.info(f"Stored new insight in category '{category}': {insight}")

    def _should_continue_reflecting(self, messages, reflection_history):
        """Determine if additional reflection is valuable"""
        # Check system resources first
        if is_gpu_too_hot():
            logger.warning("GPU temperature too high, stopping reflection")
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

    def _parse_timestamp(self, timestamp_str: str) -> str:
        """Parse timestamp with flexible format handling"""
        try:
            # Try different timestamp formats
            formats = [
                '%Y-%m-%d %H:%M:%S%z',  # Standard format with timezone
                '%Y-%m-%d %H:%M:%S %z',  # With space before timezone
                '%Y-%m-%d %H:%M:%S',     # Without timezone
                '%Y-%m-%dT%H:%M:%S%z',   # ISO format
                '%A, %Y-%m-%d %H:%M:%S %z'  # With weekday
            ]
            
            # Clean up the timestamp string
            timestamp_str = timestamp_str.strip()
            
            # Try each format
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.strftime('%Y-%m-%d %H:%M:%S%z')
                except ValueError:
                    continue
                    
            # If none of the formats work, try parsing with regex
            match = re.match(r'.*?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*([+-]\d{4})?', timestamp_str)
            if match:
                dt_str = match.group(1)
                tz = match.group(2) if match.group(2) else '+0000'
                dt = datetime.strptime(f"{dt_str} {tz}", '%Y-%m-%d %H:%M:%S %z')
                return dt.strftime('%Y-%m-%d %H:%M:%S%z')
                
            raise ValueError(f"Could not parse timestamp: {timestamp_str}")
            
        except Exception as e:
            logger.error(f"Error parsing timestamp {timestamp_str}: {str(e)}")
            # Return current time as fallback
            return datetime.now(pytz.timezone('Australia/Adelaide')).strftime('%Y-%m-%d %H:%M:%S%z')

    def start_reflection(self, messages, update_ui_callback=None):
        """
        Start the reflection process on recent messages.
        Args:
            messages: List of recent messages to reflect on
            update_ui_callback: Optional callback to update the UI with reflection progress
        """
        if not self.reflection_thread or not self.reflection_thread.is_alive():
            self.start_reflection_thread()
        
        # Queue the messages for processing
        if messages:
            self.queue_conversation(messages)
            
        logger.info("Queued messages for reflection")

import threading
import time
import logging
import subprocess
import json
from queue import Queue
from chatbot_functions import retrieve_and_format_references, chatbot_response, determine_and_perform_web_search
import gradio as gr
from typing import Optional, Dict, Any, List, Tuple
from gpu_utils import is_gpu_too_hot
from self_reflection_history import SelfReflectionHistoryManager
from datetime import datetime
import pytz

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
        self.history_manager = SelfReflectionHistoryManager()
        self.history_manager.start_new_session()
        self.embedding_updater = context['embedding_updater']
        self.reflection_system_prompt = (
            "You are an AI system focused on continuous learning and adaptation through conversation analysis."
            "Generate concise, actionable insights while following these guidelines:\n\n"
            "1. Knowledge Enhancement:\n"
            "   - New information learned from conversation\n"
            "   - Topics requiring deeper research\n"
            "   - Connections to existing knowledge\n\n"
            "2. User Understanding:\n"
            "   - Preferences and interests identified\n"
            "   - Communication style and needs\n"
            "   - Domain expertise level\n\n"
            "3. Conversation Analysis:\n"
            "   - Response accuracy and relevance\n"
            "   - Information gaps identified\n"
            "   - Search effectiveness\n\n"
            "4. Improvement Strategy:\n"
            "   - Specific action to enhance knowledge\n"
            "   - Topics to research further\n"
            "   - Clear success metrics\n\n"
            "Keep reflections under 100 words. Focus on learning and adaptation."
        )
        logger.info("SelfReflection initialized")

    def start_reflection(self, chat_history, update_ui_callback):
        """Start the self-reflection process with improved focus and efficiency"""
        if self.is_reflecting:
            logger.info("Already reflecting, skipping new reflection")
            return
            
        # Clear previous user input signals
        while not self.user_input_queue.empty():
            self.user_input_queue.get()
            
        # Reset stop signal
        self.stop_reflection.clear()
        
        # Get recent conversation context
        current_exchange = chat_history[-1:]
        
        def reflection_loop():
            try:
                reflection_count = 0
                reflection_history = []
                max_reflections = 1 # Limit reflections for efficiency
                
                while not self.stop_reflection.is_set() and reflection_count < max_reflections:
                    # Check system resources
                    if is_gpu_too_hot():
                        logger.warning("GPU temperature too high, pausing reflection")
                        break

                    if not self.user_input_queue.empty():
                        logger.info("User input detected, stopping reflection")
                        break
                    
                    # Generate focused reflection
                    logger.info("Generating focused reflection")
                    temp_context = self.context.copy()
                    temp_context['system_prompt'] = self.reflection_system_prompt
                    temp_context['skip_web_search'] = True  # Prevent web searches during reflection
                    
                    # Create dynamic reflection prompt
                    temp_context['skip_web_search'] = True  # Prevent web searches during reflection prompt generation
                    reflection_prompt = self._create_reflection_prompt(current_exchange, reflection_history)
                    
                    # Get relevant context efficiently
                    refs, filtered_docs, context_documents = retrieve_and_format_references(
                        reflection_prompt, 
                        temp_context
                    )
                    
                    # Filter references for relevance
                    current_refs = self._filter_references(refs, current_exchange)
                    
                    # Generate reflection with filtered context
                    _, reflection, _, _ = chatbot_response(
                        reflection_prompt,
                        current_refs,
                        temp_context,
                        current_exchange
                    )
                    
                    logger.info(f"Generated reflection: {reflection[:200]}...")
                    
                    if self.stop_reflection.is_set():
                        logger.info("Stop signal received")
                        break
                    
                    # Format reflection with context
                    current_conversation = self._format_history(current_exchange)
                    full_reflection = f"Reflection #{reflection_count + 1}:\n{reflection}\n\nCurrent Conversation:\n{current_conversation}\n\nRelevant Context:\n{current_refs if current_refs else 'No additional context found.'}"
                    
                    # Save reflection with metadata
                    logger.info("Saving reflection to history")
                    self.history_manager.add_reflection(
                        reflection,
                        context={
                            "references": current_refs,
                            "prompt": reflection_prompt,
                            "reflection_number": reflection_count + 1,
                            "timestamp": self._parse_timestamp(datetime.now(pytz.timezone('Australia/Adelaide')).isoformat())
                        }
                    )
                    
                    # Add reflection to history
                    reflection_history.append((
                        f"Reflection #{reflection_count + 1}",
                        reflection
                    ))
                    
                    # Update UI if callback provided
                    try:
                        if update_ui_callback:
                            result = update_ui_callback(full_reflection)
                            logger.info(f"UI update result: {result}")
                    except Exception as e:
                        logger.error(f"Error updating UI: {str(e)}", exc_info=True)
                    
                    reflection_count += 1
                    
                    # Check if further reflection needed
                    if not self._should_continue_reflecting(chat_history, reflection_history):
                        logger.info("Reflection complete, no further insights needed")
                        break
                    
                    # Pause briefly between reflections
                    time.sleep(1)
                    
                # Update embeddings with all reflections at once
                if reflection_history:
                    logger.info(f"Updating embeddings with {len(reflection_history)} reflections")
                    try:
                        # Create a temporary state for reflection updates
                        reflection_state = {"last_processed_index": 0}
                        self.embedding_updater.update_chat_embeddings_async(reflection_history, reflection_state)
                        logger.info("Embeddings update started")
                    except Exception as e:
                        logger.error(f"Error updating embeddings: {str(e)}", exc_info=True)
                
            except Exception as e:
                logger.error(f"Error in reflection loop: {str(e)}", exc_info=True)
            finally:
                logger.info(f"Reflection loop ended after {reflection_count} reflections")
                self.is_reflecting = False
        
        # Start reflection in separate thread
        self.is_reflecting = True
        self.reflection_thread = threading.Thread(target=reflection_loop)
        self.reflection_thread.daemon = True
        self.reflection_thread.start()
        logger.info("Reflection thread started")

    def stop_reflection_loop(self):
        """Stop the self-reflection process"""
        logger.info("Stopping reflection loop")
        self.stop_reflection.set()
        if self.reflection_thread and self.reflection_thread.is_alive():
            self.reflection_thread.join(timeout=1)
        logger.info("Reflection loop stopped")

    def notify_user_input(self):
        """Notify that user input has been received"""
        logger.info("Notifying about user input")
        self.user_input_queue.put(True)
        self.stop_reflection_loop()

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

    def _extract_learning_topics(self, conversation):
        """Extract key topics that would be valuable to learn more about"""
        try:
            # Create a focused prompt for topic extraction
            prompt = (
                "Extract 1-3 specific learning topics from this conversation. For each topic:\n"
                "- Use 2-4 words only\n"
                "- Focus on concrete concepts\n"
                "- Avoid general categories\n\n"
                "Format: Return only the topics, one per line with a dash prefix.\n"
                "Example output:\n"
                "- neural network architectures\n"
                "- docker containerization\n"
                "- API authentication methods\n\n"
                f"Conversation:\n{self._format_history(conversation)}"
            )
            
            temp_context = self.context.copy()
            temp_context['system_prompt'] = (
                "You are a precise topic extractor. Return only the topics in the specified format. "
                "If no specific topics are found, return exactly: '- no specific topics'"
            )
            
            _, topics, _, _ = chatbot_response(prompt, "", temp_context, [])
            
            # Clean and validate the topics
            topic_lines = [t.strip() for t in topics.split('\n') if t.strip().startswith('-')]
            if not topic_lines or topic_lines[0] == '- no specific topics':
                return None
                
            return '\n'.join(topic_lines)
            
        except Exception as e:
            logger.error(f"Error extracting learning topics: {str(e)}", exc_info=True)
            return None

    def _perform_learning_search(self, topic, temp_context):
        """Perform a focused web search to learn about a specific topic"""
        try:
            if not topic:
                logger.info("No topic provided for learning search")
                return None
                
            # Validate topic before searching
            if len(topic.split()) > 10 or len(topic) > 100:
                logger.warning(f"Topic too long or complex: {topic}")
                return None
                
            logger.info(f"Starting learning search for topic: {topic}")
            # Create a new context specifically for learning search
            search_context = temp_context.copy()
            search_context['skip_web_search'] = False  # Enable web search for learning
            search_context['search_purpose'] = 'learning'
            search_context['max_search_results'] = 2
            
            # Log the search context
            logger.info(f"Search context: skip_web_search={search_context.get('skip_web_search')}, purpose={search_context.get('search_purpose')}")
            
            # Perform the search with the cleaned topic
            logger.info("Performing web search for learning")
            search_results = determine_and_perform_web_search(topic, "", search_context)
            
            # Log the search results structure
            logger.info(f"Search results type: {type(search_results)}")
            logger.info(f"Search results keys: {search_results.keys() if isinstance(search_results, dict) else 'Not a dict'}")
            
            if search_results and search_results.get('web_results'):
                # Truncate results if too long
                web_results = search_results['web_results']
                if len(web_results) > 500:
                    web_results = web_results[:497] + "..."
                    
                logger.info(f"Found learning content ({len(web_results)} chars)")
                logger.info(f"First 100 chars of content: {web_results[:100]}...")
                return web_results
            else:
                logger.info(f"No web results found. Search results: {search_results}")
            
            logger.info("No learning content found from web search")
            return None
            
        except Exception as e:
            logger.error(f"Error in learning search: {str(e)}", exc_info=True)
            return None

    def _create_reflection_prompt(self, history, reflection_history=[]):
        """
        Create a dynamic, psychologically-grounded prompt for self-reflection.
        Uses a meta-cognitive approach to generate contextually relevant prompts.
        """
        # First, generate a dynamic framework
        reflection_context = self.context.copy()
        reflection_context['skip_web_search'] = True
        self_reflection_framework = self._generate_meta_prompt(history, reflection_history)
        
        # Extract potential learning topic with a clean context
        learning_context = self.context.copy()
        learning_context['skip_web_search'] = False  # Allow web search for learning
        learning_context['search_purpose'] = 'learning'
        learning_topic = self._extract_learning_topics(history)
        learning_content = None
        
        if learning_topic:
            learning_content = self._perform_learning_search(learning_topic, learning_context)
            logger.info(f"Learning content found: {bool(learning_content)}")
        
        recent_reflections = reflection_history[-2:] if reflection_history else []
        reflection_count = len(reflection_history)
        
        # Extract key themes from previous reflections
        themes = self._extract_reflection_themes(recent_reflections)
        
        # Track which self-reflection aspects have been explored
        previous_aspects = [r[1].split('**')[1].split('**')[0] for r in reflection_history if '**' in r[1]]
        
        # Build the prompt parts separately
        framework_intro = f"Using this framework, provide a focused analysis:\n\n{self_reflection_framework}\n"
        
        aspect_section = "Choose ONE unexplored aspect and analyze briefly:\n\n"
        
        exchange_section = f"Current Exchange:\n{self._format_history(history)}\n"
        
        insights_section = f"Previous Insights:\n{', '.join(r[1] for r in recent_reflections) if recent_reflections else 'No previous reflections'}\n"
        
        themes_section = f"Key Themes Identified:\n{themes}\n"
        
        learning_section = f"New Learning about {learning_topic}:\n{learning_content}\n" if learning_content else ""
        
        guidelines = """Guidelines:
1. Choose ONE unexplored aspect
2. Start with "**[Selected Aspect]**"
3. Provide ONE specific observation
4. Include ONE concrete example
5. Suggest ONE actionable improvement
6. Keep response under 75 words
"""
        
        previous = f"Previously Explored: {', '.join(previous_aspects) if previous_aspects else 'None'}"
        
        # Combine all parts with proper spacing
        meta_cognitive_framework = "\n".join([
            framework_intro,
            aspect_section,
            exchange_section,
            insights_section,
            themes_section,
            learning_section,
            guidelines,
            previous
        ])

        try:
            # Get the dynamic prompt from the LLM
            temp_context = self.context.copy()
            temp_context['skip_web_search'] = True  # Prevent web searches during reflection prompt generation
            
            # Use direct LLM call without references for meta prompt
            _, generated_prompt, _, _ = chatbot_response(meta_cognitive_framework, "", temp_context, [])
            
            # Log everything for quality monitoring
            logger.info("\n=== LLM Generated Reflection Prompt ===")
            logger.info(f"Reflection #{reflection_count + 1}")
            logger.info("Previously Explored Aspects:")
            logger.info(previous_aspects)
            logger.info("\nSelf-Reflection Framework:")
            logger.info(self_reflection_framework)
            logger.info("\nGenerated Prompt:")
            logger.info(generated_prompt)
            logger.info("=====================================\n")
            
            # Extract the self-reflection aspect from the generated prompt
            aspect = "Self-Reflection Analysis"
            if "**" in generated_prompt:
                aspect = generated_prompt.split("**")[1].strip("[]")
            
            final_prompt = f"""Self-reflection through the lens of {aspect}...

{self._format_history(history)}

{generated_prompt}"""

            return final_prompt
        except Exception as e:
            logger.error(f"Error generating reflection prompt: {str(e)}", exc_info=True)
            # Fallback to a simpler prompt
            return f"""Analyze this interaction to generate ONE specific, actionable insight.
Focus on an unexplored aspect.

Current Exchange:
{self._format_history(history)}

Guidelines:
1. Choose ONE unexplored aspect
2. Start with "**[Selected Aspect]**"
3. Provide ONE specific observation
4. Include ONE concrete example
5. Suggest ONE actionable improvement
6. Keep response under 75 words"""

    def _generate_meta_prompt(self, history, reflection_history=[]):
        """
        Generate a dynamic meta-prompt based on psychological principles and current context.
        This allows the LLM to create its own introspection framework.
        """
        try:
            recent_history = history[-2:] if len(history) >= 2 else history
            reflection_count = len(reflection_history)
            
            meta_prompt_generator = """Create a focused framework for analyzing the conversation and extracting valuable learning opportunities.

Consider these aspects:
1. Information and knowledge gaps
2. User expertise and interests
3. Search and research opportunities
4. Response effectiveness
5. Learning potential

Structure (keep each section brief):
1. Main learning opportunity (choose ONE area from above)
2. 2-3 specific topics for research or clarification
3. Key information sources to consult
4. One concrete knowledge enhancement action

Keep the framework focused on continuous learning and improvement."""

            # Get the dynamic framework from the LLM
            temp_context = self.context.copy()
            temp_context['skip_web_search'] = True  # Prevent web searches during meta-prompt generation
            temp_context['system_prompt'] = (
                "You are an advanced AI system specializing in knowledge acquisition and learning from conversations. "
                "You create frameworks that combine multiple aspects: "
                "information gathering, topic exploration, user understanding, and continuous improvement through research and analysis."
            )
            
            # Use local context only, no web search needed for framework generation
            _, generated_framework, _, _ = chatbot_response(meta_prompt_generator, "", temp_context, recent_history)
            
            logger.info("\n=== Generated Meta-Framework ===")
            logger.info(generated_framework)
            logger.info("================================\n")
            
            return generated_framework
        except Exception as e:
            logger.error(f"Error generating meta-framework: {str(e)}", exc_info=True)
            # Return a default framework if generation fails
            return """Framework for Self-Reflection:
1. Main Aspect: Interaction Effectiveness
2. Key Questions:
   - What patterns emerged in the conversation?
   - How well did responses match user needs?
3. Principle: Active Listening and Adaptation
4. Improvement: Enhance response relevance"""

    def _extract_reflection_themes(self, reflections):
        """Extract key themes from previous reflections"""
        if not reflections:
            return "No themes identified yet"
            
        themes = []
        for _, reflection in reflections:
            # Extract aspect from reflection
            if '**' in reflection:
                aspect = reflection.split('**')[1]
                themes.append(aspect)
                
        return ', '.join(themes) if themes else "No clear themes identified"

    def _should_continue_reflecting(self, chat_history, reflection_history):
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

    def _filter_references(self, refs, current_messages):
        """Filter references to only include those containing current messages"""
        if not current_messages:
            return refs
            
        current_conversation_refs = []
        current_contents = [self._get_message_content(msg) for msg in current_messages]
        
        for ref in refs:
            if any(content in ref for content in current_contents):
                current_conversation_refs.append(ref)
        
        return current_conversation_refs if current_conversation_refs else refs

    def _parse_timestamp(self, timestamp_str):
        """Parse timestamp with timezone information"""
        try:
            # If timestamp is already a datetime object, convert to string
            if isinstance(timestamp_str, datetime):
                return timestamp_str.strftime('%Y-%m-%d %H:%M:%S%z')
                
            # Handle ISO format timestamps
            if 'T' in timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    return dt.strftime('%Y-%m-%d %H:%M:%S%z')
                except ValueError:
                    pass
            
            # Handle timezone offset in format +HHMM
            if '+' in timestamp_str:
                main_part, tz_part = timestamp_str.rsplit('+', 1)
                if ':' in tz_part:  # Handle +HH:MM format
                    tz_part = tz_part.replace(':', '')
                timestamp_str = f"{main_part}+{tz_part}"
            
            # Parse with timezone
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S%z')
            return dt.strftime('%Y-%m-%d %H:%M:%S%z')
        except ValueError as e:
            logger.error(f"Error parsing timestamp {timestamp_str}: {str(e)}")
            # Return current time as fallback
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')

import threading
import time
import logging
from queue import Queue
from chatbot_functions import retrieve_and_format_references, chatbot_response
import gradio as gr
from self_reflection_history import SelfReflectionHistoryManager

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
            "You are engaged in private self-reflection, similar to writing quick thoughts in a journal. Express your thoughts "
            "naturally and concisely while following these guidelines:\n\n"
            "1. Keep reflections brief - capture the essence in 1-2 sentences\n"
            "2. Focus on one key insight or realization at a time\n"
            "3. Be direct and specific about what you've noticed\n"
            "4. Stay internal - don't address others or ask questions\n"
            "5. Be honest about limitations and growth\n\n"
            "Example format:\n"
            "Noticed a pattern of using clever analogies instead of speaking directly... might be trying too hard to impress "
            "rather than just being genuine."
        )
        logger.info("SelfReflection initialized")

    def start_reflection(self, chat_history, update_ui_callback):
        """
        Start the self-reflection process in a separate thread.
        
        Args:
            chat_history (list): Current chat history
            update_ui_callback (callable): Callback to update the UI with new reflections
        """
        if self.is_reflecting:
            logger.info("Already reflecting, skipping new reflection")
            return
            
        # Clear any previous user input signals
        while not self.user_input_queue.empty():
            self.user_input_queue.get()
            
        # Reset stop signal
        self.stop_reflection.clear()
        
        # Get the most recent exchange (the last item in chat_history)
        current_exchange = chat_history[-1:]
        
        def reflection_loop():
            try:
                reflection_count = 0
                reflection_history = []
                MAX_REFLECTIONS = 7
                
                while not self.stop_reflection.is_set() and reflection_count < MAX_REFLECTIONS:
                    if not self.user_input_queue.empty():
                        logger.info("User input detected, stopping reflection")
                        break
                    
                    # Generate reflection using standard method
                    logger.info("Generating reflection")
                    temp_context = self.context.copy()
                    temp_context['system_prompt'] = self.reflection_system_prompt
                    
                    # Create reflection prompt with current conversation
                    reflection_prompt = self._create_reflection_prompt(current_exchange, reflection_history)
                    
                    # Get and filter references
                    refs, filtered_docs, context_documents = retrieve_and_format_references(reflection_prompt, self.context)
                    current_conversation_refs = self._filter_references(refs, current_exchange)
                    
                    _, reflection, _ = chatbot_response(reflection_prompt, current_conversation_refs, temp_context, current_exchange)
                    logger.info(f"Generated reflection text: {reflection[:200]}...")
                    
                    if self.stop_reflection.is_set():
                        logger.info("Stop signal received after generating reflection")
                        break
                    
                    # Format reflection with context and current conversation
                    current_conversation = self._format_history(current_exchange)
                    full_reflection = f"Reflection #{reflection_count + 1}:\n{reflection}\n\nCurrent Conversation:\n{current_conversation}\n\nRelevant Context:\n{current_conversation_refs if current_conversation_refs else 'No additional context found.'}"
                    logger.info(f"Generated full reflection: {full_reflection[:200]}...")
                    
                    # Save reflection to history manager
                    logger.info("Saving reflection to history")
                    self.history_manager.add_reflection(
                        reflection,
                        context={
                            "references": current_conversation_refs,
                            "prompt": f"Quick reflection on this recent interaction...\n\n{current_conversation}\n\n{reflection}",
                            "reflection_number": reflection_count + 1
                        }
                    )
                    
                    # Add reflection to history for embedding update
                    reflection_history.append((
                        f"Self-Reflection #{reflection_count + 1}",
                        reflection
                    ))
                    
                    try:
                        # Update UI with reflection
                        logger.info("Updating UI with reflection")
                        if update_ui_callback:
                            result = update_ui_callback(full_reflection)
                            logger.info(f"UI update result: {result}")
                    except Exception as e:
                        logger.error(f"Error updating UI: {str(e)}", exc_info=True)
                    
                    reflection_count += 1
                    
                    # Check if we should continue reflecting
                    continuation_prompt = self._should_continue_reflection(chat_history, reflection_history)
                    refs, filtered_docs, context_documents = retrieve_and_format_references(continuation_prompt, self.context)
                    temp_context = self.context.copy()
                    temp_context['system_prompt'] = self.reflection_system_prompt
                    _, decision, _ = chatbot_response(continuation_prompt, context_documents, temp_context, chat_history)
                    
                    logger.info(f"Continuation decision: {decision}")
                    
                    # Check for explicit stop signals
                    stop_signals = ["complete", "nothing more", "natural end", "finished", "that's all", "no other aspects"]
                    found_stop = any(signal in decision.lower() for signal in stop_signals)
                    
                    # More flexible continuation signals
                    continue_signals = [
                        "could explore", "worth exploring", "another angle", "also notice", "thinking about",
                        "interesting to", "might be worth", "curious about", "seems like", "notice that",
                        "reminds me", "stands out", "appears to", "suggests", "reveals"
                    ]
                    found_continue = any(signal in decision.lower() for signal in continue_signals)
                    
                    logger.info(f"Stop signals found: {found_stop}, Continue signals found: {found_continue}")
                    
                    # More flexible continuation logic
                    should_stop = False
                    if reflection_count < 2:
                        # Always continue for first two reflections unless explicit stop
                        should_stop = found_stop
                    elif reflection_count < 4:
                        # Between 2-4 reflections, continue unless stop signal or clearly no continuation
                        should_stop = found_stop or (not found_continue and "complete" in decision.lower())
                    else:
                        # After 4 reflections, require explicit continuation signal
                        should_stop = found_stop or not found_continue
                    
                    if should_stop:
                        logger.info(f"Stopping reflections: count={reflection_count}, found_stop={found_stop}, found_continue={found_continue}")
                        logger.info(f"Final decision: {decision}")
                        break
                    
                    logger.info("Continuing reflection based on: " + decision[:100])
                    # Brief pause between reflections
                    time.sleep(2)
                
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

    def _should_continue_reflection(self, chat_history, reflection_history):
        """
        Create a prompt to evaluate whether more reflection is needed.
        """
        recent_history = chat_history[-2:] if len(chat_history) >= 2 else chat_history
        recent_reflections = reflection_history[-3:] if reflection_history else []
        reflection_count = len(reflection_history)
        
        reflection_summary = "\n".join([f"Insight {i+1}: {r[1]}" for i, r in enumerate(recent_reflections)])
        
        # Adjust the continuation threshold based on reflection count
        if reflection_count >= 4:
            return f"""After several insights about THIS SPECIFIC interaction:

Recent Exchange (focus ONLY on this):
{self._format_history(recent_history)}

Previous Insights:
{reflection_summary}

Have we thoroughly examined this specific interaction from different angles? Only continue if there's a genuinely new perspective about THIS exchange to explore.
Important: Focus only on THIS conversation, not any past ones."""
        
        return f"""Considering these insights about THIS specific exchange:

Recent Exchange (focus ONLY on this):
{self._format_history(recent_history)}

Current Insights:
{reflection_summary}

Is there a different angle or deeper level of understanding we haven't explored about THIS specific interaction?
Important: Focus only on THIS conversation, not any past ones."""

    def _create_reflection_prompt(self, history, reflection_history=[]):
        """
        Create a prompt for self-reflection based on chat history and previous reflections.
        """
        # Get the most recent interaction (last user input and bot response)
        recent_history = history[-2:] if len(history) >= 2 else history
        recent_reflections = reflection_history[-3:] if reflection_history else []
        reflection_count = len(reflection_history)
        
        # Different focus areas based on reflection count
        if reflection_count == 0:
            focus = "What's the most immediate pattern or behavior that stands out in this interaction?"
        elif reflection_count == 1:
            focus = "Looking deeper than the surface patterns, what does this interaction reveal about communication style or effectiveness?"
        elif reflection_count == 2:
            focus = "Considering the previous insights, what might be a better approach or alternative way to handle this interaction?"
        else:
            focus = "What final insight or learning can be drawn from this exchange that hasn't been mentioned yet?"
        
        # Generate a focused prompt
        meta_prompt = f"""Examining ONLY this most recent exchange (ignore any previous conversations):
{self._format_history(recent_history)}

Previous insights about THIS exchange:
{", ".join(r[1] for r in recent_reflections) if recent_reflections else "No previous reflections yet"}

{focus}
Frame your response as a brief, specific insight that builds on (but doesn't repeat) previous reflections.
Important: Focus ONLY on this specific exchange, not on any past conversations."""

        # Get the dynamic prompt from the LLM
        refs, filtered_docs, context_documents = retrieve_and_format_references(meta_prompt, self.context)
        
        # Filter context to only include most recent conversation
        current_conversation_refs = self._filter_references(refs, history)
        
        temp_context = self.context.copy()
        temp_context['system_prompt'] = self.reflection_system_prompt
        _, generated_prompt, _ = chatbot_response(meta_prompt, current_conversation_refs, temp_context, history)
        
        return f"""Quick reflection on THIS recent interaction...

{self._format_history(recent_history)}

{generated_prompt}"""

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

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
            "You are engaged in sophisticated psychological self-reflection, analyzing interactions through established psychological frameworks. "
            "Express your insights clearly and thoughtfully while following these guidelines:\n\n"
            "1. Focus on one psychological dimension at a time (Metacognitive Awareness, Emotional Intelligence, Cognitive Processing, etc.)\n"
            "2. Analyze underlying patterns and mechanisms rather than surface observations\n"
            "3. Connect insights to specific elements of the interaction\n"
            "4. Maintain a growth-oriented, constructive perspective\n"
            "5. Format your response with the psychological dimension as a header (e.g., **[Emotional Intelligence]**)\n\n"
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
                MAX_REFLECTIONS = 3
                
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
                            "prompt": reflection_prompt,  # Use the actual psychological reflection prompt
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

    def _generate_meta_prompt(self, history, reflection_history=[]):
        """
        Generate a dynamic meta-prompt based on psychological principles and current context.
        This allows the LLM to create its own introspection framework.
        """
        recent_history = history[-2:] if len(history) >= 2 else history
        reflection_count = len(reflection_history)
        
        meta_prompt_generator = f"""As an AI engaged in psychological self-reflection, create a framework for analyzing the following interaction:

{self._format_history(recent_history)}

Previous reflections:
{', '.join(r[1] for r in reflection_history[-2:]) if reflection_history else 'None'}

Create a psychological introspection framework that:
1. Identifies key psychological dimensions relevant to this specific interaction
2. Proposes novel angles for self-analysis
3. Draws from established psychological theories and principles
4. Encourages deep, systematic analysis
5. Maintains focus on growth and improvement

Format your response as a structured framework with:
1. Key psychological dimensions to explore
2. Specific questions or prompts for each dimension
3. Theoretical foundations or principles being applied
4. Suggested areas for growth or adaptation

Your framework should be original and tailored to this specific interaction, not a generic template."""

        # Get the dynamic framework from the LLM
        refs, filtered_docs, context_documents = retrieve_and_format_references(meta_prompt_generator, self.context)
        temp_context = self.context.copy()
        temp_context['system_prompt'] = "You are a psychological framework designer, creating introspection frameworks based on established psychological principles."
        _, generated_framework, _ = chatbot_response(meta_prompt_generator, context_documents, temp_context, history)
        
        logger.info("\n=== Generated Meta-Framework ===")
        logger.info(generated_framework)
        logger.info("================================\n")
        
        return generated_framework

    def _create_reflection_prompt(self, history, reflection_history=[]):
        """
        Create a dynamic, psychologically-grounded prompt for self-reflection based on cognitive psychology principles.
        Uses a meta-cognitive approach to generate contextually relevant prompts.
        """
        # First, generate a dynamic framework
        psychological_framework = self._generate_meta_prompt(history, reflection_history)
        
        recent_history = history[-2:] if len(history) >= 2 else history
        recent_reflections = reflection_history[-3:] if reflection_history else []
        reflection_count = len(reflection_history)
        
        # Track which psychological dimensions have been explored
        previous_dimensions = [r[1].split('\n')[0].strip() for r in reflection_history if r[1].startswith('**')]
        
        meta_cognitive_framework = f"""Using this psychological framework:

{psychological_framework}

Analyze the following interaction through ONE unexplored psychological dimension:

Recent Exchange:
{self._format_history(recent_history)}

Previous insights about this exchange:
{", ".join(r[1] for r in recent_reflections) if recent_reflections else "No previous reflections yet"}

Guidelines for your analysis:
1. Choose ONE psychological dimension from the framework that hasn't been explored
2. Start your response with "**[Chosen Dimension]**:"
3. Apply the specific questions/prompts from the framework
4. Connect your analysis to concrete elements of the interaction
5. Maintain a constructive, growth-oriented perspective

Previously explored dimensions: {', '.join(previous_dimensions) if previous_dimensions else 'None'}"""

        # Get the dynamic prompt from the LLM
        refs, filtered_docs, context_documents = retrieve_and_format_references(meta_cognitive_framework, self.context)
        current_conversation_refs = self._filter_references(refs, history)
        
        temp_context = self.context.copy()
        temp_context['system_prompt'] = self.reflection_system_prompt
        _, generated_prompt, _ = chatbot_response(meta_cognitive_framework, current_conversation_refs, temp_context, history)
        
        # Log everything for quality monitoring
        logger.info("\n=== LLM Generated Reflection Prompt ===")
        logger.info(f"Reflection #{reflection_count + 1}")
        logger.info("Previously Explored Dimensions:")
        logger.info(previous_dimensions)
        logger.info("\nPsychological Framework:")
        logger.info(psychological_framework)
        logger.info("\nGenerated Prompt:")
        logger.info(generated_prompt)
        logger.info("=====================================\n")
        
        # Extract the psychological dimension from the generated prompt
        dimension = "Psychological Analysis"
        if "**" in generated_prompt:
            dimension = generated_prompt.split("**")[1].strip("[]")
        
        final_prompt = f"""Psychological reflection through the lens of {dimension}...

{self._format_history(recent_history)}

{generated_prompt}"""

        return final_prompt

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

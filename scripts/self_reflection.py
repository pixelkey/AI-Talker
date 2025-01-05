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
            
        self.is_reflecting = True
        self.stop_reflection.clear()
        logger.info("Starting new reflection process")
        
        def reflection_loop():
            try:
                reflection_count = 0
                reflection_history = []  # Keep track of reflections for embedding updates
                MAX_REFLECTIONS = 7  # Absolute maximum to prevent endless loops
                
                while not self.stop_reflection.is_set() and reflection_count < MAX_REFLECTIONS:
                    # Check for user input
                    if not self.user_input_queue.empty():
                        logger.info("User input detected, stopping reflection")
                        break
                    
                    # Generate a reflection prompt
                    logger.info("Generating reflection prompt")
                    reflection_prompt = self._create_reflection_prompt(chat_history, reflection_history)
                    logger.info(f"Prompt: {reflection_prompt[:200]}...")
                    
                    # Get relevant context using standard method
                    logger.info("Getting relevant context")
                    refs, filtered_docs, context_documents = retrieve_and_format_references(reflection_prompt, self.context)
                    logger.info(f"Got {len(context_documents) if context_documents else 0} context documents")
                    
                    if self.stop_reflection.is_set():
                        logger.info("Stop signal received after getting context")
                        break
                    
                    # Generate reflection using standard method
                    logger.info("Generating reflection")
                    temp_context = self.context.copy()
                    temp_context['system_prompt'] = self.reflection_system_prompt
                    _, reflection, _ = chatbot_response(reflection_prompt, context_documents, temp_context, chat_history)
                    logger.info(f"Generated reflection text: {reflection[:200]}...")
                    
                    if self.stop_reflection.is_set():
                        logger.info("Stop signal received after generating reflection")
                        break
                        
                    # Format reflection with context
                    full_reflection = f"Reflection #{reflection_count + 1}:\n{reflection}\n\nRelevant Context:\n{refs if refs else 'No additional context found.'}"
                    logger.info(f"Generated full reflection: {full_reflection[:200]}...")
                    
                    # Save reflection to history manager
                    logger.info("Saving reflection to history")
                    self.history_manager.add_reflection(
                        reflection,
                        context={
                            "references": refs,
                            "prompt": reflection_prompt,
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
                    stop_signals = ["complete", "nothing more", "natural end", "finished", "enough"]
                    found_stop = any(signal in decision.lower() for signal in stop_signals)
                    
                    # Check for continuation signals
                    continue_signals = ["could explore", "worth exploring", "another angle", "also notice", "thinking about", 
                                     "interesting to consider", "might be worth", "curious about"]
                    found_continue = any(signal in decision.lower() for signal in continue_signals)
                    
                    logger.info(f"Stop signals found: {found_stop}, Continue signals found: {found_continue}")
                    
                    # Stop if we find explicit stop signals or don't find any continuation signals
                    if found_stop or (not found_continue and reflection_count >= 2):
                        logger.info(f"Stopping reflections: count={reflection_count}, found_stop={found_stop}, found_continue={found_continue}")
                        logger.info(f"Final decision: {decision}")
                        break
                    
                    logger.info("Continuing reflection based on: " + decision[:100])
                    # Brief pause between reflections
                    time.sleep(2)
                    
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
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        recent_reflections = reflection_history[-3:] if reflection_history else []
        
        reflection_summary = "\n".join([f"Previous thought: {r[1]}" for i, r in enumerate(recent_reflections)])
        reflection_count = len(reflection_history)
        
        # Add gentle pressure to conclude as reflection count increases
        pressure = ""
        if reflection_count >= 5:
            pressure = "\n\nThese reflections are becoming quite extensive. Is there truly more to uncover?"
        
        return f"""Looking at these thoughts so far...

Recent Conversation:
{self._format_history(recent_history)}

Previous Thoughts:
{reflection_summary}

Take a moment to consider: What other aspects of this interaction feel worth exploring? If you see another angle to examine, describe it briefly. If nothing more stands out, simply acknowledge that the reflection feels complete.{pressure}"""

    def _create_reflection_prompt(self, history, reflection_history=[]):
        """
        Create a prompt for self-reflection based on chat history and previous reflections.
        """
        recent_history = history[-5:] if len(history) > 5 else history
        recent_reflections = reflection_history[-3:] if reflection_history else []
        
        # Generate a completely free-form prompt
        meta_prompt = f"""Looking at this recent interaction:
{self._format_history(recent_history)}

And these previous thoughts:
{", ".join(r[1] for r in recent_reflections) if recent_reflections else "No previous reflections yet"}

What single aspect of this interaction stands out as most noteworthy? Frame this as a brief, focused prompt for reflection."""

        # Get the dynamic prompt from the LLM
        refs, filtered_docs, context_documents = retrieve_and_format_references(meta_prompt, self.context)
        temp_context = self.context.copy()
        temp_context['system_prompt'] = self.reflection_system_prompt
        _, generated_prompt, _ = chatbot_response(meta_prompt, context_documents, temp_context, history)
        
        return f"""Quick reflection on a recent interaction...

{self._format_history(recent_history)}

{generated_prompt}"""

    def _format_history(self, history):
        """Format chat history for the reflection prompt"""
        formatted = []
        for speaker, message in history:
            formatted.append(f"{speaker}: {message}")
        return "\n".join(formatted)

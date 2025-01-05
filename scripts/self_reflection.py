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
                max_reflections = 3  # Limit the number of reflections per session
                
                while not self.stop_reflection.is_set() and reflection_count < max_reflections:
                    # Check for user input
                    if not self.user_input_queue.empty():
                        logger.info("User input detected, stopping reflection")
                        break
                        
                    # Generate a reflection prompt
                    logger.info("Generating reflection prompt")
                    reflection_prompt = self._create_reflection_prompt(chat_history)
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
                    _, reflection, _ = chatbot_response(reflection_prompt, context_documents, self.context, chat_history)
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
                    
                    try:
                        # Update UI with reflection
                        logger.info("Updating UI with reflection")
                        result = update_ui_callback(full_reflection)
                        logger.info(f"UI update result: {result}")
                    except Exception as e:
                        logger.error(f"Error updating UI: {str(e)}", exc_info=True)
                    
                    reflection_count += 1
                    
                    # Sleep briefly to prevent overwhelming the system
                    if reflection_count < max_reflections:
                        logger.info(f"Sleeping before next reflection ({reflection_count}/{max_reflections})")
                        time.sleep(5)  # Increased sleep time to make reflections more noticeable
                    
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

    def _create_reflection_prompt(self, history):
        """
        Create a prompt for self-reflection based on chat history.
        
        Args:
            history (list): Current chat history
            
        Returns:
            str: Reflection prompt
        """
        # Get last few interactions for context
        recent_history = history[-5:] if len(history) > 5 else history
        
        # Create a context-aware reflection prompt
        prompts = [
            "Based on our recent conversation and my knowledge base, what insights or patterns do I notice?",
            "What assumptions am I making that I should question, considering both our conversation and my stored knowledge?",
            "How can I better integrate my knowledge base with my responses?",
            "What connections can I draw between our discussion and related information in my knowledge base?",
            "What areas in my knowledge base should I explore further to provide better assistance?",
            "Are there any inconsistencies between my responses and my knowledge base that I should address?",
            "How can I better align my understanding from both conversation and stored knowledge to serve the user's needs?"
        ]
        
        # Rotate through different reflection prompts
        prompt_index = len(history) % len(prompts)
        chosen_prompt = prompts[prompt_index]
        
        return f"""As an AI assistant engaged in self-reflection, I will analyze the recent conversation and my knowledge base:
{self._format_history(recent_history)}

{chosen_prompt}

Let me reflect on this carefully, incorporating both our conversation history and relevant information from my knowledge base."""

    def _format_history(self, history):
        """Format chat history for the reflection prompt"""
        formatted = []
        for speaker, message in history:
            formatted.append(f"{speaker}: {message}")
        return "\n".join(formatted)

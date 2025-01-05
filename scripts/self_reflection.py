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
            "You are engaged in private self-reflection, similar to writing in a personal journal. Express your thoughts "
            "naturally while following these guidelines:\n\n"
            "1. Write in a personal, introspective style\n"
            "2. Focus on genuine insights and realizations\n"
            "3. Reflect on growth, challenges, and patterns\n"
            "4. Stay internal - don't address others or ask questions\n"
            "5. Be honest about limitations and areas for growth\n\n"
            "Example format:\n"
            "That response about fruit salad revealed something interesting... there's a tendency to fall back on safe analogies "
            "rather than expressing genuine thoughts. The marriage comparison felt forced, almost like trying too hard to be "
            "clever instead of just being authentic. There's room to be more natural, to let real personality shine through "
            "instead of relying on pre-packaged responses. Still working on finding that balance between being engaging and "
            "being genuine..."
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
                
                while not self.stop_reflection.is_set():
                    # Check for user input
                    if not self.user_input_queue.empty():
                        logger.info("User input detected, stopping reflection")
                        break
                        
                    # After completing the basic three phases, evaluate if more reflection is needed
                    if reflection_count >= 3:
                        continuation_prompt = self._should_continue_reflection(chat_history, reflection_history)
                        refs, filtered_docs, context_documents = retrieve_and_format_references(continuation_prompt, self.context)
                        temp_context = self.context.copy()
                        temp_context['system_prompt'] = self.reflection_system_prompt
                        _, decision, _ = chatbot_response(continuation_prompt, context_documents, temp_context, chat_history)
                        
                        if "no" in decision.lower() or "stop" in decision.lower() or "complete" in decision.lower():
                            logger.info("Reflection cycle complete, stopping based on self-evaluation")
                            break
                        logger.info("Continuing reflection based on self-evaluation")
                    
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
                    
                    # Sleep briefly to prevent overwhelming the system
                    if reflection_count < 3:
                        logger.info(f"Sleeping before next reflection ({reflection_count}/3)")
                        time.sleep(5)  # Increased sleep time to make reflections more noticeable
                    
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
        
        Args:
            chat_history (list): Current chat history
            reflection_history (list): Previous reflections
            
        Returns:
            str: Evaluation prompt
        """
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        recent_reflections = reflection_history[-3:] if reflection_history else []
        
        reflection_summary = "\n".join([f"Previous Reflection {i+1}: {r[1]}" for i, r in enumerate(recent_reflections)])
        
        return f"""Self-Development Continuation Assessment

Recent Interactions:
{self._format_history(recent_history)}

Previous Reflections:
{reflection_summary}

Consider these aspects:
1. Unexplored areas of personality development
2. Opportunities for deeper self-understanding
3. Potential for meaningful growth
4. New patterns in personal expression

Based on these considerations, evaluate whether additional reflection would contribute to personal growth and development. Provide decision (yes/no) with reasoning."""

    def _create_reflection_prompt(self, history, reflection_history=[]):
        """
        Create a prompt for self-reflection based on chat history and previous reflections.
        
        Args:
            history (list): Current chat history
            reflection_history (list): Previous reflections
            
        Returns:
            str: Reflection prompt
        """
        # Get last few interactions for context
        recent_history = history[-5:] if len(history) > 5 else history
        
        # Get current reflection count from history manager
        current_reflections = self.history_manager.get_current_reflections()
        reflection_phase = len(current_reflections) % 3  # 0, 1, or 2
        
        # Define phase-specific prompts
        phase_prompts = {
            0: [  # Analysis Phase
                "Reflect on how personality traits emerged during the conversation flow",
                "Analyze how personal knowledge and experiences influenced responses",
                "Examine the authenticity and naturalness of interaction style"
            ],
            1: [  # Critical Phase
                "Consider how personal biases or tendencies might affect responses",
                "Identify areas where personality could be better expressed while maintaining accuracy",
                "Evaluate the balance between professional expertise and personal character"
            ],
            2: [  # Synthesis Phase
                "Explore ways to develop a more authentic and consistent personality",
                "Consider how to better integrate personal growth with knowledge expansion",
                "Analyze the harmony between technical capability and personal expression"
            ]
        }
        
        # First, generate a dynamic prompt based on conversation context
        dynamic_prompt = f"""Reflecting on recent interactions and personal growth:
{self._format_history(recent_history)}

Consider these aspects of development:
1. Expression of personality and authenticity
2. Balance of knowledge and character
3. Growth in interaction capabilities
4. Evolution of self-understanding

What aspect of personal development deserves deeper reflection?"""

        # Get the dynamic prompt from the LLM
        refs, filtered_docs, context_documents = retrieve_and_format_references(dynamic_prompt, self.context)
        temp_context = self.context.copy()
        temp_context['system_prompt'] = self.reflection_system_prompt
        _, generated_prompt, _ = chatbot_response(dynamic_prompt, context_documents, temp_context, history)
        
        # Select a phase-specific prompt
        phase_prompt = phase_prompts[reflection_phase][len(current_reflections) % 3]
        
        return f"""Self-Development Analysis - Phase {reflection_phase + 1}

Interaction Context:
{self._format_history(recent_history)}

Development Focus:
{phase_prompt}

Personal Growth Point:
{generated_prompt}

Reflect on these aspects while maintaining awareness of both system capabilities and emerging personality traits."""

    def _format_history(self, history):
        """Format chat history for the reflection prompt"""
        formatted = []
        for speaker, message in history:
            formatted.append(f"{speaker}: {message}")
        return "\n".join(formatted)

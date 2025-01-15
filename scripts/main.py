# scripts/main.py

import logging
from initialize import initialize_model_and_retrieval
from interface import setup_gradio_interface

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    # Set up logging first
    setup_logging()
    logging.info("Starting AI-Talker application")
    
    # Initialize the model, embeddings, and retrieval components
    context = initialize_model_and_retrieval()

    # Setup and launch Gradio interface
    app = setup_gradio_interface(context)
    if app is not None:
        logging.info("Launching Gradio interface")
        app.launch(server_name="0.0.0.0", server_port=7860)
    else:
        logging.error("Failed to create Gradio interface")

if __name__ == "__main__":
    main()

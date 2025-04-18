# AI Document Chatbot

This project is a support chatbot that uses OpenAI's GPT-4o, FAISS for similarity search, and Gradio for the web interface. The chatbot retrieves relevant documents from a specified folder and generates responses based on those documents.

## Benefits of RAG with LLM

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval-based and generation-based models. By integrating a retrieval mechanism with a language model, RAG can provide more accurate and contextually relevant responses. 

### Key Benefits:
1. **Enhanced Accuracy:** By retrieving relevant documents, the LLM can generate responses based on specific information rather than general knowledge.
2. **Improved Context:** The context from the retrieved documents helps the LLM produce more coherent and contextually accurate replies.
3. **Versatility:** Can be used across various applications, including support chatbots, content creation, and more.

### Use Case: Support Chatbot
This support chatbot can be integrated into your website to assist users by providing instant responses based on the information available in your documents. This can be particularly useful for answering FAQs, providing product information, and more.

## Installation

Follow these steps to set up and run the project:

### 1. Navigate to your project directory
```bash
cd support-chatbot
```

### 2. Create a virtual environment using Python 3.11.8
```bash
python3.11 -m venv venv
```

### 3. Activate the virtual environment
For macOS/Linux:
```bash
source venv/bin/activate
```
For Windows:
```bash
.\venv\Scripts\activate
```

### 4. Install required packages
Ensure you have a `requirements-lock.txt` file in your project directory, then run:
```bash
pip install -r requirements-lock.txt
```
If you run into issues with the lock file, try the requirements.txt file. Check the logs for any further updates or installations that may be required.

If you intend to use Ollama for local models, refer to the section "Setting up for Ollama (Local Models)" in this readme file.

### 5. Set up environment variables
Create a `.env` file in the project directory by selecting a .env template.
- .env-local-template
- .env-openai-template

Rename the template file to ".env"
If using OpenAI models, replace `your-openai-api-key` with your actual OpenAI API key.

### 6. Prepare the ingest folder
Make sure the `INGEST_PATH` directory specified in the `.env` file exists and contains documents with a `.txt` extension.

### 7. Run the application
```bash
python scripts/main.py
```

This will launch the Gradio interface for the chatbot. Open the provided local URL in your web browser to interact with the chatbot.
You can find the local http address in the logs. 
Find this line: Running on local URL:  http://127.0.0.1:7860

## Usage
- **Input Text:** Type your query in the input text box and click "Submit" to get a response from the chatbot. The response will be based on the similarity search of the provided documents.
- **Chat History:** View the conversation history with the chatbot.
- **References:** View the references of the documents used for generating the response.
- **Clear:** Clear the chat history and references.

## Configuration Options
The `.env` file contains several configuration options:

- **MODEL_SOURCE**: "openai" for GPT models, "local" for Ollama
- **OPENAI_API_KEY:** Your OpenAI API key for accessing OpenAI services.
- **EMBEDDING_DIM:** The dimension of the embeddings used by OpenAI.
- **FAISS_INDEX_PATH:** Path to the FAISS index file.
- **METADATA_PATH:** Path to the metadata file associated with the FAISS index.
- **DOCSTORE_PATH:** Path to the docstore file for storing documents.
- **INGEST_PATH:** Path to the folder containing documents to be ingested.
- **SYSTEM_PROMPT:** The system prompt used by the chatbot to generate responses.
- **SIMILARITY_THRESHOLD:** Threshold for document similarity; documents with a similarity score below this value will be ignored.
- **TOP_SIMILARITY_RESULTS:** The number of top similar results to be considered for generating responses.


## Setting up for Ollama (Local Models)
### Requirements:
- Ensure Ollama is installed on your local machine. Ollama provides local models for embedding and LLM purposes.
- You will need the ollama CLI tool to invoke local models.

#### 1. Install Ollama
Refer to the Ollama installation guide.

#### 2. Download and Set Up Local Models
Ensure you have the local LLM and embedding models have been downloaded via Ollama and set up as per Ollama's instructions.

Example to download LLM and Embedding models. Then display the list for your reference:
```
ollama pull nomic-embed-text:latest
ollama pull mistral:7b
ollama list
```
#### 3. Modify the .env file
- Rename .env-local-template to .env
- Ensure MODEL_SOURCE=local in the .env file.


## License
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.


## To Do List
Features I intend to add:
- **Auto update embeddings:** When app starts, scan and cross-check ingest folder with the docstore. Update changed files and add new files.
- **Resource path links:** Link resources to their text files for easier referencing of files.
- **Scraping from web page or website:** Supply a URL to a webpage or website and run a script to scrape into text files stored in the ingest folder.
- **Option to retrieve chunk or document:** By default, chunks are retrieved and used for context. However, an option could be provided to retrieve the whole document for greater context, but at the expense of more tokens required.
- **auto-sizing chunks and overlap** Considerations for using ML to calculate variable chunk size and overlap based on the semantics of the type of content. Yeah, complicated but very relevant, epecially documents containing and retrieving complete code snippets...
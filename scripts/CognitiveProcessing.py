import logging
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
import config
from gpu_utils import is_gpu_too_hot
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_llm_summary(content: str, context: Dict[str, Any]) -> str:
    """
    Generate a concise summary of the content using the LLM.
    """
    try:
        if config.MODEL_SOURCE == "openai":
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries while preserving key information. When summarizing, clearly separate and label content from 'Internal Reflections' and 'Previous Conversations' if both are present. If only one type is present, label it appropriately."},
                {"role": "user", "content": f"Please provide a concise summary of the following content. If the content contains both internal reflections and previous conversations, separate and label them clearly:\n\n{content}"}
            ]
            
            response = context["client"].chat.completions.create(
                model=context["LLM_MODEL"],
                messages=messages,
                max_tokens=min(context["LLM_MAX_TOKENS"] - len(context["encoding"].encode(str(messages))), 1000),
            )
            return response.choices[0].message.content
            
        elif config.MODEL_SOURCE == "local":
            prompt = f"""You are a helpful assistant that creates concise summaries while preserving key information. When summarizing, clearly separate and label content from 'Internal Reflections' and 'Previous Conversations' if both are present. If only one type is present, label it appropriately.

Please provide a concise summary of the following content. If the content contains both internal reflections and previous conversations, separate and label them clearly:

{content}

Summary:"""
            
            response = context["client"].chat(
                model=context["LLM_MODEL"],
                messages=[{"role": "user", "content": prompt}],
            )
            return response['message']['content']
            
    except Exception as e:
        logger.error(f"Error generating LLM summary: {str(e)}")
        return content

def summarize_rag_results(context_documents: Optional[List[Dict[str, Any]]], max_length: int = 1000, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Summarize RAG retrieval results only if they exceed the maximum length.
    Otherwise, return the original content as is.
    
    Args:
        context_documents (Optional[List[Dict[str, Any]]]): List of retrieved documents from RAG
        max_length (int): Maximum length of the summarized context in characters
        context (Optional[Dict[str, Any]]]: Context containing LLM client and settings
        
    Returns:
        str: Original or summarized context suitable for LLM input
    """
    if not context_documents:
        return ""
        
    try:
        # Sort documents by relevance score if available
        sorted_docs = sorted(
            context_documents, 
            key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0,
            reverse=True
        )
        
        # First combine all content to check total length
        all_content = []
        total_length = 0
        
        for doc in sorted_docs:
            # Extract content based on document type
            if isinstance(doc, dict):
                content = doc.get('content', '')
            elif isinstance(doc, Document):
                content = doc.page_content
            else:
                logger.warning(f"Unexpected document type: {type(doc)}")
                continue
                
            # Skip empty content
            if not content.strip():
                continue
                
            all_content.append(content)
            total_length += len(content)
        
        # If total length is within limit, return original content
        if total_length <= max_length:
            return "\n\n".join(all_content)
            
        logger.info(f"Content exceeded max length ({total_length} > {max_length})")
            
        # Check GPU temperature before using LLM for summarization
        if is_gpu_too_hot():
            logger.warning("GPU temperature too high, falling back to simple truncation")
            context = None  # Force fallback to simple truncation
            
        # If content exceeds max length and we have context, use LLM to generate summary
        if context:
            combined_content = "\n\n".join(all_content)
            logger.info("Generating LLM summary...")
            return generate_llm_summary(combined_content, context)
            
        # Fallback to simple truncation if no context provided
        logger.warning("No context provided for LLM summary. Falling back to simple truncation.")
        summarized_context = []
        current_length = 0
        
        for content in all_content:
            content_length = len(content)
            
            # If adding this content would exceed max length
            if current_length + content_length > max_length:
                # If this is the first document, take a portion of it
                if not summarized_context:
                    truncated_content = content[:max_length].rsplit(' ', 1)[0]
                    summarized_context.append(truncated_content)
                break
                
            # Add content to summary
            summarized_context.append(content)
            current_length += content_length
            
        # Join all summarized content
        final_summary = "\n\n".join(summarized_context)
        
        logger.info(f"Content exceeded max length ({total_length} > {max_length}). Truncated to {len(final_summary)} characters")
        return final_summary
        
    except Exception as e:
        logger.error(f"Error summarizing RAG results: {str(e)}")
        return "" if not context_documents else context_documents[0].get('content', '') if isinstance(context_documents[0], dict) else context_documents[0].page_content

def determine_and_perform_web_search(query: str, rag_summary: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determines if a web search is needed based on the user query and RAG summary,
    and performs the search if necessary.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - needs_web_search (bool): Whether web search was performed
            - web_results (str): Summarized results from web search if performed
    """
    # Initialize result dictionary
    result = {
        "needs_web_search": False,
        "web_results": ""
    }
    
    try:
        # For queries that explicitly request web search, bypass search decision but still optimize query
        explicit_search = any(phrase in query.lower() for phrase in ["search the web", "search online", "look up", "find online"])
        if explicit_search:
            logging.info(f"Query '{query}' explicitly requests web search")
            result["needs_web_search"] = True
            # Remove the search command phrases to get the core query
            search_topic = query.lower()
            for phrase in ["search the web for", "search online for", "look up", "find online"]:
                search_topic = search_topic.replace(phrase, "").strip()
                
            # Use LLM to optimize the search query
            prompt = f"""Convert this search request into a concise and effective search query (2-5 words).
Focus on the key terms that will yield the most relevant results.

Search request: {search_topic}

Respond with ONLY the optimized search query. Examples:
"funny jokes trending 2025"
"current weather Sydney"
"SpaceX latest launch news"
"""
            if config.MODEL_SOURCE == "openai":
                response = context["client"].chat.completions.create(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                )
                search_query = response.choices[0].message.content.strip()
            else:
                response = context["client"].chat(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                )
                search_query = response['message']['content'].strip()
                
            logging.info(f"Optimized search query: {search_query}")
        else:
            # Construct prompt to determine if web search is needed
            messages = [
                {"role": "system", "content": """You are a helpful assistant that determines if a web search is needed to answer a user's query. 
Consider:
1. If the RAG summary already provides a complete and up-to-date answer
2. If the query requires real-time or current information
3. If the query asks about topics likely not covered in the local knowledge base

Respond with EXACTLY one of these two formats:
1. If no web search is needed: "NO_SEARCH_NEEDED"
2. If web search is needed: A concise search query (2-5 words) that will help find relevant information.

Example responses:
- "NO_SEARCH_NEEDED"
- "current weather Adelaide"
- "latest SpaceX launch news"
"""},
                {"role": "user", "content": f"User Query: {query}\nRAG Summary: {rag_summary}\n\nDoes this query need a web search? Respond in the format specified above:"}
            ]
            
            # Get LLM's decision
            if config.MODEL_SOURCE == "openai":
                response = context["client"].chat.completions.create(
                    model=context["LLM_MODEL"],
                    messages=messages,
                    max_tokens=50,
                )
                llm_response = response.choices[0].message.content.strip()
            else:
                prompt = f"""You are a helpful assistant that determines if a web search is needed to answer a user's query. 
Consider:
1. If the RAG summary already provides a complete and up-to-date answer
2. If the query requires real-time or current information
3. If the query asks about topics likely not covered in the local knowledge base

Respond with EXACTLY one of these two formats:
1. If no web search is needed: "NO_SEARCH_NEEDED"
2. If web search is needed: A concise search query (2-5 words) that will help find relevant information.

Example responses:
- "NO_SEARCH_NEEDED"
- "current weather Adelaide"
- "latest SpaceX launch news"

User Query: {query}
RAG Summary: {rag_summary}

Does this query need a web search? Respond in the format specified above:"""
                
                response = context["client"].chat(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                )
                llm_response = response['message']['content'].strip()
            
            # Log LLM's decision
            logging.info(f"LLM's web search decision for query '{query}':")
            logging.info(f"RAG Summary: {rag_summary}")
            logging.info(f"LLM Response: {llm_response}")
            
            # Check if search is needed and get optimized query
            needs_search = llm_response != "NO_SEARCH_NEEDED"
            search_query = llm_response if needs_search else ""
            
            # Optimize the search query using LLM
            if needs_search:
                prompt = f"""Convert this search request into a concise and effective search query (2-5 words).
Focus on the key terms that will yield the most relevant results.

Search request: {search_query}

Respond with ONLY the optimized search query. Examples:
"funny jokes trending 2025"
"current weather Sydney"
"SpaceX latest launch news"
"""
                if config.MODEL_SOURCE == "openai":
                    response = context["client"].chat.completions.create(
                        model=context["LLM_MODEL"],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=20,
                    )
                    search_query = response.choices[0].message.content.strip()
                else:
                    response = context["client"].chat(
                        model=context["LLM_MODEL"],
                        messages=[{"role": "user", "content": prompt}],
                    )
                    search_query = response['message']['content'].strip()
                    
                logging.info(f"Optimized search query: {search_query}")
        
        result["needs_web_search"] = needs_search
        result["web_results"] = ""
        
        # Perform web search if needed
        if needs_search:
            logging.info(f"Web search deemed necessary, performing search with query: {search_query}")
            
            # Create DDGS instance and perform search
            ddgs = DDGS()
            try:
                # Try news search first
                search_results = list(ddgs.news(search_query, max_results=3))
                if not search_results:
                    # Fallback to text search if no news results
                    logging.info("No news results, falling back to text search")
                    search_results = list(ddgs.text(search_query, max_results=3))
            except Exception as e:
                logging.error(f"Error in DDGS search: {str(e)}")
                search_results = []
            
            web_content = []
            for result_item in search_results:
                try:
                    # Extract title and link from search result
                    title = result_item.get('title', '')
                    url = result_item.get('link', '')  # News search uses 'link' instead of 'url'
                    if not url:
                        url = result_item.get('url', '')  # Fallback to 'url' for text search
                    
                    # Get content from either news snippet or text body
                    content = result_item.get('body', '')
                    if not content:
                        content = result_item.get('snippet', '')  # News search might have 'snippet'
                    
                    if not url or not title or not content:
                        logging.warning(f"Skipping result due to missing data: {result_item}")
                        continue
                        
                    # Log search result metadata
                    logging.info(f"Found search result: Title='{title}' URL='{url}'")
                    logging.debug(f"Content preview: {content[:100]}...")
                    
                    web_content.append({
                        'title': title,
                        'link': url,
                        'content': content
                    })
                except Exception as e:
                    logging.warning(f"Error processing search result: {str(e)}")
                    continue
            
            # Summarize web content using LLM
            if web_content:
                web_content_str = "\n\n".join([f"Source: {item['title']}\nURL: {item['link']}\nContent: {item['content']}" for item in web_content])
                result["web_results"] = generate_llm_summary(web_content_str, context)
                logging.info(f"Summarized web search results: {result['web_results']}")
            else:
                logging.warning("No valid search results found")
                result["web_results"] = "No search results."
        else:
            logging.info("Web search not needed based on LLM decision")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in determine_and_perform_web_search: {str(e)}")
        return result  # Return the initialized result dictionary with default values

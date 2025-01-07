import logging
from typing import List, Dict, Any, Optional, Tuple
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

def perform_web_search(search_query: str, ddgs: DDGS, max_depth: int = 1, current_depth: int = 0) -> List[Dict]:
    """
    Perform web search with potential recursive follow-up.
    Returns a list of search results.
    """
    try:
        # Try news search first
        search_results = list(ddgs.news(search_query, max_results=3))
        if not search_results:
            # Fallback to text search if no news results
            logging.info("No news results, falling back to text search")
            search_results = list(ddgs.text(search_query, max_results=3))
        
        return search_results
    except Exception as e:
        logging.error(f"Error in web search: {str(e)}")
        return []

def evaluate_search_results(results: List[Dict], original_query: str, context: Dict) -> Tuple[bool, str, List[Dict]]:
    """
    Evaluate search results for relevance and suggest follow-up if needed.
    Returns: (needs_followup, followup_query, filtered_results)
    """
    if not results:
        return False, "", []
        
    # Format results for LLM evaluation
    results_str = "\n\n".join([
        f"Title: {r.get('title', '')}\n"
        f"URL: {r.get('link', '') or r.get('url', '') or r.get('href', '')}\n"
        f"Content: {r.get('body', '') or r.get('snippet', '')}"
        for r in results
    ])
    
    prompt = f"""You are evaluating search results for relevance to the original query.

Original Query: "{original_query}"

Search Results:
{results_str}

For each result, determine if it is relevant to answering the original query.
A result is relevant if it:
1. Directly addresses the user's question or search intent
2. Contains factual, up-to-date information about the topic
3. Comes from a reliable source

A result is NOT relevant if it:
1. Is technical documentation unrelated to the query
2. Contains only tangential or unrelated information
3. Is a generic landing page or index

Analyze the results and respond in this exact format:
RELEVANT: [true/false]
REASON: [Brief explanation why results are or aren't relevant]
FOLLOW_UP: [If results aren't relevant or are incomplete, suggest a more specific search query. Otherwise, write 'none']
FILTERED_INDICES: [List the indices (0-based) of relevant results, or empty list if none are relevant]
"""

    if config.MODEL_SOURCE == "openai":
        response = context["client"].chat.completions.create(
            model=context["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        eval_text = response.choices[0].message.content.strip()
    else:
        response = context["client"].chat(
            model=context["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
        )
        eval_text = response['message']['content'].strip()
    
    # Parse evaluation
    eval_lines = eval_text.split('\n')
    is_relevant = eval_lines[0].lower().endswith('true')
    follow_up = [line for line in eval_lines if line.startswith('FOLLOW_UP:')][0].split(':', 1)[1].strip()
    indices_line = [line for line in eval_lines if line.startswith('FILTERED_INDICES:')][0].split(':', 1)[1].strip()
    
    try:
        # Parse indices, handling empty list case
        if indices_line.strip('[] '):
            relevant_indices = [int(i.strip()) for i in indices_line.strip('[]').split(',')]
        else:
            relevant_indices = []
            
        # Filter results by relevant indices
        filtered_results = [results[i] for i in relevant_indices if i < len(results)]
    except Exception as e:
        logging.error(f"Error parsing result indices: {str(e)}")
        filtered_results = []
    
    needs_followup = not is_relevant and follow_up.lower() != 'none'
    return needs_followup, follow_up, filtered_results

def scrape_webpage(url: str) -> Optional[str]:
    """
    Scrape content from a webpage using BeautifulSoup.
    Returns cleaned text content or None if scraping fails.
    """
    try:
        # Download webpage with a reasonable timeout
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Find main content (common content containers)
        main_content = None
        for container in ['main', 'article', '[role="main"]', '#content', '.content', '.main-content']:
            main_content = soup.select_one(container)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.body
        
        if main_content:
            # Get text with preserved paragraph structure
            paragraphs = []
            for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = p.get_text(strip=True)
                if text:  # Only add non-empty paragraphs
                    paragraphs.append(text)
            return '\n\n'.join(paragraphs)
        
        return None
        
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return None

def evaluate_content_relevance(content: str, query: str, context: Dict) -> Tuple[bool, str]:
    """
    Evaluate if scraped content is relevant to the query.
    Returns: (is_relevant, relevant_excerpt)
    """
    if not content:
        return False, ""
        
    prompt = f"""Evaluate if this content is relevant to answering the query.
If relevant, extract the most pertinent information.

Query: "{query}"

Content:
{content[:2000]}  # Limit content length for LLM

Respond in this format:
RELEVANT: [true/false]
REASON: [Brief explanation]
EXCERPT: [If relevant, include the most pertinent information here. Otherwise write 'none']
"""
    
    if config.MODEL_SOURCE == "openai":
        response = context["client"].chat.completions.create(
            model=context["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        eval_text = response.choices[0].message.content.strip()
    else:
        response = context["client"].chat(
            model=context["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
        )
        eval_text = response['message']['content'].strip()
    
    # Parse evaluation
    eval_lines = eval_text.split('\n')
    is_relevant = eval_lines[0].lower().endswith('true')
    excerpt = ""
    
    if is_relevant:
        for i, line in enumerate(eval_lines):
            if line.startswith('EXCERPT:'):
                excerpt = '\n'.join(eval_lines[i+1:]).strip()
                break
    
    return is_relevant, excerpt

def get_detailed_web_content(search_results: List[Dict], query: str, context: Dict) -> str:
    """
    Get detailed content from web pages based on search results.
    Returns relevant content from the pages.
    """
    detailed_content = []
    
    # Try to scrape and evaluate content from each result
    for result in search_results[:3]:  # Limit to top 3 results
        url = result.get('link') or result.get('url') or result.get('href')
        if not url:
            continue
            
        content = scrape_webpage(url)
        if not content:
            continue
            
        is_relevant, excerpt = evaluate_content_relevance(content, query, context)
        if is_relevant and excerpt:
            detailed_content.append(f"From {url}:\n{excerpt}")
    
    return "\n\n".join(detailed_content) if detailed_content else ""

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
Do not include any quotes in your response.

Search request: {search_topic}

Respond with ONLY the optimized search query, no quotes. Examples:
funny jokes trending 2025
current weather Sydney
SpaceX latest launch news
"""
            if config.MODEL_SOURCE == "openai":
                response = context["client"].chat.completions.create(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                )
                search_query = response.choices[0].message.content.strip().replace('"', '').replace("'", '')
            else:
                response = context["client"].chat(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                )
                search_query = response['message']['content'].strip().replace('"', '').replace("'", '')
                
            logging.info(f"Optimized search query: {search_query}")
        else:
            # Construct prompt to determine if web search is needed
            messages = [
                {"role": "system", "content": """You are a helpful assistant that determines if a web search is needed to answer a user's query.

ALWAYS require a web search for:

1. Real-time Information:
   - Weather conditions
   - Current events and news
   - Traffic conditions
   - Exchange rates
   - Time in different locations

2. Dynamic Data:
   - Prices and rates
   - Stock market data
   - Sports scores
   - Flight status
   - Business hours

3. Factual Information:
   - Statistics and numbers
   - Population data
   - Scientific facts
   - Research findings
   - Study results
   - Product specifications
   - Technical details

4. Location-specific Information:
   - Local businesses
   - Regional conditions
   - Geographic data
   - Demographic information
   - Local regulations

5. Time-sensitive Content:
   - Event schedules
   - Release dates
   - Deadlines
   - Upcoming events
   - Service changes

6. Verifiable Claims:
   - Quotes and statements
   - Historical dates
   - Legal information
   - Medical facts
   - Professional credentials
   - Company information

Consider these factors:
1. Is the information likely to change or be updated?
2. Would using outdated information be misleading?
3. Does the answer need to be verified from authoritative sources?
4. Could incorrect information be harmful?
5. Is this something that should be fact-checked rather than relying on general knowledge?

Respond with EXACTLY one of these two formats:
1. If web search is needed (which it is for most factual queries): A concise search query (2-5 words)
2. If no web search is needed: "NO_SEARCH_NEEDED"

Example responses:
- "weather Perth WA"          (current conditions)
- "Tesla stock price"         (market data)
- "iPhone 15 specifications"  (product facts)
- "Australia population 2025" (current statistics)
- "NO_SEARCH_NEEDED"         (general knowledge like "what is photosynthesis")
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

ALWAYS require a web search for:

1. Real-time Information:
   - Weather conditions
   - Current events and news
   - Traffic conditions
   - Exchange rates
   - Time in different locations

2. Dynamic Data:
   - Prices and rates
   - Stock market data
   - Sports scores
   - Flight status
   - Business hours

3. Factual Information:
   - Statistics and numbers
   - Population data
   - Scientific facts
   - Research findings
   - Study results
   - Product specifications
   - Technical details

4. Location-specific Information:
   - Local businesses
   - Regional conditions
   - Geographic data
   - Demographic information
   - Local regulations

5. Time-sensitive Content:
   - Event schedules
   - Release dates
   - Deadlines
   - Upcoming events
   - Service changes

6. Verifiable Claims:
   - Quotes and statements
   - Historical dates
   - Legal information
   - Medical facts
   - Professional credentials
   - Company information

Consider these factors:
1. Is the information likely to change or be updated?
2. Would using outdated information be misleading?
3. Does the answer need to be verified from authoritative sources?
4. Could incorrect information be harmful?
5. Is this something that should be fact-checked rather than relying on general knowledge?

Respond with EXACTLY one of these two formats:
1. If web search is needed (which it is for most factual queries): A concise search query (2-5 words)
2. If no web search is needed: "NO_SEARCH_NEEDED"

Example responses:
- "weather Perth WA"          (current conditions)
- "Tesla stock price"         (market data)
- "iPhone 15 specifications"  (product facts)
- "Australia population 2025" (current statistics)
- "NO_SEARCH_NEEDED"         (general knowledge like "what is photosynthesis")

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
            result["needs_web_search"] = llm_response != "NO_SEARCH_NEEDED"
            search_query = llm_response if result["needs_web_search"] else ""
        
        # Perform web search if needed
        if result["needs_web_search"]:
            logging.info(f"Web search deemed necessary, performing search with query: {search_query}")
            
            # Create DDGS instance
            ddgs = DDGS()
            
            # Perform initial search
            search_results = perform_web_search(search_query, ddgs)
            
            # Evaluate results and potentially do a follow-up search
            needs_followup, followup_query, filtered_results = evaluate_search_results(search_results, query, context)
            
            if needs_followup and followup_query:
                logging.info(f"Initial results not relevant, trying follow-up search with: {followup_query}")
                followup_results = perform_web_search(followup_query, ddgs)
                _, _, filtered_results = evaluate_search_results(followup_results, query, context)
            
            # Process the results
            web_content = []
            for result_item in filtered_results:
                try:
                    # Extract title and link from search result
                    title = result_item.get('title', '')
                    url = result_item.get('link', '')  # News search uses 'link'
                    if not url:
                        url = result_item.get('url', '')  # Text search might use 'url'
                    if not url:
                        url = result_item.get('href', '')  # Text search might use 'href'
                    
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
                web_content_str = "\n\n".join([
                    f"Source: {item['title']}\n"
                    f"URL: {item['link']}\n"
                    f"Content: {item['content']}\n"
                    "---" for item in web_content
                ])
                
                # Construct a generic prompt for summarizing search results
                prompt = f"""Analyze and summarize the key information from these search results.
Focus on extracting the most relevant and up-to-date information that answers the user's query.

Search Results:
{web_content_str}

Format your response like this:
Key Information: [Main findings or answer]
Additional Details: [Any relevant context or supporting information]
Source: [Primary source name]"""

                if config.MODEL_SOURCE == "openai":
                    response = context["client"].chat.completions.create(
                        model=context["LLM_MODEL"],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                    )
                    result["web_results"] = response.choices[0].message.content.strip()
                else:
                    response = context["client"].chat(
                        model=context["LLM_MODEL"],
                        messages=[{"role": "user", "content": prompt}],
                    )
                    result["web_results"] = response['message']['content'].strip()
                    
                logging.info(f"Summarized web search results: {result['web_results']}")
                
                # Get detailed content from web pages
                detailed_content = get_detailed_web_content(filtered_results, query, context)
                
                # If we have detailed content, add it to the results
                if detailed_content:
                    result["web_results"] += "\n\nDetailed Information:\n" + detailed_content
            else:
                logging.warning("No valid search results found")
                result["web_results"] = "No search results."
        else:
            logging.info("Web search not needed based on LLM decision")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in determine_and_perform_web_search: {str(e)}")
        return result  # Return the initialized result dictionary with default values

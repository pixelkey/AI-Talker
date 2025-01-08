import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.docstore.document import Document
import config
from gpu_utils import is_gpu_too_hot
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import json
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import random

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

            # Skip reference sections
            content_parts = content.split("REFERENCES:")
            clean_content = content_parts[0].strip()  # Take only the part before REFERENCES
            
            # Skip if content is just metadata or references
            if not clean_content or clean_content.lower().startswith("references"):
                continue
                
            all_content.append(clean_content)
            total_length += len(clean_content)
        
        # If no valid content after filtering
        if not all_content:
            return ""
            
        # If total length is within limit, return filtered content
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

def perform_parallel_web_search(queries: List[str], context: Dict, max_retries: int = 3, retry_delay: int = 2) -> List[Dict]:
    """Perform multiple web searches in parallel"""
    all_results = []
    
    def search_with_retry(query: str) -> List[Dict]:
        retry_count = 0
        while retry_count < max_retries:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=10))
                    if results:
                        return results
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(retry_delay)
                    logging.info(f"Retrying search for '{query}' (attempt {retry_count + 1}/{max_retries})")
            except Exception as e:
                logging.error(f"Search attempt {retry_count + 1} for '{query}' failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(retry_delay)
        return []

    # Run searches in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_query = {executor.submit(search_with_retry, query): query for query in queries}
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
            except Exception as e:
                logging.error(f"Error in parallel search for '{query}': {str(e)}")
    
    return all_results

def perform_web_search(search_query: str, ddgs: DDGS, max_depth: int = 1, current_depth: int = 0, max_retries: int = 3, retry_delay: int = 2):
    """
    Perform web search with retries and pagination.
    Returns a list of search results.
    """
    results = []
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get search results with pagination
            search_results = list(ddgs.text(search_query, max_results=10))
            if search_results:
                results.extend(search_results)
                return results
            
            # If we get here with no results but haven't exceeded retries
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
                logging.info(f"Retrying search (attempt {retry_count + 1}/{max_retries})")
                
        except Exception as e:
            logging.error(f"Search attempt {retry_count + 1} failed: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
                logging.info(f"Retrying after error (attempt {retry_count + 1}/{max_retries})")
            else:
                raise
    
    return results

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
    
    prompt = f"""Evaluate these search results for relevance to the original query.
Consider a result relevant if it has ANY information that could help answer the query.
Be inclusive rather than exclusive in what you consider relevant.

Original Query: "{original_query}"

Search Results:
{results_str}

Analyze the results and respond in this exact format:
RELEVANT: [true/false]
REASON: [Brief explanation why results are or aren't relevant]
FOLLOW_UP: [If results aren't relevant or are incomplete, suggest a more specific search query. Otherwise, write 'none']
FILTERED_INDICES: [List the indices (0-based) of relevant results]

Example response:
RELEVANT: true
REASON: Results 0 and 2 contain recent information about the topic
FOLLOW_UP: none
FILTERED_INDICES: [0, 2]"""

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
    
    try:
        # Parse evaluation
        eval_lines = eval_text.split('\n')
        is_relevant = any(line.lower().endswith('true') for line in eval_lines if line.startswith('RELEVANT:'))
        
        # Get follow-up query
        follow_up_lines = [line for line in eval_lines if line.startswith('FOLLOW_UP:')]
        follow_up = follow_up_lines[0].split(':', 1)[1].strip() if follow_up_lines else "none"
        
        # Get indices, with fallback to using all results if parsing fails
        indices_lines = [line for line in eval_lines if line.startswith('FILTERED_INDICES:')]
        if indices_lines:
            indices_str = indices_lines[0].split(':', 1)[1].strip()
            try:
                relevant_indices = [int(i.strip()) for i in indices_str.strip('[]').split(',') if i.strip()]
            except:
                # If parsing fails, consider all results relevant
                relevant_indices = list(range(len(results)))
        else:
            relevant_indices = list(range(len(results)))
            
        # Filter results by relevant indices
        filtered_results = [results[i] for i in relevant_indices if i < len(results)]
        
        # If no results were marked relevant but we have results, use them all
        if not filtered_results and results:
            filtered_results = results
            is_relevant = True
            
    except Exception as e:
        logging.error(f"Error parsing evaluation response: {str(e)}")
        # On error, consider all results relevant
        filtered_results = results
        is_relevant = True
        follow_up = "none"
    
    needs_followup = not is_relevant and follow_up.lower() != 'none'
    return needs_followup, follow_up, filtered_results

def scrape_webpage(url: str) -> Optional[str]:
    """
    Scrape content from a webpage using BeautifulSoup.
    Returns cleaned text content or None if scraping fails.
    """
    try:
        # Small delay to avoid overwhelming servers, shorter now due to parallel processing
        time.sleep(random.uniform(0.1, 0.3))
        
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Handle common HTTP errors
        if response.status_code == 403:
            logging.warning(f"403 Forbidden error for {url}. Trying with different user agent...")
            # For 403 errors, we still want a longer delay
            time.sleep(random.uniform(1.0, 2.0))
            # Retry with different user agent
            response = requests.get(url, timeout=10, headers={
                'User-Agent': random.choice([
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
                ])
            })
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
            
        # Look for structured data first (JSON-LD, microdata)
        structured_data = []
        
        # Check JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.append(data)
            except:
                continue
                
        # Check microdata
        for element in soup.find_all(attrs={'itemtype': True}):
            data = {}
            for prop in element.find_all(attrs={'itemprop': True}):
                data[prop['itemprop']] = prop.get_text(strip=True)
            if data:
                structured_data.append(data)
                
        # If we found structured data, include it first
        content_parts = []
        if structured_data:
            content_parts.append("Structured Data:")
            content_parts.append(json.dumps(structured_data, indent=2))
            
        # Find main content area
        main_content = None
        for container in ['main', 'article', '[role="main"]', '#content', '.content', '.main-content']:
            main_content = soup.select_one(container)
            if main_content:
                break
                
        # If no main content found, use body
        if not main_content:
            main_content = soup.body
            
        if main_content:
            # Get text with semantic structure preserved
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'dl', 'div']):
                # Skip empty or navigation elements
                if not element.get_text(strip=True) or element.find_parent(['nav', 'header', 'footer']):
                    continue
                    
                # Check if element has any special attributes or classes that might indicate importance
                attrs = element.attrs
                if any(attr in str(attrs).lower() for attr in ['price', 'date', 'time', 'current', 'value', 'result']):
                    content_parts.append(f"[Important] {element.get_text(strip=True)}")
                else:
                    content_parts.append(element.get_text(strip=True))
                    
            # Look for data in tables
            for table in main_content.find_all('table'):
                rows = []
                for row in table.find_all('tr'):
                    cols = [col.get_text(strip=True) for col in row.find_all(['td', 'th'])]
                    if any(cols):  # Skip empty rows
                        rows.append(' | '.join(cols))
                if rows:
                    content_parts.append("\nTable Data:\n" + '\n'.join(rows))
                    
        return '\n\n'.join(content_parts) if content_parts else None
        
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

def get_detailed_web_content(search_results: List[Dict], query: str, context: Dict, max_pages: int = 3, timeout_seconds: int = 30) -> str:
    """
    Get detailed content from web pages based on search results.
    Uses ThreadPoolExecutor for parallel processing with improved concurrency.
    """
    all_content = []
    urls_tried = set()
    start_time = time.time()
    
    def is_timed_out() -> bool:
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            logging.warning(f"Web search timed out after {elapsed:.1f} seconds")
            return True
        return False

    def process_single_url(url: str) -> Optional[str]:
        """Process a single URL with existing scrape_webpage function"""
        if is_timed_out() or not url or url in urls_tried:
            return None
            
        if not url.startswith(('http://', 'https://')):
            logging.warning(f"Invalid URL format: {url}")
            return None
            
        urls_tried.add(url)
        try:
            content = scrape_webpage(url)
            if content:
                is_relevant, excerpt = evaluate_content_relevance(content, query, context)
                if is_relevant:
                    return excerpt
        except Exception as e:
            logging.warning(f"Error processing {url}: {str(e)}")
        return None

    def process_urls_parallel(urls: List[str]) -> None:
        """Process URLs in parallel, adding results as they complete"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(process_single_url, url): url for url in urls}
            
            # Process results as they complete instead of waiting for all
            for future in concurrent.futures.as_completed(future_to_url):
                if is_timed_out() or len(all_content) >= max_pages:
                    # Cancel remaining futures if we have enough content
                    for f in future_to_url:
                        f.cancel()
                    break
                
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        all_content.append(result)
                except Exception as e:
                    logging.error(f"Error processing {url}: {str(e)}")

    # Get all valid URLs upfront
    all_urls = []
    for result in search_results:
        if len(all_urls) >= max_pages * 2:  # Get 2x the URLs we need
            break
        url = result.get('link', '')  # News search uses 'link'
        if not url:
            url = result.get('url', '')  # Text search might use 'url'
        if not url:
            url = result.get('href', '')  # Text search might use 'href'
        
        if url and url not in urls_tried and url.startswith(('http://', 'https://')):
            all_urls.append(url)

    # Process initial batch
    if all_urls:
        process_urls_parallel(all_urls)

    # If we need more content, try alternative queries
    if not is_timed_out() and len(all_content) < max_pages:
        alt_queries = generate_alternative_queries(query, context)
        if alt_queries:
            # Perform alternative searches in parallel
            alt_results = perform_parallel_web_search(alt_queries, context)
            if alt_results:
                alt_urls = []
                for result in alt_results:
                    if len(alt_urls) >= max_pages:
                        break
                    url = result.get('link', '')  # News search uses 'link'
                    if not url:
                        url = result.get('url', '')  # Text search might use 'url'
                    if not url:
                        url = result.get('href', '')  # Text search might use 'href'
                    
                    if url and url not in urls_tried and url.startswith(('http://', 'https://')):
                        alt_urls.append(url)
                
                if alt_urls:
                    process_urls_parallel(alt_urls)

    elapsed = time.time() - start_time
    logging.info(f"Web search completed in {elapsed:.1f} seconds, found {len(all_content)} relevant results")
    return "\n\n".join(all_content) if all_content else ""

def generate_alternative_queries(original_query: str, context: Dict[str, Any]) -> List[str]:
    """
    Generate alternative search queries if the original doesn't yield good results.
    """
    prompt = f"""Generate 2-3 alternative search queries that might find relevant information.
Make them more specific or use different terms. Remove any unnecessary words.
Original query: {original_query}

Format each query on a new line, no quotes or punctuation.
Example:
query term1 term2
another search query
final search terms"""

    try:
        if config.MODEL_SOURCE == "openai":
            response = context["client"].chat.completions.create(
                model=context["LLM_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            alt_queries = response.choices[0].message.content.strip().split('\n')
        else:
            response = context["client"].chat(
                model=context["LLM_MODEL"],
                messages=[{"role": "user", "content": prompt}],
            )
            alt_queries = response['message']['content'].strip().split('\n')
        
        # Clean up queries
        alt_queries = [q.strip().replace('"', '').replace("'", "").replace(",", "") for q in alt_queries if q.strip()]
        return alt_queries[:3]  # Limit to top 3 alternatives
        
    except Exception as e:
        logging.error(f"Error generating alternative queries: {str(e)}")
        return []

def determine_and_perform_web_search(query: str, rag_summary: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determines if a web search is needed based on:
    1. Whether information is available in RAG/LLM context
    2. If the information needs to be current/real-time
    3. If the query explicitly requests external information
    """
    result = {
        "needs_web_search": False,
        "web_results": ""
    }
    
    try:
        # Get recent conversation history
        recent_messages = []
        if "memory" in context:
            try:
                chat_history = context["memory"].chat_memory.messages[-5:]
                recent_messages = [f"{msg.type}: {msg.content}" for msg in chat_history]
            except:
                logging.warning("Could not retrieve chat history")
        
        # First, check for explicit search requests or real-time information needs
        explicit_search_terms = ["search", "look up", "find", "what is", "how to", "current", "latest", "news", "weather", "price", "status"]
        time_related_terms = ["time", "today", "now", "current", "latest"]
        
        query_lower = query.lower()
        is_explicit_search = any(term in query_lower for term in explicit_search_terms)
        needs_current_time = any(term in query_lower for term in time_related_terms)
        
        # Immediate search for time-related queries
        if needs_current_time and any(word in query_lower for word in ["time", "hour", "today", "now"]):
            result["needs_web_search"] = True
            search_query = query
        elif is_explicit_search:
            # Use the query directly, just clean it up
            search_query = query_lower
            for term in ["search for", "look up", "find", "what is", "how to"]:
                search_query = search_query.replace(term, "").strip()
            result["needs_web_search"] = True
        else:
            # Evaluate if information is available in context or needs web search
            prompt = f"""Analyze if this query requires a web search. Be very strict about avoiding unnecessary searches.

Key Decision Points:
1. Query Type:
   - Is this a conversational query (greetings, opinions, personal questions)?
   - Is this small talk or casual conversation?
   - Is this asking about the AI assistant's capabilities or personality?
   
2. Information Source:
   - Can this be answered from common knowledge or conversation?
   - Does this need real-time or external information?
   - Is this asking for factual information not present in context?

3. Information Completeness:
   - Does the RAG context ACTUALLY CONTAIN the specific information asked for? 
   - Just mentioning a topic/name is NOT enough - we need the specific details requested
   - If the RAG only mentions something exists but doesn't provide details, we NEED a web search

4. Information Quality:
   - Is the information complete and detailed enough to fully answer the query?
   - Is it current enough to be reliable?
   - Do we need additional details or verification?

5. Critical Assessment:
   - Don't assume information exists just because a topic is mentioned
   - If RAG only shows partial information, we should still search
   - When in doubt, prefer to search to get complete information

NO Web Search Needed For:
- Greetings (hi, hello, hey)
- How are you / What's up
- Thank you / You're welcome
- Personal questions about the AI
- Simple yes/no responses
- Opinion questions
- General chitchat
- Help requests about capabilities
- Very short follow-up questions

Current Query: {query}
RAG Summary: {rag_summary if rag_summary else "No relevant information found in RAG"}
Recent Messages: {recent_messages[-3:] if recent_messages else "No recent messages"}

Respond in this format:
NEEDS_SEARCH: [YES/NO]
REASON: [Explain why search is or isn't needed, including information completeness]
SEARCH_TERMS: [If YES or if RAG only has partial info, provide 2-5 key search terms without quotes]"""

            if config.MODEL_SOURCE == "openai":
                response = context["client"].chat.completions.create(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                )
                llm_response = response.choices[0].message.content.strip()
            else:
                response = context["client"].chat(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                )
                llm_response = response['message']['content'].strip()

            # Log the response
            logging.info(f"LLM's web search decision for query '{query}':")
            logging.info(f"RAG Summary: {rag_summary}")
            logging.info(f"LLM Response: {llm_response}")

            # Parse response
            lines = [line.strip() for line in llm_response.split('\n') if line.strip()]
            if not lines:
                # No valid response, default to search if RAG is empty or unclear
                result["needs_web_search"] = True
                search_query = query
            else:
                needs_search_line = lines[0].lower()
                reason_line = next((line for line in lines if line.startswith("REASON:")), "")
                
                # Default to search if:
                # 1. LLM says we need search
                # 2. Reason indicates information is partial/missing
                # 3. RAG is empty or just mentions existence without details
                result["needs_web_search"] = (
                    "yes" in needs_search_line or
                    "partial" in reason_line.lower() or
                    "only mentions" in reason_line.lower() or
                    "doesn't provide" in reason_line.lower() or
                    not rag_summary
                )
                
                if result["needs_web_search"]:
                    # Get search terms, prioritizing LLM's suggestion but falling back to query
                    search_terms_line = next((line for line in lines if line.startswith("SEARCH_TERMS:")), "")
                    if search_terms_line and len(search_terms_line.split(":")) > 1:
                        search_query = search_terms_line.split(":", 1)[1].strip()
                    else:
                        search_query = query
                        
                    # Clean up search query - remove quotes and extra punctuation
                    search_query = search_query.replace('"', '').replace("'", "").replace(",", "")
                    search_query = ' '.join(search_query.split()[:5])  # Limit to 5 words
        # Perform web search if needed
        if result["needs_web_search"]:
            logging.info(f"Web search deemed necessary, performing search with query: {search_query}")
            
            # Create DDGS instance
            with DDGS() as ddgs:
                search_results = perform_web_search(search_query, ddgs)
                
            logging.info(f"Web search deemed necessary, performing search with query: {search_query}")
            
            # Initialize filtered results
            filtered_results = []
            web_content = []
            
            # Process search results
            if not search_results:
                logging.warning("No search results found")
                result["web_results"] = "No search results found."
                return result
                
            # Evaluate search results for relevance
            needs_followup, followup_query, filtered_results = evaluate_search_results(search_results, query, context)
            
            # Process each filtered result
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
                # First evaluate RAG results for relevance
                if rag_summary:
                    rag_eval_prompt = f"""Evaluate if this RAG context contains information that directly answers or is relevant to the user's query.
IMPORTANT: IGNORE any references or citations sections. Only evaluate the actual content.
If you see "REFERENCES:" or similar sections, completely disregard them.

Query: {query}

RAG Context:
{rag_summary}

Response format:
RELEVANT: [yes/no]
SUMMARY: [1-2 sentence summary of ONLY the relevant content, or 'NOT_RELEVANT' if nothing relevant found]

Example response:
If the RAG context only contains references or chat history that's not directly relevant, respond with:
RELEVANT: no
SUMMARY: NOT_RELEVANT"""

                    if config.MODEL_SOURCE == "openai":
                        rag_response = context["client"].chat.completions.create(
                            model=context["LLM_MODEL"],
                            messages=[{"role": "user", "content": rag_eval_prompt}],
                            max_tokens=100,
                        )
                        rag_eval = rag_response.choices[0].message.content.strip()
                    else:
                        rag_response = context["client"].chat(
                            model=context["LLM_MODEL"],
                            messages=[{"role": "user", "content": rag_eval_prompt}],
                        )
                        rag_eval = rag_response['message']['content'].strip()

                    # Parse RAG evaluation
                    rag_lines = rag_eval.split('\n')
                    is_rag_relevant = any('yes' in line.lower() for line in rag_lines if line.startswith('RELEVANT:'))
                    if is_rag_relevant:
                        summary_lines = [line for line in rag_lines if line.startswith('SUMMARY:')]
                        if summary_lines and 'NOT_RELEVANT' not in summary_lines[0].upper():
                            rag_summary = summary_lines[0].replace('SUMMARY:', '').strip()
                        else:
                            rag_summary = None
                    else:
                        rag_summary = None

                web_content_str = "\n\n".join([
                    f"Source: {item['title']}\n"
                    f"URL: {item['link']}\n"
                    f"Content: {item['content']}\n"
                    "---" for item in web_content
                ])
                
                # Construct prompt including relevant RAG context if any
                prompt = f"""Analyze and summarize the key information from these search results that answers the user's query.
Focus on the most recent and relevant information.

Query: {query}
"""

                if rag_summary:
                    prompt += f"\nRelevant Context:\n{rag_summary}\n"

                prompt += f"""
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

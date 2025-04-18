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
import re
import threading

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
            prompt = f"""You are a helpful assistant that creates concise summaries while preserving key details and examples. Focus on maintaining the practical value of the content.

Please provide a detailed summary of the following content, preserving key information, examples, and practical details:

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

def is_content_relevant(content: str, query: str, context: Dict) -> bool:
    """
    Quick binary check if content is relevant to the query.
    Returns: True if relevant, False if not
    """
    if not content.strip():
        return False
        
    start_time = time.time()
    prompt = f"""Determine if this content is meaningfully relevant to answering the query.

Consider content relevant ONLY if it:
- Contains specific information that directly answers the query
- Provides important context or background knowledge needed to understand the topic
- Has clear examples or explanations related to the main subject
- Contains factual information that would help form a complete answer
- Discusses key concepts or terms central to the query

Consider content NOT relevant if it:
- Only mentions terms from the query without meaningful context
- Contains very general or tangential information
- Is too vague or high-level to be useful
- Only has superficial keyword matches
- Would not contribute meaningful information to the answer

Query: "{query}"

Content:
{content[:1000]}

Respond ONLY with TRUE or FALSE. Default to FALSE if unsure."""
    
    try:
        if config.MODEL_SOURCE == "openai":
            response = context["client"].chat.completions.create(
                model=context["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": "You are a relevance filter that identifies content that could help answer the user's query either directly or by providing useful context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content.strip().upper()
        else:
            response = context["client"].chat(
                model=context["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": "You are a relevance filter that identifies content that could help answer the user's query either directly or by providing useful context."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response['message']['content'].strip().upper()
        
        is_relevant = result == "TRUE"
        duration = time.time() - start_time
        logger.debug(f"Relevance check took {duration:.2f}s - Query: '{query[:50]}...' - Result: {is_relevant}")
        return is_relevant
    except Exception as e:
        logger.error(f"Error in relevance check: {str(e)}")
        return True  # Default to including content if check fails

def summarize_rag_results(context_documents: Optional[List[Dict[str, Any]]], max_length: int = 1000, context: Optional[Dict[str, Any]] = None, query: Optional[str] = None) -> str:
    """
    Summarize RAG retrieval results only if they exceed the maximum length.
    Otherwise, return the original content as is.
    """
    if not context_documents:
        logger.info("No context documents provided")
        return ""
        
    try:
        logger.info(f"Processing {len(context_documents)} documents")
        # Sort documents by relevance score if available
        sorted_docs = sorted(
            context_documents, 
            key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0,
            reverse=True
        )
        logger.info(f"Sorted {len(sorted_docs)} documents")
        
        # Process and filter documents
        filtered_contents = []
        
        for doc in sorted_docs:
            # Extract content based on document type
            if isinstance(doc, dict):
                content = doc.get('content', '')
                logger.info(f"Dict document keys: {doc.keys()}")
            elif isinstance(doc, Document):
                content = doc.page_content
                logger.info("Found Document type")
            else:
                logger.warning(f"Unexpected document type: {type(doc)}")
                continue
                
            # Skip empty content
            if not content.strip():
                logger.info("Empty content, skipping")
                continue
                
            # Skip reference sections
            content_parts = content.split("REFERENCES:")
            clean_content = content_parts[0].strip()
            
            # Skip if content is just metadata or references
            if not clean_content or clean_content.lower().startswith("references"):
                logger.info("Content is just references, skipping")
                continue
                
            # Check relevance if we have query and context
            if query and context:
                if is_content_relevant(clean_content, query, context):
                    filtered_contents.append(clean_content)
                    logger.info("Added relevant content")
                else:
                    logger.info("Content not relevant, skipping")
            else:
                filtered_contents.append(clean_content)
                logger.info("Added content (no relevance check)")
        
        # If no valid content after filtering
        if not filtered_contents:
            logger.warning("No valid content after filtering")
            # Return most relevant document's content as fallback
            if sorted_docs:
                first_doc = sorted_docs[0]
                if isinstance(first_doc, dict):
                    return first_doc.get('content', '')[:max_length]
                elif isinstance(first_doc, Document):
                    return first_doc.page_content[:max_length]
            return ""
            
        # Join all content
        all_content = "\n\n".join(filtered_contents)
        total_length = len(all_content)
        logger.info(f"Total content length: {total_length}")
            
        # If total length is within limit, return filtered content
        if total_length <= max_length:
            logger.info("Content within length limit")
            return all_content
            
        logger.info(f"Content exceeded max length ({total_length} > {max_length})")
            
        # Check GPU temperature before using LLM for summarization
        if is_gpu_too_hot():
            logger.warning("GPU temperature too high, falling back to simple truncation")
            return all_content[:max_length]
            
        # If content exceeds max length and we have context, use LLM to generate summary
        if context:
            logger.info("Generating LLM summary...")
            summary = generate_llm_summary(all_content, context)
            if summary and len(summary.strip()) > 0:
                return summary
                
        # Fallback to simple truncation if summarization fails
        logger.warning("Summarization failed or no context provided. Falling back to simple truncation.")
        return all_content[:max_length]
        
    except Exception as e:
        logger.error(f"Error in summarize_rag_results: {str(e)}")
        if context_documents:
            # Return first document's content as fallback
            first_doc = context_documents[0]
            if isinstance(first_doc, dict):
                return first_doc.get('content', '')[:max_length]
            elif isinstance(first_doc, Document):
                return first_doc.page_content[:max_length]
        return ""

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
    Returns a list of search results with fresh content scraped directly from websites.
    """
    results = []
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get initial search results
            logging.info(f"Performing DuckDuckGo search for: {search_query}")
            raw_results = list(ddgs.text(search_query, max_results=10))
            
            # Extract and validate URLs from HTML response
            search_results = []
            for result in raw_results:
                # DuckDuckGo sometimes returns results in different formats
                # Try both 'link' and 'href' fields
                url = result.get('link') or result.get('href')
                title = result.get('title') or result.get('text', '')
                snippet = result.get('body') or result.get('snippet', '')
                
                if url and isinstance(url, str) and ('http://' in url or 'https://' in url):
                    search_results.append({
                        'link': url,
                        'title': title,
                        'body': snippet
                    })
            
            if search_results:
                logging.info(f"Found {len(search_results)} valid search results")
                
                start_time = time.time()
                
                # Process URLs in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=5) as executor:
                    # Create a list of (url, title) tuples to process
                    url_tasks = [(result['link'], result['title']) for result in search_results]
                    
                    logging.info(f"Starting parallel scraping of {len(url_tasks)} URLs with {min(5, len(url_tasks))} threads")
                    
                    # Submit all scraping tasks
                    future_to_url = {
                        executor.submit(lambda x: (
                            x[0], 
                            x[1], 
                            (logging.info(f"Thread {threading.current_thread().name}: Starting scrape of {x[0]}"),
                             time.time(),
                             scrape_webpage(x[0]),
                             time.time())[2:]  # Return only the content and end time
                        ), url_task): url_task[0] 
                        for url_task in url_tasks
                    }
                    
                    # Process completed tasks as they finish
                    for future in concurrent.futures.as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            url, title, (content, end_time) = future.result()
                            scrape_time = end_time - start_time
                            logging.info(f"Thread {threading.current_thread().name}: Completed scrape of {url} in {scrape_time:.2f}s")
                            
                            if content and len(content.strip()) > 100:  # Ensure we have substantial content
                                # Create new result with fresh content
                                enriched_result = {
                                    'url': url,
                                    'title': title,
                                    'fresh_content': content,
                                    'source': 'direct_scrape',
                                    'scrape_time': scrape_time
                                }
                                results.append(enriched_result)
                                logging.info(f"Successfully scraped content from {url} (size: {len(content)} chars)")
                                
                                # Limit to first 3 successful scrapes for quality results
                                if len(results) >= 3:
                                    total_time = time.time() - start_time
                                    logging.info(f"Reached target number of scraped results in {total_time:.2f}s")
                                    return results
                            else:
                                logging.warning(f"Insufficient content scraped from {url}")
                        except Exception as scrape_error:
                            logging.warning(f"Failed to scrape {url}: {str(scrape_error)}")
                            continue
                
                total_time = time.time() - start_time
                logging.info(f"Completed all scraping in {total_time:.2f}s")
                
                # If we got any results at all, return them
                if results:
                    logging.info(f"Returning {len(results)} scraped results")
                    return results
                
                # Only if we failed to get ANY scraped content, fall back to search snippets
                if not results:
                    logging.warning("Failed to scrape any pages, falling back to search snippets")
                    fallback_results = []
                    for r in search_results[:3]:
                        if r.get('link') and r.get('title') and r.get('body'):
                            fallback_result = {
                                'url': r['link'],
                                'title': r['title'],
                                'fresh_content': r['body'],
                                'source': 'search_snippet'
                            }
                            fallback_results.append(fallback_result)
                            logging.info(f"Added fallback result from {r['link']}")
                    
                    if fallback_results:
                        return fallback_results
                    else:
                        logging.warning("No valid search results found")
            else:
                logging.warning("No valid URLs found in search results")
                # Try direct news site URLs for news queries
                if any(term in search_query.lower() for term in ['news', 'latest', 'update', 'current']):
                    perth_news_sites = [
                        'https://www.perthnow.com.au/news',
                        'https://www.watoday.com.au',
                        'https://www.abc.net.au/news/wa',
                        'https://thewest.com.au'
                    ]
                    logging.info("Trying direct news site URLs")
                    for url in perth_news_sites:
                        try:
                            fresh_content = scrape_webpage(url)
                            if fresh_content and len(fresh_content.strip()) > 100:
                                enriched_result = {
                                    'url': url,
                                    'title': f"Latest news from {url.split('//')[1].split('/')[0]}",
                                    'fresh_content': fresh_content,
                                    'source': 'direct_news'
                                }
                                results.append(enriched_result)
                                if len(results) >= 2:  # Limit direct news site results
                                    return results
                        except Exception as e:
                            logging.warning(f"Failed to scrape news site {url}: {str(e)}")
                            continue
            
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

def scrape_webpage(url: str) -> Optional[str]:
    """
    Scrape content from a webpage using BeautifulSoup.
    Returns cleaned text content or None if scraping fails.
    """
    try:
        # Configure session with common headers
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1'
        }
        session.headers.update(headers)
        
        # Small delay to avoid overwhelming servers
        time.sleep(random.uniform(0.5, 1.0))
        
        response = session.get(url, timeout=15, allow_redirects=True)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Handle common HTTP errors
        if response.status_code in [403, 429]:
            logging.warning(f"{response.status_code} error for {url}. Trying with different user agent...")
            time.sleep(random.uniform(1.0, 2.0))
            # Retry with different user agent
            headers['User-Agent'] = random.choice([
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ])
            session.headers.update(headers)
            response = session.get(url, timeout=15)
            
        # Check if we got HTML content
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            logging.warning(f"Unexpected content type for {url}: {content_type}")
            return None
            
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
            
        content_parts = []
        
        # First try to find article content (news sites)
        article_content = None
        for selector in ['article', '.article', '.article-content', '.story-content', '.post-content', 
                        '[data-testid="article-body"]', '.article-body', '.entry-content']:
            article_content = soup.select_one(selector)
            if article_content:
                break
                
        if article_content:
            # Extract article text with structure
            for element in article_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = element.get_text(strip=True)
                if text and len(text) > 20:  # Skip very short fragments
                    content_parts.append(text)
        else:
            # Fallback to main content area
            main_content = None
            for container in ['main', '[role="main"]', '#content', '.content', '.main-content']:
                main_content = soup.select_one(container)
                if main_content:
                    break
                    
            # If still no main content, use body but be more selective
            if not main_content:
                main_content = soup.body
                
            if main_content:
                # Get text with semantic structure preserved
                for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    # Skip navigation and unwanted elements
                    if element.find_parent(['nav', 'header', 'footer', 'aside']):
                        continue
                        
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:  # Skip very short fragments
                        content_parts.append(text)
        
        # Look for publication date and author
        metadata = []
        for meta in soup.find_all('meta'):
            if meta.get('property') in ['article:published_time', 'article:modified_time', 'article:author']:
                metadata.append(f"{meta['property']}: {meta.get('content', '')}")
                
        if metadata:
            content_parts.insert(0, "Article Metadata:\n" + "\n".join(metadata) + "\n")
            
        # Join all parts and clean up
        if content_parts:
            content = '\n\n'.join(content_parts)
            # Clean up excessive whitespace
            content = re.sub(r'\n{3,}', '\n\n', content)
            return content
            
        return None
        
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return None

def evaluate_search_results(results: List[Dict], original_query: str, context: Dict) -> Tuple[bool, Optional[str], List[Dict]]:
    """
    Evaluate search results for relevance and suggest follow-up if needed.
    Returns: (needs_followup, followup_query, filtered_results)
    """
    if not results:
        return True, "Try a more general search", []

    # Format results for LLM evaluation with comprehensive details
    results_str = "\n\n".join([
        f"Title: {r.get('title', '')}\n"
        f"URL: {r.get('link', '') or r.get('url', '') or r.get('href', '')}\n"
        f"Content: {r.get('body', '') or r.get('snippet', '')}"
        for r in results[:5]  # Limit to first 5 for evaluation
    ])

    prompt = f"""Evaluate if these search results are relevant to the query.
A result is relevant if it:
- Contains information that helps answer the query
- Provides useful context or background
- Has current or factual information about the topic
- Mentions specific details related to the query

Query: {original_query}

Search Results:
{results_str}

Respond in this format:
RELEVANT: [true/false for each result, numbered]
REASON: [brief reason for each]
FOLLOWUP: [suggested followup query if results aren't ideal, or 'none' if good]
FILTERED_INDEX: [list indices of relevant results, e.g. [0,2,4]]"""

    try:
        if config.MODEL_SOURCE == "openai":
            response = context["client"].chat.completions.create(
                model=context["LLM_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            eval_response = response.choices[0].message.content
        else:
            response = context["client"].chat(
                model=context["LLM_MODEL"],
                messages=[{"role": "user", "content": prompt}]
            )
            eval_response = response['message']['content']

        logger.info("LLM relevance evaluation response:\n" + eval_response)

        # Extract filtered indices
        filtered_index_match = re.search(r'FILTERED_INDEX:\s*\[([\d\s,]*)\]', eval_response)
        if filtered_index_match:
            try:
                filtered_indices = [int(i.strip()) for i in filtered_index_match.group(1).split(',') if i.strip()]
                # Always include at least one result if we found any
                if not filtered_indices and results:
                    filtered_indices = [0]
                filtered_results = [results[i] for i in filtered_indices if i < len(results)]
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing filtered indices: {str(e)}")
                filtered_results = results[:3]  # Default to first 3 results
        else:
            filtered_results = results[:3]  # Default to first 3 results if no indices found

        # Extract followup suggestion
        followup_match = re.search(r'FOLLOWUP:\s*(.+?)(?:\n|$)', eval_response)
        followup = followup_match.group(1).strip() if followup_match else None
        needs_followup = followup and followup.lower() != 'none'

        # Return at least some results
        if not filtered_results and results:
            logger.info("No results passed relevance filter, using first result as fallback")
            filtered_results = [results[0]]
            needs_followup = True
            followup = "Try a more specific search"

        # Log URLs for debugging
        if filtered_results:
            logger.info(f"Web search relevance check - Query: '{original_query[:50]}...' - Found {len(filtered_results)} relevant results")
            logger.info(f"Relevant result URLs: {[r.get('link', '') or r.get('url', '') or r.get('href', '') for r in filtered_results]}")

        return needs_followup, followup if needs_followup else None, filtered_results

    except Exception as e:
        logger.error(f"Error in search result evaluation: {str(e)}")
        return False, None, results[:3]  # Return first 3 results on error

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
    
    # Add debug logging for LLM response
    logging.info(f"LLM relevance evaluation response:\n{eval_text}")
    
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
    Uses ThreadPoolExecutor for parallel processing with improved concurrency.
    Returns content with clickable source links.
    """
    all_content = []
    processed_urls = set()
    start_time = time.time()
    url_index = 1  # Counter for source references
    
    def is_timed_out() -> bool:
        elapsed = time.time() - start_time
        if elapsed >= 30:
            logging.warning(f"Web search timed out after {elapsed:.1f} seconds")
            return True
        return False

    # First process any results that already have fresh content
    for result in search_results:
        if len(all_content) >= 3 or is_timed_out():
            break
            
        url = result.get('url') or result.get('link')
        if not url or url in processed_urls:
            continue
            
        processed_urls.add(url)
        
        # Use existing fresh_content if available
        content = result.get('fresh_content')
        if content:
            is_relevant, excerpt = evaluate_content_relevance(content, query, context)
            if is_relevant:
                logging.info(f"Using existing fresh content from {url}")
                # Format content with source reference
                formatted_content = f"[{url_index}] {excerpt}\n[Source {url_index}]: {url}"
                all_content.append(formatted_content)
                url_index += 1
                continue

    def process_single_url(url: str, index: int) -> Optional[str]:
        """Process a single URL with existing scrape_webpage function"""
        if is_timed_out() or not url or url in processed_urls:
            return None
            
        if not url.startswith(('http://', 'https://')):
            logging.warning(f"Invalid URL format: {url}")
            return None
            
        processed_urls.add(url)
        try:
            content = scrape_webpage(url)
            if content:
                is_relevant, excerpt = evaluate_content_relevance(content, query, context)
                if is_relevant:
                    # Format content with source reference
                    return f"[{index}] {excerpt}\n[Source {index}]: {url}"
        except Exception as e:
            logging.warning(f"Error processing {url}: {str(e)}")
        return None

    # Only scrape additional URLs if we need more content
    if len(all_content) < 3 and not is_timed_out():
        remaining_urls = []
        for result in search_results:
            if len(remaining_urls) >= (3 - len(all_content)) * 2:  # Get 2x the URLs we still need
                break
            url = result.get('url') or result.get('link')
            if url and url not in processed_urls and url.startswith(('http://', 'https://')):
                remaining_urls.append((url, url_index))
                url_index += 1

        if remaining_urls:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(process_single_url, url, idx): url for url, idx in remaining_urls}
                
                for future in concurrent.futures.as_completed(future_to_url):
                    if is_timed_out() or len(all_content) >= 3:
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

    elapsed = time.time() - start_time
    logging.info(f"Web content processing completed in {elapsed:.1f} seconds, found {len(all_content)} relevant results")
    
    # Join all content with double newlines for readability
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
        alt_queries = [q.strip().replace('"', '').replace("'", "").replace(",", "").strip() for q in alt_queries if q.strip()]
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
    # Early return if query is None, empty, or N/A
    if not query or query.strip() == "" or query.strip().upper() in ["N/A", "NA", "NONE"]:
        return {
            "needs_web_search": False,
            "web_results": ""
        }
    
    # Early return if web search is disabled
    if context.get('skip_web_search', False):
        return {
            "needs_web_search": False,
            "web_results": ""
        }
        
    result = {
        "needs_web_search": False,
        "web_results": ""
    }
    
    try:
        # Get recent conversation history
        recent_messages = []
        if "memory" in context:
            try:
                memory = context["memory"]
                if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
                    chat_history = memory.chat_memory.messages[-5:]
                elif hasattr(memory, "buffer"):
                    # Handle ConversationBufferMemory
                    chat_history = [{"type": "human" if i % 2 == 0 else "assistant", "content": msg} 
                                  for i, msg in enumerate(memory.buffer.split('\n')[-10:] if memory.buffer else [])]
                else:
                    chat_history = []
                recent_messages = [f"{msg['type']}: {msg['content']}" for msg in chat_history]
            except Exception as e:
                logging.warning(f"Could not retrieve chat history: {str(e)}")
        
        # First, check for explicit search requests or real-time information needs
        explicit_search_terms = ["search", "look up", "find", "what is", "how to", "current", "latest", "news", "weather", "price", "status"]
        time_related_terms = ["time", "today", "now", "current", "latest"]
        
        query_lower = query.lower()
        is_explicit_search = any(term in query_lower for term in explicit_search_terms)
        needs_current_time = any(term in query_lower for term in time_related_terms)
        
        # Initialize search terms list
        search_terms = []
        
        # Immediate search for time-related queries
        if needs_current_time and any(word in query_lower for word in ["time", "hour", "today", "now"]):
            result["needs_web_search"] = True
        elif is_explicit_search:
            # Use the query directly, just clean it up
            cleaned_query = query_lower
            for term in ["search for", "look up", "find", "what is", "how to"]:
                cleaned_query = cleaned_query.replace(term, "").strip()
            result["needs_web_search"] = True
            search_terms = [cleaned_query]  # Use cleaned query as a search term suggestion
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
   - For location-specific information (restaurants, businesses, etc), RAG must contain CURRENT details
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
- Greetings and farewells
- Questions about feelings or opinions
- Personal questions about the AI
- Simple yes/no responses
- Questions about capabilities
- Very short follow-up questions
- General chitchat
- Anything that can be answered from conversation history

Current Query: {query}
RAG Summary: {rag_summary if rag_summary else "No relevant information found in RAG"}
Recent Messages: {recent_messages}

Respond in this format:
NEEDS_SEARCH: [YES/NO]
REASON: [Explain why search is or isn't needed, including information completeness]
SEARCH_TERMS: [If YES or if RAG only has partial info, provide 2-5 key search terms without quotes]"""

            if config.MODEL_SOURCE == "openai":
                response = context["client"].chat.completions.create(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.1  # Lower temperature for more consistent decisions
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
                    search_terms = []
                    in_search_terms = False
                    for line in lines:
                        if line.startswith("SEARCH_TERMS:"):
                            in_search_terms = True
                            # Get the rest of the line after the colon
                            terms = line.split(":", 1)[1].strip()
                            if terms and not terms.startswith("*"):  # If terms are on same line
                                search_terms.append(terms)
                        elif in_search_terms and line.startswith("*"):
                            # Get terms from bullet points
                            terms = line.strip("* ").strip('"').strip()
                            if terms:
                                search_terms.append(terms)
                        elif in_search_terms and not line.startswith("*"):
                            # End of search terms section
                            break
                    
                    if search_terms:
                        search_query = ' '.join(search_terms)
                    else:
                        search_query = query  # Fall back to original query if no terms found
                    
        # If search is needed, generate optimized search query using LLM
        if result["needs_web_search"]:
            search_query_prompt = f"""Generate a clear and effective search query for this user request.
The query should be specific, focused, and optimized for web search.
Remove any conversational elements and focus on the key information needed.

User Request: {query}
{"Suggested Search Terms: " + ", ".join(search_terms) if search_terms else ""}

Response format:
SEARCH_QUERY: [2-5 word optimized search query]

Example:
User: "What's the weather like in New York?"
SEARCH_QUERY: new york weather forecast"""

            if config.MODEL_SOURCE == "openai":
                response = context["client"].chat.completions.create(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": search_query_prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                search_query_response = response.choices[0].message.content.strip()
            else:
                response = context["client"].chat(
                    model=context["LLM_MODEL"],
                    messages=[{"role": "user", "content": search_query_prompt}],
                )
                search_query_response = response['message']['content'].strip()

            # Extract search query
            search_query = ""
            for line in search_query_response.split('\n'):
                if line.startswith('SEARCH_QUERY:'):
                    search_query = line.replace('SEARCH_QUERY:', '').strip()
                    break
            
            if not search_query:  # If LLM fails to generate query
                search_query = ' '.join(query.lower().split()[:5])  # Use first 5 words as fallback
                logging.warning(f"LLM failed to generate search query, using simplified: {search_query}")
            else:
                logging.info(f"Using LLM generated search query: {search_query}")
            
            # Skip search if terms are N/A or empty
            if not search_query or search_query.upper() in ["N/A", "NA", "NONE"]:
                return {
                    "needs_web_search": False,
                    "web_results": ""
                }
            
            # Clean up search query
            search_query = search_query.replace('"', '').replace("'", "").replace(",", "").strip()
        else:
            search_query = ""
        
        # Perform web search if needed
        if result["needs_web_search"]:
            logging.info(f"Web search deemed necessary, performing search with query: {search_query}")
            
            # Play the searching sound when NEEDS_SEARCH is YES
            try:
                continuous_listener = context.get("continuous_listener")
                if continuous_listener and hasattr(continuous_listener, 'play_searching_sound'):
                    logging.info("Playing searching sound since web search is needed")
                    continuous_listener.play_searching_sound()
            except Exception as e:
                logging.error(f"Error playing searching sound: {e}")
            
            # Get DDGS instance from context
            ddgs = context.get('ddgs')
            if not ddgs:
                # Create new instance if not in context
                ddgs = DDGS()
            
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
                
            # Ensure search_results is a list of dictionaries
            if isinstance(search_results, list):
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
                        
                        # Get content from fresh_content first, then fall back to body/snippet
                        content = result_item.get('fresh_content', '')  # Try fresh content first
                        if not content:
                            content = result_item.get('body', '')  # Fall back to body
                        if not content:
                            content = result_item.get('snippet', '')  # Fall back to snippet
                        
                        if not url or not title or not content:
                            logging.warning(f"Skipping result due to missing data: {result_item}")
                            continue
                        
                        # Log search result metadata
                        logging.info(f"Found search result: Title='{title}' URL='{url}'")
                        logging.debug(f"Content preview: {content[:100]}...")
                        
                        # Only include substantial content
                        if len(content.strip()) > 100:  # Ensure we have meaningful content
                            web_content.append({
                                'title': title,
                                'link': url,
                                'content': content[:2000]  # Increased from 2000 to allow more context
                            })
                            logging.info(f"Added content from: {title}")
                    except Exception as e:
                        logging.warning(f"Error processing search result: {str(e)}")
                        continue
            
                # Summarize web content using LLM
                if web_content:
                    web_content_str = "\n\n".join([
                        f"Source: {item['title']}\n"
                        f"URL: {item['link']}\n"
                        f"Content: {item['content']}\n"
                        "---" for item in web_content[:3]  # Limit to top 3 sources for better summaries
                    ])
                    
                    # Construct prompt including relevant RAG context if any
                    prompt = f"""Analyze and summarize the key information from these search results that answers the user's query.
Focus on the most recent and relevant information from ALL provided sources. Combine information from different sources to provide a comprehensive summary.

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
                            max_tokens=500,  # Increased from 150 to allow for longer summaries
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
                    detailed_content = get_detailed_web_content(search_results, query, context)  # Pass original search_results instead of filtered_results
                    
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

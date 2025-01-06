import logging
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def summarize_rag_results(context_documents: Optional[List[Dict[str, Any]]], max_length: int = 1000) -> str:
    """
    Summarize RAG retrieval results only if they exceed the maximum length.
    Otherwise, return the original content as is.
    
    Args:
        context_documents (Optional[List[Dict[str, Any]]]): List of retrieved documents from RAG
        max_length (int): Maximum length of the summarized context in characters
        
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
            
        # Otherwise, perform summarization
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
        
        logger.info(f"Content exceeded max length ({total_length} > {max_length}). Summarized to {len(final_summary)} characters")
        return final_summary
        
    except Exception as e:
        logger.error(f"Error summarizing RAG results: {str(e)}")
        return "" if not context_documents else context_documents[0].get('content', '') if isinstance(context_documents[0], dict) else context_documents[0].page_content

"""
Common utility functions for the Financial Document Intelligence System.
"""
import re
import hashlib
import time
from typing import List, Dict, Any, Optional
from functools import wraps
from datetime import datetime


def clean_text(text: str) -> str:
    """
    Clean and normalize text from SEC filings.
    
    Removes excessive whitespace, HTML artifacts, and normalizes characters.
    """
    if not text:
        return ""
    
    # Remove HTML tags if any remain
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def generate_doc_id(content: str, metadata: dict = None) -> str:
    """Generate a unique document ID from content and metadata."""
    hash_input = content[:500]
    if metadata:
        hash_input += str(sorted(metadata.items()))
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = None
) -> List[str]:
    """
    Split text into overlapping chunks with smart boundary detection.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        separators: Priority list of separators for splitting
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    if separators is None:
        separators = ["\n\n", "\n", ". ", ", ", " "]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to find a good breaking point
        best_break = end
        for sep in separators:
            # Look for separator near the end of the chunk
            search_start = max(start + chunk_size // 2, start)
            idx = text.rfind(sep, search_start, end)
            if idx != -1:
                best_break = idx + len(sep)
                break
        
        chunk = text[start:best_break].strip()
        if chunk:
            chunks.append(chunk)
        
        start = best_break - chunk_overlap
    
    return chunks


def format_citations(sources: List[Dict[str, Any]]) -> str:
    """Format source documents as numbered citations."""
    if not sources:
        return "No sources available."
    
    citations = []
    for i, source in enumerate(sources, 1):
        company = source.get("company", "Unknown")
        filing_type = source.get("filing_type", "N/A")
        date = source.get("filing_date", "N/A")
        section = source.get("section", "N/A")
        
        citations.append(
            f"[{i}] {company} | {filing_type} | {date} | Section: {section}"
        )
    
    return "\n".join(citations)


def timer_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


def extract_filing_metadata(text: str, filename: str = "") -> Dict[str, str]:
    """Extract metadata from filing text or filename."""
    metadata = {
        "filing_type": "",
        "company": "",
        "filing_date": "",
        "cik": "",
    }
    
    # Try to extract from filename
    if filename:
        type_match = re.search(r'(10-K|10-Q|8-K)', filename, re.IGNORECASE)
        if type_match:
            metadata["filing_type"] = type_match.group(1).upper()
    
    # Try to extract from text
    type_match = re.search(
        r'FORM\s+(10-K|10-Q|8-K)', text[:2000], re.IGNORECASE
    )
    if type_match:
        metadata["filing_type"] = type_match.group(1).upper()
    
    date_match = re.search(
        r'(?:filed|date)[:\s]+(\d{4}-\d{2}-\d{2})', text[:2000], re.IGNORECASE
    )
    if date_match:
        metadata["filing_date"] = date_match.group(1)
    
    return metadata


def calculate_token_estimate(text: str) -> int:
    """Rough estimate of tokens (1 token ≈ 4 characters)."""
    return len(text) // 4


def batch_process(items: list, batch_size: int = 32):
    """Yield batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

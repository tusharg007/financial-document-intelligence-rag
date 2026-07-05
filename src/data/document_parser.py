"""
Document parser and chunker for SEC filings.

Handles HTML/text parsing, section extraction, and intelligent chunking
with financial-domain awareness.
"""
import re
from typing import List, Dict, Any, Optional
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

from src.utils.logger import get_logger
from src.utils.helpers import clean_text, generate_doc_id, chunk_text
from config.settings import settings

logger = get_logger("document_parser")


# SEC filing section patterns
SECTION_PATTERNS = {
    "10-K": {
        "Risk Factors": [
            r"(?:ITEM\s*1A[\.\s\-]*RISK\s*FACTORS)",
            r"(?:Risk\s*Factors)",
        ],
        "Management Discussion and Analysis": [
            r"(?:ITEM\s*7[\.\s\-]*MANAGEMENT)",
            r"(?:Management.s?\s*Discussion\s*and\s*Analysis)",
        ],
        "Financial Statements": [
            r"(?:ITEM\s*8[\.\s\-]*FINANCIAL\s*STATEMENTS)",
            r"(?:Consolidated\s*Statements?\s*of\s*(?:Income|Operations))",
        ],
        "Business": [
            r"(?:ITEM\s*1[\.\s\-]*BUSINESS(?!\s*1A))",
        ],
        "Properties": [
            r"(?:ITEM\s*2[\.\s\-]*PROPERTIES)",
        ],
    },
    "10-Q": {
        "Management Discussion and Analysis": [
            r"(?:ITEM\s*2[\.\s\-]*MANAGEMENT)",
            r"(?:Management.s?\s*Discussion)",
        ],
        "Risk Factors": [
            r"(?:ITEM\s*1A[\.\s\-]*RISK\s*FACTORS)",
            r"(?:Risk\s*Factors)",
        ],
        "Financial Statements": [
            r"(?:ITEM\s*1[\.\s\-]*FINANCIAL\s*STATEMENTS)",
        ],
    },
    "8-K": {
        "Material Event": [
            r"(?:ITEM\s*\d+\.\d+)",
        ],
        "Earnings Release": [
            r"(?:press\s*release|earnings\s*release)",
        ],
    },
}


class DocumentParser:
    """Parse and chunk SEC filings into indexed documents."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def parse_html_filing(self, html_content: str) -> str:
        """
        Extract clean text from an HTML SEC filing.
        
        Args:
            html_content: Raw HTML content of the filing
            
        Returns:
            Cleaned plain text
        """
        try:
            if BeautifulSoup is None:
                return clean_text(re.sub(r"<[^>]+>", " ", html_content or ""))
            soup = BeautifulSoup(html_content, "lxml")
        except Exception:
            soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()

        # Extract text
        text = soup.get_text(separator="\n")
        text = clean_text(text)

        logger.debug(f"Parsed HTML: {len(html_content)} chars -> {len(text)} chars")
        return text

    def extract_sections(
        self,
        text: str,
        filing_type: str
    ) -> Dict[str, str]:
        """
        Extract named sections from a filing.
        
        Args:
            text: Plain text of the filing
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            
        Returns:
            Dict mapping section names to their text content
        """
        patterns = SECTION_PATTERNS.get(filing_type, {})
        sections = {}

        for section_name, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = match.start()
                    # Find the next section start or use a reasonable chunk
                    end = min(start + 15000, len(text))
                    
                    # Try to find next section header
                    next_match = re.search(
                        r"(?:ITEM\s*\d+[A-Z]?[\.\s\-])", 
                        text[match.end():end],
                        re.IGNORECASE
                    )
                    if next_match:
                        end = match.end() + next_match.start()

                    sections[section_name] = clean_text(text[start:end])
                    break

        if not sections:
            # If no sections found, treat entire text as one section
            sections["Full Document"] = text

        logger.info(f"Extracted {len(sections)} sections from {filing_type}")
        return sections

    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split a document into overlapping chunks with metadata.
        
        Args:
            text: Document text to chunk
            metadata: Base metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if metadata is None:
            metadata = {}

        chunks = chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        result = []
        for i, chunk_content in enumerate(chunks):
            if not chunk_content.strip():
                continue

            doc_id = generate_doc_id(chunk_content, metadata)
            chunk_doc = {
                "content": chunk_content,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **metadata
            }
            result.append(chunk_doc)

        logger.debug(f"Created {len(result)} chunks from {len(text)} chars")
        return result

    def process_filing(
        self,
        content: str,
        metadata: Dict[str, Any],
        is_html: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Full processing pipeline: parse -> extract sections -> chunk.
        
        Args:
            content: Raw filing content (HTML or text)
            metadata: Filing metadata (company, type, date, etc.)
            is_html: Whether content is HTML
            
        Returns:
            List of processed document chunks with metadata
        """
        # Parse HTML if needed
        if is_html:
            text = self.parse_html_filing(content)
        else:
            text = clean_text(content)

        if not text:
            logger.warning("Empty document after parsing")
            return []

        # Extract sections
        filing_type = metadata.get("filing_type", "")
        sections = self.extract_sections(text, filing_type)

        # Chunk each section
        all_chunks = []
        for section_name, section_text in sections.items():
            section_metadata = {
                **metadata,
                "section": section_name,
            }
            chunks = self.chunk_document(section_text, section_metadata)
            all_chunks.extend(chunks)

        logger.info(
            f"Processed {metadata.get('filing_type', 'unknown')} filing: "
            f"{len(sections)} sections, {len(all_chunks)} chunks"
        )
        return all_chunks

    def process_sample_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process pre-loaded sample documents (already chunked).
        Adds doc_ids and standardizes format.
        
        Args:
            documents: List of sample document dictionaries
            
        Returns:
            Processed documents with IDs
        """
        processed = []
        for i, doc in enumerate(documents):
            doc_id = generate_doc_id(doc["content"], doc)
            processed_doc = {
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(documents),
                **doc
            }
            processed.append(processed_doc)

        logger.info(f"Processed {len(processed)} sample documents")
        return processed


# Module-level instance
_parser = None

def get_parser() -> DocumentParser:
    """Get the global DocumentParser instance."""
    global _parser
    if _parser is None:
        _parser = DocumentParser()
    return _parser

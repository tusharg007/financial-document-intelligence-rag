"""
Company comparison agent.

Compares financial data across multiple companies' SEC filings,
generating structured side-by-side analyses.
"""
from typing import List, Dict, Any, Optional

from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.retrieval.reranker import get_reranker
from src.utils.logger import get_logger
from src.utils.helpers import format_citations
from config.settings import settings

logger = get_logger("comparison_agent")


class ComparisonAgent:
    """
    Generates side-by-side comparisons of company filings.
    
    Retrieves matching sections from multiple companies and produces
    structured comparison tables with key differences highlighted.
    """

    def __init__(self):
        self.retriever = get_hybrid_retriever()
        self.reranker = get_reranker()

    def compare_companies(
        self,
        companies: List[str],
        topic: str,
        filing_type: str = "10-K"
    ) -> Dict[str, Any]:
        """
        Compare companies on a specific topic.
        
        Args:
            companies: List of company names/tickers to compare
            topic: Topic to compare (e.g., "risk factors", "revenue")
            filing_type: Type of filing to search
            
        Returns:
            Structured comparison with per-company findings
        """
        logger.info(f"Comparing {companies} on topic: {topic}")
        
        comparison = {
            "topic": topic,
            "companies": {},
            "summary": "",
            "key_differences": [],
            "citations": [],
        }
        
        all_citations = []
        
        for company in companies:
            # Build company-specific query
            query = f"{company} {topic} {filing_type}"
            
            # Retrieve documents for this company
            results = self.retriever.retrieve(query, top_k=10)
            
            # Filter to this company
            company_docs = [
                doc for doc in results
                if company.lower() in doc.get("metadata", {}).get("company", "").lower()
                or company.lower() in doc.get("metadata", {}).get("ticker", "").lower()
            ]
            
            if not company_docs:
                company_docs = results[:3]
            
            # Rerank
            reranked = self.reranker.rerank(
                f"{company} {topic}", company_docs, top_k=3
            )
            
            # Extract key findings
            findings = []
            for doc in reranked:
                content = doc["content"]
                # Extract key sentences with numbers/data
                sentences = [
                    s.strip() for s in content.split(".")
                    if len(s.strip()) > 20
                ]
                findings.extend(sentences[:3])
                
                meta = doc.get("metadata", {})
                all_citations.append({
                    "company": meta.get("company", company),
                    "filing_type": meta.get("filing_type", filing_type),
                    "filing_date": meta.get("filing_date", "N/A"),
                    "section": meta.get("section", "N/A"),
                    "relevance_score": round(doc.get("rerank_score", 0), 4),
                })
            
            comparison["companies"][company] = {
                "findings": findings[:5],
                "num_sources": len(reranked),
                "documents": reranked,
            }
        
        # Generate comparison summary
        comparison["summary"] = self._generate_comparison_summary(
            comparison["companies"], topic
        )
        comparison["key_differences"] = self._extract_differences(
            comparison["companies"], topic
        )
        comparison["citations"] = all_citations
        
        return comparison

    def _generate_comparison_summary(
        self,
        company_data: Dict[str, Any],
        topic: str
    ) -> str:
        """Generate a text summary of the comparison."""
        summary_parts = [f"## Comparison: {topic.title()}\n"]
        
        for company, data in company_data.items():
            findings = data.get("findings", [])
            summary_parts.append(f"\n### {company}")
            
            if findings:
                for finding in findings[:3]:
                    summary_parts.append(f"- {finding}.")
            else:
                summary_parts.append("- No specific findings available.")
        
        return "\n".join(summary_parts)

    def _extract_differences(
        self,
        company_data: Dict[str, Any],
        topic: str
    ) -> List[str]:
        """Identify key differences between companies."""
        differences = []
        
        companies = list(company_data.keys())
        if len(companies) < 2:
            return differences
        
        # Simple keyword-based difference detection
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                comp_a = companies[i]
                comp_b = companies[j]
                
                findings_a = " ".join(company_data[comp_a].get("findings", []))
                findings_b = " ".join(company_data[comp_b].get("findings", []))
                
                # Check for contrasting signals
                if "increase" in findings_a.lower() and "decrease" in findings_b.lower():
                    differences.append(
                        f"{comp_a} shows increases while {comp_b} reports decreases in {topic}"
                    )
                if "loss" in findings_a.lower() and "profit" in findings_b.lower():
                    differences.append(
                        f"Profitability differs: {comp_a} reports losses vs {comp_b}'s profits"
                    )
                if "growth" in findings_a.lower() and "decline" in findings_b.lower():
                    differences.append(
                        f"Growth trajectory differs between {comp_a} and {comp_b}"
                    )
        
        if not differences:
            differences.append(
                f"Both companies address {topic} but with different emphasis and detail levels"
            )
        
        return differences

    def generate_comparison_table(
        self,
        companies: List[str],
        metrics: List[str]
    ) -> str:
        """Generate a markdown comparison table."""
        # Header
        header = "| Metric | " + " | ".join(companies) + " |"
        separator = "|" + "|".join(["---"] * (len(companies) + 1)) + "|"
        
        rows = [header, separator]
        
        for metric in metrics:
            row_data = [metric]
            for company in companies:
                # Retrieve metric data
                query = f"{company} {metric}"
                results = self.retriever.retrieve(query, top_k=3)
                
                # Extract value from best result
                if results:
                    content = results[0]["content"]
                    # Try to extract a short value
                    import re
                    numbers = re.findall(r'\$[\d,]+\.?\d*\s*(?:billion|million|%)?', content)
                    if numbers:
                        row_data.append(numbers[0])
                    else:
                        row_data.append(content[:50] + "...")
                else:
                    row_data.append("N/A")
            
            rows.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join(rows)


# Singleton
_agent = None

def get_comparison_agent() -> ComparisonAgent:
    """Get the global ComparisonAgent instance."""
    global _agent
    if _agent is None:
        _agent = ComparisonAgent()
    return _agent

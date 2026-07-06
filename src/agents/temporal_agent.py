"""
Temporal analysis agent.

Tracks how company narratives, metrics, and risk disclosures change
across filing periods, generating timeline-based analysis.
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.retrieval.pipeline import get_retrieval_pipeline
from src.retrieval.reranker import get_reranker
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("temporal_agent")


class TemporalAgent:
    """
    Analyzes how company disclosures evolve over time.
    
    Tracks changes in risk factors, financial metrics, and strategic
    narratives across quarterly and annual filings.
    """

    def __init__(self):
        self.retriever = get_retrieval_pipeline()
        self.reranker = get_reranker()

    def analyze_temporal_changes(
        self,
        company: str,
        topic: str,
        periods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze how a topic changes across filing periods.
        
        Args:
            company: Company name or ticker
            topic: Topic to track (e.g., "risk factors", "revenue")
            periods: Specific periods to analyze (e.g., ["2022", "2023", "2024"])
            
        Returns:
            Timeline analysis with per-period findings
        """
        logger.info(f"Temporal analysis: {company} | {topic}")
        
        # Retrieve all relevant documents
        query = f"{company} {topic}"
        all_docs = self.retriever.retrieve(query, top_k=20)
        
        # Filter to company
        company_docs = [
            doc for doc in all_docs
            if company.lower() in doc.get("metadata", {}).get("company", "").lower()
            or company.lower() in doc.get("metadata", {}).get("ticker", "").lower()
        ]
        
        if not company_docs:
            company_docs = all_docs
        
        # Group by fiscal period
        period_groups = defaultdict(list)
        for doc in company_docs:
            meta = doc.get("metadata", {})
            year = meta.get("fiscal_year", "Unknown")
            quarter = meta.get("fiscal_quarter", "")
            period_key = f"{year} {quarter}".strip()
            period_groups[period_key].append(doc)
        
        # Sort periods chronologically
        sorted_periods = sorted(period_groups.keys())
        
        if periods:
            sorted_periods = [p for p in sorted_periods if any(
                period in p for period in periods
            )]
        
        # Analyze each period
        timeline = []
        all_citations = []
        
        for period in sorted_periods:
            docs = period_groups[period]
            
            # Rerank for this period
            reranked = self.reranker.rerank(
                f"{company} {topic} {period}", docs, top_k=3
            )
            
            # Extract key points
            key_points = []
            metrics = []
            
            for doc in reranked:
                content = doc["content"]
                sentences = [
                    s.strip() for s in content.split(".")
                    if len(s.strip()) > 20
                ]
                
                for sent in sentences[:3]:
                    # Categorize as metric or narrative
                    import re
                    if re.search(r'\$[\d,]+|[\d.]+%|\d+\s*billion|\d+\s*million', sent):
                        metrics.append(sent + ".")
                    else:
                        key_points.append(sent + ".")
                
                meta = doc.get("metadata", {})
                all_citations.append({
                    "period": period,
                    "company": meta.get("company", company),
                    "filing_type": meta.get("filing_type", "N/A"),
                    "section": meta.get("section", "N/A"),
                    "filing_date": meta.get("filing_date", "N/A"),
                })
            
            timeline.append({
                "period": period,
                "key_points": key_points[:3],
                "metrics": metrics[:3],
                "num_sources": len(docs),
                "documents": reranked,
            })
        
        # Generate trend summary
        trend_summary = self._generate_trend_summary(
            company, topic, timeline
        )
        
        return {
            "company": company,
            "topic": topic,
            "timeline": timeline,
            "trend_summary": trend_summary,
            "num_periods": len(timeline),
            "citations": all_citations,
        }

    def _generate_trend_summary(
        self,
        company: str,
        topic: str,
        timeline: List[Dict]
    ) -> str:
        """Generate a narrative summary of trends over time."""
        if not timeline:
            return f"No temporal data available for {company} on {topic}."
        
        summary_parts = [
            f"## Temporal Analysis: {company} - {topic.title()}",
            f"\nAnalyzed across {len(timeline)} filing periods.\n"
        ]
        
        for entry in timeline:
            period = entry["period"]
            summary_parts.append(f"### {period}")
            
            if entry["metrics"]:
                summary_parts.append("**Key Metrics:**")
                for metric in entry["metrics"][:2]:
                    summary_parts.append(f"- {metric}")
            
            if entry["key_points"]:
                summary_parts.append("**Key Developments:**")
                for point in entry["key_points"][:2]:
                    summary_parts.append(f"- {point}")
            
            summary_parts.append("")
        
        # Add trend observation
        if len(timeline) >= 2:
            summary_parts.append("### Trend Observations")
            first = timeline[0]["period"]
            last = timeline[-1]["period"]
            summary_parts.append(
                f"From {first} to {last}, the data shows evolving "
                f"patterns in {company}'s {topic} disclosures."
            )
        
        return "\n".join(summary_parts)

    def compare_periods(
        self,
        company: str,
        topic: str,
        period_a: str,
        period_b: str
    ) -> Dict[str, Any]:
        """Compare a company's disclosures between two specific periods."""
        analysis = self.analyze_temporal_changes(
            company, topic, periods=[period_a, period_b]
        )
        
        timeline = analysis.get("timeline", [])
        
        comparison = {
            "company": company,
            "topic": topic,
            "period_a": period_a,
            "period_b": period_b,
            "period_a_findings": [],
            "period_b_findings": [],
            "changes": [],
        }
        
        for entry in timeline:
            if period_a in entry["period"]:
                comparison["period_a_findings"] = (
                    entry.get("metrics", []) + entry.get("key_points", [])
                )
            elif period_b in entry["period"]:
                comparison["period_b_findings"] = (
                    entry.get("metrics", []) + entry.get("key_points", [])
                )
        
        return comparison


# Singleton
_agent = None

def get_temporal_agent() -> TemporalAgent:
    """Get the global TemporalAgent instance."""
    global _agent
    if _agent is None:
        _agent = TemporalAgent()
    return _agent

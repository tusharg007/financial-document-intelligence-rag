"""
SEC EDGAR API Client for downloading financial filings.

Implements rate-limited access to the SEC EDGAR database following
the SEC Fair Access Policy (max 10 req/sec, User-Agent required).
"""
import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("edgar_client")


class EdgarClient:
    """Client for accessing SEC EDGAR filings API."""

    BASE_URL = "https://data.sec.gov"
    FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    
    # Rate limiting: max 10 requests per second
    MIN_REQUEST_INTERVAL = 0.12  # ~8 req/sec to be safe

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.sec_edgar_user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time = 0
        self._ticker_cache = {}
        self.raw_dir = Path(settings.raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str) -> Optional[requests.Response]:
        """Make a rate-limited GET request."""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        """
        Look up a company's CIK number by ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            CIK number as zero-padded string, or None if not found
        """
        if not self._ticker_cache:
            logger.info("Loading company tickers from SEC...")
            resp = self._get(self.COMPANY_TICKERS_URL)
            if resp:
                data = resp.json()
                for _, entry in data.items():
                    t = entry.get("ticker", "").upper()
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    self._ticker_cache[t] = cik

        ticker = ticker.upper()
        cik = self._ticker_cache.get(ticker)
        if cik:
            logger.info(f"Found CIK {cik} for {ticker}")
        else:
            logger.warning(f"Ticker {ticker} not found in SEC database")
        return cik

    def get_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get a list of filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types to filter (e.g., ['10-K', '10-Q'])
            limit: Maximum number of filings to return
            
        Returns:
            List of filing metadata dictionaries
        """
        if filing_types is None:
            filing_types = ["10-K", "10-Q", "8-K"]

        cik = self.get_cik_by_ticker(ticker)
        if not cik:
            return []

        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        resp = self._get(url)
        if not resp:
            return []

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})

        filings = []
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        for i in range(min(len(forms), 100)):
            form_type = forms[i] if i < len(forms) else ""
            
            # Filter by filing type
            if form_type not in filing_types:
                continue

            filing = {
                "company": data.get("name", ""),
                "ticker": ticker.upper(),
                "cik": cik,
                "filing_type": form_type,
                "filing_date": dates[i] if i < len(dates) else "",
                "accession_number": accessions[i] if i < len(accessions) else "",
                "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                "description": descriptions[i] if i < len(descriptions) else "",
            }
            filings.append(filing)

            if len(filings) >= limit:
                break

        logger.info(f"Found {len(filings)} filings for {ticker}")
        return filings

    def download_filing(self, filing: Dict[str, Any]) -> Optional[str]:
        """
        Download the full text of a filing.
        
        Args:
            filing: Filing metadata dict from get_company_filings()
            
        Returns:
            Filing text content, or None if download failed
        """
        accession = filing["accession_number"].replace("-", "")
        cik = filing["cik"].lstrip("0")
        primary_doc = filing["primary_document"]

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik}/{accession}/{primary_doc}"
        )

        resp = self._get(url)
        if not resp:
            return None

        # Save to local cache
        safe_name = (
            f"{filing['ticker']}_{filing['filing_type']}_"
            f"{filing['filing_date']}.html"
        )
        filepath = self.raw_dir / safe_name
        filepath.write_text(resp.text, encoding="utf-8")
        logger.info(f"Downloaded filing to {filepath}")

        return resp.text

    def search_filings(
        self,
        query: str,
        filing_types: List[str] = None,
        date_range: tuple = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search EDGAR full-text search for filings matching a query.
        
        Args:
            query: Search query string
            filing_types: Filter by filing types
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            limit: Maximum results
            
        Returns:
            List of matching filing metadata
        """
        params = {
            "q": query,
            "dateRange": "custom",
            "startdt": date_range[0] if date_range else "2023-01-01",
            "enddt": date_range[1] if date_range else datetime.now().strftime("%Y-%m-%d"),
        }
        
        if filing_types:
            params["forms"] = ",".join(filing_types)

        url = "https://efts.sec.gov/LATEST/search-index"
        
        try:
            self._rate_limit()
            resp = self.session.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                results = []
                for hit in hits[:limit]:
                    source = hit.get("_source", {})
                    results.append({
                        "company": source.get("entity_name", ""),
                        "filing_type": source.get("form_type", ""),
                        "filing_date": source.get("file_date", ""),
                        "description": source.get("file_description", ""),
                    })
                return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
        
        return []


# Singleton client
_client = None

def get_edgar_client() -> EdgarClient:
    """Get the global EdgarClient instance."""
    global _client
    if _client is None:
        _client = EdgarClient()
    return _client

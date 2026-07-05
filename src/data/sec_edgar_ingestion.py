"""SEC EDGAR ingestion pipeline."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from config.settings import PROJECT_ROOT, settings


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dash}/{primary_doc}"


class SecEdgarIngestor:
    """Rate-limited SEC EDGAR downloader."""

    def __init__(self, user_agent: str | None = None, raw_dir: str | Path | None = None):
        self.user_agent = user_agent if user_agent is not None else (os.getenv("SEC_EDGAR_USER_AGENT") or settings.sec_edgar_user_agent)
        if not self.user_agent or "@" not in self.user_agent:
            raise ValueError("SEC_EDGAR_USER_AGENT must identify your app and email, e.g. 'Name email@example.com'.")
        self.raw_dir = Path(raw_dir or PROJECT_ROOT / "data" / "raw" / "sec")
        self.processed_dir = PROJECT_ROOT / "data" / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        import requests
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0
        self._ticker_map: Dict[str, Dict[str, Any]] = {}

    def _get(self, url: str) -> requests.Response:
        elapsed = time.time() - self._last
        if elapsed < 0.12:
            time.sleep(0.12 - elapsed)
        self._last = time.time()
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response

    def ticker_map(self) -> Dict[str, Dict[str, Any]]:
        if not self._ticker_map:
            data = self._get(SEC_TICKERS_URL).json()
            self._ticker_map = {
                row["ticker"].upper(): {
                    "cik": str(row["cik_str"]).zfill(10),
                    "company": row["title"],
                }
                for row in data.values()
            }
        return self._ticker_map

    def list_filings(
        self,
        ticker: str,
        forms: Iterable[str],
        start_year: int,
        end_year: int,
        limit_per_company: int | None = None,
    ) -> List[Dict[str, Any]]:
        ticker = ticker.upper()
        info = self.ticker_map().get(ticker)
        if not info:
            raise ValueError(f"Ticker {ticker} not found in SEC ticker map.")
        data = self._get(SEC_SUBMISSIONS_URL.format(cik=info["cik"])).json()
        recent = data.get("filings", {}).get("recent", {})
        rows: List[Dict[str, Any]] = []
        forms_set = {f.upper() for f in forms}
        for i, form in enumerate(recent.get("form", [])):
            filing_date = recent.get("filingDate", [""])[i]
            year = int(filing_date[:4]) if filing_date[:4].isdigit() else 0
            if form.upper() not in forms_set or not (start_year <= year <= end_year):
                continue
            accession = recent.get("accessionNumber", [""])[i]
            primary_doc = recent.get("primaryDocument", [""])[i]
            source_url = SEC_ARCHIVE_URL.format(
                cik_int=str(int(info["cik"])),
                accession_no_dash=accession.replace("-", ""),
                primary_doc=primary_doc,
            )
            rows.append({
                "accession_number": accession,
                "company": data.get("name") or info["company"],
                "ticker": ticker,
                "cik": info["cik"],
                "form_type": form,
                "filing_date": filing_date,
                "fiscal_year": year,
                "source_url": source_url,
                "primary_document": primary_doc,
            })
            if limit_per_company and len(rows) >= limit_per_company:
                break
        return rows

    def download_filings(self, filings: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        manifest: List[Dict[str, Any]] = []
        for filing in filings:
            ext = ".html" if filing.get("primary_document", "").lower().endswith((".htm", ".html")) else ".txt"
            out_dir = self.raw_dir / filing["ticker"] / filing["form_type"] / filing["accession_number"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"filing{ext}"
            if not out_path.exists():
                response = self._get(filing["source_url"])
                out_path.write_text(response.text, encoding="utf-8", errors="ignore")
            manifest.append({**filing, "local_path": str(out_path)})
        return manifest

    def ingest(self, tickers: Iterable[str], forms: Iterable[str], start_year: int, end_year: int, limit_per_company: int | None = None) -> pd.DataFrame:
        all_filings: List[Dict[str, Any]] = []
        for ticker in tickers:
            all_filings.extend(self.list_filings(ticker, forms, start_year, end_year, limit_per_company))
        manifest = self.download_filings(all_filings)
        df = pd.DataFrame(manifest)
        csv_path = self.processed_dir / "filing_manifest.csv"
        parquet_path = self.processed_dir / "filing_manifest.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--limit-per-company", type=int, default=None)
    args = parser.parse_args()
    df = SecEdgarIngestor().ingest(args.tickers, args.forms, args.start_year, args.end_year, args.limit_per_company)
    print(f"Wrote {len(df)} filings to data/processed/filing_manifest.csv and .parquet")


if __name__ == "__main__":
    main()

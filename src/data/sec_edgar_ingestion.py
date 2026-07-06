"""SEC EDGAR ingestion pipeline."""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from config.settings import PROJECT_ROOT, settings
from src.data.chunking import REQUIRED_METADATA, chunk_sections, save_chunks
from src.data.sec_parser import parse_filing_file


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{filename}"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dash}/{primary_doc}"

logger = logging.getLogger("findoc.sec_ingestion")


class _UrllibResponse:
    def __init__(self, url: str, status_code: int, body: bytes):
        import gzip
        import zlib

        self.url = url
        self.status_code = status_code
        if body.startswith(b"\x1f\x8b"):
            body = gzip.decompress(body)
        else:
            try:
                body = zlib.decompress(body)
            except Exception:
                pass
        self.content = body
        self.text = body.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code} for {self.url}")

    def json(self) -> Any:
        import json
        try:
            return json.loads(self.text)
        except Exception as exc:
            preview = self.text[:500].replace("\n", " ")
            raise RuntimeError(f"Failed to parse JSON from {self.url}. Response preview: {preview}") from exc


class _UrllibSession:
    def __init__(self):
        self.headers: Dict[str, str] = {}

    def get(self, url: str, timeout: int = 30):
        from urllib.request import Request, urlopen

        request = Request(url, headers=self.headers)
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return _UrllibResponse(url, status, response.read())


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
        try:
            import requests
            self.session = requests.Session()
        except Exception:
            self.session = _UrllibSession()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0
        self._ticker_map: Dict[str, Dict[str, Any]] = {}

    def _get(self, url: str) -> requests.Response:
        elapsed = time.time() - self._last
        if elapsed < 0.12:
            time.sleep(0.12 - elapsed)
        self._last = time.time()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except Exception as exc:
            raise RuntimeError(f"SEC download failed for {url}: {exc}") from exc

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
        forms_set = {f.upper() for f in forms}
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_from_recent(recent: Dict[str, List[Any]]) -> None:
            forms_list = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])
            report_dates = recent.get("reportDate", [])
            for i, form in enumerate(forms_list):
                filing_date = dates[i] if i < len(dates) else ""
                year = int(filing_date[:4]) if filing_date[:4].isdigit() else 0
                if form.upper() not in forms_set or not (start_year <= year <= end_year):
                    continue
                accession = accessions[i] if i < len(accessions) else ""
                if not accession or accession in seen:
                    continue
                seen.add(accession)
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""
                source_url = SEC_ARCHIVE_URL.format(
                    cik_int=str(int(info["cik"])),
                    accession_no_dash=accession.replace("-", ""),
                    primary_doc=primary_doc,
                )
                fiscal_period = "FY" if form.upper() == "10-K" else ""
                fiscal_year = year
                report_date = report_dates[i] if i < len(report_dates) else ""
                if report_date[:4].isdigit():
                    fiscal_year = int(report_date[:4])
                rows.append({
                    "accession_number": accession,
                    "company": data.get("name") or info["company"],
                    "ticker": ticker,
                    "cik": info["cik"],
                    "form_type": form,
                    "filing_date": filing_date,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "source_url": source_url,
                    "primary_document": primary_doc,
                })

        add_from_recent(data.get("filings", {}).get("recent", {}))
        if limit_per_company and len(rows) >= limit_per_company:
            return rows[:limit_per_company]

        for file_info in data.get("filings", {}).get("files", []):
            name = file_info.get("name")
            if not name:
                continue
            filing_to = file_info.get("filingTo", "")
            filing_from = file_info.get("filingFrom", "")
            if filing_to[:4].isdigit() and int(filing_to[:4]) < start_year:
                continue
            if filing_from[:4].isdigit() and int(filing_from[:4]) > end_year:
                continue
            logger.info("Loading SEC submissions archive %s for %s", name, ticker)
            archive = self._get(SEC_SUBMISSIONS_FILE_URL.format(filename=name)).json()
            add_from_recent(archive)
            if limit_per_company and len(rows) >= limit_per_company:
                break
        return rows[:limit_per_company] if limit_per_company else rows

    def download_filings(self, filings: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        manifest: List[Dict[str, Any]] = []
        for filing in filings:
            ext = ".html" if filing.get("primary_document", "").lower().endswith((".htm", ".html")) else ".txt"
            out_dir = self.raw_dir / filing["ticker"] / filing["form_type"] / filing["accession_number"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"filing{ext}"
            if not out_path.exists():
                logger.info("Downloading %s %s %s from %s", filing["ticker"], filing["form_type"], filing["accession_number"], filing["source_url"])
                response = self._get(filing["source_url"])
                out_path.write_text(response.text, encoding="utf-8", errors="ignore")
            else:
                logger.info("Using cached filing %s", out_path)
            manifest.append({**filing, "local_path": str(out_path)})
        return manifest

    def ingest(self, tickers: Iterable[str], forms: Iterable[str], start_year: int, end_year: int, limit_per_company: int | None = None) -> pd.DataFrame:
        all_filings: List[Dict[str, Any]] = []
        for ticker in tickers:
            ticker_filings = self.list_filings(ticker, forms, start_year, end_year, limit_per_company)
            logger.info("Found %s filings for %s", len(ticker_filings), ticker)
            all_filings.extend(ticker_filings)
        if not all_filings:
            raise RuntimeError("SEC ingestion returned zero filings for the requested tickers/forms/date range.")
        manifest = self.download_filings(all_filings)
        df = pd.DataFrame(manifest)
        csv_path = self.processed_dir / "filing_manifest.csv"
        parquet_path = self.processed_dir / "filing_manifest.parquet"
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to write {parquet_path}. Install pyarrow or fastparquet. Error: {exc}") from exc

        sections: List[Dict[str, Any]] = []
        parse_errors: List[str] = []
        for filing in manifest:
            metadata = {key: filing.get(key, "") for key in REQUIRED_METADATA}
            missing = [
                key for key in REQUIRED_METADATA
                if key not in {"section", "fiscal_period"} and metadata.get(key, "") in ("", None)
            ]
            if missing:
                raise RuntimeError(
                    f"Chunking metadata missing for {filing.get('local_path')}: {missing}"
                )
            try:
                parsed_sections = parse_filing_file(filing["local_path"], metadata)
                if not parsed_sections:
                    parse_errors.append(f"{filing['local_path']}: parser returned zero sections")
                sections.extend(parsed_sections)
                logger.info("Parsed %s into %s sections/tables", filing["local_path"], len(parsed_sections))
            except Exception as exc:
                parse_errors.append(f"{filing.get('local_path')}: {exc}")
        if parse_errors:
            raise RuntimeError("SEC parsing failed:\n" + "\n".join(parse_errors))

        chunks = chunk_sections(sections)
        if not chunks:
            raise RuntimeError("Chunking produced zero chunks from parsed SEC filings.")
        for chunk in chunks:
            missing = [key for key in REQUIRED_METADATA if key != "fiscal_period" and chunk.get(key, "") in ("", None)]
            if missing:
                raise RuntimeError(f"Chunking failed for doc_id={chunk.get('doc_id')}: missing metadata {missing}")
        paths = save_chunks(chunks, self.processed_dir)
        if not Path(paths["parquet"]).name == "chunks.parquet":
            raise RuntimeError(f"Failed to write data/processed/chunks.parquet. Details written to {paths['parquet']}")
        logger.info("Wrote %s chunks to %s and %s", len(chunks), paths["parquet"], paths["jsonl"])
        return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--limit-per-company", type=int, default=None)
    args = parser.parse_args()
    df = SecEdgarIngestor().ingest(args.tickers, args.forms, args.start_year, args.end_year, args.limit_per_company)
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
    chunk_count = len(pd.read_parquet(chunks_path)) if chunks_path.exists() else 0
    print(f"Wrote {len(df)} filings to data/processed/filing_manifest.csv and .parquet")
    print(f"Wrote {chunk_count} chunks to data/processed/chunks.parquet and chunks.jsonl")


if __name__ == "__main__":
    main()

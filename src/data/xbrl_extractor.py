"""Extract standardized XBRL facts from SEC companyfacts data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

from config.settings import PROJECT_ROOT, settings


def flatten_companyfacts(payload: Dict[str, Any], ticker: str = "", accession_filter: set[str] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cik = str(payload.get("cik", "")).zfill(10)
    for taxonomy in payload.get("facts", {}).values():
        for concept, data in taxonomy.items():
            label = data.get("label", concept)
            for unit, facts in data.get("units", {}).items():
                for fact in facts:
                    accession = fact.get("accn", "")
                    if accession_filter and accession not in accession_filter:
                        continue
                    rows.append({
                        "ticker": ticker,
                        "cik": cik,
                        "fiscal_year": fact.get("fy", ""),
                        "fiscal_period": fact.get("fp", ""),
                        "concept": concept,
                        "label": label,
                        "value": fact.get("val", ""),
                        "unit": unit,
                        "form": fact.get("form", ""),
                        "filed_date": fact.get("filed", ""),
                        "accession_number": accession,
                    })
    return rows


def extract_companyfacts(cik: str, ticker: str = "", user_agent: str | None = None) -> pd.DataFrame:
    headers = {"User-Agent": user_agent or settings.sec_edgar_user_agent}
    response = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json", headers=headers, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(flatten_companyfacts(response.json(), ticker=ticker))


def save_xbrl_facts(df: pd.DataFrame, output: str | Path | None = None) -> Path:
    output = Path(output or PROJECT_ROOT / "data" / "processed" / "xbrl_facts.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cik", required=True)
    parser.add_argument("--ticker", default="")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "processed" / "xbrl_facts.parquet"))
    args = parser.parse_args()
    path = save_xbrl_facts(extract_companyfacts(args.cik, args.ticker), args.output)
    print(f"Wrote XBRL facts to {path}")


if __name__ == "__main__":
    main()

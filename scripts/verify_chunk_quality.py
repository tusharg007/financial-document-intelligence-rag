"""Verify SEC chunk quality characteristics after parsing/chunking."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROJECT_ROOT
from src.data.chunking import boilerplate_score, content_quality_score, is_toc_like


def main() -> None:
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
    if not chunks_path.exists():
        raise SystemExit(f"Chunk file not found: {chunks_path}")

    df = pd.read_parquet(chunks_path).fillna("")
    if df.empty:
        raise SystemExit("Chunk quality verification failed: chunks.parquet is empty.")

    if "is_toc_like" not in df.columns:
        df["is_toc_like"] = df["content"].astype(str).map(is_toc_like)
    else:
        df["is_toc_like"] = df["is_toc_like"].astype(bool)

    if "boilerplate_score" not in df.columns:
        df["boilerplate_score"] = df["content"].astype(str).map(boilerplate_score)
    else:
        df["boilerplate_score"] = pd.to_numeric(df["boilerplate_score"], errors="coerce").fillna(0.0)

    if "content_quality_score" not in df.columns:
        df["content_quality_score"] = [
            content_quality_score(str(content), section=str(section))
            for content, section in zip(df["content"], df["section"])
        ]
    else:
        df["content_quality_score"] = pd.to_numeric(df["content_quality_score"], errors="coerce").fillna(0.0)

    rf = df[df["section"].astype(str) == "Risk Factors"].copy()
    toc_count = int(df["is_toc_like"].sum())
    boilerplate_count = int((df["boilerplate_score"] >= 0.7).sum())
    substantive_risk = rf[(~rf["is_toc_like"]) & (rf["boilerplate_score"] < 0.7) & (rf["content_quality_score"] >= 0.45)]
    warnings = []
    if rf.empty:
        warnings.append("No Risk Factors chunks were found.")
    else:
        toc_rate = float(rf["is_toc_like"].mean())
        boilerplate_rate = float((rf["boilerplate_score"] >= 0.7).mean())
        if toc_rate > 0.25:
            warnings.append(f"Risk Factors TOC-like rate is high: {toc_rate:.3f}")
        if boilerplate_rate > 0.35:
            warnings.append(f"Risk Factors boilerplate-heavy rate is high: {boilerplate_rate:.3f}")
        if len(substantive_risk) < max(20, int(len(rf) * 0.2)):
            warnings.append("Risk Factors substantive chunk count is lower than expected.")

    sample_cols = [
        col for col in [
            "ticker",
            "form_type",
            "filing_date",
            "section",
            "is_toc_like",
            "boilerplate_score",
            "content_quality_score",
            "section_confidence",
            "source_url",
            "content",
        ]
        if col in df.columns
    ]
    good_samples = substantive_risk.sort_values("content_quality_score", ascending=False).head(3)
    bad_samples = df[(df["is_toc_like"]) | (df["boilerplate_score"] >= 0.7)].sort_values(
        ["is_toc_like", "boilerplate_score"], ascending=False
    ).head(3)

    output = {
        "total chunks": int(len(df)),
        "chunks by section": {str(k): int(v) for k, v in df["section"].astype(str).value_counts().to_dict().items()},
        "TOC-like chunk count": toc_count,
        "TOC-like chunk rate": round(float(df["is_toc_like"].mean()), 4),
        "boilerplate-heavy chunk count": boilerplate_count,
        "boilerplate-heavy chunk rate": round(float((df["boilerplate_score"] >= 0.7).mean()), 4),
        "average content quality score": round(float(df["content_quality_score"].mean()), 4),
        "Risk Factors substantive chunk count": int(len(substantive_risk)),
        "sample good Risk Factors chunks": [
            {
                **{col: row[col] for col in sample_cols if col != "content"},
                "content": str(row["content"])[:400],
            }
            for _, row in good_samples[sample_cols].iterrows()
        ],
        "sample bad/filtered chunks": [
            {
                **{col: row[col] for col in sample_cols if col != "content"},
                "content": str(row["content"])[:400],
            }
            for _, row in bad_samples[sample_cols].iterrows()
        ],
        "warnings": warnings,
    }
    print(json.dumps(output, indent=2, default=str))

    if not substantive_risk.empty and df["source_url"].astype(str).str.strip().eq("").any():
        raise SystemExit("Chunk quality verification failed: some chunks are missing source_url.")
    if rf.empty:
        raise SystemExit("Chunk quality verification failed: no Risk Factors chunks were found.")


if __name__ == "__main__":
    main()

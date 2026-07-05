"""Build instruction-tuning data for LoRA from available processed datasets."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import PROJECT_ROOT


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_examples() -> List[Dict[str, Any]]:
    processed = PROJECT_ROOT / "data" / "processed"
    examples: List[Dict[str, Any]] = []
    for name in ["financebench", "tatqa", "finqa", "fiqa", "demo"]:
        for row in _read_jsonl(processed / f"eval_{name}.jsonl"):
            question = row.get("question") or row.get("query")
            answer = row.get("answer") or row.get("ground_truth")
            if question and answer:
                examples.append({
                    "instruction": "Answer the financial question using the provided evidence.",
                    "input": row.get("evidence") or row.get("context") or "",
                    "output": answer,
                    "source_dataset": name,
                    "evidence": row.get("evidence") or row.get("context") or "",
                    "task_type": "qa",
                })
    for row in _read_jsonl(processed / "chunks.jsonl"):
        if row.get("content") and row.get("source_url"):
            examples.append({
                "instruction": f"Summarize the {row.get('section', 'filing')} evidence with citation support.",
                "input": row["content"],
                "output": row["content"][:500],
                "source_dataset": "sec_chunks",
                "evidence": row["content"],
                "task_type": "cited_summary",
            })
    return examples


def write_splits(examples: List[Dict[str, Any]], output_dir: Path) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.Random(42).shuffle(examples)
    n = len(examples)
    splits = {
        "train": examples[: int(n * 0.8)],
        "val": examples[int(n * 0.8): int(n * 0.9)],
        "test": examples[int(n * 0.9):],
    }
    for name, rows in splits.items():
        (output_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    return {name: len(rows) for name, rows in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed" / "lora"))
    args = parser.parse_args()
    examples = build_examples()
    counts = write_splits(examples, Path(args.output_dir))
    print(json.dumps({"total": len(examples), "splits": counts, "target_met": len(examples) >= 1000}, indent=2))


if __name__ == "__main__":
    main()

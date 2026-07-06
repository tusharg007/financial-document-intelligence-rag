"""CLI for evaluating a single RAG response payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_eval_kit import evaluate_rag_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = evaluate_rag_response(
        question=payload["question"],
        answer=payload["answer"],
        contexts=payload.get("contexts", []),
        ground_truth=payload.get("ground_truth"),
        expected_keywords=payload.get("expected_keywords"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "guardrail_status": result.guardrail_status}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from rag_eval_kit import evaluate_rag_response


ROOT = Path(__file__).resolve().parents[2]
payload = json.loads((ROOT / "examples" / "rag_eval_cookbook" / "sample_eval_input.json").read_text(encoding="utf-8"))
result = evaluate_rag_response(
    question=payload["question"],
    answer=payload["answer"],
    contexts=payload["contexts"],
    expected_keywords=payload.get("expected_keywords"),
)
print(json.dumps(result.to_dict(), indent=2))

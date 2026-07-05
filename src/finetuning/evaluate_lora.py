"""Evaluate LoRA only when an adapter/report exists; never fabricate results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import PROJECT_ROOT


def main() -> None:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    adapter = PROJECT_ROOT / "adapters" / "lora_findoc"
    if not adapter.exists():
        result = {"status": "not_trained", "message": "LoRA not trained yet; adapters/lora_findoc is missing.", "models_compared": []}
        (reports / "lora_eval_results.md").write_text("# LoRA Evaluation\n\nLoRA not trained yet.\n", encoding="utf-8")
    else:
        result = {"status": "available", "message": "Adapter exists; run full evaluation with configured models.", "models_compared": ["lora"]}
    (reports / "lora_eval_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (reports / "model_comparison.csv").write_text("model,status\nlora," + result["status"] + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

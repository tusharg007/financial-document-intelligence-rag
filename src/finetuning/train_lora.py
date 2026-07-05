"""QLoRA training entrypoint.

Runs only when the required GPU/model dependencies and training data are present.
No final metrics are fabricated; missing prerequisites produce a clear report.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import PROJECT_ROOT


def main() -> None:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    train_path = PROJECT_ROOT / "data" / "processed" / "lora" / "train.jsonl"
    started = time.time()
    result = {
        "status": "not_started",
        "training_time": 0,
        "hardware": platform.platform(),
        "message": "",
    }
    if not train_path.exists():
        result["status"] = "blocked"
        result["message"] = "Training data missing. Run python src/finetuning/build_lora_dataset.py first."
    else:
        try:
            import torch
            if not torch.cuda.is_available():
                result["status"] = "smoke_test_only"
                result["message"] = "GPU unavailable; skipped full LoRA training."
            else:
                result["status"] = "ready_for_training"
                result["message"] = "GPU detected. Configure training hyperparameters before a full run."
        except Exception as exc:
            result["status"] = "blocked"
            result["message"] = f"Training dependencies unavailable: {exc}"
    result["training_time"] = time.time() - started
    (reports / "lora_training_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

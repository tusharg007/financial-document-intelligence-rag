"""Verify local dense-indexing dependencies for ChromaDB and OpenTelemetry."""
from __future__ import annotations

import json
import gc
import shutil
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    report = {
        "import chromadb": False,
        "import opentelemetry.proto.collector.logs.v1.logs_service_pb2": False,
        "created Chroma PersistentClient": False,
        "created or loaded test collection": False,
        "upserted test documents": False,
        "test collection count": 0,
        "cleaned up test directory": False,
        "test directory": "",
        "chromadb version": "",
    }

    temp_dir = Path(tempfile.mkdtemp(prefix="chroma-deps-", dir=str(Path.cwd())))
    report["test directory"] = str(temp_dir)

    exit_code = 0

    try:
        import chromadb

        report["import chromadb"] = True
        report["chromadb version"] = getattr(chromadb, "__version__", "")

        import opentelemetry.proto.collector.logs.v1.logs_service_pb2  # noqa: F401

        report["import opentelemetry.proto.collector.logs.v1.logs_service_pb2"] = True

        client = chromadb.PersistentClient(path=str(temp_dir))
        report["created Chroma PersistentClient"] = True

        collection = client.get_or_create_collection(name="dependency_smoke_test")
        report["created or loaded test collection"] = True

        collection.upsert(
            ids=["dep-1", "dep-2"],
            documents=["alpha filing risk factors", "beta revenue growth"],
            embeddings=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
            metadatas=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        )
        report["upserted test documents"] = True

        report["test collection count"] = collection.count()
        if report["test collection count"] != 2:
            raise RuntimeError(
                f"Expected dependency smoke-test collection count 2, got {report['test collection count']}"
            )
        client.delete_collection("dependency_smoke_test")
        client.clear_system_cache()
        del collection
        del client
        gc.collect()

    except Exception as exc:
        exit_code = 1
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for _ in range(5):
            shutil.rmtree(temp_dir, ignore_errors=True)
            if not temp_dir.exists():
                break
            gc.collect()
            time.sleep(0.2)
        report["cleaned up test directory"] = not temp_dir.exists()
        print(json.dumps(report, indent=2))

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

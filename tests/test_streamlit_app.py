from pathlib import Path


def test_lora_runtime_status_is_optional_when_adapter_missing(monkeypatch, tmp_path):
    import app

    monkeypatch.setattr(app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app, "REPORTS", tmp_path / "reports")

    status = app.lora_runtime_status()

    assert status["adapter_exists"] is False
    assert status["status"] == "not_trained"
    assert "optional" in status["message"].lower()
    assert "not been trained" in status["message"].lower()
    assert status["adapter_path"] == str(tmp_path / "adapters" / "lora_findoc")

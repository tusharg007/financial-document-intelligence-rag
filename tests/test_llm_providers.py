def test_extractive_provider_labels_fallback():
    from src.llm.factory import get_llm

    llm = get_llm("extractive")
    result = llm.generate([{"role": "user", "content": "[Source 1] Tesla revenue"}])
    assert result["provider"] == "extractive"
    assert "Fallback mode used" in result["text"]


def test_lora_missing_health_is_clear(tmp_path):
    from src.llm.lora_provider import LoRAProvider

    provider = LoRAProvider(adapter_path=str(tmp_path / "missing"))
    health = provider.health_check()
    assert health["ok"] is False
    assert "not trained" in health["message"].lower()

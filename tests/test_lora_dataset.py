def test_lora_split_writer(tmp_path):
    from src.finetuning.build_lora_dataset import write_splits

    rows = [{"instruction": "i", "input": "x", "output": "y", "source_dataset": "demo", "evidence": "x", "task_type": "qa"} for _ in range(10)]
    counts = write_splits(rows, tmp_path)
    assert counts["train"] == 8
    assert (tmp_path / "train.jsonl").exists()

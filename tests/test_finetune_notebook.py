"""finetune/train_lora.ipynb: JSON válido, não executado e com as peças certas."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "finetune" / "train_lora.ipynb"


def test_notebook_e_json_valido_e_nao_foi_executado():
    with open(str(NOTEBOOK), encoding="utf-8") as handle:
        nb = json.load(handle)
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code, "notebook sem células de código"
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in code)
    source = "".join("".join(c["source"]) for c in nb["cells"])
    for needed in ("unsloth", "render_example", "q4_k_m", "lora_alpha", "train_on", "val.jsonl"):
        assert needed in source, needed

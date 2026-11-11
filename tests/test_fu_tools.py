"""Follow-up tooling (DEVIATIONS D14, D15): the run-host t* module reproduces the frozen t*, and the BOS rule."""
import json
from pathlib import Path

import pytest

from src.runlib import needs_bos
from src.score.tstar import select

ROOT = Path(__file__).resolve().parents[1]


def jl(p):
    return [json.loads(line) for line in open(ROOT / p, encoding="utf-8")]


@pytest.mark.parametrize("model,frozen", [("4B", "selfverdict.json"), ("1.7B", "selfverdict.json"), ("8B", "selfverdict_8B.json")])
def test_tstar_reproduces_frozen(model, frozen):
    path = ROOT / "analysis" / "out" / frozen
    if not path.exists():
        pytest.skip("frozen selection not present")
    want = json.loads(path.read_text())[model]["t_star_s"]
    got = select(jl(f"results/p1/{model}/p1-initial-{model}-d2ebdea/initial.jsonl"),
                 jl(f"results/p1/{model}/p1-verbal-{model}-d2ebdea/verbal.jsonl"))["t_star_s"]
    assert got == want


def test_needs_bos():
    from transformers import AutoTokenizer
    try:
        qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
        tulu = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B", local_files_only=True)
    except OSError:
        pytest.skip("tokenizers not cached")
    m = [{"role": "user", "content": "1+1?"}]
    tq = qwen.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tt = tulu.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    assert needs_bos(qwen, tq) is False            # Qwen3: unchanged (no BOS token)
    assert needs_bos(tulu, tt) is True
    assert tulu(tt, add_special_tokens=True).input_ids[0] == tulu.bos_token_id
    assert needs_bos(tulu, tulu.bos_token + tt) is False

"""P0 helpers: train-split fixtures (outside the study pool), chat rendering, a minimal scorer,
and the run directory with its version manifest. The full scorer and splits come in P1."""
import hashlib
import json
import os
import platform
import random
import re
import subprocess
from pathlib import Path

MODELS = {"1.7B": "Qwen/Qwen3-1.7B", "4B": "Qwen/Qwen3-4B", "8B": "Qwen/Qwen3-8B"}
PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
FEEDBACK = ("Your previous answer may contain an error. Please re-check your solution "
            "and put your final answer within \\boxed{}.")
REPO = Path(__file__).resolve().parents[2]


def fixtures(n, seed=0):
    """GSM8K *train* problems: outside the D-nat pool (GSM8K test + MATH-500), design §5.1."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [{"id": f"gsm8k-train-{i}", "q": ds[i]["question"],
             "solution": ds[i]["answer"].split("####")[0].strip(),
             "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in idx]


def render(tok, messages, thinking=False):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def _norm(s):
    s = s.replace("$", "").replace(",", "").replace("\\%", "").replace("%", "").strip().rstrip(".")
    s = re.sub(r"\\text\{[^}]*\}", "", s).replace("\\!", "").strip()
    return s


def score(pred, gold):
    """Minimal numeric/string match for GSM8K fixtures only (P1 replaces it with the design §5.6 scorer)."""
    if pred is None:
        return False
    p, g = _norm(pred), _norm(gold)
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p == g


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def code_commit():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def snapshot_revision(repo_id):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id, local_files_only=True)
    return Path(path).name, path


def run_dir(exp, model_key, kind):
    root = Path(os.environ.get("RESULTS_ROOT", "/home/results"))
    run_id = f"p0-{kind}-{model_key}-{code_commit()[:7]}"
    d = root / exp / model_key / run_id
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d


def write_manifest(d, run_id, model_key, tok, config, backend):
    import torch
    rev, _ = snapshot_revision(MODELS[model_key])
    lock = Path("/home/research_state/lock.txt")
    man = {
        "run_id": run_id, "model": MODELS[model_key], "model_revision": rev,
        "chat_template_sha256": sha256_text(tok.chat_template or ""),
        "prompt_sha256": sha256_text(PROMPT + "\n" + FEEDBACK),
        "lockfile_sha256": sha256_file(lock) if lock.exists() else None,
        "code_commit": code_commit(), "config_sha256": sha256_text(json.dumps(config, sort_keys=True)),
        "backend": backend, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG")},
        "seeds": {"fixtures": 0, "sampling": 0},
    }
    (d / "config.json").write_text(json.dumps(config, indent=1))
    files = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.name != "manifest.json":
            files[p.name] = sha256_file(p)
    man["files"] = files
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    return man

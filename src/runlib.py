"""Run directories and version manifests for study runs (CLAUDE.md rule 7)."""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = {"1.7B": "Qwen/Qwen3-1.7B", "4B": "Qwen/Qwen3-4B", "8B": "Qwen/Qwen3-8B",
          "T8B": "allenai/Llama-3.1-Tulu-3-8B"}   # T8B: second family, follow-up (DEVIATIONS D15)


def needs_bos(tok, text):
    """Whether HF tokenization of a rendered chat must add the BOS token. vLLM adds it when it tokenizes a prompt;
    the Qwen3 tokenizers have none (unchanged: False), the Llama 3.1 (Tulu 3) template does not emit it."""
    return bool(tok.bos_token) and not text.startswith(tok.bos_token)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def code_commit():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def run_dir(exp, model_key, kind):
    root = Path(os.environ.get("RESULTS_ROOT", "/home/results"))
    run_id = f"{exp}-{kind}-{model_key}-{code_commit()[:7]}"
    d = root / exp / model_key / run_id
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d


def register(run_id, status, **fields):
    """Append an event to the run host's run registry (CLAUDE.md §4 runs.jsonl). The control host merges it into
    the repository's runs.jsonl, keeping the latest event per run_id."""
    import socket
    from datetime import datetime, timezone
    path = Path(os.environ.get("RESULTS_ROOT", "/home/results")) / "runs.jsonl"
    row = {"run_id": run_id, "status": status, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "host": socket.gethostname(), "code_commit": code_commit(), **fields}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def require_batch_invariant():
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise SystemExit("VLLM_BATCH_INVARIANT=1 is required for all vLLM generation (design §4)")


def write_manifest(d, run_id, model_key, tok, config, backend, prompts):
    """Version manifest: model/tokenizer/template, data, scorer, prompts, lockfile, code commit, env, GPU."""
    import torch
    from huggingface_hub import snapshot_download
    rev = Path(snapshot_download(MODELS[model_key], local_files_only=True)).name
    lock = Path("/home/research_state/lock.txt")
    data_manifest = REPO / "data" / "processed" / "build_manifest.json"
    man = {
        "run_id": run_id, "model": MODELS[model_key], "model_revision": rev,
        "chat_template_sha256": sha256_text(tok.chat_template or ""),
        "prompts_sha256": {k: sha256_text(v) for k, v in prompts.items()},
        "data_build_manifest_sha256": sha256_file(data_manifest) if data_manifest.exists() else None,
        "scorer_sha256": sha256_file(REPO / "src" / "score" / "scorer.py"),
        "lockfile_sha256": sha256_file(lock) if lock.exists() else None,
        "code_commit": code_commit(), "config_sha256": sha256_text(json.dumps(config, sort_keys=True)),
        "backend": backend, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT",)},
    }
    (d / "config.json").write_text(json.dumps(config, indent=1))
    man["files"] = {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    return man

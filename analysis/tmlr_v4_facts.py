"""Extract the configuration facts that manuscript_v4 (TMLR) states but the v3 evidence package lacked.

Reads only committed repository files (run configs and the pinned lockfile) and writes
paper/TMLR/evidence_v4/configs/v4_facts.json plus paper/TMLR/evidence_v4/manifest_v4.json.
No inference, fitting or rescoring. Host paths are not copied: each source is named by its
repository-relative path and the sha256 of its LF-normalized bytes.

Facts:
- sampling settings of the five agreement samples (Math Qwen3-4B; MMLU-Pro Qwen3-4B and Tulu-3-8B);
- the pinned math-verify version used by the symbolic scorer.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "paper" / "TMLR" / "evidence_v4"

SAMPLING_SOURCES = {
    "math_4B": "results/fu/4B/fu-h2a-consistency-test-4B-aac4cc6/config.json",
    "mmlupro_4B": "results/v2/fu/4B/fu-mc-consist-test-4B-9221b24/config.json",
    "mmlupro_T8B": "results/v2/fu/T8B/fu-mc-consist-test-T8B-9221b24/config.json",
}
LOCKFILE = "configs/lock.txt"
SAMPLING_KEYS = ("k", "temperature", "top_p", "top_k", "seed_stream")


def lf_bytes(rel: str) -> bytes:
    return (REPO / rel).read_bytes().replace(b"\r\n", b"\n")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    sampling = {}
    for setting, rel in SAMPLING_SOURCES.items():
        raw = lf_bytes(rel)
        cfg = json.loads(raw)
        sampling[setting] = {key: cfg[key] for key in SAMPLING_KEYS}
        sampling[setting]["source"] = rel
        sampling[setting]["source_sha256_lf"] = sha256(raw)
    shared = {key: sampling["math_4B"][key] for key in ("k", "temperature", "top_p", "top_k")}
    for setting, values in sampling.items():
        for key, value in shared.items():
            if values[key] != value:
                raise SystemExit(f"{setting}: {key}={values[key]!r} differs from math_4B {value!r}")

    lock_raw = lf_bytes(LOCKFILE)
    versions = re.findall(r"^math-verify==([0-9][0-9.]*)\s*$", lock_raw.decode("utf-8"), re.M)
    if len(versions) != 1:
        raise SystemExit(f"expected one math-verify pin in {LOCKFILE}, found {versions}")

    facts = {
        "purpose": "Configuration facts stated in manuscript_v4; extracted from committed run configs and the lockfile.",
        "generator": "analysis/tmlr_v4_facts.py (parent repository; not part of the companion archive)",
        "agreement_sampling": {"shared": shared, "per_setting": sampling},
        "math_verify": {"version": versions[0], "source": LOCKFILE, "source_sha256_lf": sha256(lock_raw)},
    }
    OUT_DIR.joinpath("configs").mkdir(parents=True, exist_ok=True)
    facts_bytes = (json.dumps(facts, indent=2) + "\n").encode("utf-8")
    facts_path = OUT_DIR / "configs" / "v4_facts.json"
    facts_path.write_bytes(facts_bytes)
    manifest = {"files": [{"path": "evidence_v4/configs/v4_facts.json", "sha256": sha256(facts_bytes)}]}
    (OUT_DIR / "manifest_v4.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"shared_sampling": shared, "math_verify": versions[0]}))


if __name__ == "__main__":
    main()

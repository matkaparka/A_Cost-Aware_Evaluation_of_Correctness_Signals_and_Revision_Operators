"""Shared helpers for the ARR-round analyses (DEVIATIONS D20; docs/planning/arr_round/plan.md).

Stored outputs only: no generation, no training, no model imports.

- Gold-blind answer candidates and complete-link plurality groups follow the ACL revision's frozen rules
  (paper/1区/acl2027/protocol/math_vote_rules.md, "Gold-blind candidates and plurality", items 1-4).
- Symbolic equivalence runs in a disposable worker subprocess with the same guards as the ACL replay
  (20 CPU-seconds and +8 GiB resident memory per call; a 600 s wall watchdog aborts). Results are cached in
  analysis/out/arr/equivalence_cache.jsonl; the ACL replay's cache is read (never written) as a seed when the
  scorer file hash matches.
- Paired cluster bootstrap over canonical problem IDs with fixed memberships.

Run as `python -B analysis/arr/common.py --worker` only from Equivalence (internal).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# ARR_OUT redirects outputs and the equivalence cache (the run host writes under RESULTS_ROOT, never into a worktree).
OUT = Path(os.environ.get("ARR_OUT", str(REPO / "analysis" / "out" / "arr")))
PLAN = REPO / "docs" / "planning" / "arr_round" / "plan.md"
ACL = REPO / "paper" / "1区" / "acl2027"
J1 = REPO / "paper" / "1区"
sys.path.insert(0, str(REPO))
from src.score import scorer  # noqa: E402

MODELS_MATH = ("1.7B", "4B", "8B", "T8B")
MODELS_MC = ("4B", "T8B")
OPS = ("F0", "F2", "F2-scratch")
SEED = 20261110
N_BOOT = 4000
INPUTS: dict[str, dict] = {}


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_lf(path) -> str:
    """sha256 of the file with CR removed (Windows checkouts may hold CRLF copies of LF files)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r", b"")).hexdigest()


def pin(path) -> Path:
    """Record an input's sha256 for the run manifest and return the path."""
    path = Path(path)
    INPUTS[str(path.relative_to(REPO)).replace("\\", "/")] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return path


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def unique(values, key="item_id"):
    out = {}
    for row in values:
        assert row[key] not in out, ("duplicate", row[key])
        out[row[key]] = row
    return out


def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_rows_gz(path, values):
    """Deterministic gzip (mtime 0) of a jsonl stream."""
    import gzip
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        for value in values:
            gz.write((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))


def rows_gz(path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8",
                          newline="\n")


def run_dir(run_id: str) -> Path:
    """Locate a results directory by run ID anywhere under results/."""
    hits = [p for p in (REPO / "results").glob(f"**/{run_id}") if p.is_dir()]
    assert len(hits) == 1, (run_id, hits)
    return hits[0]


def plan_hash() -> str:
    assert PLAN.exists(), "the analysis plan must exist (and be committed) before any ARR analysis runs"
    return sha_lf(PLAN)


def plan_core_hash() -> str:
    """sha256 of the frozen part of the plan: the LF text before the "## Amendments" heading. Amendments are only
    appended, so this hash must never change; outputs record it and downstream scripts compare it."""
    text = PLAN.read_bytes().replace(b"\r", b"")
    head, sep, _ = text.partition(b"\n## Amendments\n")
    assert sep, "plan has no Amendments section"
    return hashlib.sha256(head).hexdigest()


# ----------------------------------------------------------------------------- equivalence worker
def worker():
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    import math_verify  # noqa: F401  (load once before the parent starts timing calls)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        a, b = json.loads(line)
        print(json.dumps({"value": bool(scorer.equivalent(a, b))}), flush=True)


class Equivalence:
    """Directional scorer.equivalent(a, b) with caching and resource guards; guard hits are listed, never hidden."""

    def __init__(self):
        import psutil
        self.psutil = psutil
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "tmp").mkdir(exist_ok=True)
        self.scorer_sha = sha(Path(scorer.__file__))
        self.cache, self.guards, self.seeded, self.calls = {}, [], 0, 0
        self.process = None
        acl_cache = ACL / "artifacts" / "math_vote" / "equivalence_cache.jsonl"
        if acl_cache.exists():
            pin(acl_cache)
            for row in rows(acl_cache):
                if row.get("scorer_sha256") == self.scorer_sha and not row.get("guard"):
                    self.cache[(row["a"], row["b"])] = bool(row["value"])
                    self.seeded += 1
        own = OUT / "equivalence_cache.jsonl"
        if own.exists():
            for row in rows(own):
                if row.get("scorer_sha256") == self.scorer_sha:
                    self.cache[(row["a"], row["b"])] = bool(row["value"])
                    if row.get("guard"):
                        self.guards.append(row["guard"])
        self.cache_file = own.open("a", encoding="utf-8", newline="\n")
        self.log = (OUT / "worker_stderr.log").open("a", encoding="utf-8")

    def start(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMP=str(OUT / "tmp"), TEMP=str(OUT / "tmp"))
        self.process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                        encoding="utf-8", cwd=REPO, env=env)
        self.queue = queue.Queue()
        out_queue, process = self.queue, self.process

        def reader():
            for line in process.stdout:
                out_queue.put(line)
            out_queue.put(None)
        threading.Thread(target=reader, daemon=True).start()
        ready = self.queue.get(timeout=120)
        assert ready and json.loads(ready).get("ready"), ("worker init failed", ready)
        # The guards measure the worker's whole process tree. On Windows the venv's python.exe is a launcher that runs
        # the interpreter as a child process; until 2026-11-09 the guards watched the launcher alone, which never uses
        # CPU or memory, so only the wall watchdog could stop a runaway comparison (plan Amendment 5).
        root = self.psutil.Process(self.process.pid)
        self.tree = [root] + root.children(recursive=True)

    def usage(self):
        """CPU seconds (user + system) and resident bytes of the worker's process tree."""
        cpu = mem = 0
        for q in self.tree:
            t = q.cpu_times()
            cpu += t.user + t.system
            mem += q.memory_info().rss
        return cpu, mem

    def close_process(self):
        if self.process is not None:
            self.process.kill()
            self.process.wait()
            self.process = None

    def __call__(self, a, b) -> bool:
        if a is None or b is None:
            return False
        if scorer._norm_str(a) == scorer._norm_str(b):
            return True
        if len(a) > scorer.MAX_SYMBOLIC_CHARS or len(b) > scorer.MAX_SYMBOLIC_CHARS:
            return False
        key = (a, b)
        if key in self.cache:
            return self.cache[key]
        if self.process is None:
            self.start()
        self.calls += 1
        cpu0, mem0 = self.usage()
        start = time.monotonic()
        self.process.stdin.write(json.dumps([a, b]) + "\n")
        self.process.stdin.flush()
        guard = None
        while True:
            try:
                response = self.queue.get(timeout=0.1)
                if response is None:
                    raise RuntimeError("equivalence worker exited unexpectedly; no label assigned")
                value = bool(json.loads(response)["value"])
                break
            except queue.Empty:
                cpu, mem = self.usage()
                cpu -= cpu0
                if cpu >= 20 or mem > mem0 + (8 << 30):
                    guard = {"a": a, "b": b, "reason": "cpu" if cpu >= 20 else "memory"}
                    self.guards.append(guard)
                    self.close_process()
                    value = False
                    break
                if time.monotonic() - start > 600:
                    self.close_process()
                    raise RuntimeError("wall watchdog: abort rather than assign an equivalence label")
        self.cache[key] = value
        self.cache_file.write(json.dumps({"a": a, "b": b, "value": value, "guard": guard,
                                          "scorer_sha256": self.scorer_sha}, ensure_ascii=False) + "\n")
        self.cache_file.flush()
        return value

    def symmetric(self, a, b) -> bool:
        return self(a, b) and self(b, a)

    def close(self):
        self.close_process()
        self.cache_file.close()
        self.log.close()


REVIEWED_GUARDS = Path(__file__).resolve().parent / "reviewed_guards.json"
GUARDS_REPORTED = []


def guard_check(guards):
    """An ARR analysis may use its labels only when every guard hit has been hand-checked (plan Amendment 5).
    A hit counts as "math-verify does not confirm" (scorer 1.1.1, DEVIATIONS D1): the answer is correct only if its
    normalized string equals the gold, and the Equivalence returns False. The reviewed pairs and their hand checks are
    listed in analysis/arr/reviewed_guards.json; any other hit stops the analysis. The hits go into the manifest."""
    reviewed = {}
    if REVIEWED_GUARDS.exists():
        reviewed = {(r["a"], r["b"]): r for r in json.loads(pin(REVIEWED_GUARDS).read_text(encoding="utf-8"))["pairs"]}
    pending = [g for g in guards if (g["a"], g["b"]) not in reviewed]
    assert not pending, ("scorer guard hits without a hand check: review them, then list them in reviewed_guards.json",
                         pending)
    seen = {}
    for g in guards:
        seen.setdefault((g["a"], g["b"]), {**g, "hand_check": reviewed[(g["a"], g["b"])]["hand_check"]})
    GUARDS_REPORTED[:] = list(seen.values())
    return GUARDS_REPORTED


# ----------------------------------------------------------------------------- candidates and votes
def candidate(text, finish_reason, out_tokens, eq: Equivalence) -> dict:
    """Gold-blind candidate (ACL math_vote_rules.md item 2)."""
    pred, status, boxes = scorer.extract(text, last_number=False)
    raw = status
    if finish_reason == "length":
        status = "truncated"
    elif not pred or status not in ("ok", "alternatives"):
        status = "unparsed"
    elif status == "alternatives" and not eq.symmetric(boxes[-2], boxes[-1]):
        status = "conflict"
    else:
        status = "ok"
    return {"pred": pred, "status": status, "raw_extraction_status": raw, "final_boxes": boxes,
            "out_tokens": out_tokens, "finish_reason": finish_reason}


def groups_of(cands, match) -> list[list[int]]:
    """Complete-link groups over valid candidates in order (ACL item 4). `match(i, j)` compares candidates i, j."""
    groups = []
    for j, c in enumerate(cands):
        if c["status"] != "ok":
            continue
        for g in groups:
            if all(match(m, j) for m in g):
                g.append(j)
                break
        else:
            groups.append([j])
    return groups


def winner(groups) -> tuple[int | None, bool, int]:
    """Largest group, ties to the earliest-formed group; returns (representative index, tied, largest size)."""
    if not groups:
        return None, False, 0
    size = max(map(len, groups))
    largest = [g for g in groups if len(g) == size]
    return largest[0][0], len(largest) > 1, size


def prefix_groups(groups, k) -> list[list[int]]:
    """Groups of the first k candidates. Membership is decided sequentially, so a prefix's groups are the
    restriction of the full groups to indices < k (empty groups dropped, formation order kept)."""
    out = [[i for i in g if i < k] for g in groups]
    return [g for g in out if g]


# ----------------------------------------------------------------------------- bootstrap
class Boot:
    """Paired cluster bootstrap: canonical problems drawn with replacement, all rows of a drawn cluster kept."""

    def __init__(self, canonical_ids, n_boot=N_BOOT, seed=SEED):
        import numpy as np
        self.np = np
        _, cid = np.unique(list(canonical_ids), return_inverse=True)
        g = int(cid.max()) + 1
        rng = np.random.default_rng(seed)
        counts = np.stack([np.bincount(rng.integers(g, size=g), minlength=g) for _ in range(n_boot)]).astype(np.float32)
        self.W = counts[:, cid]
        self.den = self.W.sum(axis=1, dtype=np.float64)
        self.n, self.clusters, self.n_boot, self.seed = len(cid), g, n_boot, seed

    def means(self, v):
        return (self.W @ self.np.asarray(v, dtype=self.np.float32)).astype(self.np.float64) / self.den

    def ci(self, v, scale=100.0) -> dict:
        np = self.np
        v = np.asarray(v, dtype=float)
        b = self.means(v)
        lo, hi = np.quantile(b, [0.025, 0.975])
        p = min(1.0, 2 * min(float((b <= 0).mean()), float((b >= 0).mean())))
        return {"estimate": scale * float(v.mean()), "lo": scale * float(lo), "hi": scale * float(hi), "p_boot": p}

    def ratio_ci(self, num, den, scale=100.0) -> dict:
        np = self.np
        num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
        est = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
        bn = (self.W @ num.astype(np.float32)).astype(np.float64)
        bd = (self.W @ den.astype(np.float32)).astype(np.float64)
        ok = bd > 0
        lo, hi = np.quantile(bn[ok] / bd[ok], [0.025, 0.975]) if ok.any() else (float("nan"), float("nan"))
        return {"estimate": scale * est, "lo": scale * float(lo), "hi": scale * float(hi), "n_den": int(den.sum())}


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values for a dict name -> p."""
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(names), 0.0, {}
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def p_floor() -> float:
    """Smallest resolvable two-sided bootstrap p: 2 / replicates. A p of 0 is reported as <= this and enters Holm as it."""
    return 2 / N_BOOT


def pfmt(p: float) -> str:
    return f"≤{p_floor():.4f}" if p < p_floor() else f"{p:.4f}"


def tie_key(item_id: str) -> int:
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:12], 16)


def manifest(name: str, extra: dict | None = None) -> dict:
    """Write analysis/out/arr/<name>_manifest.json with input hashes, plan and script hashes."""
    import importlib.metadata
    import subprocess as sp
    head = sp.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = sp.run(["git", "status", "--porcelain", "--", "analysis/arr", "docs/planning/arr_round", "src"], cwd=REPO,
                   capture_output=True, text=True).stdout.strip()
    value = {"analysis": name, "plan_sha256_lf": plan_hash(), "plan_core_sha256": plan_core_hash(), "git_head": head, "code_dirty": bool(dirty),
             "scripts": {str(p.relative_to(REPO)).replace("\\", "/"): sha_lf(p) for p in sorted((REPO / "analysis/arr").glob("*.py"))},
             "scorer_sha256": sha(Path(scorer.__file__)),
             "versions": {p: importlib.metadata.version(p) for p in ("numpy", "sympy", "math-verify", "psutil")},
             "bootstrap": {"replicates": N_BOOT, "seed": SEED, "unit": "canonical problem"},
             "inputs": dict(sorted(INPUTS.items())), "scorer_guards": list(GUARDS_REPORTED), **(extra or {})}
    dump(OUT / f"{name}_manifest.json", value)
    return value


if __name__ == "__main__" and "--worker" in sys.argv:
    worker()

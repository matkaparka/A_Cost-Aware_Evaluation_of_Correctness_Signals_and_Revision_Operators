"""The ARR job list (cloud/arr_jobs.py) and the multi-GPU runner (cloud/arr_parallel.py). CPU only: fake jobs."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cloud"))
import arr_jobs as J  # noqa: E402
import arr_parallel as P  # noqa: E402


def groups(jobs):
    out = {}
    for j in jobs:
        if "chunk" in j:
            c, k, size = j["chunk"]
            base = j["name"][: -len(f"-c{c}")] if k > 1 else j["name"]
            out.setdefault(base, []).append(j)
    return out


def test_chunks_cover_every_item_once():
    for base, js in groups(J.build("abc1234")).items():
        k, size = js[0]["chunk"][1], js[0]["chunk"][2]
        assert sorted(j["chunk"][0] for j in js) == list(range(k)), base
        n = sum(j["items"] for j in js)
        assert all(j["items"] == size for j in js if j["chunk"][0] < k - 1) and 0 < js[-1]["items"] <= size, base
        if k > 1:
            assert "--chunk-size" in js[0]["cmd"] and js[0]["cmd"][js[0]["cmd"].index("--chunk-size") + 1] == str(size)
        assert n > 0


def test_estimates_match_the_price_plan():
    prices = json.loads((REPO / "docs/planning/arr_round/prices.json").read_text(encoding="utf-8"))
    planned = prices["total_gpu_h"] - prices["rows_gpu_h"]["smoke and parity block, model downloads"]
    est = sum(j["est_h"] for j in J.build("abc1234") if j["gpu"])
    assert abs(est - planned) / planned < 0.05, (est, planned)


def test_dependencies_and_chunk_sizes():
    jobs = J.build("abc1234")
    names = {j["name"] for j in jobs}
    for j in jobs:
        assert set(j["deps"]) <= names, j["name"]
        if j["gpu"] and j["chunk"][1] > 1:
            assert j["est_h"] <= 0.75 * 1.05, (j["name"], j["est_h"])
    by = {j["name"]: j for j in jobs}
    assert by["b4-critique-math-8B"]["deps"] == ["b4-select-math-8B"]
    sel = by["b4-select-math-8B"]
    assert sel["deps"] and all(d.startswith("b1-test-k5-8B") for d in sel["deps"])
    assert all(f"b1/8B/b1-samples-test-k5-h2a-consistency-t4096" in a for a in sel["cmd"] if a.endswith("samples.jsonl"))
    assert by["b4-select-math-4B"]["deps"] == [] and not by["b4-select-math-4B"]["gpu"]
    assert {j["priority"] for j in jobs if j["name"].startswith("c1-think-math-8B")} == {4}


def test_reenv_adds_main_wording_runs():
    base = {j["name"] for j in J.build("abc1234")}
    more = {j["name"] for j in J.build("abc1234", reenv=True)} - base
    assert more and all(n.startswith("b3-main-") for n in more)
    assert {n.split("-")[2] for n in more} == {"4B", "8B", "T8B"}


def run_fake(tmp_path, jobs, gpus=(0, 1)):
    r = P.Runner(jobs, list(gpus), tmp_path / "logs", fake_scale=1.0, poll=0.02)
    rc = r.run()
    ev = [json.loads(l) for l in open(tmp_path / "logs" / "jobs.jsonl", encoding="utf-8")]
    return rc, r, ev


def test_runner_respects_dependencies_and_slots(tmp_path):
    jobs = [{"name": "a", "cmd": [], "est_h": 0.2, "priority": 1},
            {"name": "b", "cmd": [], "est_h": 0.1, "priority": 1},
            {"name": "c", "cmd": [], "est_h": 0.05, "priority": 2, "deps": ["a"]},
            {"name": "sel", "cmd": [], "est_h": 0.05, "gpu": False, "deps": ["b"]},
            {"name": "d", "cmd": [], "est_h": 0.05, "priority": 2, "deps": ["sel"]}]
    rc, r, ev = run_fake(tmp_path, jobs)
    assert rc == 0 and all(v == "done" for v in r.state.values())
    t = {(e["job"], e["status"]): i for i, e in enumerate(ev)}
    assert t[("a", "done")] < t[("c", "start")] and t[("sel", "done")] < t[("d", "start")] and t[("b", "done")] < t[("sel", "start")]
    running = {}
    for e in ev:
        if e["status"] == "start":
            assert e["slot"] not in running.values(), "two jobs on one slot"
            running[e["job"]] = e["slot"]
        elif e["status"] in ("done", "failed"):
            running.pop(e["job"], None)
    assert next(e for e in ev if e["job"] == "sel" and e["status"] == "start")["slot"] == "cpu"


def test_runner_skips_dependents_of_a_failure(tmp_path):
    jobs = [{"name": "a", "cmd": [], "est_h": 0.05, "fake_exit": 1},
            {"name": "b", "cmd": [], "est_h": 0.05, "deps": ["a"]},
            {"name": "c", "cmd": [], "est_h": 0.05, "deps": ["b"]},
            {"name": "x", "cmd": [], "est_h": 0.05}]
    rc, r, _ = run_fake(tmp_path, jobs)
    assert rc == 2 and r.state == {"a": "failed", "b": "skipped", "c": "skipped", "x": "done"}

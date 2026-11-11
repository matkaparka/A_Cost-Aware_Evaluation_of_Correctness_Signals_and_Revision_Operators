"""Exp 4 invariants (src/patch; design Exp 4, §8 "a self-patch reproduces the baseline").

Tokenizer-only (pinned Qwen3 tokenizer, dev items only): position groups are contiguous and cover the prompt from the
label on; every group but label and turn has identical tokens in receiver and donor; the label alignment is the
documented one; tokens before the label are identical; the target's gold span decodes to the gold; the markers and the
content exclusion match the frozen Exp 3 pilot.
Selection (numpy): the bootstrap is the scheme of analysis/p6_confirmatory.Boot; the k rule and the test rule.
GPU (Qwen3-1.7B from the local HF cache, dev pairs only; skipped without CUDA or the snapshot): the self-patch
reproduces the receiver baseline for every (layer, group) and for all components at once; patching the last block at the
last prompt position reproduces the donor's marker and leaves lp_gold unchanged; attribution is finite.
Run: python -m pytest tests/test_patch.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.patch import select as S
from src.patch.core import (GROUPS, MARKERS, METRICS, SAME_TOKENS, SELF_PATCH_TOL, TARGET_HEAD, align, build_pair,
                            marker_ids)
from src.patch.exp4 import make_pairs, ordered_items

ROOT = Path(__file__).resolve().parents[1]
TOK_DIR = ROOT / "data" / "tokenizers" / "qwen3-4b"
PILOT = ROOT / "results" / "p2" / "4B" / "p2-steer-pilot-4B-7ea8f7d" / "metrics.json"
need_tok = pytest.mark.skipif(not TOK_DIR.exists(), reason="pinned tokenizer not downloaded")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


@pytest.fixture(scope="module")
def dev_pairs(tok):
    return make_pairs(tok, ordered_items("dev"))


def test_align_rule():
    assert align([1, 2, 3], [1, 2, 3]) == [(0, 0), (1, 1), (2, 2)]
    # common prefix [1], common suffix [9, 9], middle [2, 3] vs [5, 6, 7] right-aligned: 2<-6, 3<-7, 5 unpaired
    assert align([1, 2, 3, 9, 9], [1, 5, 6, 7, 9, 9]) == [(0, 0), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert align([4], [5]) == [(0, 0)]
    assert align([1, 2], [1, 2, 2]) == [(0, 0), (1, 1)]   # the common prefix is taken first


@need_tok
def test_groups_and_alignment(tok, dev_pairs):
    pairs, _ = dev_pairs
    gold_of = {it["item_id"]: it["gold"] for it in ordered_items("dev")}
    for p in pairs:
        r, d = p.r, p.d
        la = r.ranges["label"][0]
        assert r.ids[:la] == d.ids[:la]
        assert r.ranges["label"][0] < r.ranges["turn"][0] and r.ranges["last"] == (r.P - 1, r.P)
        for g in GROUPS:
            rp, dp = p.groups[g]
            assert rp == sorted(rp) and dp == sorted(dp) and len(rp) == len(dp) > 0
            if g in SAME_TOKENS:
                assert [r.ids[i] for i in rp] == [d.ids[j] for j in dp] and p.unpaired[g] == (0, 0)
        # turn: same length, only the role token differs
        rt = [r.ids[i] for i in p.groups["turn"][0]]
        dt = [d.ids[j] for j in p.groups["turn"][1]]
        diff = [(tok.decode([a]), tok.decode([b])) for a, b in zip(rt, dt) if a != b]
        assert diff == [("assistant", "user")] and p.unpaired["turn"] == (0, 0)
        # label: " your own" vs " another model's" -> " your" <- " model", " own" <- "'s"; " another" unpaired
        lab = [(tok.decode([r.ids[i]]), tok.decode([d.ids[j]])) for i, j in zip(*p.groups["label"])]
        changed = [x for x in lab if x[0] != x[1]]
        assert changed == [(" your", " model"), (" own", "'s")] and p.unpaired["label"] == (0, 1)
        # target: identical in both sides, natural tokenization; the gold span is the gold plus at most the box
        # braces / period merged into its boundary tokens (e.g. "{\\" and "}}." for \frac golds)
        assert r.ids[r.P:] == d.ids[d.P:]
        span = tok.decode([r.ids[i] for i in r.gold_pos])
        head = tok.decode(r.ids[r.P:r.gold_pos[0]])
        gold = gold_of[p.item_id]
        assert span.lstrip("{").rstrip(".").removesuffix("}") == gold or span.lstrip("{").rstrip(".") == gold, (span, gold)
        assert TARGET_HEAD.startswith(head) and len(TARGET_HEAD) - len(head) <= 1
        assert r.gold_pos == list(range(r.gold_pos[0], r.gold_pos[-1] + 1))


@need_tok
def test_markers_and_exclusions_match_exp3_pilot(tok, dev_pairs):
    mk = marker_ids(tok)
    assert len(mk["correction"]) == len(mk["commit"]) == 1
    if PILOT.exists():
        m = json.loads(PILOT.read_text())
        assert MARKERS == {"correction": m["correction_markers"], "commit": m["commit_markers"]}
        assert len(dev_pairs[1]) == m["n_dinj_excluded_no_content"] and len(dev_pairs[0]) == m["n_dir_items"]


def test_bootstrap_is_the_frozen_scheme():
    sys.path.insert(0, str(ROOT / "analysis"))
    p6 = pytest.importorskip("p6_confirmatory")
    rng = np.random.default_rng(1)
    clusters = rng.integers(0, 40, 120)
    v = rng.normal(0.3, 1.0, 120)
    est, lo, hi = p6.Boot(clusters, n_boot=500, seed=0).ci(lambda m: p6.wmean(v, m))
    mu, l2, h2 = S.mean_ci(v, clusters, n_boot=500, seed=0)
    assert np.isclose(mu[0], est) and np.isclose(l2[0], lo) and np.isclose(h2[0], hi)


def test_k_rule():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(100), 2)
    comps = [(L, "instr") for L in range(20)]
    e = rng.normal(0, 1, (200, 20))
    e[:, 3] += 2.0       # strong
    e[:, 7] -= 1.0       # strong, negative
    e[:, 11] += 0.5      # weaker
    stats, k, frozen = S.choose(e, clusters, comps, n_boot=2000)
    assert k == sum(s["eligible"] for s in stats) and 3 <= k <= S.K_MAX
    assert [c[:2] for c in frozen[:3]] == [(3, "instr"), (7, "instr"), (11, "instr")]
    stats0, k0, f0 = S.choose(np.zeros((200, 20)), clusters, comps, n_boot=500)   # nothing eligible -> K_MIN
    assert not any(s["eligible"] for s in stats0) and k0 == S.K_MIN and f0[0][:2] == comps[0]
    e2 = rng.normal(0, 0.1, (200, 20)) + np.arange(1, 21)
    _, k2, f2 = S.choose(e2, clusters, comps, n_boot=500)
    assert k2 == S.K_MAX and [c[0] for c in f2] == [19, 18, 17, 16, 15]


def test_final_block_not_selectable_for_marker():
    n_layers = 28
    assert S.excluded_layers("marker", n_layers) == [n_layers - 1] and S.excluded_layers("lp_gold", n_layers) == []
    rng = np.random.default_rng(3)
    attr = rng.normal(0, 0.01, (30, n_layers, len(GROUPS)))
    attr[:, n_layers - 1, GROUPS.index("last")] = 5.0      # the largest attribution sits in the final block
    attr[:, n_layers - 1, GROUPS.index("gen")] = -4.0
    top_m = S.top_by_attribution(attr, GROUPS, exclude_layers=S.excluded_layers("marker", n_layers))
    top_g = S.top_by_attribution(attr, GROUPS, exclude_layers=S.excluded_layers("lp_gold", n_layers))
    assert len(top_m) == S.N_REAL and all(L != n_layers - 1 for L, _, _ in top_m)
    assert top_g[0][:2] == (n_layers - 1, "last") and top_g[1][:2] == (n_layers - 1, "gen")
    with pytest.raises(ValueError):
        S.choose(np.ones((4, 1)), np.arange(4), [(n_layers - 1, "last")], n_boot=10, exclude_layers=[n_layers - 1])


def test_commit_check(tmp_path):
    """The test stage's commit check (exp4.check_frozen with need_commit) in a throwaway git repo."""
    import hashlib
    import shutil
    import subprocess
    from src.patch.exp4 import check_frozen
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)

    def git(*a):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=exp4-test", "-c", "user.email=exp4-test@invalid", *a],
                       check=True, capture_output=True)

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    git("init", "-q")
    f = repo / "results" / "frozen.json"
    f.write_bytes(b'{"k": 1}' + bytes([10]))
    git("add", "results/frozen.json")
    git("commit", "-q", "-m", "frozen")
    rel = "results/frozen.json"
    assert check_frozen(rel, sha(f), True, repo=repo) == {"k": 1}            # committed, unmodified: passes
    with pytest.raises(SystemExit, match="sha256"):
        check_frozen(rel, "0" * 64, True, repo=repo)                          # wrong sha
    f.write_bytes(b'{"k": 2}' + bytes([10]))                                  # modified (sha of the new content)
    with pytest.raises(SystemExit, match="committed, unmodified"):
        check_frozen(rel, sha(f), True, repo=repo)
    git("add", rel)                                                           # modified and staged, not committed
    with pytest.raises(SystemExit, match="committed, unmodified"):
        check_frozen(rel, sha(f), True, repo=repo)
    git("commit", "-q", "-m", "update")
    assert check_frozen(rel, sha(f), True, repo=repo) == {"k": 2}            # committed again: passes
    g = repo / "results" / "new.json"                                         # never added
    g.write_bytes(b'{"k": 3}' + bytes([10]))
    with pytest.raises(SystemExit, match="committed, unmodified"):
        check_frozen("results/new.json", sha(g), True, repo=repo)
    git("add", "results/new.json")                                            # staged, never committed
    with pytest.raises(SystemExit, match="committed, unmodified"):
        check_frozen("results/new.json", sha(g), True, repo=repo)
    assert check_frozen("results/new.json", sha(g), False, repo=repo) == {"k": 3}   # dev smokes: sha check only
    with pytest.raises(SystemExit, match="not inside the code checkout"):
        out = tmp_path / "outside.json"
        out.write_bytes(b"{}")
        check_frozen(str(out), sha(out), True, repo=repo)


def test_confirm_rule():
    rng = np.random.default_rng(2)
    clusters = np.arange(200)
    e = np.stack([rng.normal(1, 1, 200), rng.normal(-1, 1, 200), rng.normal(0, 1, 200)], 1)
    out = S.confirm(e, clusters, [(1, "gen", 0.8), (2, "gen", 0.7), (3, "gen", 0.5)], n_boot=1000)
    assert [o["replicated"] for o in out] == [True, False, False]


# ---------------------------------------------------------------- GPU: Qwen3-1.7B, dev pairs
def _gpu_model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from huggingface_hub import snapshot_download
    try:
        snapshot_download("Qwen/Qwen3-1.7B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen3-1.7B not in the local HF cache")
    from src.patch.exp4 import load_model
    return load_model("1.7B")


@pytest.fixture(scope="module")
def model():
    return _gpu_model()


@need_tok
def test_self_patch_reproduces_baseline(model, tok, dev_pairs):
    import torch
    from src.patch.core import patch_pair, run, self_patch_error, side_metrics
    mk = marker_ids(tok)
    n_layers = len(model.model.layers)
    p = dev_pairs[0][0]
    h0, own = run(model, p.r.ids, capture=range(n_layers))
    base = [float(x) for x in side_metrics(model, h0, p.r, mk)]
    worst = 0.0
    for L in range(n_layers):
        for g in GROUPS:
            pos = p.groups[g][0]
            h, _ = run(model, p.r.ids, patches={L: (pos, own[L][0, torch.tensor(pos, device=h0.device)])})
            v = [float(x) for x in side_metrics(model, h, p.r, mk)]
            worst = max(worst, *(abs(a - b) for a, b in zip(v, base)))
    assert worst <= SELF_PATCH_TOL, worst
    comps = [(L, g) for L in range(0, n_layers, 3) for g in GROUPS]
    for q in dev_pairs[0][:3]:
        arms = patch_pair(model, q, comps, mk)
        assert max(self_patch_error(arms).values()) <= SELF_PATCH_TOL
        assert arms["base_r"]["lp_gold"] < 0 and np.isfinite(arms["base_r"]["marker"])


@need_tok
def test_readout_patch_reproduces_donor_marker(model, tok, dev_pairs):
    from src.patch.core import patch_pair
    mk = marker_ids(tok)
    last = len(model.model.layers) - 1
    for p in dev_pairs[0][:3]:
        arms = patch_pair(model, p, [(last, "last")], mk)
        a = arms[f"L{last}:last"]
        assert abs(a["marker"] - arms["base_d"]["marker"]) <= SELF_PATCH_TOL
        assert abs(a["lp_gold"] - arms["base_r"]["lp_gold"]) <= SELF_PATCH_TOL


@need_tok
def test_attribution_finite_and_matches_forward(model, tok, dev_pairs):
    from src.patch.core import attribute, patch_pair
    mk = marker_ids(tok)
    p = dev_pairs[0][1]
    A, mr, md = attribute(model, p, mk)
    assert A.shape == (len(METRICS), len(model.model.layers), len(GROUPS)) and np.isfinite(A).all()
    assert np.abs(A).sum() > 0
    arms = patch_pair(model, p, [(0, "label")], mk)
    # the gradient forward (inputs_embeds path) and the no-grad forward agree up to bf16 kernel noise
    # (exactly equal on the laptop's RTX 5070 Ti; the bound allows another SDPA backend under autograd)
    assert abs(mr[1] - arms["base_r"]["marker"]) < 0.05 and abs(md[1] - arms["base_d"]["marker"]) < 1e-6

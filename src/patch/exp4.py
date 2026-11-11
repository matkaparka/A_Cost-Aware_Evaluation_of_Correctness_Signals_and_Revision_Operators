"""Exp 4 - activation patching (exploratory localization; design Exp 4, prereg §3.3; blocks.yaml 4B-exp4: priority 10,
chunk 100, min_n 100, planned n 200 [analysis/p3_min_n.py]). Written after the freeze (DEVIATIONS entry pending).
Primitives: src/patch/core.py (pairs, position groups, metrics, hooks). Selection and the test rule: src/patch/select.py.

Two stages, separate invocations; the dev stage's frozen.json is synced and committed between them.
  dev   every D-inj dev pair: attribution patching over (layer x position group) -> the top N_REAL = 20 selectable
        components per metric (the marker excludes the final block; src/patch/select.py step 0) ->
        real patching of their union -> the k rule -> frozen.json (frozen components per metric, k, the dev statistics).
  test  the first --n (200) D-inj test pairs: real patching of the frozen components (union over metrics), the receiver
        and donor baselines and the self-patch. Chunks of --chunk (100) pairs, each complete (every arm of every pair)
        before the next; a re-run resumes after the last complete chunk. Refuses to run unless frozen.json matches
        --frozen-sha256 and is committed, unmodified, in the code checkout. `test --split dev` runs the same code on
        dev pairs (smoke tests; no git check).
Items: D-inj items of the split with solution content (more than SKIP_FIRST solution tokens; others are excluded and
counted, as in every Exp 3 stage), in the seeded order of the split: (rank of the canonical ID in split.json "order",
item_id), as src/exp/runner.py orders the alt-wrapper subset.
Runtime invariant (design §8): for every pair, the self-patch reproduces the receiver baseline within SELF_PATCH_TOL
for both metrics; otherwise the run stops and is registered failed.
Outputs. dev: pairs.jsonl (per-pair baselines of the attribution pass), attr.npz (attr [pairs, metric, layer, group]),
patch_rows.jsonl (one row per pair x arm), frozen.json, metrics.json. test: rows.jsonl, metrics.json. Both: config.json,
manifest.json. Statistics: analysis/p4_exp4.py.
Usage (run host, from the run's code worktree):
  python -m src.patch.exp4 dev  --model 4B [--limit N]
  python -m src.patch.exp4 test --model 4B --frozen results/p2/4B/<dev run>/frozen.json --frozen-sha256 <sha256>
                                [--n 200] [--chunk 100] [--split dev --limit N]
  python -m src.patch.exp4 {dev,test} --model 4B --dry-run [--tokenizer PATH]   (renders and counts pairs; no model)
"""
import argparse
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..gen import render as R
from ..runlib import MODELS, REPO, code_commit, register, run_dir, sha256_file, sha256_text, write_manifest
from ..steer.steer import SKIP_FIRST
from . import select as S
from .core import (DONOR, GROUPS, MARKERS, METRICS, RECEIVER, SELF_PATCH_TOL, TARGET_HEAD, TARGET_TAIL, NoContent,
                   attribute, build_pair, marker_ids, patch_pair, self_patch_error)

N_TEST, CHUNK = 200, 100   # blocks.yaml 4B-exp4 chunk 100; planned n 200 (analysis/p3_min_n.py)
# The code that defines pairs, groups, metrics, patching and the selection rule; the test stage refuses to run if any
# of these differs from the dev run's (frozen.json "code_sha256").
SCIENCE = ("src/patch/core.py", "src/patch/select.py", "src/gen/render.py", "src/steer/steer.py")


def science_sha256():
    return {f: sha256_file(REPO / f) for f in SCIENCE}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def ordered_items(split):
    P = REPO / "data" / "processed"
    rank = {c: i for i, c in enumerate(json.loads((P / "split.json").read_text())["order"][split])}
    return sorted((r for r in jl(P / "dinj.jsonl") if r["split"] == split), key=lambda r: (rank[r["canonical_id"]], r["item_id"]))


def make_pairs(tok, items):
    pairs, excluded = [], []
    for it in items:
        try:
            pairs.append(build_pair(tok, it))
        except NoContent:
            excluded.append(it["item_id"])
    return pairs, excluded


def arm_name(L, g):
    return f"L{L}:{g}"


def check_frozen(path, sha, need_commit, repo=REPO):
    """sha256 check; with need_commit, the file must be tracked in `repo` (the code checkout) and equal to HEAD."""
    p = Path(path) if Path(path).is_absolute() else Path(repo) / path
    got = sha256_file(p)
    if got != sha:
        raise SystemExit(f"{p}: sha256 {got} is not --frozen-sha256 {sha}")
    if need_commit:
        try:
            rel = str(p.resolve().relative_to(Path(repo).resolve())).replace("\\", "/")
        except ValueError:
            raise SystemExit(f"{p} is not inside the code checkout {repo}") from None
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode == 0
        if not (tracked and clean):
            raise SystemExit(f"{rel} must be committed, unmodified, at the locked commit before any test run")
    return json.loads(p.read_text(encoding="utf-8"))


def write_row(f, p, arm, v):
    L, g = (int(arm[1:].split(":")[0]), arm.split(":")[1]) if arm.startswith("L") else (None, None)
    f.write(json.dumps({"item_id": p.item_id, "canonical_id": p.canonical_id, "dataset": p.dataset, "arm": arm,
                        "layer": L, "group": g, **v}) + "\n")


def invariant(run_id, args, p, arms):
    err = self_patch_error(arms)
    if not all(math.isfinite(arms[a][m]) for a in arms for m in METRICS):
        register(run_id, "failed", block=f"{args.model}-exp4", exp="exp4", model=args.model)
        raise SystemExit(f"{p.item_id}: non-finite metric")
    if max(err.values()) > SELF_PATCH_TOL:
        register(run_id, "failed", block=f"{args.model}-exp4", exp="exp4", model=args.model)
        raise SystemExit(f"{p.item_id}: self-patch differs from the receiver baseline: {err} > {SELF_PATCH_TOL}")
    return err


def load_model(key):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODELS[key], dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    return model


def stage_dev(args, tok, pairs, excluded, cfg):
    import transformers
    mk = marker_ids(tok)
    model = load_model(args.model)
    n_layers = len(model.model.layers)
    run_id, d = run_dir("p2", args.model, "patch-dev" + ("-smoke" if args.limit else ""))
    register(run_id, "running", block=f"{args.model}-exp4", exp="exp4", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()

    # 1. Attribution patching on every dev pair.
    A = np.zeros((len(pairs), len(METRICS), n_layers, len(GROUPS)))
    with open(d / "pairs.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            A[i], mr, md = attribute(model, p, mk)
            if not (np.isfinite(A[i]).all() and all(map(math.isfinite, mr + md))):
                register(run_id, "failed", block=f"{args.model}-exp4", exp="exp4", model=args.model)
                raise SystemExit(f"{p.item_id}: non-finite attribution")
            f.write(json.dumps({"item_id": p.item_id, "canonical_id": p.canonical_id, "dataset": p.dataset,
                                "prompt_tokens_r": p.r.P, "prompt_tokens_d": p.d.P, "gold_tokens": len(p.r.gold_pos),
                                "r": dict(zip(METRICS, mr)), "d": dict(zip(METRICS, md)),
                                "paired": {g: len(p.groups[g][0]) for g in GROUPS}, "unpaired": p.unpaired}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} attribution {i + 1}/{len(pairs)}", flush=True)
    np.savez_compressed(d / "attr.npz", attr=A.astype(np.float32), item_id=np.array([p.item_id for p in pairs]),
                        canonical_id=np.array([p.canonical_id for p in pairs]), metrics=np.array(METRICS),
                        groups=np.array(GROUPS))
    t_attr = time.time() - t0

    # 2. Real patching of the union of the per-metric top N_REAL, on every dev pair.
    excl = {m: S.excluded_layers(m, n_layers) for m in METRICS}
    top = {m: S.top_by_attribution(A[:, k], GROUPS, exclude_layers=excl[m]) for k, m in enumerate(METRICS)}
    union = sorted({(L, g) for m in METRICS for L, g, _ in top[m]}, key=lambda c: (c[0], GROUPS.index(c[1])))
    eff = np.zeros((len(pairs), len(METRICS), len(union)))
    err_max = dict.fromkeys(METRICS, 0.0)
    total = np.zeros((len(pairs), len(METRICS)))
    with open(d / "patch_rows.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            arms = patch_pair(model, p, union, mk)
            for arm, v in arms.items():
                write_row(f, p, arm, v)
            f.flush()
            err = invariant(run_id, args, p, arms)
            err_max = {m: max(err_max[m], err[m]) for m in METRICS}
            for k, m in enumerate(METRICS):
                total[i, k] = arms["base_d"][m] - arms["base_r"][m]
                eff[i, k] = [arms[arm_name(L, g)][m] - arms["base_r"][m] for L, g in union]
            if (i + 1) % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} real patching {i + 1}/{len(pairs)}", flush=True)
    t_real = time.time() - t0 - t_attr

    # 3. The k rule per metric; frozen.json.
    clusters = [p.canonical_id for p in pairs]
    sel, test_comps = {}, set()
    for k, m in enumerate(METRICS):
        comps = [(L, g) for L, g, _ in top[m]]
        e = eff[:, k, [union.index(c) for c in comps]]
        stats, kk, frozen = S.choose(e, clusters, comps, exclude_layers=excl[m])
        sel[m] = {"attr_top": [{"layer": L, "group": g, "mean_attr": a} for L, g, a in top[m]], "real": stats, "k": kk,
                  "frozen": [{"layer": L, "group": g, "dev_mean": mu} for L, g, mu in frozen],
                  "not_selectable_attr": [{"layer": L, "group": g, "mean_attr": float(A[:, k, L, j].mean())}
                                          for L in excl[m] for j, g in enumerate(GROUPS)]}
        test_comps |= {(L, g) for L, g, _ in frozen}
    test_comps = sorted(test_comps, key=lambda c: (c[0], GROUPS.index(c[1])))
    frozen = {"model": args.model, "dev_run_id": run_id, "code_commit": code_commit(), "code_sha256": science_sha256(),
              "n_dev_pairs": len(pairs),
              "receiver": RECEIVER, "donor": DONOR, "groups": GROUPS, "metrics": METRICS, "markers": MARKERS,
              "marker_ids": mk, "target": TARGET_HEAD + "<gold>" + TARGET_TAIL, "self_patch_tol": SELF_PATCH_TOL,
              "rule": {"n_real": S.N_REAL, "k_max": S.K_MAX, "k_min": S.K_MIN, "n_boot": S.N_BOOT, "seed": S.SEED,
                       "not_selectable": {m: excl[m] for m in METRICS},
                       "not_selectable_reason": "marker: final block (L = n_layers - 1); patching it at the last prompt "
                                                "position reproduces the donor's marker by construction; kept in the "
                                                "attribution table, not in the top-N_REAL pool or the k ranking",
                       "test": "replicated iff the test mean has the dev sign and its 95 % cluster-bootstrap CI excludes 0"},
              "test_plan": {"n": N_TEST, "chunk": CHUNK}, "selection": sel,
              "test_components": [{"layer": L, "group": g} for L, g in test_comps]}
    (d / "frozen.json").write_text(json.dumps(frozen, indent=1))

    # Descriptive dev summaries: total effect D - R, and attribution vs real patching over the real-patched comps.
    tot_mu, tot_lo, tot_hi = S.mean_ci(total, clusters)
    agree = {}
    for k, m in enumerate(METRICS):
        a = np.array([A[:, k, L, GROUPS.index(g)].mean() for L, g in union])
        r = eff[:, k].mean(0)
        agree[m] = {"pearson": float(np.corrcoef(a, r)[0, 1]) if len(union) > 1 else None,
                    "sign_agreement": float(np.mean(np.sign(a) == np.sign(r)))}
    metrics = {"model": args.model, "split": "dev", "pairs": len(pairs), "excluded_no_content": len(excluded),
               "layers": n_layers, "groups": GROUPS, "real_patched": len(union),
               "k": {m: sel[m]["k"] for m in METRICS}, "test_components": len(test_comps),
               "total_effect": {m: {"mean": float(tot_mu[k]), "lo": float(tot_lo[k]), "hi": float(tot_hi[k])}
                                for k, m in enumerate(METRICS)},
               "attr_vs_real": agree, "self_patch_max_abs_err": err_max, "self_patch_tol": SELF_PATCH_TOL,
               "unpaired_tokens": {g: [int(sum(p.unpaired[g][0] for p in pairs)), int(sum(p.unpaired[g][1] for p in pairs))]
                                   for g in GROUPS},
               "passes": {"attr_fwd_bwd": len(pairs), "attr_backward": len(pairs) * len(METRICS), "attr_donor_fwd": len(pairs),
                          "real_fwd": len(pairs) * (len(union) + 3)},
               "attr_s": t_attr, "real_s": t_real, "wall_s": time.time() - t0, "gpu": torch.cuda.get_device_name(0),
               "transformers": transformers.__version__}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {"wording": cfg["wording_sha256"]})
    register(run_id, "done", block=f"{args.model}-exp4", exp="exp4", model=args.model)
    print(json.dumps({"run_id": run_id, "frozen_sha256": sha256_file(d / "frozen.json"),
                      **{k: metrics[k] for k in ("pairs", "real_patched", "k", "test_components", "self_patch_max_abs_err", "wall_s")}},
                     indent=1))


def stage_test(args, tok, pairs, excluded, cfg, frozen):
    import transformers
    mk = marker_ids(tok)
    if frozen["model"] != args.model or frozen["marker_ids"] != mk or list(frozen["groups"]) != list(GROUPS):
        raise SystemExit("frozen.json does not match this model, the markers or the position groups")
    if frozen["code_sha256"] != science_sha256():
        raise SystemExit(f"code differs from the dev run's: {frozen['code_sha256']} vs {science_sha256()}")
    comps = [(c["layer"], c["group"]) for c in frozen["test_components"]]
    arms_all = {"base_r", "base_d", "self_patch"} | {arm_name(L, g) for L, g in comps}
    chunks = [pairs[i:i + args.chunk] for i in range(0, len(pairs), args.chunk)]
    model = load_model(args.model)
    kind = ("patch-test" if args.split == "test" else "patch-test-on-dev") + ("-smoke" if args.limit else "")
    run_id, d = run_dir("main" if args.split == "test" else "p2", args.model, kind)
    register(run_id, "running", block=f"{args.model}-exp4", exp="exp4", model=args.model,
             config_sha256=sha256_text(json.dumps(cfg, sort_keys=True)))
    t0 = time.time()
    rows_f = d / "rows.jsonl"
    done = defaultdict(set)
    if rows_f.exists():
        for r in jl(rows_f):
            done[r["item_id"]].add(r["arm"])
    complete = [all(done[p.item_id] >= arms_all for p in ch) for ch in chunks]
    keep = {p.item_id for ch, ok in zip(chunks, complete) if ok for p in ch}
    if rows_f.exists():   # rows of an incomplete chunk are dropped; the chunk is redone in full
        rows_f.write_text("".join(l for l in open(rows_f, encoding="utf-8") if json.loads(l)["item_id"] in keep), encoding="utf-8")
    with open(rows_f, "a", encoding="utf-8") as f:
        for c, (ch, ok) in enumerate(zip(chunks, complete)):
            if ok:
                continue
            for p in ch:
                arms = patch_pair(model, p, comps, mk)
                invariant(run_id, args, p, arms)
                for arm, v in arms.items():
                    write_row(f, p, arm, v)
            f.flush()
            print(f"{time.strftime('%H:%M:%S')} chunk {c + 1}/{len(chunks)} complete", flush=True)
    rows = jl(rows_f)
    base = {r["item_id"]: r for r in rows if r["arm"] == "base_r"}
    err = {m: max(abs(r[m] - base[r["item_id"]][m]) for r in rows if r["arm"] == "self_patch") for m in METRICS}
    metrics = {"model": args.model, "split": args.split, "pairs": len(pairs), "excluded_no_content": len(excluded),
               "chunks": len(chunks), "chunks_complete": len(chunks), "components": len(comps),
               "self_patch_max_abs_err": err, "self_patch_tol": SELF_PATCH_TOL,
               "passes": {"real_fwd": len(pairs) * (len(comps) + 3)}, "wall_s": time.time() - t0,
               "gpu": torch.cuda.get_device_name(0), "transformers": transformers.__version__}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_manifest(d, run_id, args.model, tok, cfg, {"name": "hf-transformers", "version": transformers.__version__},
                   {"wording": cfg["wording_sha256"]})
    register(run_id, "done", block=f"{args.model}-exp4", exp="exp4", model=args.model)
    print(json.dumps({"run_id": run_id, **{k: metrics[k] for k in ("pairs", "chunks", "components", "self_patch_max_abs_err", "wall_s")}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "test"])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", default=None, choices=["dev", "test"], help="test stage only; dev = smoke test of its code")
    ap.add_argument("--frozen", default=None, help="test stage: the dev run's frozen.json (repo-relative)")
    ap.add_argument("--frozen-sha256", default=None)
    ap.add_argument("--n", type=int, default=N_TEST, help="test stage: pairs in the seeded order")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: first N pairs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokenizer", default=None, help="dry runs: local tokenizer path")
    args = ap.parse_args()
    split = "dev" if args.stage == "dev" else (args.split or "test")
    args.split = split
    if args.stage == "dev" and (args.frozen or args.split != "dev"):
        raise SystemExit("the dev stage takes no --frozen and runs on dev only")

    frozen = None   # checked before any item of the split is rendered
    if args.stage == "test" and (args.frozen or not args.dry_run):
        if not (args.frozen and args.frozen_sha256):
            raise SystemExit("the test stage needs --frozen and --frozen-sha256")
        frozen = check_frozen(args.frozen, args.frozen_sha256, need_commit=split == "test" and not args.dry_run)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or MODELS[args.model])
    items = ordered_items(split)
    pairs, excluded = make_pairs(tok, items)
    if args.stage == "test":
        pairs = pairs[:args.n]
    if args.limit:
        pairs = pairs[:args.limit]
    cfg = {**vars(args), "receiver": RECEIVER, "donor": DONOR, "groups": GROUPS, "metrics": METRICS,
           "target": TARGET_HEAD + "<gold>" + TARGET_TAIL, "markers": MARKERS, "skip_first": SKIP_FIRST,
           "self_patch_tol": SELF_PATCH_TOL, "n_real": S.N_REAL, "k_max": S.K_MAX, "k_min": S.K_MIN,
           "n_boot": S.N_BOOT, "boot_seed": S.SEED, "order": "split.json order, then item_id",
           "wording_sha256": sha256_text(json.dumps([R.PROMPT, R.LABELS, R.INSTRUCTION], sort_keys=True))}
    plan = {"stage": args.stage, "split": split, "items": len(items), "excluded_no_content": len(excluded),
            "pairs": len(pairs), "datasets": dict(Counter(p.dataset for p in pairs)),
            "chunks": math.ceil(len(pairs) / args.chunk) if args.stage == "test" else 1,
            "unpaired_tokens": {g: [sum(p.unpaired[g][0] for p in pairs), sum(p.unpaired[g][1] for p in pairs)]
                                for g in GROUPS},
            "prompt_tokens_max": max((p.d.P for p in pairs), default=0)}
    if frozen:
        cfg["frozen_sha256"] = args.frozen_sha256
        plan["components"] = len(frozen["test_components"])
        plan["real_fwd"] = len(pairs) * (len(frozen["test_components"]) + 3)
    if args.dry_run:
        print(json.dumps(plan, indent=1))
        return
    if args.stage == "dev":
        stage_dev(args, tok, pairs, excluded, cfg)
    else:
        stage_test(args, tok, pairs, excluded, cfg, frozen)


if __name__ == "__main__":
    main()

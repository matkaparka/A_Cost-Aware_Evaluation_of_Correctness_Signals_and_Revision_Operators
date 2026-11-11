"""Paper figures (P7; CLAUDE.md rule 1): drawn from the generated analysis files, no new statistics.
-> paper/figures/fig_forest.pdf   family quantities with 95 % CIs, 1.7B / 4B / 8B (analysis/out/06_confirmatory_4B.json,
                                  p4_{1.7B,8B}_tier1.json)
   paper/figures/fig_streams.pdf  greedy dev accuracy of the persistent streams by checkpoint (p4_4B_streams.json)
   paper/figures/fig_streams_controls.pdf  the follow-up stream controls nll, matched next to self, ext (fu_03_group2.json)
Palette: the validated categorical slots 1-4 (dataviz reference palette, light mode); two slots are below 3:1 contrast
against the surface, so every series also has its own marker and a legend or direct label.
Usage: python analysis/p7_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "analysis" / "out", ROOT / "paper" / "figures"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"
MARK = ["o", "s", "^", "D"]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42})


def jload(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def forest():
    conf = jload("06_confirmatory_4B")["main"]
    rep = {m: jload(f"p4_{m}_tier1")["estimates"] for m in ("1.7B", "8B")}
    rows = ["H1a", "H1b", "H1d", "H2a", "H3a-role", "H3a-label", "H3b"]
    models = [("1.7B", rep["1.7B"]), ("4B", conf), ("8B", rep["8B"])]
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    for j, (m, src) in enumerate(models):
        for i, h in enumerate(rows):
            if h not in src or "est" not in src[h]:
                continue
            r = src[h]
            y = i + (j - 1) * 0.24
            ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=SLOT[j], lw=1.4, solid_capstyle="round")
            ax.plot(100 * r["est"], y, MARK[j], color=SLOT[j], ms=4.5, mec="white", mew=0.6,
                    label=m if i == 0 else None)
    for x, ls in ((0, "-"), (-3, "--"), (3, "--")):
        ax.axvline(x, color=MUTED if x == 0 else GRID, lw=0.7 if x == 0 else 0.9, ls=ls, zorder=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlabel("Estimate, pp (H2a: AUROC difference × 100)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", handlelength=1.2, title="Model", title_fontsize=7, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_forest.pdf")
    plt.close(fig)


def streams():
    st = jload("p4_4B_streams")
    ks = sorted(st["eval"]["self"], key=int)
    x = [int(k) for k in ks]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ends = {}
    for j, c in enumerate(("context", "random", "ext", "self")):
        seeds = list(zip(*[st["eval"][c][k]["acc_greedy"] for k in ks]))
        for s in seeds:
            ax.plot(x, [100 * v for v in s], color=SLOT[j], lw=0.6, alpha=0.45)
        mean = [100 * sum(st["eval"][c][k]["acc_greedy"]) / len(st["eval"][c][k]["acc_greedy"]) for k in ks]
        ax.plot(x, mean, color=SLOT[j], lw=1.6, marker=MARK[j], ms=3.5, mec="white", mew=0.5, label=c)
        ends[c] = mean[-1]
    # direct labels at the line ends, pushed apart so that close end values do not collide
    placed = []
    for c, y in sorted(ends.items(), key=lambda kv: -kv[1]):
        y = min([y] + [p - 6 for p in placed if abs(p - y) < 6])
        placed.append(y)
        ax.annotate(c, (x[-1], ends[c]), xytext=(x[-1] + 6, y), textcoords="data", va="center", fontsize=7, color=INK)
    ax.set_xlabel("Stream step $k$ (evaluation checkpoint)")
    ax.set_ylabel("Greedy dev accuracy (%)")
    ax.set_xlim(0, x[-1] + 30)
    ax.set_ylim(-3, 100)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=2, fontsize=7, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_streams.pdf")
    plt.close(fig)


def streams_controls():
    """The two follow-up stream controls next to self and ext (analysis/out/fu_03_group2.json; DEVIATIONS D16)."""
    st = jload("fu_03_group2")["streams"]
    ks = sorted(st["eval"]["self"], key=int)
    x = [int(k) for k in ks]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ends = {}
    for j, c in ((3, "self"), (0, "matched"), (2, "ext"), (1, "nll")):
        acc = [st["eval"][c][k]["acc_greedy"] for k in ks]
        for s in zip(*acc):
            ax.plot(x, [100 * v for v in s], color=SLOT[j], lw=0.6, alpha=0.45)
        mean = [100 * sum(a) / len(a) for a in acc]
        ax.plot(x, mean, color=SLOT[j], lw=1.6, marker=MARK[j], ms=3.5, mec="white", mew=0.5, label=c)
        ends[c] = mean[-1]
    placed = []
    for c, y in sorted(ends.items(), key=lambda kv: -kv[1]):
        y = min([y] + [p - 6 for p in placed if abs(p - y) < 6])
        placed.append(y)
        ax.annotate(c, (x[-1], ends[c]), xytext=(x[-1] + 6, y), textcoords="data", va="center", fontsize=7, color=INK)
    ax.set_xlabel("Stream step $k$ (evaluation checkpoint)")
    ax.set_ylabel("Greedy dev accuracy (%)")
    ax.set_xlim(0, x[-1] + 34)
    ax.set_ylim(-3, 100)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Direct labels at the line ends name every series; a legend box would sit on the lines.
    fig.tight_layout()
    fig.savefig(FIG / "fig_streams_controls.pdf")
    plt.close(fig)


def recheck_cost():
    """Net accuracy change against incremental generated tokens per item for re-check rankings with F2
    (analysis/out/fu_05_recheck_cost.json; DEVIATIONS D18)."""
    o = jload("fu_05_recheck_cost")["policies"]["F2"]
    bs = sorted(o["budgets"], key=float)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    series = [("verbal s", SLOT[0], MARK[0], "verbal"), ("probe", SLOT[1], MARK[1], "probe"),
              ("agreement with a0", SLOT[2], MARK[2], "agreement (5 samples)"),
              ("dev-fitted net gain, cheap", SLOT[3], MARK[3], "dev-fitted net gain")]
    for key, col, mk, lab in series:
        xs = [o["budgets"][b][key]["incremental generated / item"] for b in bs]
        ys = [100 * o["budgets"][b][key]["acc change"][0] for b in bs]
        ax.plot(xs, ys, color=col, marker=mk, ms=3.5, lw=1.3, mec="white", mew=0.5, label=lab)
    xs = [o["budgets"][b]["random (expected)"]["incremental generated / item"] for b in bs]
    ys = [100 * o["budgets"][b]["random (expected)"]["acc change"][0] for b in bs]
    ax.plot(xs, ys, color=MUTED, ls="--", lw=1.0, label="random")
    e = o["every item"]
    ax.plot([e["generated / item"]], [100 * e["acc change"][0]], "*", color=INK, ms=8, label="F2 on every item")
    ax.set_xscale("log")
    ax.set_xlabel("Incremental generated tokens per item (log scale)")
    ax.set_ylabel("Accuracy change (pp)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_recheck_cost.pdf")
    plt.close(fig)


def _settings():
    """(label, a0 accuracy, {cond: (net CI, W->R, R->W)}, F2 policies, verbal key, probe key) for the three settings."""
    fu5, v2 = jload("fu_05_recheck_cost"), jload("v2_02_mc_results")
    out = [("Math, Qwen3-4B\n(seen test split)", 1 - fu5["n wrong test"] / fu5["n test"],
            {c: (v["acc change"], v["W->R"], v["R->W"]) for c, v in fu5["cost per condition"].items()},
            fu5["policies"], "verbal s", "probe")]
    for m, lab in (("4B", "MMLU-Pro, Qwen3-4B\n(frozen addendum)"), ("T8B", "MMLU-Pro, Tulu-3-8B\n(frozen addendum)")):
        o = v2["models"][m]
        out.append((lab, o["a0 accuracy"], {c: (v["net change"], v["W->R"], v["R->W"]) for c, v in o["conditions"].items()},
                    o["policies"], "verbal W0", "task probe"))
    return out


def repair_damage():
    """Repairs and damages as shares of all items, with the net change and its CI, per condition and setting
    (fu_05_recheck_cost.json, v2_02_mc_results.json; manuscript Figure, DEVIATIONS D19)."""
    conds = ["F0", "F1", "F2", "F2-scratch", "F3(1.0)"]
    S = _settings()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
    for ax, (lab, acc, d, *_rest) in zip(axes, S):
        x = list(range(len(conds)))
        rep = [100 * d[c][1] * (1 - acc) for c in conds]
        dam = [-100 * d[c][2] * acc for c in conds]
        ax.bar(x, rep, width=0.6, color=SLOT[0], label="repairs (W→R, % of items)")
        ax.bar(x, dam, width=0.6, color=SLOT[1], label="damages (R→W, % of items)")
        for i, c in enumerate(conds):
            e, lo, hi = (100 * v for v in d[c][0])
            ax.plot([i, i], [lo, hi], color=INK, lw=1.0)
            ax.plot(i, e, "o", color=INK, ms=3.5, mec="white", mew=0.5, label="net change, 95% CI" if i == 0 else None)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_xticks(x, conds, rotation=35, ha="right")
        ax.set_title(lab, fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Share of all items (pp)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig_repair_damage.pdf")
    plt.close(fig)


def signal_vs_policy():
    """AUROC difference against the verbal score (x) next to the re-check gain difference against the verbal order at
    the addendum's 10 % share (y), with the 95 % CIs stored in the analysis files (06_confirmatory_4B, fu_02_group1,
    fu_05_recheck_cost, v2_02_mc_results). The math agreement point has no stored AUROC-difference CI: its x is the
    difference of two stored point estimates, drawn hollow without a horizontal bar."""
    conf = jload("06_confirmatory_4B")["main"]["H2a"]
    fu2, fu5, v2 = jload("fu_02_group1")["H2a comparators"], jload("fu_05_recheck_cost"), jload("v2_02_mc_results")
    pts = []   # (label, x triple or point, y triple, colour, marker, hollow)
    pol = fu5["policies"]["F2"]["budgets"]["10"]
    pts.append(("math, probe", [conf["est"], conf["lo"], conf["hi"]], pol["probe"]["minus verbal s"], SLOT[0], MARK[0], False))
    ax_ = fu2["auroc"]["5-sample consistency"] - fu2["auroc"]["verbal s"]
    pts.append(("math, agreement", [ax_, None, None], pol["agreement with a0"]["minus verbal s"], SLOT[0], MARK[2], True))
    for m, col in (("4B", SLOT[1]), ("T8B", SLOT[2])):
        o = v2["models"][m]
        p10 = o["policies"]["F2"]["budgets"]["10"]
        pts.append((f"MMLU-Pro {m}, probe", o["task probe - verbal W0"], p10["task probe"]["minus verbal W0"], col, MARK[0], False))
        pts.append((f"MMLU-Pro {m}, agreement", o["agreement - verbal W0"], p10["agreement with a0"]["minus verbal W0"], col, MARK[2], False))
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for lab, xv, yv, col, mk, hollow in pts:
        x, y = 100 * xv[0], 100 * yv[0]
        if xv[1] is not None:
            ax.plot([100 * xv[1], 100 * xv[2]], [y, y], color=col, lw=1.0)
        ax.plot([x, x], [100 * yv[1], 100 * yv[2]], color=col, lw=1.0)
        ax.plot(x, y, mk, color=col, mfc="white" if hollow else col, ms=5, mec=col if hollow else "white", mew=0.8, label=lab)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("AUROC difference vs verbal score (× 100)")
    ax.set_ylabel("Gain difference vs verbal order,\nF2 on 10% of items (pp)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_signal_policy.pdf", bbox_inches="tight")
    plt.close(fig)


def cost_curves():
    """Accuracy change against incremental generated tokens per item for selective F2 re-checks, three settings, with
    KEEP (zero line), ALL (F2 on every item) and F0 on every item (fu_05_recheck_cost.json, v2_02_mc_results.json)."""
    S = _settings()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=False)
    for ax, (lab, acc, d, pols, vb, pr) in zip(axes, S):
        o = pols["F2"]
        bs = [b for b in sorted(o["budgets"], key=float) if b != "100"]
        series = [(vb, SLOT[0], MARK[0], "verbal"), (pr, SLOT[1], MARK[1], "probe"), ("agreement with a0", SLOT[2], MARK[2], "agreement"),
                  ("random (expected)", MUTED, None, "random")]
        for key, col, mk, name in series:
            xs = [o["budgets"][b][key]["incremental generated / item"] for b in bs]
            ys = [100 * o["budgets"][b][key]["acc change"][0] for b in bs]
            ax.plot(xs, ys, color=col, marker=mk, ms=3.2, lw=1.2 if mk else 1.0, ls="-" if mk else "--", mec="white", mew=0.4, label=name)
        e = o["every item"]
        ax.plot([e["generated / item"]], [100 * e["acc change"][0]], "*", color=INK, ms=8, label="ALL (F2 on every item)")
        f0 = pols["F0"]["every item"]
        ax.plot([f0["generated / item"]], [100 * f0["acc change"][0]], "P", color=SLOT[3], ms=5, mec="white", mew=0.4, label="F0 on every item")
        ax.axhline(0, color=INK, lw=0.8)
        lo, hi = ax.get_ylim()
        ax.axhspan(min(lo, -0.2), 0, color=SLOT[1], alpha=0.07, lw=0)
        ax.set_ylim(min(lo, -0.2), hi)
        ax.set_xscale("log")
        ax.set_title(lab, fontsize=7.5)
        ax.set_xlabel("Incremental tokens / item (log)")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Accuracy change vs KEEP (pp)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_cost_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    forest()
    streams()
    streams_controls()
    recheck_cost()
    repair_damage()
    signal_vs_policy()
    cost_curves()
    print("wrote fig_forest, fig_streams, fig_streams_controls, fig_recheck_cost, fig_repair_damage, fig_signal_policy, fig_cost_curves")

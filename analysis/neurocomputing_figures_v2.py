"""Vector figures from the existing study records; no inference or new statistics."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parents[1]
    default = here if here.name == "supplementary" else here / "paper/Neurocomputing_v2/supplementary"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=default)
    args = parser.parse_args()
    package = args.package.resolve()
    sys.path.insert(0, str(package / "analysis"))
    from neurocomputing_revision import legacy_module
    gen = legacy_module(package)
    G = gen.load_all()
    out = package / "out/figures"
    out.mkdir(parents=True, exist_ok=True)
    audit_dir = package / "analysis/out/neurocomputing_v2"
    audit_dir.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.titlesize": 9, "axes.labelsize": 8})
    outputs = {}
    def save(fig, name):
        target = out / name
        fig.savefig(target, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
        outputs[str(target.relative_to(package)).replace("\\", "/")] = sha(target)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 2.95))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=.015",
                                   linewidth=.8, edgecolor=color, facecolor=color+"12"))
        ax.text(x+w/2,y+h-.055,title,ha="center",va="top",fontweight="bold",fontsize=9,color=color)
        ax.text(x+w/2,y+h/2-.03,body,ha="center",va="center",fontsize=8,linespacing=1.45)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=10,color="#53616f",lw=1))
    box(.02,.60,.25,.32,"Shared initial answer", "$a_0$: fixed across policies\nCorrectness, output length\nInitial generation is charged", "#285780")
    box(.37,.60,.25,.32,"Signal and selection", "Verbal / probe / length\nAgreement / vote margin\nSamples are acquired once", "#285780")
    box(.72,.60,.26,.32,"Triggered action", "Review / critique / re-solve\nSelective thinking\nKeep unselected answers", "#2a7567")
    arrow((.28,.76),(.36,.76)); arrow((.63,.76),(.71,.76))
    box(.02,.10,.60,.32,"Standalone alternatives from the question", "Sampling and voting  |  Uniform thinking\nThese pipelines can bypass the initial answer", "#715283")
    box(.72,.05,.26,.39,"Paired evaluation", "Repair and damage\nCompletion and format effects\nOutput and prompt accounting\nAccuracy-cost trade-offs", "#896523")
    arrow((.85,.59),(.85,.45)); arrow((.63,.26),(.71,.26))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    save(fig,"fig_overview.pdf")

    parts = ["repairs_completion", "repairs_content", "damage_format_loss", "damage_content"]
    labels = ["Completion", "Content repair", "Format loss", "Content damage"]
    colors = ["#8fc4df", "#26779c", "#edbd76", "#c26b52"]
    settings = ["math 4B", "math 8B", "MMLU-Pro 4B", "MMLU-Pro T8B"]
    records = []
    fig, axes = plt.subplots(2,2,figsize=(7.1,4.35))
    for ax, setting in zip(axes.flat,settings):
        P = gen.policies(G, setting)
        if setting in G["c1_think"]["settings"]:
            P = {**P, **G["c1_think"]["settings"][setting]["policies"]}
        names = ["F0 ALL", "F2 ALL", "F2-scratch ALL", "V5"]
        if "THINK ALL" in P: names.append("THINK ALL")
        x = np.arange(len(names)); positive = np.zeros(len(names)); negative = positive.copy()
        for j, (part,label,color) in enumerate(zip(parts,labels,colors)):
            vals = np.array([P[name][part]["estimate"] for name in names])
            if j<2:
                ax.bar(x,vals,bottom=positive,color=color,label=label,width=.6); positive += vals
            else:
                ax.bar(x,-vals,bottom=negative,color=color,label=label,width=.6); negative -= vals
        net = np.array([P[n]["delta_vs_keep"]["estimate"] for n in names])
        assert np.allclose(positive+negative, net, atol=1e-8)
        lo = np.array([P[n]["delta_vs_keep"]["lo"] for n in names])
        hi = np.array([P[n]["delta_vs_keep"]["hi"] for n in names])
        ax.errorbar(x,net,yerr=[net-lo,hi-net],fmt="D",color="#222222",ms=3,elinewidth=.8,capsize=2,label="Net gain (95% CI)")
        ax.axhline(0,color="#555555",lw=.7); ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
        ax.set_xticks(x,["Review","Critique","Re-solve","Vote-5","Think"][:len(names)],fontsize=7)
        ax.set_title(gen.setting_name(setting),loc="left"); ax.set_ylabel("Percentage points")
        records += [{"setting":setting,"policy":n,"net":P[n]["delta_vs_keep"],
                     **{p:P[n][p] for p in parts}} for n in names]
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",ncol=3,frameon=False,fontsize=7.5)
    fig.tight_layout(rect=(0,.12,1,1),h_pad=1.5)
    save(fig,"fig_decomposition.pdf")

    total = G["c1bt"] if "c1bt" in G else json.loads((package/"analysis/out/arr/c1b_total.json").read_text())
    fig, axes = plt.subplots(1,3,figsize=(7.1,2.65),sharey=True)
    for ax, setting in zip(axes,["math 4B","math 8B","MMLU-Pro 4B"]):
        B = total["settings"][setting]["budgets"]
        budgets = sorted(B,key=int)
        for j,(signal,color) in enumerate([("verbal","#2574a0"),("probe","#bc692c")]):
            rows = [B[b]["equal_total_selective"][signal]["selective_minus_uniform"] for b in budgets]
            y=np.array([r["estimate"] for r in rows])
            ax.errorbar(np.arange(len(budgets))+(j-.5)*.10,y,
                        yerr=[y-np.array([r["lo"] for r in rows]),np.array([r["hi"] for r in rows])-y],
                        fmt="o-",color=color,ms=3,lw=.9,capsize=2,label=signal.capitalize())
        ax.axhline(0,color="#555555",lw=.7)
        ax.set_xticks(np.arange(len(budgets)),budgets); ax.set_xlabel("Uniform thinking cap")
        ax.set_title(gen.setting_name(setting),loc="left"); ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Selective − uniform (pp)")
    # One figure-level legend below the panels: an axes legend at a negative offset landed on the middle panel's
    # x-axis label.
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.14,1,1))
    save(fig,"fig_budget_gap.pdf")
    (audit_dir/"figure_data.json").write_text(json.dumps({"decomposition":records,"budget":total},indent=1)+"\n",encoding="utf8")
    manifest={"analysis":"analysis/neurocomputing_figures_v2.py", "new_model_runs":False,
              "script_sha256":sha(Path(__file__)), "hash_scope":"raw file bytes",
              "inputs":{rel:sha(package/rel) for rel in gen.INPUTS}, "outputs":outputs,
              "checks":["All plotted repair-minus-damage totals equal stored net estimates.","Intervals copied from existing estimates; no new hypothesis tests."]}
    (audit_dir/"figures_manifest.json").write_text(json.dumps(manifest,indent=1)+"\n",encoding="utf8")
    print(json.dumps({"figures":list(outputs),"checks":manifest["checks"]}))


if __name__ == "__main__":
    main()

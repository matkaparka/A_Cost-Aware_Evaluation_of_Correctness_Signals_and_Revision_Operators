"""Existing-record cost audit for Neurocomputing v2; CPU only, no model execution.

One-time extraction (local pinned tokenizer, no network):
  python -B analysis/neurocomputing_cost_v2.py --prepare-from E:/research
Portable reanalysis from supplementary/: 
  python -B analysis/neurocomputing_cost_v2.py --ms out
P is logical input tokens excluding batch padding, D is generated tokens. D + alpha*P is a
sensitivity index, not measured latency, FLOPs, energy, or a monetary price.
"""
from __future__ import annotations
import argparse, ast, gzip, hashlib, importlib.util, json, os, shutil
from pathlib import Path
import numpy as np

SETTINGS = ("math 4B", "math 8B", "MMLU-Pro 4B")
SLUG = {"math 4B":"m4", "math 8B":"m8", "MMLU-Pro 4B":"c4"}
BUDGETS = (256,512,1024,2048)
ALPHAS = (0.,.01,.05,.1,.25,.5,1.)
VERSION = "neurocomputing-v2-existing-cost-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1,allow_nan=False)+"\n",encoding="utf8",newline="\n")

def rows(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def gzwrite(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,filename="") as stream:
        for rec in records:
            stream.write((json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n").encode())

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def prepare(root, package):
    """Extract measured token records and retokenize recorded prompts; no weights."""
    evidence=package/"analysis/out/cost_v2/evidence"
    evidence.mkdir(parents=True,exist_ok=True)
    inputs={}
    def read(rel):
        path=root/rel;inputs[rel]=sha(path);return path
    def jl(rel): return rows(read(rel))
    def js(rel): return json.loads(read(rel).read_text(encoding="utf8"))
    render=load_module(read("src/data/mmlupro.py"),"cost_mmlupro")
    # Literal extraction avoids importing generation modules and torch.
    def literal(rel,name):
        tree=ast.parse(read(rel).read_text(encoding="utf8"))
        return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    prompt=literal("src/gen/render.py","PROMPT")
    query=literal("src/score/verbal.py","QUERY")
    revisions={"4B":"1cfa9a7208912126459214e8b04321603b3df60c","8B":"b968826d9c46dd6066d109eabc6255188de91218"}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B",revision=revisions["4B"],local_files_only=True)
    snapshot=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots"/revisions["4B"]
    tokenizer_files={n:sha(snapshot/n) for n in ("tokenizer.json","tokenizer_config.json")}
    other=snapshot.parents[2]/"models--Qwen--Qwen3-8B/snapshots"/revisions["8B"]
    # Both Qwen settings use the identical tokenizer and chat template.
    assert sha(other/"tokenizer.json")==tokenizer_files["tokenizer.json"]
    assert json.loads((other/"tokenizer_config.json").read_text(encoding="utf8"))["chat_template"]==tok.chat_template
    candidate_ids=[tok(c,add_special_tokens=False).input_ids for c in literal("src/score/verbal.py","YES")+literal("src/score/verbal.py","NO")]
    assert all(len(c)==1 for c in candidate_ids) and len({c[0] for c in candidate_ids})==len(candidate_ids)
    questions={r["item_id"]:r["question"] for rel in ("data/processed/dnat.jsonl","data/processed/ext/dnat_ext.jsonl") for r in jl(rel)}
    mc={r["item_id"]:r for r in jl("data/processed/mmlupro/items.jsonl")}
    outputs=[];setup={};prefix_examples={}
    for setting in SETTINGS:
        model=setting.split()[1]
        if setting.startswith("math"):
            files=[f"results/p1/{model}/p1-initial-{model}-d2ebdea/initial.jsonl",f"results/p1/{model}/p1-initial-ext-{model}-45575c2/initial.jsonl"]
            all_initial=[r for f in files for r in jl(f)]
        else:
            all_initial=jl("results/v2/fu/4B/fu-mc-initial-test-4B-9221b24/initial.jsonl")+jl("results/v2/fu-dev/4B/fu-dev-mc-initial-dev-4B-ef3e485/initial.jsonl")
        initial=[r for r in all_initial if r["split"]=="test"]
        dev=[r for r in all_initial if r["split"]=="dev"]
        assert dev,setting
        setup[setting]={"dev_items":len(dev),"activation_input_tokens":sum(r["prompt_tokens"]+len(r["a0_token_ids"]) for r in dev),
                        "scope":"one-off development activation collection; CPU probe fitting is not timed separately"}
        for r in sorted(initial,key=lambda r:r["item_id"]):
            q=questions[r["item_id"]] if setting.startswith("math") else render.render_question(mc[r["item_id"]])
            user={"role":"user","content":prompt.format(q=q)}
            base=tok.apply_chat_template([user],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            baseids=tok(base,add_special_tokens=False).input_ids
            assert len(baseids)==r["prompt_tokens"],(setting,r["item_id"],"initial prompt tokens")
            verdict=tok.apply_chat_template([user,{"role":"assistant","content":r["a0"]},{"role":"user","content":query}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            verdictids=tok(verdict,add_special_tokens=False).input_ids
            previous=baseids+r["a0_token_ids"]
            common=next((i for i,(a,b) in enumerate(zip(previous,verdictids)) if a!=b),min(len(previous),len(verdictids)))
            if setting not in prefix_examples:
                prefix_examples[setting]={"item_id":r["item_id"],"common_prefix_tokens":common,"initial_prompt_tokens":len(baseids),
                                          "original_at_first_mismatch_ids":previous[common:common+12],"verbal_at_first_mismatch_ids":verdictids[common:common+12],
                                          "original_at_first_mismatch_text":tok.decode(previous[common:common+12]),"verbal_at_first_mismatch_text":tok.decode(verdictids[common:common+12])}
            outputs.append({"setting":setting,"item_id":r["item_id"],"a0_D":len(r["a0_token_ids"]),"a0_P":len(baseids),
                            "verbal_P_uncached":len(verdictids),"verbal_P_prefix_cache":len(verdictids)-common,
                            "cached_prefix_tokens":common,"probe_P_separate":len(previous),"probe_P_reused":0})
        print(f"Extracted prompt costs: {setting} ({len(initial)} items)",flush=True)
    gzwrite(evidence/"signal_costs.jsonl.gz",outputs)
    dump(evidence/"probe_setup.json",setup)
    budget_records=[]
    for setting in SETTINGS:
        model=setting.split()[1];task="math" if setting.startswith("math") else "mc"
        for budget in BUDGETS:
            rel=f"results/arr/c1b/{model}/c1b-think-budget-{task}-test-b{budget}-{model}-a596548/generations.jsonl"
            for r in jl(rel):
                budget_records.append({"setting":setting,"budget":budget,"item_id":r["item_id"],"reused":r["reused"],
                                       "trace_tokens_used":r.get("trace_tokens_used",0),"generated_tokens":r["generated_tokens"],"prompt_tokens":r["prompt_tokens"]})
    gzwrite(evidence/"uniform_budget_costs.jsonl.gz",budget_records)
    # Exact metrics/config/manifests are small and independently inspectable.
    dirs=[]
    for model in ("1.7B","4B","8B"):
        dirs += [f"results/p0/{model}/p0-hf-{model}-66fb8df"]
        dirs += [f"results/p1/{model}/p1-verbal-{model}-d2ebdea",f"results/p1/{model}/p1-verbal-ext-{model}-45575c2"]
        commit="a7a0579" if model=="4B" else "3fd21bc"
        dirs += [f"results/main/{model}/main-acts-test-{model}-{commit}",f"results/main/{model}/main-acts-test-ext-{model}-{commit}"]
    dirs += ["results/p0/4B/p0-vllm-4B-5c43f13","results/p0/8B/p0-vllm-8B-66fb8df"]
    dirs += [f"results/fu2/p1/T8B/p1-verbal{suffix}-T8B-1809070" for suffix in ("","-ext")]
    dirs += [f"results/fu2/main/T8B/main-acts-test{suffix}-T8B-1809070" for suffix in ("","-ext")]
    dirs += [f"results/v2/fu/{m}/fu-mc-hf-{m}-9221b24" for m in ("4B","T8B")]
    dirs += [str(p.relative_to(root)).replace("\\","/") for p in (root/"results/arr/c1").glob("*/*") if p.is_dir()]
    dirs += [str(p.relative_to(root)).replace("\\","/") for p in (root/"results/arr/c1b").glob("*/*") if p.is_dir()]
    copied=[]
    for rel in sorted(set(dirs)):
        if not (root/rel/"metrics.json").exists():continue
        for name in ("metrics.json","config.json","manifest.json"):
            if (root/rel/name).exists():
                target=evidence/"records"/rel/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(read(rel+"/"+name),target)
        copied.append(rel)
    for rel in ("analysis/out/fu_05_recheck_cost.json","cloud/ledger.jsonl","reports/00_compute.md","reports/00_compute_rates.md","src/probe/cache.py","src/score/verbal.py","src/gen/render.py","src/exp/arr_runs.py","src/exp/mc_run.py"):
        target=evidence/"records"/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(read(rel),target)
    manifest={"id":VERSION,"kind":"CPU extraction from existing records; no new model outputs","tokenizer":{"repo":"Qwen/Qwen3-4B","revision":revisions["4B"],"files_sha256":tokenizer_files},
              "source_files":inputs,"timing_run_dirs":copied,"verbal_first_token_shortcut_verified":True,"prefix_examples":prefix_examples,
              "outputs":{p.name:sha(p) for p in (evidence/"signal_costs.jsonl.gz",evidence/"uniform_budget_costs.jsonl.gz",evidence/"probe_setup.json")}}
    dump(evidence/"extraction_manifest.json",manifest)


def analyze(package, ms):
    evidence=package/"analysis/out/cost_v2/evidence";out=package/"analysis/out/cost_v2"
    inputs={};checks=[]
    def read(rel):
        p=package/rel;inputs[rel]=sha(p);return p
    def jl(rel):return rows(read(rel))
    def js(rel):return json.loads(read(rel).read_text(encoding="utf8"))
    ev="analysis/out/cost_v2/evidence/"
    em=js(ev+"extraction_manifest.json")
    for name,h in em["outputs"].items():assert sha(evidence/name)==h
    for rel,h in em["source_files"].items():
        if (evidence/"records"/rel).exists():assert sha(read(ev+"records/"+rel))==h
    signals=jl(ev+"signal_costs.jsonl.gz");budgetraw=jl(ev+"uniform_budget_costs.jsonl.gz")
    setup=js(ev+"probe_setup.json")
    c1=jl("analysis/out/arr/c1_labels.jsonl.gz");forced=jl("analysis/out/arr/c1b_labels.jsonl.gz")
    reference=js("analysis/out/arr/c1b_total.json")
    a1=js("analysis/out/arr/a1_policies.json")
    b1=js("analysis/out/arr/b1_policies.json")
    mathlabels=jl("analysis/out/arr/math_common_labels.jsonl.gz")
    mathmetrics=jl("paper/1区/artifacts/math/item_evaluations.jsonl")
    fu=js(ev+"records/analysis/out/fu_05_recheck_cost.json")
    budgets={};populations={};accounting=[];sensitivity=[];headroom=[]
    for setting in SETTINGS:
        S={r["item_id"]:r for r in signals if r["setting"]==setting};ids=sorted(S);n=len(ids)
        T={r["item_id"]:r for r in c1 if r["source"]=="C1" and r["setting"]==setting}
        assert set(T)==set(ids)
        model=setting.split()[1]
        if setting.startswith("math"):
            L={r["item_id"]:r for r in mathlabels if r["model"]==model}
            F={r["item_id"]:r for r in mathmetrics if r["model"]==model and r["condition"]=="F2"}
            y0=np.array([L[i]["initial"]["correct"] for i in ids]);scores={"verbal":[F[i]["verbal_log_odds"] for i in ids],"probe":[F[i]["probe_p_correct"] for i in ids]}
        else:
            F={r["item_id"]:r for r in jl("paper/1区/artifacts/mc/4B_items.jsonl") if r["condition"]=="F2"}
            y0=np.array([F[i]["y0"] for i in ids]);scores={"verbal":[F[i]["verbal_W0"] for i in ids],"probe":[F[i]["task_probe"] for i in ids]}
        arrays={k:np.array([S[i][k] for i in ids],float) for k in ("a0_D","a0_P","verbal_P_uncached","verbal_P_prefix_cache","probe_P_separate")}
        td=np.array([T[i]["out_tokens"] for i in ids],float);tp=np.array([T[i]["prompt_tokens"] for i in ids],float);dy=np.array([T[i]["correct"] for i in ids])-y0
        ad,ap=arrays["a0_D"].mean(),arrays["a0_P"].mean()
        assert abs(ad-reference["settings"][setting]["a0_tokens_per_item"])<1e-8
        if setting=="math 4B":
            assert abs(arrays["verbal_P_uncached"].mean()-fu["signal cost"]["verbal s"]["processed (no cache)"])<1e-8
            assert abs(arrays["probe_P_separate"].mean()-fu["signal cost"]["probe"]["processed (separate prefill)"])<1e-8
            assert setup[setting]["activation_input_tokens"]==fu["signal cost"]["probe training (one-off)"]["prefill tokens"]
        tie=[int(hashlib.sha256(i.encode()).hexdigest()[:12],16) for i in ids]
        curves={}
        for sig,score in scores.items():
            order=np.lexsort((tie,score));cd=np.r_[0,np.cumsum(td[order])]/n;cp=np.r_[0,np.cumsum(tp[order])]/n;cy=np.r_[0,np.cumsum(dy[order])]/n*100
            curves[sig]=[{"q":q,"D":ad+cd[round(n*q/100)],"P_base":ap+cp[round(n*q/100)],"delta_pp":cy[round(n*q/100)]} for q in range(101)]
        nc={"verbal":float(arrays["verbal_P_uncached"].mean()),"probe":float(arrays["probe_P_separate"].mean())}
        retained={"verbal":float(arrays["verbal_P_prefix_cache"].mean()),"probe":0.}
        populations[setting]={"n":n,"a0_D":ad,"a0_P":ap,"signal_prefill_uncached":nc,"signal_prefill_retained":retained,
                              "think_all_D":float(td.mean()),"think_all_P":float(tp.mean()),
                              "verbal_common_prefix_tokens":float(np.mean([S[i]["cached_prefix_tokens"] for i in ids]))}
        def append_policy(policy,D,P,delta,scope):
            accounting.append({"setting":setting,"policy":policy,"D":float(D),"P":float(P),"delta_pp":float(delta),"scope":scope})
        append_policy("KEEP",ad,ap,0,"initial-answer execution")
        pp={**a1["settings"][setting]["policies"],**b1.get("a1",{}).get(setting,{}).get("policies",{})}
        for policy in ("F2 ALL","V5"):
            rec=pp[policy];needs=policy=="F2 ALL"
            append_policy(policy,rec["generated_tokens_per_item"]+(ad if needs else 0),rec["prompt_tokens_per_item"]+(ap if needs else 0),rec["delta_vs_keep"]["estimate"],"all prompt turns; no cross-request prefix reuse")
        append_policy("THINK ALL",td.mean(),tp.mean(),dy.mean()*100,"standalone thinking; no initial answer")
        for sig in scores:
            pt=curves[sig][20]
            for cache,extra in (("uncached",nc[sig]),("retained",retained[sig])):
                append_policy(f"{sig} THINK 20% ({cache})",pt["D"],pt["P_base"]+extra,pt["delta_pp"],"retained signal cache is an assumed execution; no observed online latency")
        budgets[setting]={}
        for budget in BUDGETS:
            R={r["item_id"]:r for r in budgetraw if r["setting"]==setting and r["budget"]==budget}
            B={r["item_id"]:r for r in forced if r["setting"]==setting and r["budget"]==budget}
            assert set(R)==set(ids)
            yd=[];ud=[];up=[];replay=[];stop=[]
            for i in ids:
                r=R[i];ud.append(r["generated_tokens"])
                if r["reused"]:
                    assert i not in B and r["generated_tokens"]==T[i]["out_tokens"]
                    yd.append(T[i]["correct"]);up.append(T[i]["prompt_tokens"]);replay.append(T[i]["prompt_tokens"])
                else:
                    assert r["generated_tokens"]==B[i]["out_tokens"]
                    extra=r["prompt_tokens"]-r["trace_tokens_used"]-T[i]["prompt_tokens"]
                    assert extra>0
                    stop.append(extra);yd.append(B[i]["correct"])
                    # The generated trace is retained in KV: only the external stop instruction is new input.
                    up.append(T[i]["prompt_tokens"]+extra)
                    # If the trace cache is discarded, its generation prompt and a fresh answer prefill are both needed.
                    replay.append(T[i]["prompt_tokens"]+r["prompt_tokens"])
            uniform={"D":float(np.mean(ud)),"P_retained_trace":float(np.mean(up)),"P_replayed_trace":float(np.mean(replay)),
                     "delta_pp":float(100*np.mean(np.array(yd)-y0)),"forced_items":len(stop),"stop_instruction_tokens":sorted(set(stop))}
            assert len(uniform["stop_instruction_tokens"])==1
            ref=reference["settings"][setting]["budgets"][str(budget)]
            assert abs(uniform["D"]-ref["uniform"]["total_tokens_per_item"])<1e-8
            assert abs(uniform["delta_pp"]-ref["uniform"]["delta_vs_keep"]["estimate"])<1e-8
            budgets[setting][str(budget)]=uniform
            append_policy(f"uniform B={budget}",uniform["D"],uniform["P_retained_trace"],uniform["delta_pp"],"trace KV retained; original question and inserted stop instruction counted once")
            for sig in scores:
                for cache,extra in (("uncached",nc[sig]),("retained",retained[sig])):
                    for alpha in ALPHAS:
                        cap=uniform["D"]+alpha*uniform["P_retained_trace"]
                        pts=[{**pt,"P":pt["P_base"]+(extra if pt["q"] else 0)} for pt in curves[sig]]
                        fit=[pt for pt in pts if pt["D"]+alpha*pt["P"]<=cap+1e-8]
                        if not fit:
                            selected=None;gap=None
                        else:
                            selected=max(fit,key=lambda p:p["q"]);gap=selected["delta_pp"]-uniform["delta_pp"]
                        if alpha==0:
                            old=ref["equal_total_selective"][sig]
                            assert selected["q"]==old["share_pct"]
                            assert abs(gap-old["selective_minus_uniform"]["estimate"])<1e-8
                        sensitivity.append({"setting":setting,"budget":budget,"signal":sig,"signal_cache":cache,"alpha":alpha,
                                            "uniform_cost":cap,"selected":selected,"gap_pp":gap,
                                            "execution_mode":"infeasible" if selected is None else ("KEEP_without_signal" if selected["q"]==0 else "scored_gate"),
                                            "signal_acquired":selected is not None and selected["q"]>0,"gate_feasible":any(pt["q"]>0 for pt in fit),
                                            "a0_plus_signal_fits":bool(ad+alpha*(ap+extra)<=cap+1e-8)})
                    # Headroom of the previously published D-matched point, before any additional unpriced overhead h.
                    oldq=ref["equal_total_selective"][sig]["share_pct"];pt=curves[sig][oldq]
                    dp=pt["P_base"]+(extra if oldq else 0)-uniform["P_retained_trace"]
                    slack=uniform["D"]-pt["D"]
                    headroom.append({"setting":setting,"budget":budget,"signal":sig,"signal_cache":cache,"original_q":oldq,
                                     "D_slack":slack,"additional_P":dp,"alpha_break_even":slack/dp if dp>0 else None,
                                     "h_headroom_at_alpha_0.1":slack-.1*dp})
    checks+= ["Initial and signal prefill counts reproduce the earlier independent Qwen3-4B cost audit.",
              "All twelve uniform generated-token totals and gains reproduce c1b_total.json.",
              "All alpha=0 selections and gain differences reproduce the earlier generated-token comparison for both signals.",
              "Uniform thinking input accounting separates the generated trace from new prompt tokens, preventing double charging under retained KV."]
    timing=[]
    for rel in em["timing_run_dirs"]:
        m=js(ev+"records/"+rel+"/metrics.json");cfg=js(ev+"records/"+rel+"/config.json") if (evidence/"records"/rel/"config.json").exists() else {}
        man=js(ev+"records/"+rel+"/manifest.json") if (evidence/"records"/rel/"manifest.json").exists() else {}
        entry={"run":rel,"metrics":m,"config":cfg,"gpu":man.get("gpu"),"backend":man.get("backend"),"batch_invariant":man.get("env",{}).get("VLLM_BATCH_INVARIANT")}
        if "/p0/" in rel:entry["scope"]="train fixtures; teacher-forced/HF activations and full generation blocks, not separate pure prefill/decode rates"
        elif "mc-hf" in rel:entry["scope"]="joint verbal readout and all-layer activation extraction over dev+test; analysis_s also includes CPU probe selection"
        elif "verbal" in rel:entry["scope"]="standalone batched HF verbal-scoring pass over dev+test; load and online orchestration excluded"
        elif "acts-test" in rel:entry["scope"]="serial all-layer activation extraction on test; offline feature collection, not in-generation probe overhead"
        elif "/c1b/" in rel:entry["scope"]="forced-answer run reusing stored thinking traces; gen_s does not include generating those traces"
        else:entry["scope"]="historical full-pool thinking chunk; batched generation, not request latency or a selected-policy replay"
        timing.append(entry)
    result={"id":VERSION,"kind":"descriptive CPU-only reanalysis of existing recorded outputs","alphas":ALPHAS,
            "cost_definition":"D + alpha*P; P is logical input-token volume excluding batch padding. Alpha is a hypothetical input/decode relative weight, not a calibrated timing ratio",
            "cache_assumptions":{"uncached":"verbal query or separate probe activation extraction reprocesses the complete recorded context",
                                 "retained":"verbal query reuses the exact common token-prefix cache; probe P=0 is an optimistic lower bound conditional on target hidden-state availability, not a measured execution",
                                 "uniform":"thinking trace KV retained in all primary weighted-cost comparisons; reprocessing workload separately recorded"},
            "missing_costs":["per-policy end-to-end latency","KV retention/eviction overhead","hidden-state instrumentation and logistic readout overhead","final-token evaluation for capped outputs and backend target-state availability","batch padding and sequence-length-dependent attention work","scheduler/batching effects","energy"],
            "populations":populations,"probe_setup":setup,"prefix_examples":em["prefix_examples"],"uniform_budgets":budgets,"accounting":accounting,"sensitivity":sensitivity,"break_even":headroom,"historical_timing":timing,"checks":checks}
    out.mkdir(parents=True,exist_ok=True);dump(out/"cost_v2.json",result)
    files=render_outputs(result,ms)
    report=out/"cost_report.md";report.write_text(render_report(result),encoding="utf8",newline="\n");files.append(report)
    files.append(out/"cost_v2.json")
    inputs["analysis/neurocomputing_cost_v2.py"]=sha(package/"analysis/neurocomputing_cost_v2.py")
    manifest={"id":VERSION,"new_model_runs":False,"new_hypothesis_tests":False,"inputs":inputs,
              "outputs":{os.path.relpath(p,package).replace("\\","/"):sha(p) for p in files},"checks":checks}
    dump(out/"cost_v2_manifest.json",manifest)
    print(json.dumps({"id":VERSION,"checks":checks,"outputs":len(files)},indent=1))
    return result


def render_outputs(result,ms):
    gen=ms/"generated";fig=ms/"figures";gen.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
    paths=[];mac={}
    def number(x,d=0):return f"{x:,.{d}f}"
    def signed(x):return f"{x:+.2f}".replace("-",r"$-$")
    def table(name,cols,head,body,caption,label):
        path=gen/name
        path.write_text("\n".join([r"\begin{table*}[!htbp]",r"\centering\footnotesize",r"\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{"+cols+"}",r"\toprule",head+r" \\",r"\midrule"]+body+[r"\bottomrule",r"\end{tabular}",r"\caption{"+caption+"}",r"\label{"+label+"}",r"\end{table*}",""]),encoding="utf8",newline="\n");paths.append(path)
    body=[]
    labels={"math 4B":"Math 4B","math 8B":"Math 8B","MMLU-Pro 4B":"MMLU-Pro 4B"}
    keep={"KEEP","F2 ALL","V5","THINK ALL","verbal THINK 20% (uncached)","verbal THINK 20% (retained)","probe THINK 20% (uncached)","probe THINK 20% (retained)","uniform B=1024","uniform B=2048"}
    display={"F2 ALL":"Critique all","V5":"Standalone $V_5$","THINK ALL":"Standalone think","verbal THINK 20% (uncached)":r"Verbal 20\%, separate prefill","verbal THINK 20% (retained)":r"Verbal 20\%, exact prefix reuse","probe THINK 20% (uncached)":r"Probe 20\%, separate prefill","probe THINK 20% (retained)":r"Probe 20\%, optimistic state reuse"}
    # Display names only: the record keys ("uniform B=1024") stay as they are in the JSON and the report.
    display.update({f"uniform B={b}":f"Uniform thinking, $B$ = {b:,}" for b in BUDGETS})
    for setting in SETTINGS:
        for r in result["accounting"]:
            if r["setting"]==setting and r["policy"] in keep:
                body.append(" & ".join([labels[setting],display.get(r["policy"],r["policy"]),number(r["D"]),number(r["P"]),signed(r["delta_pp"])])+r" \\")
        if setting!=SETTINGS[-1]:body.append(r"\midrule")
        for name,value in result["populations"][setting].items():
            if isinstance(value,(int,float)):mac[f"costv2.{SLUG[setting]}.{name}"]=number(value)
        for cache in ("uncached","retained"):
            for sig,value in result["populations"][setting]["signal_prefill_"+cache].items():mac[f"costv2.{SLUG[setting]}.{sig}.P.{cache}"]=number(value)
        mac[f"costv2.{SLUG[setting]}.probe.dev_n"]=number(result["probe_setup"][setting]["dev_items"])
        mac[f"costv2.{SLUG[setting]}.probe.dev_P"]=number(result["probe_setup"][setting]["activation_input_tokens"])
    stops={x for budgets in result["uniform_budgets"].values() for r in budgets.values() for x in r["stop_instruction_tokens"]}
    assert len(stops)==1;mac["costv2.stop_P"]=number(next(iter(stops)))
    table("tab_cost_accounting_full.tex","llrrr","Setting & Policy & Generated $D$ & Input $P$ & $\\Delta$ (pp)",body,
          r"Full-pipeline token accounting from existing records (per item): $P$ counts logical input tokens, excluding batch padding. Initial-answer-dependent policies include $a_0$; standalone voting/thinking omit it. Separate prefill counts complete signal contexts. Verbal reuse is limited to the exact common token prefix; chat re-rendering breaks reuse at the empty thinking tag. Probe state reuse is an optimistic lower bound conditional on target-state availability, excluding instrumentation and possible final-token evaluation. Uniform budgets retain trace KV and count the question plus inserted stop instruction as new input. Other cross-request reuse and one-off probe fitting are excluded. $D+\alpha P$ is a sensitivity index, not latency.","tab:cost-accounting-full")
    body=[]
    for r in result["accounting"]:
        if r["setting"]=="math 4B" and r["policy"] in keep and "(retained)" not in r["policy"]:
            body.append(" & ".join([display.get(r["policy"],r["policy"]),number(r["D"]),number(r["P"]),signed(r["delta_pp"])])+r" \\")
    table("tab_cost_accounting.tex","lrrr","Policy & Generated $D$ & Input $P$ & $\\Delta$ (pp)",body,
          r"Representative full-pipeline token volumes for Qwen3-4B mathematics (per item). $P$ is logical input length, excluding batch padding. Critique and gated thinking include the initial answer; standalone voting/thinking do not. Signals use separate prefill. Uniform budgets retain their generated trace KV and count the original question plus the inserted stop instruction as new input. No other cross-request cache reuse is credited. These volumes support cost sensitivity, not measured end-to-end latency. All settings and alternative signal-cache assumptions are in Table~\ref{tab:cost-accounting-full}.","tab:cost-accounting")
    idx={(r["setting"],r["budget"],r["signal"],r["signal_cache"],r["alpha"]):r for r in result["sensitivity"]}
    body=[]
    for setting in SETTINGS:
        for b in (1024,2048):
            cells=[labels[setting],f"{b:,}"]
            for sig,alpha in (("verbal",0.),("verbal",.1),("verbal",1.),("probe",.1),("probe",1.)):
                r=idx[setting,b,sig,"uncached",alpha];p=r["selected"]
                cells.append("--" if p is None else f"{p['q']}\\% / {signed(r['gap_pp'])}")
            body.append(" & ".join(cells)+r" \\")
    table("tab_cost_sensitivity.tex","lrrrrrr",r"Setting & $B$ & Verbal, $\alpha=0$ & Verbal, $.1$ & Verbal, $1$ & Probe, $.1$ & Probe, $1$",body,
          r"Largest affordable share / accuracy difference from uniform thinking (pp), under the uniform policy's $D+\alpha P$ cap with separate signal prefill. Shares use the original ranking and a one-percentage-point grid. A 0\% entry is KEEP chosen before signal acquisition, with no signal cost; it is not a scored gate. A dash means even KEEP exceeds the indexed budget. All four budgets, seven weights, both cache assumptions and explicit feasibility flags are in the supplementary JSON and Fig.~\ref{fig:cost-sensitivity}. These post hoc estimates add no hypothesis test or deployment validation.","tab:cost-sensitivity")
    body=[]
    for model in ("1.7B","4B","8B","T8B"):
        for kind,caption in (("verbal","Verbal HF, batch 16"),("acts-test","All-layer states, serial")):
            rs=[r for r in result["historical_timing"] if kind in r["run"] and f"/{model}/" in r["run"] and "/p0/" not in r["run"] and "wall_s" in r["metrics"]]
            if rs:
                assert all(r["gpu"]=="NVIDIA A100-SXM4-80GB" and r["backend"]=={"name":"hf-transformers","version":"5.20.2"} for r in rs)
                if kind=="verbal":assert all(r["config"]["batch"]==16 and r["metrics"]["first_token_shortcut"] for r in rs)
                n=sum(r["metrics"]["n"] for r in rs);wall=sum(r["metrics"]["wall_s"] for r in rs)
                body.append(" & ".join([model,caption,number(n),number(wall,2),number(wall/n,3),"dev + test" if kind=="verbal" else "test"])+r" \\")
                mac[f"costv2.timing.{model}.{kind}.n"]=number(n);mac[f"costv2.timing.{model}.{kind}.seconds"]=number(wall,2)
    table("tab_cost_timing.tex","llrrrl",r"Model & Measured operation & Items & Batch wall s & s/item & Population",body,
          r"Historical measured GPU-pass times on the mathematics pools (A100-SXM4-80GB; HF Transformers 5.20.2), aggregated from original metrics/config/manifests. Verbal scoring used batched forward passes; state extraction collected every layer serially. Item-normalised times are throughput summaries, not individual-request latency or integrated-gate overhead. Loading is excluded. Per-run provenance, P0 fixtures, combined MMLU-Pro passes and ARR chunk timings remain in the cost JSON; heterogeneous backend timings are not added together.","tab:cost-timing")
    body=[]
    hf=next(r for r in result["historical_timing"] if "p0-hf-4B-" in r["run"])["metrics"]
    vl=next(r for r in result["historical_timing"] if "p0-vllm-4B-" in r["run"])["metrics"]
    for label,rec,processed,generated in [("HF teacher-forced",hf["teacher_forced"],hf["teacher_forced"]["tokens"],None),("HF all-layer states",hf["activations"],None,None)]+[("vLLM "+label,vl[key],vl[key]["prompt_tokens"],vl[key]["output_tokens"]) for label,key in (("initial answer","round1"),("feedback revision","round2"),("thinking","thinking"))]:
        body.append(" & ".join([label,number(rec["n"]),"--" if processed is None else number(processed),"--" if generated is None else number(generated),number(rec["wall_s"],3)])+r" \\")
    table("tab_cost_fixture.tex","lrrrr",r"Recorded 4B operation & Items & Input tokens & Output tokens & Batch wall s",body,
          r"Representative P0 measurements on GSM8K training fixtures outside the study pool (Qwen3-4B, A100-SXM4-80GB). HF teacher forcing processes supplied sequences; it is not autoregressive decoding. vLLM blocks include prompt processing and generation; their output-token throughput is not a pure decode rate. These early vLLM timings use the original benchmark configuration, before mandatory batch-invariant study generation, and cannot calibrate a serving-time $\alpha$. A dash denotes an unreported token total; loading is excluded. Exact configurations and all models are retained in the cost evidence.","tab:cost-fixture")
    for alpha in ALPHAS:
        for sig in ("verbal","probe"):
            for cache in ("uncached","retained"):
                rr=[r for r in result["sensitivity"] if r["alpha"]==alpha and r["signal"]==sig and r["signal_cache"]==cache and r["budget"]>=512 and r["setting"].startswith("math") and r["gap_pp"] is not None]
                tag=str(alpha).replace(".","p")
                mac[f"costv2.math.{sig}.{cache}.alpha{tag}.min"]=signed(min(r["gap_pp"] for r in rr))
                mac[f"costv2.math.{sig}.{cache}.alpha{tag}.max"]=signed(max(r["gap_pp"] for r in rr))
    macro=gen/"cost_values.tex";macro.write_text("% Generated from existing records by analysis/neurocomputing_cost_v2.py.\n"+"\n".join(f"\\expandafter\\def\\csname val@{k}\\endcsname{{{v}}}" for k,v in sorted(mac.items()))+"\n",encoding="utf8",newline="\n");paths.append(macro)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcdefaults();plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    f,axes=plt.subplots(2,3,figsize=(7.2,4.6));colors=("#848484","#2a78b8","#d77625","#19866a")
    for row,sig in enumerate(("verbal","probe")):
        for col,setting in enumerate(SETTINGS):
            ax=axes[row,col]
            for b,color in zip(BUDGETS,colors):
                for cache,style in (("uncached","-"),("retained","--")):
                    vals=[idx[setting,b,sig,cache,a]["gap_pp"] for a in ALPHAS]
                    ax.plot(ALPHAS,[np.nan if v is None else v for v in vals],color=color,ls=style,lw=1.1)
            ax.axhline(0,color="#444444",lw=.6);ax.grid(axis="y",color="#dddddd",lw=.4)
            ax.set_title(labels[setting]+" / "+sig,loc="left");ax.set_xscale("symlog",linthresh=.01);ax.set_xlim(0,1);ax.set_xticks([0,.01,.1,1],labels=["0",".01",".1","1"])
            if row==1:ax.set_xlabel(r"Input-token weight $\alpha$")
            if col==0:ax.set_ylabel("Policy − uniform (pp)")
    handles=[Line2D([],[],color=c,label=f"B={b}") for b,c in zip(BUDGETS,colors)]+[Line2D([],[],color="#333333",ls=s,label=n) for s,n in (("-","separate signal prefill"),("--","signal prefix/state reused"))]
    f.legend(handles=handles,ncol=3,loc="lower center",frameon=False,fontsize=8,bbox_to_anchor=(.5,0));f.tight_layout(rect=(0,.12,1,1),w_pad=.8,h_pad=1)
    path=fig/"fig_cost_sensitivity.pdf";f.savefig(path,metadata={"Creator":None,"Producer":None,"CreationDate":None,"ModDate":None});paths.append(path)
    preview=fig/"fig_cost_sensitivity.png";f.savefig(preview,dpi=180);paths.append(preview);plt.close(f)
    return paths


def render_report(result):
    lines=["# Existing-record cost audit (generated)","",
           "This module is a deterministic CPU-only reanalysis. It executes no model and makes no cloud/API call. No new hypothesis test is introduced.","",
           "## Definitions and supported claims", "",
           "D is the full pipeline's generated-token count. P is its logical input-token volume, excluding batch padding. Re-reading a generated answer in a later forward pass adds to P. Thus P is neither FLOPs nor the number of GPU token positions. C_alpha = D + alpha P is a hypothetical sensitivity index, not latency, energy or money. No alpha value has been calibrated from the historical timings.","",
           "For a positive gated-thinking share q, D = mean(D_a0) + mean(1[selected] D_think), and P = mean(P_a0) + mean(P_signal) + mean(1[selected] P_think). A zero-share fallback is selected before acquiring the signal and pays only a0. The largest affordable share on the existing one-percentage-point grid is used; no accuracy-dependent share optimisation is performed. Every JSON sensitivity row records execution_mode, signal_acquired, gate_feasible and a0_plus_signal_fits. If even KEEP is unaffordable the selection and accuracy gap are null.","",
           "Uniform-budget primary P assumes retaining the generated trace KV: question tokens plus the inserted stop instruction for forced items. D already includes trace plus answer; the trace is not double-charged as fresh input. P_replayed_trace separately describes discarding the trace cache and reprocessing the answer prefix, in addition to the original question prefill. Neither scenario is an observed complete online execution.","",
           "Separate signal prefill follows the existing HF scoring inputs. Exact-prefix reuse is a hypothetical alternative, computed by token-level longest common prefix, and cannot be substituted with a chat-message-level assumption. The verbal prompt re-rendering omits the initial empty thinking segment, breaking the common prefix before a0. Probe reuse is an optimistic lower bound conditional on the requested last-content-token hidden state being available. Capped outputs may require evaluating their last emitted token; instrumentation, state transfer, the classifier readout and KV handling are unmeasured.","",
           "## Recorded populations and one-off setup", "",
           "| Setting | Test n | a0 D | a0 P | Verbal P separate | Verbal P exact-prefix reuse | Probe P separate | Dev n | Dev activation P |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s,p in result["populations"].items():
        z=result["probe_setup"][s]
        lines.append(f"| {s} | {p['n']} | {p['a0_D']:.3f} | {p['a0_P']:.3f} | {p['signal_prefill_uncached']['verbal']:.3f} | {p['signal_prefill_retained']['verbal']:.3f} | {p['signal_prefill_uncached']['probe']:.3f} | {z['dev_items']} | {z['activation_input_tokens']} |")
    lines += ["", "Dev activation input is a one-off setup volume, not a per-request serving charge; CPU fitting is not separately timed. The earlier Qwen3-4B audit's uncached counts reproduce exactly. Its nominal verbal cache charge is superseded here by the exact token-prefix calculation, with illustrative token mismatches in extraction_manifest.json and cost_v2.json.","", "## Matched-index sensitivity", "",
              "| Setting | B | Signal | alpha | Mode | Share | Policy minus uniform (pp) |", "|---|---:|---|---:|---|---:|---:|"]
    for r in result["sensitivity"]:
        if r["signal_cache"]=="uncached" and r["alpha"] in (0.,.1,1.):
            p=r["selected"];q="--" if p is None else str(p["q"]);gap="--" if r["gap_pp"] is None else f"{r['gap_pp']:+.3f}"
            lines.append(f"| {r['setting']} | {r['budget']} | {r['signal']} | {r['alpha']:g} | {r['execution_mode']} | {q} | {gap} |")
    lines += ["", "Full seven-weight results and both signal-cache assumptions are in cost_v2.json. break_even describes the previously D-matched fixed share: alpha_break_even = (D_uniform - D_gate)/(P_gate - P_uniform), when the denominator is positive. This is the affordability boundary of that fixed share, not a boundary for positive accuracy gain. h_headroom_at_alpha_0.1 is remaining budget in index units for additional unpriced per-item overhead; a negative number means that old fixed share already exceeds the indexed budget.","",
              "## Historical measured time: useful but a different estimand", "",
              "metrics/config/manifests preserve actual batch wall times for mathematics verbal scoring, all-layer activation extraction, P0 fixture operations, combined MMLU-Pro passes and ARR chunks. The generated tab_cost_timing.tex gives the mathematics HF times; tab_cost_fixture.tex gives representative P0 measurements. All math timing rows in the generated table are validated to be A100-SXM4-80GB, HF Transformers 5.20.2; verbal batch size is checked against configs. Dividing wall time by items gives a throughput summary, not request latency. Model loading is excluded.","",
              "P0 teacher-forced token throughput is a supplied-sequence forward pass, whereas vLLM generation blocks include prefill and decoding. They use different batching, context lengths and early backend configurations, and cannot identify separate prefill/decode coefficients. Historical vLLM P0 timings predate mandatory batch-invariant study generation. MMLU-Pro gpu_s combines verbal readout and all-layer state extraction. ARR c1b gen_s reuses saved thinking prefixes and does not include generating the traces. These timings are not added across backends to claim per-policy serving latency.","",
              "cloud/ledger.jsonl records historical whole research sessions, including setup/idle/storage/provider accounting. It is retained as evidence of existing costs but is not allocated to any policy and is not used to price C_alpha.","",
              "## Replay and provenance", "",
              "From supplementary/: `python -B analysis/neurocomputing_cost_v2.py --ms out`. This uses numpy and matplotlib, existing compact token counts and existing label/score arrays. No tokenizer, model, GPU or network is needed for replay. The optional original-repository extraction is `python -B analysis/neurocomputing_cost_v2.py --prepare-from E:/research --package E:/research/paper/Neurocomputing_v2/supplementary --ms out`; it uses a locally cached pinned tokenizer only. Extraction hashes the underlying records; the portable manifest hashes every analysis input and generated artifact.","",
              "Inputs under evidence/ include exact metrics/config/manifests, compact per-item cost arrays, probe setup totals, relevant prompt/scoring source files and the older independent Qwen3-4B cost audit. The compact arrays preserve the cost quantities required for replay, rather than redistributing every raw answer. Underlying raw-answer/token-record hashes are retained in extraction_manifest.json.","",
              "## Checks",""]
    lines += ["- "+c for c in result["checks"]]
    lines += ["", "## Remaining unmeasured quantities",""]+["- "+c for c in result["missing_costs"]]
    return "\n".join(lines)+"\n"


def main():
    here=Path(__file__).resolve().parents[1]
    default=here if here.name=="supplementary" else here/"paper/Neurocomputing_v2/supplementary"
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--package",type=Path,default=default);parser.add_argument("--ms",type=Path,default=Path("out"));parser.add_argument("--prepare-from",type=Path)
    args=parser.parse_args();package=args.package.resolve();assert package.exists(),"Create revision directory first"
    if args.prepare_from:prepare(args.prepare_from.resolve(),package)
    ms=args.ms if args.ms.is_absolute() else package/args.ms
    analyze(package,ms)

if __name__=="__main__":main()

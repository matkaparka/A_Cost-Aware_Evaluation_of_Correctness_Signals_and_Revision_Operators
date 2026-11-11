# Re-checking frozen language models: code, run outputs and protocol records

Code, run configurations, per-item run outputs and protocol records for a controlled study of LLM self-correction as a
revision decision with frozen models (Qwen3-1.7B/4B/8B and Llama-3.1-Tulu-3-8B). The study was pre-registered and
**run by an autonomous AI agent (Claude Code) under human supervision**. The author, Lihan Xu, set the research
question and approved the design and the claims.

Companion manuscript: *When Does Re-checking Pay? A Cost-Aware Evaluation of Correctness Signals and Revision Operators*.

## This is a curated public snapshot

The work was done in a private development repository. This repository is a snapshot of selected files from it, with
no history. Every included file is byte-identical to the file in the private repository at the time of the snapshot,
except `README.md`, `.gitignore`, `LICENSE-DATA.md` (written for this snapshot), `DEVIATIONS.md`, and
`docs/research_design.md`. In `DEVIATIONS.md`, six verbatim quotes of the author's instructions and one budget
figure were replaced by neutral wording and are marked "quote omitted" or reworded. In the research design, a
reference to background notes not distributed here was replaced by a neutral provenance note.

Not included:
- the agent's working records (decision log, status files, hand-over notes) and the run log;
- scripts that operated rented GPU hosts, and session logs;
- manuscript sources and working plans;
- personal background notes;
- the git history.

Consequences:
- Run manifests record a "locked code commit" and some run IDs end in a short commit hash. Those hashes belong to the
  private history and cannot be resolved here. To check that the code in a run's `src/` path is the code that ran,
  compare the `scorer_sha256`, script and config hashes in the run's `manifest.json` with the files here.
- The pre-registration `prereg.md` is included unchanged. Its sha256 is
  `ed3b274540ba9966de2319a8dd3fca6ad1fef79a5666873d836b5b549a8223e8`; it was frozen on 2026-11-05 at commit
  `08657caf44035d3772bb68ba6d38c823b664e304` of the private history. `DEVIATIONS.md` lists every later change.
- Some text refers to files that are not published (`DECISIONS.md`, `STATUS.md`, `CLAUDE.md`, `cloud/`).

## What is in the repository

| Path | Content |
|---|---|
| `src/` | generation, scoring, probing, steering, patching and test-time-update code (vLLM / Hugging Face) |
| `analysis/` | analysis scripts and their outputs (`analysis/out/`); every number in the papers comes from here |
| `configs/` | pinned model revisions, block manifest, package lockfile (`configs/lock.txt`) |
| `data/processed/` | the benchmark items actually used (see `LICENSE-DATA.md`) |
| `results/` | per-run outputs: configs, manifests with sha256, scores, generations (jsonl / jsonl.gz) |
| `prereg.md`, `DEVIATIONS.md`, `docs/`, `reports/` | pre-registration, deviation log, design, evidence ledger, AI-reviewer outputs and responses, phase reports |
| `theory/` | technical framework notes that motivate the hypotheses |
| `tests/` | invariant and scorer tests |

Not in git (too large): hidden-state tensors, adapters, model weights, and a few very large run outputs.
`reports/09_artifacts.md` lists the large outputs with sizes and sha256, and each run's `manifest.json` lists the
files it produced together with the inputs needed to rebuild them.

One more run output is left out of git because of its size (89 MB):
`results/fu3/main/4B/main-streams-test-controls-4B-1809070/eval_texts.jsonl`. It is attached to the release `data-v1`
as `fu3__main__4B__main-streams-test-controls-4B-1809070__eval_texts.jsonl` (sha256
`1d0dc951e9d9f7890241e661531653a43d30b573d26f95249e5e6780a86e3391`). Put it back at its path to rebuild anything that
reads it; the other files of that run are in git.

## Replaying the analyses

The tables, figures and numbers of the paper are regenerated from per-item records without loading a model. Some of
those records are not in git; they are in the data package `supplementary.zip`, attached to the release `data-v1` of
this repository (the checksum is in the release notes). Unpack it and follow its `README.md` (Python 3.12,
`requirements.txt`). Regenerating the *generations* needs the model weights and a GPU (A100 class) and is not part of
the replay.

Two files, `results/p3/{1.7B,4B}/p3-probe-dev-ext-*/probe_selection.json`, are stored with LF line endings, while the
sha256 in the manifests that list them was computed on a CRLF copy written on a Windows machine. To check those two
hashes, convert the file to CRLF first.

## Data and licenses

- Code (`src/`, `analysis/`, `tests/`, `scripts/`): MIT, see `LICENSE`.
- Our own data and text: CC BY 4.0, except the lore material (CC BY-NC 4.0). Third-party datasets, model outputs and
  files keep their own terms; the details, including three dataset pages that state no license, are in
  `LICENSE-DATA.md`.
- Outputs of Llama-3.1-Tulu-3-8B are subject to the Llama 3.1 Community License. Built with Llama.

## Size

About 0.45 GB packed, mostly `results/`. The largest file in git is 61 MB.

## Citation

To be added after publication.

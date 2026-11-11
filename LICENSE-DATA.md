# Licenses of the non-code material

Licenses chosen by the author, Lihan Xu, on 2026-11-08. Third-party notes and the data-package paragraph added on
2026-11-11 for the public release of the repository.

| Material | License |
|---|---|
| Code: `src/`, `analysis/`, `scripts/`, `tests/` | MIT (see `LICENSE`) |
| Lore QA material drawn from the author's unpublished fictional world (the Dragontooth texts of the Siawadria setting): `data/processed/lore/`, `results/lore/`, the lore runs under `results/fu*/` (run IDs containing `lore`), and `docs/planning/lore_qa/prereg_addendum.md` | CC BY-NC 4.0. No commercial use. https://creativecommons.org/licenses/by-nc/4.0/ |
| Everything else of our own: prompts, configs, protocols (`prereg.md`, addenda), logs, reports, the framework notes in `theory/`, the paper sources (not the third-party LaTeX files below), and the generations and scores of our own runs | CC BY 4.0. https://creativecommons.org/licenses/by/4.0/ |

## Third-party material

Third-party material keeps its own license.

**Datasets** (pages and upstream sources read on 2026-11-11):

| Dataset (as used) | License stated on the dataset page | Note |
|---|---|---|
| GSM8K (`openai/gsm8k`) | MIT | upstream `openai/grade-school-math`: MIT |
| MATH (`EleutherAI/hendrycks_math`) | MIT | upstream `hendrycks/math`: MIT |
| MMLU-Pro (`TIGER-Lab/MMLU-Pro`) | MIT | built from several upstream sources whose terms were not checked |
| Self-Correction Bench `scli5` (`kenhktsui/scli5`) | MIT | |
| MATH-500 (`HuggingFaceH4/MATH-500`) | none stated | the page says it is the 500-problem subset of MATH that OpenAI used in "Let's Verify Step by Step"; `openai/prm800k` is MIT |
| `kenhktsui/gsm8k_sc` | none stated | built on GSM8K; the injected errors were generated with `gpt-4.1-2025-04-14` |
| `kenhktsui/prm800k_sc` | none stated | built on PRM800K; the check answers were generated with `gemini-2.5-flash-preview-05-20` |

The items of these datasets that we used are included in `data/processed/` and may be quoted in stored outputs.
Decision of 2026-11-11: the three datasets without a stated license stay in the repository with this attribution, and
their original terms apply. We make no claim over them.

**Model outputs.** Outputs remain subject to the license of the model that produced them.
- Qwen3-1.7B/4B/8B: Apache-2.0.
- Llama-3.1-Tulu-3-8B: Llama 3.1 Community License (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
  Built with Llama.

**LaTeX files shipped in `paper/`.** `elsarticle.cls`, `elsarticle-num-names.bst` (Elsevier) and `acl_natbib.bst`: LPPL 1.3 (see the
file headers). `acl.sty`: from https://github.com/acl-org/acl-style-files, whose license was not found by the GitHub
API. These files are not covered by the CC BY 4.0 grant above.

## Data package and release files

The supplementary data package (`supplementary.zip`) and the files attached to the GitHub releases are derived from the
material above. Our own derived records (scores, per-item outcomes, analysis outputs) are CC BY 4.0; any third-party
text they contain keeps the terms in the tables above.

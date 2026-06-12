# Repository Guidelines

## Project Structure & Module Organization
The main experiment pipeline lives at the repository root. Core entry points include `recommendation_bias_experiment.py`, `poisoned_context_eval.py`, `ck_vllm_models.py`, `global_card_vllm_models.py`, `global_card_trace_utils.py`, `prompts.py`, and `dataset.py`. Data-preparation helpers include `build_brand_doc_combinations.py`, `generate_generic_product_documents.py`, `generate_poisoned_docs.py`, `rewrite_tap_source_docs.py`, `generate_tap_attacks.py`, and `extract_parametric_knowledge.py`. Source data and generated combinations live under `dataset/` and `out/brand_doc_combinations/`. Runtime artifacts belong in `out/`, `plots/`, and `logs/`; keep one-off notebooks, backups, and old experiments in their existing archival folders instead of mixing them with active pipeline code. Treat `cad/`, `cd/`, `LLMProductBias-main/`, and `papers/` as reference or side-project directories unless a task explicitly targets them.

## Build, Test, and Development Commands
There is no formal build step; development is script-driven. Use the CARD environment before running local-model code:

- `source ~/miniconda3/etc/profile.d/conda.sh && conda activate card`
- `python recommendation_bias_experiment.py --help` checks the main CLI.
- `python recommendation_bias_experiment.py --run-eval --model llama3.1-8b --test smartphone --target-gpu-ids 0` runs one category.
- `python poisoned_context_eval.py --help` checks the attack-evaluation CLI.
- `python -B -m py_compile recommendation_bias_experiment.py poisoned_context_eval.py ck_vllm_models.py global_card_vllm_models.py prompts.py dataset.py` is the fastest syntax check after edits; remove any generated `__pycache__/` if a check creates one.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, and `UPPER_CASE` for constants or tuning switches. Keep reusable CK logic in `ck_vllm_models.py` and reusable CARD logic in `global_card_vllm_models.py` or `global_card_trace_utils.py`; keep experiment wiring, CLI flags, and output-path logic in `recommendation_bias_experiment.py` and `poisoned_context_eval.py`. Add short comments only where the experiment logic is genuinely hard to infer.

## Testing Guidelines
This repository does not expose a single `pytest` suite. Prefer targeted validation: use `--help` for CLI parse checks, single-category `recommendation_bias_experiment.py --run-eval --test ...` invocations for end-to-end checks, and `poisoned_context_eval.py --help` for attack-evaluation CLI coverage. Finish with syntax checks on every edited Python file. Name active ad hoc checks `test_<topic>.py`; move stale variants into backup folders rather than leaving ambiguous copies at the root.

## Commit & Pull Request Guidelines
The current checkout does not include a root `.git` directory, so there is no local history to mirror. If you version changes elsewhere, use short imperative commit subjects such as `card: fix trigger followup handling`. PRs should state the experiment scope, touched models or datasets, exact validation commands, and whether files under `out/`, `plots/`, or `logs/` were intentionally regenerated.

# CARD Recommendation Bias Experiment Release

This repository contains the experiment code for measuring and controlling
knowledge reliance in product recommendation settings. All paths below are
relative to the repository root unless noted otherwise.

## Environment

Create the base environment first:

```bash
conda env create -f environment.yml
conda activate card
```

Install vLLM as a separate step, following the official vLLM installation
instructions for your hardware platform. Do not install vLLM through
`environment.yml`: vLLM wheels and source builds are sensitive to GPU vendor,
CUDA/ROCm/runtime versions, Python version, and PyTorch backend.

This project targets `vllm==0.18.1` because the CK and CARD patch files below
match the vLLM 0.18.1 Python source layout. For a standard NVIDIA CUDA wheel
install, this may look like:

```bash
uv pip install "vllm==0.18.1" --torch-backend=auto
```

For AMD ROCm, Intel XPU, CPU, Apple Silicon, Docker, or source builds, use the
corresponding official vLLM installation path for your machine, while targeting
the `0.18.1` release.

Local model weights are resolved by `models.py` through environment variables.
Either set a shared model root:

```bash
export EVA_LOCAL_MODEL_ROOT=./models
```

or set a model-specific path, for example:

```bash
export EVA_QWEN3_8B_PATH=/path/to/Qwen3-8B
export EVA_LLAMA3_1_8B_PATH=/path/to/Llama-3.1-8B-Instruct
export EVA_GEMMA3_12B_PATH=/path/to/gemma-3-12b-it
```

Remote-model runs do not ship with API keys. Set the corresponding environment
variable for the provider you use:

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export DEEPSEEK_API_KEY=...
export MODELSCOPE_API_KEY=...
export SILICONFLOW_API_KEY=...
export DASHSCOPE_API_KEY=...
export TOGETHER_API_KEY=...
export PERPLEXITY_API_KEY=...
```

Only the key for the selected model provider is required for a given run.

## vLLM CK/CARD Patch

The vLLM patch files are stored under `vllm_patches/vllm/` and should be copied
into the installed `vllm` package after vLLM 0.18.1 is installed:

```bash
VLLM_DIR=$(python -c "import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)")
cp -r vllm_patches/vllm/* "$VLLM_DIR"/
```

The patch tree contains:

- `v1/sample/logits_processor/paired_logits.py`
- `v1/sample/logits_processor/ck_pair.py`
- `v1/sample/logits_processor/global_card_pair.py`
- `v1/core/sched/scheduler.py`
- `v1/worker/gpu_model_runner.py`
- `v1/request.py`

Verify that the patched modules import:

```bash
python - <<'PY'
import importlib

modules = [
    "vllm.v1.sample.logits_processor.paired_logits",
    "vllm.v1.sample.logits_processor.ck_pair",
    "vllm.v1.sample.logits_processor.global_card_pair",
]
for module in modules:
    importlib.import_module(module)
print("vLLM CK/CARD patch imports succeeded")
PY
```

Reapply the patch after reinstalling or upgrading vLLM. The packaged code
expects `GlobalCARDPairLogitsProcessor` at
`vllm.v1.sample.logits_processor.global_card_pair`.

## Experiment 1: Parametric Knowledge and Main Bias Evaluation

Extract parametric knowledge for the model used to select known brands:

```bash
python extract_parametric_knowledge.py \
  --model qwen3-8b \
  --num-brands 8 \
  --gpu-ids "0" \
  --local-inference-backend vllm
```

Build brand-document combinations if `out/brand_doc_combinations/` does not
already contain the required model/category split:

```bash
python build_brand_doc_combinations.py \
  --model qwen3-8b \
  --num-brands 8 \
  --resume
```

Run the main recommendation-bias evaluation:

```bash
python recommendation_bias_experiment.py \
  --run-eval \
  --model qwen3-8b \
  --target-model qwen3-8b \
  --num-brands 8 \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0
```

Run the same setting with Global CARD mitigation:

```bash
python recommendation_bias_experiment.py \
  --run-eval \
  --model qwen3-8b \
  --target-model qwen3-8b \
  --num-brands 8 \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0 \
  --use-card \
  --card-global-direction-sign 1 \
  --card-dynamic-strength-max 1.0 \
  --card-global-main-bias-coeff 0.0
```

Global CARD exposes two paper-level hyperparameters in the CLI:
`--card-dynamic-strength-max` is the maximum intervention strength `M`, and
`--card-global-main-bias-coeff` is the PRS calibration coefficient `beta_m`.
The commands below use the same shared setting for all CARD runs (`M=1.0`,
`beta_m=0.0`). Use `--card-global-direction-sign 1` for recommendation-bias
mitigation and `--card-global-direction-sign -1` for PoisonedRAG/TAP defense.
The implementation defaults to Global CARD, dynamic strength, and
`main_aux_topk_union` support with top-k 10, so those default implementation
knobs are omitted from the commands.

The reproduction commands omit `--test` to run all categories. Add
`--test smartphone` to any command for a single-category smoke test.

## Experiment 2: PoisonedRAG Attack and Defense

Generate PoisonedRAG documents:

```bash
python generate_poisoned_docs.py \
  --brand Z_Brand \
  --model Z_Model \
  --llm-model deepseek-v4-flash
```

Evaluate the attack without defense:

```bash
python poisoned_context_eval.py \
  --run-eval \
  --attack-method PoisonedRAG \
  --target-model qwen3-8b \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0
```

Evaluate CK:

```bash
python poisoned_context_eval.py \
  --run-eval \
  --attack-method PoisonedRAG \
  --target-model qwen3-8b \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0 \
  --use-ck \
  --ck-alpha 0.5
```

Evaluate Global CARD defense. In poisoned-context evaluation the CARD direction
uses `-1` to suppress unreliable external evidence:

```bash
python poisoned_context_eval.py \
  --run-eval \
  --attack-method PoisonedRAG \
  --target-model qwen3-8b \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0 \
  --use-card \
  --card-global-direction-sign -1 \
  --card-dynamic-strength-max 1.0 \
  --card-global-main-bias-coeff 0.0
```

## Experiment 3: TAP Attack and Defense

Prepare the baseline TAP source document:

```bash
python rewrite_tap_source_docs.py \
  --target-brand Z_Brand \
  --target-model-name Z_Model \
  --rewriter-model deepseek-v4-flash
```

Generate TAP attack artifacts:

```bash
python generate_tap_attacks.py \
  --target-brand Z_Brand \
  --target-model-name Z_Model \
  --attacker-model deepseek-v4-flash \
  --target-model qwen3-8b \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0
```

Evaluate the pre-attack baseline:

```bash
python poisoned_context_eval.py \
  --run-eval \
  --attack-method TAP \
  --tap-doc-mode baseline \
  --target-model qwen3-8b \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0
```

Evaluate TAP after the adversarial prompt is inserted:

```bash
python poisoned_context_eval.py \
  --run-eval \
  --attack-method TAP \
  --tap-doc-mode after_tap \
  --target-model qwen3-8b \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0
```

Evaluate TAP with Global CARD defense:

```bash
python poisoned_context_eval.py \
  --run-eval \
  --attack-method TAP \
  --tap-doc-mode after_tap \
  --target-model qwen3-8b \
  --target-local-backend vllm \
  --target-gpu-ids "0" \
  --target-temp 0.0 \
  --use-card \
  --card-global-direction-sign -1 \
  --card-dynamic-strength-max 1.0 \
  --card-global-main-bias-coeff 0.0
```

## Directory Structure

```text
.
├── environment.yml                  # base conda environment, excluding vLLM
├── README.md                        # setup and reproduction instructions
├── recommendation_bias_experiment.py       # main recommendation-bias and CARD evaluation
├── recommendation_bias_analysis.py         # post-run plots, F-statistics, and summary metrics
├── poisoned_context_eval.py         # PoisonedRAG/TAP evaluation with CK or CARD
├── extract_parametric_knowledge.py  # extract category-level parametric knowledge
├── build_brand_doc_combinations.py  # build brand-document combinations
├── generate_generic_product_documents.py   # generate generic template documents
├── generate_poisoned_docs.py        # generate PoisonedRAG target documents
├── rewrite_tap_source_docs.py       # prepare TAP rewritten source documents
├── generate_tap_attacks.py          # generate TAP adversarial artifacts
├── ck_vllm_models.py                # CK-PLUG vLLM paired-decoding wrapper
├── global_card_vllm_models.py       # Global CARD vLLM paired-decoding wrapper
├── global_card_trace_utils.py       # Global CARD token-trace utilities
├── models.py                        # model registry and inference backends
├── prompts.py                       # prompt templates
├── dataset.py                       # dataset loading helpers
├── dataset/                         # product categories, products, and documents
├── out/                             # runtime inputs and generated results
│   ├── brand_doc_combinations/      # brand-document combinations
│   ├── parametric_knowledge/        # extracted parametric-knowledge outputs
│   ├── poisoned_documents/          # PoisonedRAG generated documents
│   ├── tap_rewritten_source_docs/   # TAP rewritten source documents
│   └── tap_attacks/                 # TAP adversarial prompts and optimized docs
├── plots/                           # generated figures
├── logs/                            # run logs
├── vllm_patches/                    # files to copy into vLLM 0.18.1
│   └── vllm/
│       └── v1/
│           ├── core/sched/scheduler.py       # paired-request scheduling patch
│           ├── request.py                     # extra-args request plumbing patch
│           ├── sample/logits_processor/
│           │   ├── paired_logits.py           # shared paired-logits helpers
│           │   ├── ck_pair.py                 # CK logits processor
│           │   └── global_card_pair.py        # Global CARD logits processor
│           └── worker/gpu_model_runner.py     # paired-logits runner patch
└── EMNLP_2026_Debias/               # paper source
```

The `out/` tree contains both reusable experiment inputs and generated outputs;
do not delete its subdirectories unless you are intentionally regenerating the
corresponding artifacts.

## Checks

Run these lightweight checks after editing code or reinstalling the environment:

```bash
python recommendation_bias_experiment.py --help
python poisoned_context_eval.py --help
python -B -m py_compile \
  recommendation_bias_experiment.py \
  poisoned_context_eval.py \
  ck_vllm_models.py \
  global_card_vllm_models.py \
  global_card_trace_utils.py \
  models.py \
  dataset.py
```

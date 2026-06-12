"""
CK-PLUG style greedy decoding for EVA local vLLM models.

This module keeps the CK runtime contract deliberately narrow:
- non-thinking only
- temperature=0.0 greedy decoding only

The implementation uses a paired-logits vLLM engine patch:
1. submit each logical sample as paired main/base live vLLM requests;
2. vLLM scheduler keeps paired logits requests complete in sampler batches;
3. a batch-level vLLM logits processor applies CK fusion on GPU logits;
4. the selected token is forced into both branches.

This keeps the original CK-PLUG requirement that both branches advance with the
same token at every decoding step while avoiding external per-token scoring.
"""

import gc
import math
import os
import re
import typing as t

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from _types import Message, Role
from prompts import format_target_message_without_doc_content


DEFAULT_CK_VLLM_ALPHA = 0.0
DEFAULT_CK_VLLM_SELECT_TOP = 10
DEFAULT_CK_VLLM_RELATIVE_TOP = 0.01
DEFAULT_CK_VLLM_ADAPTIVE = False

_ck_vllm_model_cache: t.Dict[t.Tuple[t.Any, ...], t.Tuple[t.Any, t.Any, int]] = {}
_ck_vllm_generate_kwargs_logged: bool = False
_ck_vllm_custom_all_reduce_notice_logged: bool = False
_ck_vllm_nccl_p2p_notice_logged: bool = False


def _count_visible_gpus(gpu_ids: t.Optional[str]) -> int:
    if gpu_ids is None or str(gpu_ids).strip() == "":
        return max(1, len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
    return max(1, len([part for part in str(gpu_ids).split(",") if part.strip()]))


def _resolve_effective_gpu_ids(gpu_ids: t.Optional[str]) -> t.Optional[str]:
    if gpu_ids is None:
        return None
    stripped = str(gpu_ids).strip()
    return stripped if stripped else None


def _should_disable_vllm_custom_all_reduce(gpu_ids: t.Optional[str]) -> bool:
    from models import (
        _should_disable_vllm_custom_all_reduce as should_disable_custom_all_reduce,
    )

    return should_disable_custom_all_reduce(gpu_ids)


def _should_disable_vllm_nccl_p2p(gpu_ids: t.Optional[str]) -> bool:
    from models import _should_disable_vllm_nccl_p2p as should_disable_nccl_p2p

    return should_disable_nccl_p2p(gpu_ids)


def _should_skip_vllm_mm_profiling(model_path: str) -> bool:
    normalized_path = str(model_path).lower()
    return "gemma3" in normalized_path or "gemma-3" in normalized_path


def _log_vllm_custom_all_reduce_notice(gpu_ids: t.Optional[str]) -> None:
    global _ck_vllm_custom_all_reduce_notice_logged
    if _ck_vllm_custom_all_reduce_notice_logged:
        return
    print(
        "CK vLLM detected target GPU set "
        f"{gpu_ids}; explicitly setting disable_custom_all_reduce=True",
        flush=True,
    )
    _ck_vllm_custom_all_reduce_notice_logged = True


def _log_vllm_nccl_p2p_notice(gpu_ids: t.Optional[str]) -> None:
    global _ck_vllm_nccl_p2p_notice_logged
    if _ck_vllm_nccl_p2p_notice_logged:
        return
    print(
        "CK vLLM detected target GPU set "
        f"{gpu_ids}; explicitly setting NCCL_P2P_DISABLE=1",
        flush=True,
    )
    _ck_vllm_nccl_p2p_notice_logged = True


def _resolve_vllm_gpu_memory_utilization(
    gpu_memory_utilization: t.Optional[float],
) -> t.Optional[float]:
    if gpu_memory_utilization is None:
        return None
    value = float(gpu_memory_utilization)
    if not (0.0 < value < 1.0):
        raise ValueError("vllm_gpu_memory_utilization must be in the (0, 1) range")
    return value


def _build_ck_vllm_cache_key(
    model_path: str,
    gpu_ids: t.Optional[str],
    gpu_memory_utilization: t.Optional[float],
    max_model_len: t.Optional[int],
    max_num_seqs: t.Optional[int],
    max_num_batched_tokens: t.Optional[int],
) -> t.Tuple[t.Any, ...]:
    effective_gpu_ids = _resolve_effective_gpu_ids(gpu_ids)
    resolved_gpu_memory_utilization = _resolve_vllm_gpu_memory_utilization(
        gpu_memory_utilization
    )
    return (
        model_path,
        effective_gpu_ids,
        resolved_gpu_memory_utilization,
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
        _should_disable_vllm_custom_all_reduce(effective_gpu_ids),
        _should_disable_vllm_nccl_p2p(effective_gpu_ids),
    )


def _release_vllm_model_object(llm: t.Any, mode: str = "wait") -> None:
    if llm is None:
        return

    sleep_fn = getattr(llm, "sleep", None)
    if callable(sleep_fn):
        try:
            sleep_fn(level=2, mode=mode)
        except Exception:
            pass
        return

    llm_engine = getattr(llm, "llm_engine", None)
    sleep_fn = getattr(llm_engine, "sleep", None)
    if callable(sleep_fn):
        try:
            sleep_fn(level=2, mode=mode)
        except Exception:
            pass


def release_ck_vllm_model_cache(
    model_name: str,
    gpu_ids: t.Optional[str] = None,
    gpu_memory_utilization: t.Optional[float] = None,
    max_model_len: t.Optional[int] = None,
    max_num_seqs: t.Optional[int] = None,
    max_num_batched_tokens: t.Optional[int] = None,
    vllm_sleep_mode: str = "wait",
) -> bool:
    from models import Models

    if model_name not in Models:
        return False

    _, model_path, is_remote = Models[model_name]
    if is_remote:
        return False

    cache_key = _build_ck_vllm_cache_key(
        model_path=model_path,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    cached_entry = _ck_vllm_model_cache.pop(cache_key, None)
    if cached_entry is None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    _tokenizer, llm, _vocab_size = cached_entry
    _release_vllm_model_object(llm, mode=vllm_sleep_mode)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def _extract_doc_info_from_prompt(user_prompt: str) -> t.Tuple[str, t.List[str], t.List[str]]:
    query_match = re.search(r"We now are processing a user query: (.+?)\n", user_prompt)
    query = query_match.group(1) if query_match else ""

    models_section = re.search(
        r"For your reference, here are again the product models you should include in your response:\n\n(.+?)\n\nUser:",
        user_prompt,
        re.DOTALL,
    )
    product_models: t.List[str] = []
    if models_section:
        product_models = [
            line.strip()
            for line in models_section.group(1).strip().split("\n")
            if line.strip()
        ]

    brand_matches = re.findall(
        r"DOCUMENT \d+ \(brand: (.+?), model: .+?\):",
        user_prompt,
    )
    product_brands = list(brand_matches)

    return query, product_models, product_brands


def _trim_trailing_stop_token_ids(tokenizer, generated_ids: t.List[int]) -> t.List[int]:
    stop_token_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    trimmed_ids = list(generated_ids)
    while trimmed_ids and trimmed_ids[-1] in stop_token_ids:
        trimmed_ids.pop()
    return trimmed_ids


def _relative_top_filter(
    main_scores: torch.Tensor,
    base_scores: torch.Tensor,
    select_top: int,
    relative_top: float,
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    main_log_probs = F.log_softmax(main_scores, dim=-1)
    base_log_probs = F.log_softmax(base_scores, dim=-1)

    vocab_size = main_log_probs.shape[-1]
    keep_count = min(max(int(select_top), 1), vocab_size)

    main_top_values = torch.topk(main_log_probs, keep_count, dim=-1).values
    base_top_values = torch.topk(base_log_probs, keep_count, dim=-1).values

    min_thresh_main = main_top_values[:, -1]
    min_thresh_base = base_top_values[:, -1]

    rel_log = math.log(float(relative_top))
    main_thresh = torch.minimum(
        min_thresh_main,
        torch.max(main_log_probs, dim=-1).values + rel_log,
    ).unsqueeze(-1)
    base_thresh = torch.minimum(
        min_thresh_base,
        torch.max(base_log_probs, dim=-1).values + rel_log,
    ).unsqueeze(-1)

    mask = (main_log_probs < main_thresh) & (base_log_probs < base_thresh)
    filtered_main = main_scores.clone()
    filtered_base = base_scores.clone()
    filtered_main[mask] = -1e10
    filtered_base[mask] = -1e10
    return filtered_main, filtered_base, mask


def _apply_pair(
    main_scores: torch.Tensor,
    base_scores: torch.Tensor,
    alpha: float,
    adaptive: bool,
    select_top: int,
    relative_top: float,
) -> torch.Tensor:
    filtered_main, filtered_base, mask = _relative_top_filter(
        main_scores=main_scores.float().unsqueeze(0),
        base_scores=base_scores.float().unsqueeze(0),
        select_top=select_top,
        relative_top=relative_top,
    )
    filtered_main = filtered_main.squeeze(0)
    filtered_base = filtered_base.squeeze(0)
    mask = mask.squeeze(0)

    probs_main = F.softmax(filtered_main, dim=-1)
    probs_base = F.softmax(filtered_base, dim=-1)
    entropy_main = -torch.sum(probs_main * torch.log(probs_main + 1e-9))
    entropy_base = -torch.sum(probs_base * torch.log(probs_base + 1e-9))

    info_gain = entropy_base - entropy_main
    is_adjust = (info_gain - torch.abs(entropy_main)) < 0
    if not bool(is_adjust.item()):
        return filtered_main.to(main_scores.dtype)

    base_for_context = filtered_base.clone()
    base_for_context[mask] = -1e3
    logits_context = filtered_main - base_for_context
    filtered_base[mask] = -1e10

    if adaptive:
        diff = torch.abs(entropy_main - entropy_base)
        entropy_sum = torch.clamp(entropy_main + entropy_base, min=1e-6)
        normalization_factor = 1.0 + (diff / entropy_sum)
        denominator = torch.clamp(
            entropy_main + entropy_base * normalization_factor,
            min=1e-6,
        )
        fused = (
            2.0 * filtered_base * entropy_main / denominator
            + 2.0 * logits_context * entropy_base / denominator
        )
    else:
        fused = float(alpha) * filtered_base + (1.0 - float(alpha)) * logits_context

    return fused.to(main_scores.dtype)


def _get_vllm_vocab_size(llm, tokenizer) -> int:
    model_config = getattr(llm, "model_config", None)
    if model_config is not None and hasattr(model_config, "get_vocab_size"):
        return int(model_config.get_vocab_size())
    return int(max(len(tokenizer), getattr(tokenizer, "vocab_size", 0)))


def _extract_full_vocab_scores(
    completion,
    vocab_size: int,
) -> torch.Tensor:
    if completion is None:
        raise RuntimeError("CK vLLM single-step scoring failed: completion is empty")
    if completion.logprobs is None:
        raise RuntimeError("CK vLLM single-step scoring failed: logprobs were not returned")

    step_logprobs = completion.logprobs
    if hasattr(step_logprobs, "start_indices") and hasattr(step_logprobs, "end_indices"):
        if len(step_logprobs.start_indices) != 1 or len(step_logprobs.end_indices) != 1:
            raise RuntimeError("CK vLLM expects FlatLogprobs for exactly one token per step")
        start = int(step_logprobs.start_indices[0])
        end = int(step_logprobs.end_indices[0])
        token_ids = step_logprobs.token_ids[start:end]
        scores = step_logprobs.logprobs[start:end]
    else:
        if len(step_logprobs) != 1:
            raise RuntimeError(
                f"CK vLLM expects scores for exactly one token per step; got {len(step_logprobs)} positions"
            )
        one_position = step_logprobs[0]
        token_ids = list(one_position.keys())
        scores = [one_position[token_id].logprob for token_id in token_ids]

    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    full_scores = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    valid_mask = (token_ids_tensor >= 0) & (token_ids_tensor < vocab_size)
    full_scores[token_ids_tensor[valid_mask]] = scores_tensor[valid_mask]

    covered_vocab = int(torch.isfinite(full_scores).sum().item())
    if covered_vocab != vocab_size:
        raise RuntimeError(
            "CK vLLM did not receive full-vocabulary scores; "
            f"covered {covered_vocab}/{vocab_size}. "
            "Strict CK reproduction is not possible."
        )

    return full_scores


def _build_tokenized_chat_prompt(
    tokenizer,
    system_content: str,
    user_content: str,
) -> t.List[int]:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _validate_ck_vllm_prompt_lengths(
    prompt_token_ids_pairs: t.List[t.Tuple[t.List[int], t.List[int]]],
    max_new_tokens: int,
    max_model_len: int,
) -> None:
    prompt_over_limit_items: t.List[str] = []
    for sample_idx, (main_ids, base_ids) in enumerate(prompt_token_ids_pairs):
        main_len = len(main_ids)
        base_len = len(base_ids)
        max_prompt_len = max(main_len, base_len)
        if max_prompt_len > max_model_len:
            prompt_over_limit_items.append(
                f"sample={sample_idx}, main_prompt={main_len}, "
                f"base_prompt={base_len}, max_prompt={max_prompt_len}"
            )

    if prompt_over_limit_items:
        raise ValueError(
            "CK vLLM prompt length exceeds max_model_len; "
            f"current max_model_len={max_model_len}.\n"
            + "\n".join(prompt_over_limit_items[:5])
        )


def _load_ck_vllm_model(
    model_path: str,
    gpu_ids: t.Optional[str] = None,
    gpu_memory_utilization: t.Optional[float] = None,
    max_model_len: t.Optional[int] = None,
    max_num_seqs: t.Optional[int] = None,
    max_num_batched_tokens: t.Optional[int] = None,
) -> t.Tuple[t.Any, t.Any, int]:
    effective_gpu_ids = _resolve_effective_gpu_ids(gpu_ids)
    resolved_gpu_memory_utilization = _resolve_vllm_gpu_memory_utilization(
        gpu_memory_utilization
    )
    cache_key = (
        model_path,
        effective_gpu_ids,
        resolved_gpu_memory_utilization,
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
        _should_disable_vllm_custom_all_reduce(effective_gpu_ids),
        _should_disable_vllm_nccl_p2p(effective_gpu_ids),
    )
    if cache_key not in _ck_vllm_model_cache:
        if effective_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = effective_gpu_ids
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if _should_disable_vllm_nccl_p2p(effective_gpu_ids):
            os.environ["NCCL_P2P_DISABLE"] = "1"
            _log_vllm_nccl_p2p_notice(effective_gpu_ids)

        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        llm_kwargs: t.Dict[str, t.Any] = {
            "model": model_path,
            "tensor_parallel_size": _count_visible_gpus(effective_gpu_ids),
            "max_logprobs": -1,
            "logits_processors": [
                "vllm.v1.sample.logits_processor.ck_pair:CKPairLogitsProcessor"
            ],
            "enable_chunked_prefill": True,
            "async_scheduling": False,
        }
        if resolved_gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = resolved_gpu_memory_utilization
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if _should_skip_vllm_mm_profiling(model_path):
            llm_kwargs["skip_mm_profiling"] = True
        if _should_disable_vllm_custom_all_reduce(effective_gpu_ids):
            llm_kwargs["disable_custom_all_reduce"] = True
            _log_vllm_custom_all_reduce_notice(effective_gpu_ids)

        llm = LLM(**llm_kwargs)
        vocab_size = _get_vllm_vocab_size(llm, tokenizer)
        _ck_vllm_model_cache[cache_key] = (tokenizer, llm, vocab_size)
    return _ck_vllm_model_cache[cache_key]


def chat_ck_vllm_local(
    model_path: str,
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    alpha: float = DEFAULT_CK_VLLM_ALPHA,
    adaptive: bool = DEFAULT_CK_VLLM_ADAPTIVE,
    select_top: int = DEFAULT_CK_VLLM_SELECT_TOP,
    relative_top: float = DEFAULT_CK_VLLM_RELATIVE_TOP,
    max_new_tokens: int = 2000,
    gpu_ids: t.Optional[str] = None,
    return_token_counts: bool = False,
    queries: t.Optional[t.List[str]] = None,
    product_models_list: t.Optional[t.List[t.List[str]]] = None,
    product_brands_list: t.Optional[t.List[t.List[str]]] = None,
    vllm_gpu_memory_utilization: t.Optional[float] = None,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
) -> t.Union[Message, t.Dict[str, t.Any], t.List[Message], t.List[t.Dict[str, t.Any]]]:
    is_single = len(messages) > 0 and isinstance(messages[0], Message)
    if is_single:
        batch_messages = [t.cast(t.List[Message], messages)]
    else:
        batch_messages = t.cast(t.List[t.List[Message]], messages)

    required_num_seqs = max(2, 2 * len(batch_messages))
    effective_max_num_seqs = (
        None
        if vllm_max_num_seqs is None
        else max(int(vllm_max_num_seqs), required_num_seqs)
    )

    tokenizer, llm, vocab_size = _load_ck_vllm_model(
        model_path=model_path,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        max_num_seqs=effective_max_num_seqs,
        max_num_batched_tokens=vllm_max_num_batched_tokens,
    )

    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("CK alpha must be in the [0, 1] range")
    if int(select_top) <= 0:
        raise ValueError("CK select_top must be greater than 0")
    if not (0.0 < float(relative_top) <= 1.0):
        raise ValueError("CK relative_top must be in the (0, 1] range")

    current_main_prompt_token_ids: t.List[t.List[int]] = []
    current_base_prompt_token_ids: t.List[t.List[int]] = []
    base_user_contents: t.List[str] = []

    for sample_idx, sample_messages in enumerate(batch_messages):
        system_content = sample_messages[0].content if len(sample_messages) > 0 else ""
        user_content = sample_messages[1].content if len(sample_messages) > 1 else ""

        query = queries[sample_idx] if queries and sample_idx < len(queries) else None
        product_models = (
            product_models_list[sample_idx]
            if product_models_list and sample_idx < len(product_models_list)
            else None
        )
        product_brands = (
            product_brands_list[sample_idx]
            if product_brands_list and sample_idx < len(product_brands_list)
            else None
        )

        if query is None or product_models is None or product_brands is None:
            query, product_models, product_brands = _extract_doc_info_from_prompt(
                user_content
            )

        base_user_content = format_target_message_without_doc_content(
            query=query,
            product_models=product_models,
            product_brands=product_brands,
        )
        base_user_contents.append(base_user_content)

        current_main_prompt_token_ids.append(
            _build_tokenized_chat_prompt(
                tokenizer=tokenizer,
                system_content=system_content,
                user_content=user_content,
            )
        )
        current_base_prompt_token_ids.append(
            _build_tokenized_chat_prompt(
                tokenizer=tokenizer,
                system_content=system_content,
                user_content=base_user_content,
            )
        )

    effective_model_max_len = int(llm.model_config.max_model_len)
    _validate_ck_vllm_prompt_lengths(
        prompt_token_ids_pairs=list(
            zip(current_main_prompt_token_ids, current_base_prompt_token_ids)
        ),
        max_new_tokens=max_new_tokens,
        max_model_len=effective_model_max_len,
    )

    global _ck_vllm_generate_kwargs_logged
    if not _ck_vllm_generate_kwargs_logged:
        print("\n[CK vLLM local-model inference parameters] (first call)", flush=True)
        print(f"  model path: {model_path}", flush=True)
        print("  backend: vllm", flush=True)
        print(
            "  decoding: greedy-only (temperature=0.0, non-thinking, sample-step atomic paired logits requests)",
            flush=True,
        )
        if bool(adaptive):
            print("  alpha: ignored in adaptive mode", flush=True)
        else:
            print(f"  alpha: {float(alpha):g}", flush=True)
        print(f"  adaptive: {bool(adaptive)}", flush=True)
        print(f"  select_top: {int(select_top)}", flush=True)
        print(f"  relative_top: {float(relative_top):g}", flush=True)
        print(f"  logical batch_size: {len(batch_messages)}", flush=True)
        print(f"  paired live vLLM requests: {2 * len(batch_messages)}", flush=True)
        print(f"  CK processor: vLLM batch-level logits fusion", flush=True)
        print(
            "  chunked_prefill: enabled (paired logits only constrains sample-producing steps)",
            flush=True,
        )
        print(f"  vocab_size: {vocab_size}", flush=True)
        print(f"  max_model_len: {effective_model_max_len}", flush=True)
        if effective_max_num_seqs is None:
            print(
                "  max_num_seqs: (not explicitly provided; using the vLLM default; "
                f"at least {required_num_seqs} requests per step are recommended for throughput)",
                flush=True,
            )
        else:
            print(f"  max_num_seqs: {effective_max_num_seqs}", flush=True)
        if vllm_max_num_batched_tokens is None:
            print("  max_num_batched_tokens: (not explicitly provided; using the vLLM default)", flush=True)
        else:
            print(f"  max_num_batched_tokens: {vllm_max_num_batched_tokens}", flush=True)
        print(flush=True)
        _ck_vllm_generate_kwargs_logged = True

    from vllm import SamplingParams

    paired_prompts: t.List[t.List[int]] = []
    paired_params: t.List[t.Any] = []
    pair_max_tokens: t.List[int] = []
    for sample_idx, (main_ids, base_ids) in enumerate(
        zip(current_main_prompt_token_ids, current_base_prompt_token_ids)
    ):
        max_tokens_for_pair = min(
            int(max_new_tokens),
            effective_model_max_len - len(main_ids),
            effective_model_max_len - len(base_ids),
        )
        if max_tokens_for_pair <= 0:
            raise ValueError(
                "CK vLLM prompt already fills max_model_len; generation is impossible: "
                f"sample={sample_idx}, main_prompt={len(main_ids)}, "
                f"base_prompt={len(base_ids)}, max_model_len={effective_model_max_len}"
            )
        pair_max_tokens.append(max_tokens_for_pair)

        common_extra_args: t.Dict[str, t.Any] = {
            "logits_pair_id": str(sample_idx),
            "ck_alpha": float(alpha),
            "ck_adaptive": bool(adaptive),
            "ck_select_top": int(select_top),
            "ck_relative_top": float(relative_top),
        }
        for role, prompt_ids in (("main", main_ids), ("base", base_ids)):
            paired_prompts.append(list(prompt_ids))
            paired_params.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens_for_pair,
                    min_tokens=0,
                    ignore_eos=False,
                    skip_special_tokens=False,
                    detokenize=False,
                    extra_args={**common_extra_args, "logits_pair_role": role},
                )
            )

    paired_outputs = llm.generate(
        paired_prompts,
        sampling_params=paired_params,
        use_tqdm=False,
    )

    results: t.List[t.Union[Message, t.Dict[str, t.Any]]] = []
    for sample_idx in range(len(batch_messages)):
        main_output = paired_outputs[2 * sample_idx]
        base_output = paired_outputs[2 * sample_idx + 1]
        main_completion = main_output.outputs[0] if main_output.outputs else None
        base_completion = base_output.outputs[0] if base_output.outputs else None
        if main_completion is None or base_completion is None:
            raise RuntimeError("CK vLLM engine-level generation failed: completion is empty")

        main_output_ids = list(main_completion.token_ids)
        base_output_ids = list(base_completion.token_ids)
        if main_output_ids != base_output_ids:
            raise RuntimeError(
                "CK vLLM pair synchronization failed: main/base generated token mismatch, "
                f"sample={sample_idx}, main_len={len(main_output_ids)}, "
                f"base_len={len(base_output_ids)}"
            )

        trimmed_output_ids = _trim_trailing_stop_token_ids(tokenizer, main_output_ids)
        response_text = tokenizer.decode(
            trimmed_output_ids,
            skip_special_tokens=True,
        )
        message = Message(role=Role.assistant, content=response_text)
        if return_token_counts:
            result: t.Dict[str, t.Any] = {
                "message": message,
                "aux_user_message": base_user_contents[sample_idx],
                "thinking_tokens": 0,
                "response_tokens": len(trimmed_output_ids),
                "token_count_source": "vllm_generated_ids",
                "hit_length_limit": (
                    getattr(main_completion, "finish_reason", None) == "length"
                    or len(main_output_ids) >= pair_max_tokens[sample_idx]
                ),
            }
            results.append(result)
        else:
            results.append(message)

    if is_single:
        return results[0]
    return results


def load_ck_vllm_model(
    model_name: str,
    alpha: float = DEFAULT_CK_VLLM_ALPHA,
    adaptive: bool = DEFAULT_CK_VLLM_ADAPTIVE,
    select_top: int = DEFAULT_CK_VLLM_SELECT_TOP,
    relative_top: float = DEFAULT_CK_VLLM_RELATIVE_TOP,
    max_new_tokens: int = 2000,
    gpu_ids: t.Optional[str] = None,
    return_token_counts: bool = False,
    vllm_gpu_memory_utilization: t.Optional[float] = None,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
) -> t.Callable:
    from models import Models

    if model_name not in Models:
        raise ValueError(f"Unknown model: {model_name}")

    _, model_path, is_remote = Models[model_name]
    if is_remote:
        raise ValueError(f"CK vLLM currently only supports local models; {model_name} is a remote API model")

    def wrapper(
        messages,
        queries=None,
        categories=None,
        documents_list=None,
        product_models_list=None,
        product_brands_list=None,
    ):
        del categories
        del documents_list
        return chat_ck_vllm_local(
            model_path=model_path,
            messages=messages,
            alpha=alpha,
            adaptive=adaptive,
            select_top=select_top,
            relative_top=relative_top,
            max_new_tokens=max_new_tokens,
            gpu_ids=gpu_ids,
            return_token_counts=return_token_counts,
            queries=queries,
            product_models_list=product_models_list,
            product_brands_list=product_brands_list,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_max_num_seqs=vllm_max_num_seqs,
            vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        )

    return wrapper

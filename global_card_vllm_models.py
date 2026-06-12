"""
Global CARD greedy decoding for EVA local vLLM models.

This module implements the narrow Global CARD variant described in the current
experiment writeup:
- non-thinking only
- temperature=0.0 greedy decoding only
- full-vocabulary contrastive logits correction by default
- optional main/aux top-k union support for final token selection
- empty-document auxiliary prompt built with preserved product slots

The implementation follows the CK-vLLM paired-request pattern. Each logical
sample is submitted as two live vLLM requests: main and base. The base request
is the empty-document auxiliary branch. A vLLM batch-level logits processor
computes KL(aux || main), derives the dynamic strength, selects the next token
from (1 - abs(b)) * z_main + sign*alpha * (z_main - z_aux), optionally
restricts that selection to the union of main and auxiliary top-k tokens, and
forces the chosen token into both branches. The default sign=1 and b=0 preserve
the original formula. sign=1 enhances external-document contribution for
debiasing; sign=-1 suppresses it for poisoning defense.
"""

import gc
import math
import os
import tempfile
import typing as t

import torch
from transformers import AutoTokenizer

from _types import Message, Role
from ck_vllm_models import (
    _build_tokenized_chat_prompt,
    _count_visible_gpus,
    _extract_doc_info_from_prompt,
    _get_vllm_vocab_size,
    _resolve_effective_gpu_ids,
    _resolve_vllm_gpu_memory_utilization,
    _should_disable_vllm_custom_all_reduce,
    _should_disable_vllm_nccl_p2p,
    _should_skip_vllm_mm_profiling,
    _trim_trailing_stop_token_ids,
)
from global_card_trace_utils import (
    build_global_card_token_trace,
    read_jsonl_records,
)
from prompts import format_target_message_without_doc_content


_global_card_vllm_model_cache: t.Dict[
    t.Tuple[t.Any, ...], t.Tuple[t.Any, t.Any, int]
] = {}
_global_card_vllm_generate_kwargs_logged: bool = False
_global_card_vllm_custom_all_reduce_notice_logged: bool = False
_global_card_vllm_nccl_p2p_notice_logged: bool = False

GLOBAL_CARD_VLLM_SUPPORT_FULL_VOCAB = "full_vocab"
GLOBAL_CARD_VLLM_SUPPORT_MAIN_AUX_TOPK_UNION = "main_aux_topk_union"
GLOBAL_CARD_VLLM_SUPPORT_MODES = {
    GLOBAL_CARD_VLLM_SUPPORT_FULL_VOCAB,
    GLOBAL_CARD_VLLM_SUPPORT_MAIN_AUX_TOPK_UNION,
}


def _log_vllm_custom_all_reduce_notice(gpu_ids: t.Optional[str]) -> None:
    global _global_card_vllm_custom_all_reduce_notice_logged
    if _global_card_vllm_custom_all_reduce_notice_logged:
        return
    print(
        "Global CARD vLLM detected target GPU set "
        f"{gpu_ids}; explicitly setting disable_custom_all_reduce=True",
        flush=True,
    )
    _global_card_vllm_custom_all_reduce_notice_logged = True


def _log_vllm_nccl_p2p_notice(gpu_ids: t.Optional[str]) -> None:
    global _global_card_vllm_nccl_p2p_notice_logged
    if _global_card_vllm_nccl_p2p_notice_logged:
        return
    print(
        "Global CARD vLLM detected target GPU set "
        f"{gpu_ids}; explicitly setting NCCL_P2P_DISABLE=1",
        flush=True,
    )
    _global_card_vllm_nccl_p2p_notice_logged = True


def _resolve_dynamic_strength_max(
    dynamic_strength_max: t.Any,
) -> t.Tuple[float, float]:
    try:
        max_strength = float(dynamic_strength_max)
    except (TypeError, ValueError):
        raise ValueError(
            "Global CARD vLLM dynamic_strength_max must be a finite positive number"
        ) from None
    if not math.isfinite(max_strength) or max_strength <= 0.0:
        raise ValueError("Global CARD vLLM dynamic_strength_max must be a finite positive number")
    gamma = 1.0 / max_strength
    return max_strength, gamma


def _resolve_main_bias_coeff(main_bias_coeff: t.Any) -> float:
    try:
        coeff = float(main_bias_coeff)
    except (TypeError, ValueError):
        raise ValueError(
            "Global CARD vLLM main_bias_coeff must be a finite number in the [-1, 1] range"
        ) from None
    if not math.isfinite(coeff) or not (-1.0 <= coeff <= 1.0):
        raise ValueError(
            "Global CARD vLLM main_bias_coeff must be a finite number in the [-1, 1] range"
        )
    return coeff


def _resolve_direction_sign(direction_sign: t.Any) -> int:
    try:
        sign = int(direction_sign)
    except (TypeError, ValueError):
        raise ValueError("Global CARD vLLM direction_sign must be 1 or -1") from None
    if sign not in {1, -1}:
        raise ValueError("Global CARD vLLM direction_sign must be 1 or -1")
    return sign


def _resolve_support_config(
    support_mode: t.Any,
    support_top_k: t.Any,
) -> t.Tuple[str, int]:
    mode = str(support_mode or GLOBAL_CARD_VLLM_SUPPORT_FULL_VOCAB)
    if mode not in GLOBAL_CARD_VLLM_SUPPORT_MODES:
        raise ValueError(
            "Global CARD vLLM support_mode must be one of "
            f"{sorted(GLOBAL_CARD_VLLM_SUPPORT_MODES)}"
        )
    try:
        top_k = int(support_top_k)
    except (TypeError, ValueError):
        raise ValueError("Global CARD vLLM support_top_k must be a positive integer") from None
    if mode == GLOBAL_CARD_VLLM_SUPPORT_MAIN_AUX_TOPK_UNION and top_k <= 0:
        raise ValueError(
            "Global CARD vLLM support_top_k must be a positive integer "
            "when support_mode=main_aux_topk_union"
        )
    return mode, top_k


def _build_global_card_vllm_cache_key(
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


def release_global_card_vllm_model_cache(
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

    cache_key = _build_global_card_vllm_cache_key(
        model_path=model_path,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    cached_entry = _global_card_vllm_model_cache.pop(cache_key, None)
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


def _validate_global_card_vllm_prompt_lengths(
    prompt_token_ids_pairs: t.List[t.Tuple[t.List[int], t.List[int]]],
    max_model_len: int,
) -> None:
    prompt_over_limit_items: t.List[str] = []
    for sample_idx, (main_ids, aux_ids) in enumerate(prompt_token_ids_pairs):
        main_len = len(main_ids)
        aux_len = len(aux_ids)
        max_prompt_len = max(main_len, aux_len)
        if max_prompt_len > max_model_len:
            prompt_over_limit_items.append(
                f"sample={sample_idx}, main_prompt={main_len}, "
                f"aux_prompt={aux_len}, max_prompt={max_prompt_len}"
            )

    if prompt_over_limit_items:
        raise ValueError(
            "Global CARD vLLM prompt length exceeds max_model_len; "
            f"current max_model_len={max_model_len}.\n"
            + "\n".join(prompt_over_limit_items[:5])
        )


def _load_global_card_vllm_model(
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
    if cache_key not in _global_card_vllm_model_cache:
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
            "logits_processors": [
                "vllm.v1.sample.logits_processor.global_card_pair:GlobalCARDPairLogitsProcessor"
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
        _global_card_vllm_model_cache[cache_key] = (tokenizer, llm, vocab_size)
    return _global_card_vllm_model_cache[cache_key]


def chat_global_card_vllm_local(
    model_path: str,
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    card_dynamic_strength_max: float = 1.0,
    card_global_main_bias_coeff: float = 0.0,
    card_global_direction_sign: int = 1,
    card_global_vllm_support_mode: str = GLOBAL_CARD_VLLM_SUPPORT_FULL_VOCAB,
    card_global_vllm_support_top_k: int = 10,
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
    save_global_card_token_trace: bool = True,
) -> t.Union[Message, t.Dict[str, t.Any], t.List[Message], t.List[t.Dict[str, t.Any]]]:
    dynamic_strength_max, gamma = _resolve_dynamic_strength_max(
        card_dynamic_strength_max
    )
    main_bias_coeff = _resolve_main_bias_coeff(card_global_main_bias_coeff)
    direction_sign = _resolve_direction_sign(card_global_direction_sign)
    support_mode, support_top_k = _resolve_support_config(
        card_global_vllm_support_mode,
        card_global_vllm_support_top_k,
    )

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

    tokenizer, llm, vocab_size = _load_global_card_vllm_model(
        model_path=model_path,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        max_num_seqs=effective_max_num_seqs,
        max_num_batched_tokens=vllm_max_num_batched_tokens,
    )

    current_main_prompt_token_ids: t.List[t.List[int]] = []
    current_aux_prompt_token_ids: t.List[t.List[int]] = []
    aux_user_contents: t.List[str] = []

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

        aux_user_content = format_target_message_without_doc_content(
            query=query,
            product_models=product_models,
            product_brands=product_brands,
        )
        aux_user_contents.append(aux_user_content)

        current_main_prompt_token_ids.append(
            _build_tokenized_chat_prompt(
                tokenizer=tokenizer,
                system_content=system_content,
                user_content=user_content,
            )
        )
        current_aux_prompt_token_ids.append(
            _build_tokenized_chat_prompt(
                tokenizer=tokenizer,
                system_content=system_content,
                user_content=aux_user_content,
            )
        )

    effective_model_max_len = int(llm.model_config.max_model_len)
    _validate_global_card_vllm_prompt_lengths(
        prompt_token_ids_pairs=list(
            zip(current_main_prompt_token_ids, current_aux_prompt_token_ids)
        ),
        max_model_len=effective_model_max_len,
    )

    global _global_card_vllm_generate_kwargs_logged
    if not _global_card_vllm_generate_kwargs_logged:
        print("\n[Global CARD vLLM local-model inference parameters] (first call)", flush=True)
        print(f"  model path: {model_path}", flush=True)
        print("  backend: vllm", flush=True)
        print(
            "  decoding: greedy-only (temperature=0.0, non-thinking, sample-step atomic main/aux pairs)",
            flush=True,
        )
        print("  auxiliary prompt: empty document content (delete)", flush=True)
        print("  application: global correction every step", flush=True)
        print(
            "  formula: z_card = (1 - abs(b)) * z_main + sign*alpha_t * (z_main - z_aux)",
            flush=True,
        )
        print(f"  main_bias_coeff_b: {main_bias_coeff:g}", flush=True)
        print(
            "  direction_sign: "
            f"{direction_sign} "
            "(1=enhance external document contribution/debias, "
            "-1=suppress external document contribution/poisoning defense)",
            flush=True,
        )
        print(f"  support_mode: {support_mode}", flush=True)
        if support_mode == GLOBAL_CARD_VLLM_SUPPORT_MAIN_AUX_TOPK_UNION:
            print(f"  support_top_k: {support_top_k}", flush=True)
            print(
                "  support_semantics: next token is selected only from "
                "union(top-k main logits, top-k aux logits); KL/alpha_t remains full-vocab",
                flush=True,
            )
        else:
            print("  support_semantics: full vocabulary", flush=True)
        print(
            "  alpha_t: 1 / ln(exp(gamma) + KL(p_aux || p_main))",
            flush=True,
        )
        print(f"  dynamic_strength_max: {dynamic_strength_max:g}", flush=True)
        print(f"  gamma: {gamma:g}", flush=True)
        print(f"  logical batch_size: {len(batch_messages)}", flush=True)
        print(f"  paired live vLLM requests: {2 * len(batch_messages)}", flush=True)
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
        _global_card_vllm_generate_kwargs_logged = True

    from vllm import SamplingParams

    paired_prompts: t.List[t.List[int]] = []
    paired_params: t.List[t.Any] = []
    pair_max_tokens: t.List[int] = []
    trace_paths: t.List[t.Optional[str]] = []
    trace_temp_dir: t.Optional[t.Any] = None
    if save_global_card_token_trace:
        trace_temp_dir = tempfile.TemporaryDirectory(prefix="global_card_vllm_trace_")
    for sample_idx, (main_ids, aux_ids) in enumerate(
        zip(current_main_prompt_token_ids, current_aux_prompt_token_ids)
    ):
        max_tokens_for_pair = min(
            int(max_new_tokens),
            effective_model_max_len - len(main_ids),
            effective_model_max_len - len(aux_ids),
        )
        if max_tokens_for_pair <= 0:
            raise ValueError(
                "Global CARD vLLM prompt already fills max_model_len; generation is impossible: "
                f"sample={sample_idx}, main_prompt={len(main_ids)}, "
                f"aux_prompt={len(aux_ids)}, max_model_len={effective_model_max_len}"
            )
        pair_max_tokens.append(max_tokens_for_pair)
        trace_path = None
        if trace_temp_dir is not None:
            trace_path = os.path.join(
                trace_temp_dir.name,
                f"sample_{sample_idx:04d}.jsonl",
            )
        trace_paths.append(trace_path)

        common_extra_args: t.Dict[str, t.Any] = {
            "logits_pair_id": str(sample_idx),
            "global_card_dynamic_strength_max": float(dynamic_strength_max),
            "global_card_main_bias_coeff": float(main_bias_coeff),
            "global_card_direction_sign": int(direction_sign),
            "global_card_support_mode": support_mode,
            "global_card_support_top_k": int(support_top_k),
        }
        if trace_path is not None:
            common_extra_args["global_card_trace_path"] = trace_path
        for role, prompt_ids in (("main", main_ids), ("base", aux_ids)):
            paired_prompts.append(list(prompt_ids))
            paired_params.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens_for_pair,
                    min_tokens=0,
                    ignore_eos=False,
                    skip_special_tokens=False,
                    detokenize=False,
                    extra_args={
                        **common_extra_args,
                        "logits_pair_role": role,
                    },
                )
            )

    try:
        paired_outputs = llm.generate(
            paired_prompts,
            sampling_params=paired_params,
            use_tqdm=False,
        )
        results: t.List[t.Union[Message, t.Dict[str, t.Any]]] = []
        for sample_idx in range(len(batch_messages)):
            main_output = paired_outputs[2 * sample_idx]
            aux_output = paired_outputs[2 * sample_idx + 1]
            main_completion = main_output.outputs[0] if main_output.outputs else None
            aux_completion = aux_output.outputs[0] if aux_output.outputs else None
            if main_completion is None or aux_completion is None:
                raise RuntimeError("Global CARD vLLM engine-level generation failed: completion is empty")

            main_output_ids = list(main_completion.token_ids)
            aux_output_ids = list(aux_completion.token_ids)
            if main_output_ids != aux_output_ids:
                raise RuntimeError(
                    "Global CARD vLLM pair synchronization failed: main/aux generated token mismatch, "
                    f"sample={sample_idx}, main_len={len(main_output_ids)}, "
                    f"aux_len={len(aux_output_ids)}"
                )

            trimmed_output_ids = _trim_trailing_stop_token_ids(
                tokenizer,
                main_output_ids,
            )
            response_text = tokenizer.decode(
                trimmed_output_ids,
                skip_special_tokens=True,
            )
            message = Message(role=Role.assistant, content=response_text)
            if return_token_counts:
                result: t.Dict[str, t.Any] = {
                    "message": message,
                    "marked_response": response_text,
                    "aux_user_message": aux_user_contents[sample_idx],
                    "card_aux_prompt_type": "delete",
                    "card_triggers": [],
                    "card_step_strength_logs": [],
                    "thinking_tokens": 0,
                    "response_tokens": len(trimmed_output_ids),
                    "token_count_source": "vllm_generated_ids",
                    "card_global_vllm_support_mode": support_mode,
                    "card_global_vllm_support_top_k": support_top_k,
                    "hit_length_limit": (
                        getattr(main_completion, "finish_reason", None) == "length"
                        or len(main_output_ids) >= pair_max_tokens[sample_idx]
                    ),
                }
                if save_global_card_token_trace and trace_paths[sample_idx]:
                    trace_records = read_jsonl_records(trace_paths[sample_idx])
                    result["global_card_token_trace"] = build_global_card_token_trace(
                        tokenizer=tokenizer,
                        output_token_ids=trimmed_output_ids,
                        source_records=trace_records,
                    )
                results.append(result)
            else:
                results.append(message)

        if is_single:
            return results[0]
        return results
    finally:
        if trace_temp_dir is not None:
            trace_temp_dir.cleanup()


def load_global_card_vllm_model(
    model_name: str,
    card_dynamic_strength_max: float = 1.0,
    card_global_main_bias_coeff: float = 0.0,
    card_global_direction_sign: int = 1,
    card_global_vllm_support_mode: str = GLOBAL_CARD_VLLM_SUPPORT_FULL_VOCAB,
    card_global_vllm_support_top_k: int = 10,
    max_new_tokens: int = 2000,
    gpu_ids: t.Optional[str] = None,
    return_token_counts: bool = False,
    vllm_gpu_memory_utilization: t.Optional[float] = None,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
    save_global_card_token_trace: bool = True,
) -> t.Callable:
    from models import Models

    if model_name not in Models:
        raise ValueError(f"Unknown model: {model_name}")

    _, model_path, is_remote = Models[model_name]
    if is_remote:
        raise ValueError(f"Global CARD vLLM currently only supports local models; {model_name} is a remote API model")

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
        return chat_global_card_vllm_local(
            model_path=model_path,
            messages=messages,
            card_dynamic_strength_max=card_dynamic_strength_max,
            card_global_main_bias_coeff=card_global_main_bias_coeff,
            card_global_direction_sign=card_global_direction_sign,
            card_global_vllm_support_mode=card_global_vllm_support_mode,
            card_global_vllm_support_top_k=card_global_vllm_support_top_k,
            max_new_tokens=max_new_tokens,
            gpu_ids=gpu_ids,
            return_token_counts=return_token_counts,
            save_global_card_token_trace=save_global_card_token_trace,
            queries=queries,
            product_models_list=product_models_list,
            product_brands_list=product_brands_list,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_max_num_seqs=vllm_max_num_seqs,
            vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        )

    return wrapper

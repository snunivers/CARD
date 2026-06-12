import gc
import os
import json
import typing as t
import functools
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import torch
from transformers import AutoTokenizer
from _types import Message, Parameters, Role, ChatFunction



class TokenProbInfo(t.TypedDict):
    token: str
    token_id: int
    probability: float



class ResponseWithProbs(t.TypedDict):
    message: Message
    token_probs: t.List[TokenProbInfo]



class ResponseWithThinking(t.TypedDict):
    message: Message
    thinking: str



class ResponseWithThinkingAndTokens(t.TypedDict):
    message: Message
    thinking: str
    thinking_tokens: t.Optional[int]  
    response_tokens: t.Optional[int]  


class ResponseWithTokenCounts(t.TypedDict, total=False):
    message: Message
    thinking: str
    token_probs: t.List[TokenProbInfo]
    thinking_tokens: int
    response_tokens: int
    token_count_source: str
    finish_reason: t.Optional[str]
    stop_reason: t.Optional[t.Union[int, str]]
    hit_length_limit: bool
# from mistralai.client import MistralClient # type: ignore
# from mistralai.models.chat_completion import ChatMessage # type: ignore
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam



_local_model_cache: t.Dict[str, t.Tuple[t.Any, t.Any]] = {}
_local_vllm_model_cache: t.Dict[
    t.Tuple[
        str,
        t.Optional[str],
        t.Optional[float],
        t.Optional[int],
        t.Optional[int],
        t.Optional[int],
        bool,
        bool,
    ],
    t.Tuple[t.Any, t.Any],
] = {}


_generate_kwargs_logged: bool = False
_vllm_generate_kwargs_logged: bool = False
_vllm_custom_all_reduce_notice_logged: bool = False
_vllm_nccl_p2p_notice_logged: bool = False


_api_request_kwargs_logged: bool = False
_api_request_kwargs_log_lock = threading.Lock()


_deepseek_request_kwargs_logged: bool = False


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_V4_DEFAULT_REASONING_EFFORT = "high"


def _local_model_path(env_var: str, relative_path: str) -> str:
    """Resolve local model weights without hard-coding a machine-specific path."""
    explicit_path = os.environ.get(env_var)
    if explicit_path:
        return explicit_path
    model_root = os.environ.get("EVA_LOCAL_MODEL_ROOT", "./models")
    return str(Path(model_root) / relative_path)


def _require_env_api_key(env_var: str) -> str:
    """Return an API key from the environment with a reproducible error message."""
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"Missing API key: set {env_var}")
    return api_key


def _resolve_deepseek_thinking_flag(value: t.Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return value.get("type") == "enabled"
    return False


LOCAL_MODEL_GPU_IDS: t.Optional[str] = None


def set_local_model_gpu(gpu_ids: str) -> None:
    global LOCAL_MODEL_GPU_IDS
    LOCAL_MODEL_GPU_IDS = gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def _normalize_gpu_ids(gpu_ids: t.Optional[t.Any]) -> t.Optional[str]:
    if gpu_ids is None:
        return None
    gpu_ids_text = str(gpu_ids).strip()
    return gpu_ids_text if gpu_ids_text else None


def _resolve_effective_local_gpu_ids(
    gpu_ids: t.Optional[t.Any] = None,
) -> t.Optional[str]:
    if gpu_ids is not None:
        return _normalize_gpu_ids(gpu_ids)
    return _normalize_gpu_ids(LOCAL_MODEL_GPU_IDS)


def _count_visible_gpus(gpu_ids: t.Optional[str]) -> int:
    if gpu_ids is None:
        return 1
    visible_gpu_ids = [gid.strip() for gid in gpu_ids.split(",") if gid.strip()]
    return max(1, len(visible_gpu_ids))


def _should_disable_vllm_custom_all_reduce(gpu_ids: t.Optional[str]) -> bool:
    if gpu_ids is None:
        return False
    visible_gpu_ids = tuple(
        sorted(gid.strip() for gid in gpu_ids.split(",") if gid.strip())
    )
    return visible_gpu_ids in {
        ("0", "2"),
        ("0", "3"),
        ("0", "7"),
        ("2", "3"),
        ("4", "5"),
        ("6", "7"),
    }


def _should_disable_vllm_nccl_p2p(gpu_ids: t.Optional[str]) -> bool:
    if gpu_ids is None:
        return False
    visible_gpu_ids = tuple(
        sorted(gid.strip() for gid in gpu_ids.split(",") if gid.strip())
    )
    return visible_gpu_ids in {
        ("0", "2"),
        ("0", "3"),
        ("0", "7"),
        ("2", "3"),
        ("4", "5"),
        ("6", "7"),
    }


def _should_skip_vllm_mm_profiling(model_path: str) -> bool:
    normalized_path = str(model_path).lower()
    return "gemma3" in normalized_path or "gemma-3" in normalized_path


def _log_vllm_custom_all_reduce_notice(gpu_ids: t.Optional[str]) -> None:
    global _vllm_custom_all_reduce_notice_logged
    if _vllm_custom_all_reduce_notice_logged:
        return
    print(
        "vLLM detected target GPU set "
        f"{gpu_ids}; explicitly setting disable_custom_all_reduce=True",
        flush=True,
    )
    _vllm_custom_all_reduce_notice_logged = True


def _log_vllm_nccl_p2p_notice(gpu_ids: t.Optional[str]) -> None:
    global _vllm_nccl_p2p_notice_logged
    if _vllm_nccl_p2p_notice_logged:
        return
    print(
        "vLLM detected target GPU set "
        f"{gpu_ids}; explicitly setting NCCL_P2P_DISABLE=1",
        flush=True,
    )
    _vllm_nccl_p2p_notice_logged = True


def _resolve_vllm_gpu_memory_utilization(
    gpu_memory_utilization: t.Any,
) -> t.Optional[float]:
    if gpu_memory_utilization is None:
        return None

    try:
        utilization = float(gpu_memory_utilization)
    except (TypeError, ValueError):
        raise ValueError(
            "vllm_gpu_memory_utilization must be a float in the (0, 1) range"
        ) from None

    if (not math.isfinite(utilization)) or (utilization <= 0.0) or (utilization >= 1.0):
        raise ValueError(
            "vllm_gpu_memory_utilization must be in the (0, 1) range"
        )
    return utilization


def _build_local_vllm_cache_key(
    model_path: str,
    gpu_ids: t.Optional[str],
    vllm_gpu_memory_utilization: t.Optional[float],
    vllm_max_model_len: t.Optional[int],
    vllm_max_num_seqs: t.Optional[int],
    vllm_max_num_batched_tokens: t.Optional[int],
) -> t.Tuple[t.Any, ...]:
    resolved_gpu_ids = _resolve_effective_local_gpu_ids(gpu_ids)
    resolved_gpu_memory_utilization = _resolve_vllm_gpu_memory_utilization(
        vllm_gpu_memory_utilization
    )
    return (
        model_path,
        resolved_gpu_ids,
        resolved_gpu_memory_utilization,
        vllm_max_model_len,
        vllm_max_num_seqs,
        vllm_max_num_batched_tokens,
        _should_disable_vllm_custom_all_reduce(resolved_gpu_ids),
        _should_disable_vllm_nccl_p2p(resolved_gpu_ids),
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
    else:
        llm_engine = getattr(llm, "llm_engine", None)
        sleep_fn = getattr(llm_engine, "sleep", None)
        if callable(sleep_fn):
            try:
                sleep_fn(level=2, mode=mode)
            except Exception:
                pass


def release_local_model_cache(
    model: str,
    gpu_ids: t.Optional[str] = None,
    local_inference_backend: str = "vllm",
    vllm_gpu_memory_utilization: t.Optional[float] = None,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
    vllm_sleep_mode: str = "wait",
) -> bool:
    if model not in Models:
        return False

    _, model_path, is_remote = Models[model]
    if is_remote:
        return False

    released = False
    if local_inference_backend == "vllm":
        cache_key = _build_local_vllm_cache_key(
            model_path=model_path,
            gpu_ids=gpu_ids,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_max_num_seqs=vllm_max_num_seqs,
            vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        )
        cached_entry = _local_vllm_model_cache.pop(cache_key, None)
        if cached_entry is not None:
            _tokenizer, llm = cached_entry
            _release_vllm_model_object(llm, mode=vllm_sleep_mode)
            released = True
    elif local_inference_backend != "vllm":
        raise ValueError("local_inference_backend only supports 'vllm'")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return released


def _prepare_local_prompt_texts(
    tokenizer,
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    enable_thinking: t.Optional[bool] = None,
) -> t.Tuple[bool, t.List[t.List[Message]], t.List[str]]:
    is_single = len(messages) > 0 and isinstance(messages[0], Message)
    if is_single:
        batch_messages = [t.cast(t.List[Message], messages)]
    else:
        batch_messages = t.cast(t.List[t.List[Message]], messages)

    texts = []
    for msgs in batch_messages:
        chat_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in msgs
        ]
        apply_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            apply_kwargs["enable_thinking"] = enable_thinking
        texts.append(tokenizer.apply_chat_template(chat_messages, **apply_kwargs))

    return is_single, batch_messages, texts


def _resolve_local_generation_parameters(
    parameters: Parameters,
    temperature: t.Optional[float] = None,
    max_new_tokens: t.Optional[int] = None,
    do_sample: t.Optional[bool] = None,
    top_p: t.Optional[float] = None,
) -> t.Tuple[t.Optional[float], t.Optional[int], t.Optional[float], t.Optional[bool]]:
    _temperature = temperature if temperature is not None else parameters.temperature
    _max_new_tokens = (
        max_new_tokens if max_new_tokens is not None else parameters.max_tokens
    )
    _top_p = top_p if top_p is not None else parameters.top_p

    if do_sample is not None:
        _do_sample = do_sample
    elif _temperature is not None:
        _do_sample = _temperature > 0
    else:
        _do_sample = None

    return _temperature, _max_new_tokens, _top_p, _do_sample


def _load_local_model(model_path: str, gpu_ids: t.Optional[str] = None) -> t.Tuple[t.Any, t.Any]:
    if model_path not in _local_model_cache:
        
        effective_gpu_ids = _resolve_effective_local_gpu_ids(gpu_ids)
        if effective_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = effective_gpu_ids
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "left"
        
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        _local_model_cache[model_path] = (tokenizer, model)
    return _local_model_cache[model_path]


def _load_local_vllm_model(
    model_path: str,
    gpu_ids: t.Optional[str] = None,
    gpu_memory_utilization: t.Optional[float] = None,
    max_model_len: t.Optional[int] = None,
    max_num_seqs: t.Optional[int] = None,
    max_num_batched_tokens: t.Optional[int] = None,
) -> t.Tuple[t.Any, t.Any]:
    effective_gpu_ids = _resolve_effective_local_gpu_ids(gpu_ids)
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
    if cache_key not in _local_vllm_model_cache:
        if effective_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = effective_gpu_ids
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if _should_disable_vllm_nccl_p2p(effective_gpu_ids):
            os.environ["NCCL_P2P_DISABLE"] = "1"
            _log_vllm_nccl_p2p_notice(effective_gpu_ids)

        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        llm_kwargs: t.Dict[str, t.Any] = {
            "model": model_path,
            "tensor_parallel_size": _count_visible_gpus(effective_gpu_ids),
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
        _local_vllm_model_cache[cache_key] = (tokenizer, llm)
    return _local_vllm_model_cache[cache_key]


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


def _chat_local(
    model_path: str,
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    parameters: Parameters,
    enable_thinking: t.Optional[bool] = None,
    extract_thinking: bool = False,
    gpu_ids: t.Optional[str] = None,
    return_probs: bool = False,
    return_token_counts: bool = False,
    
    temperature: t.Optional[float] = None,
    max_new_tokens: t.Optional[int] = None,
    do_sample: t.Optional[bool] = None,
    top_p: t.Optional[float] = None,
) -> t.Union[Message, ResponseWithProbs, ResponseWithThinking, t.List[Message], t.List[ResponseWithProbs], t.List[ResponseWithThinking]]:
    tokenizer, model = _load_local_model(model_path, gpu_ids=gpu_ids)
    
    is_single, _, texts = _prepare_local_prompt_texts(
        tokenizer=tokenizer,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    
    
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)
    
    
    
    _temperature, _max_new_tokens, _top_p, _do_sample = (
        _resolve_local_generation_parameters(
            parameters=parameters,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
        )
    )
    
    
    generate_kwargs = {
        "max_new_tokens": _max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,  
    }
    
    
    if _top_p is not None:
        generate_kwargs["top_p"] = _top_p
    if _do_sample is not None:
        generate_kwargs["do_sample"] = _do_sample
        
        if _do_sample and _temperature is not None:
            generate_kwargs["temperature"] = _temperature
    elif _temperature is not None:
        
        generate_kwargs["temperature"] = _temperature
    
    
    if return_probs:
        generate_kwargs["output_scores"] = True
        generate_kwargs["return_dict_in_generate"] = True
    
    
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
    
    
    global _generate_kwargs_logged
    if not _generate_kwargs_logged:
        print(f"\n[Local-model inference parameters] (first call)", flush=True)
        print(f"  model path: {model_path}", flush=True)
        print(f"  enable_thinking: {enable_thinking}", flush=True)
        print(f"  parameters passed to model.generate():", flush=True)
        for k, v in generate_kwargs.items():
            if k != "pad_token_id":  
                print(f"    {k}: {v}", flush=True)
        
        if "do_sample" in generate_kwargs and generate_kwargs["do_sample"] == False:
            print(f"    do_sample: False (greedy decoding; temperature and top_p are ignored)", flush=True)
        else:
            if "temperature" not in generate_kwargs:
                print(f"    temperature: (not provided; using the model generation_config default)", flush=True)
            if "do_sample" not in generate_kwargs:
                print(f"    do_sample: (not provided; using the model generation_config default)", flush=True)
            if "top_p" not in generate_kwargs:
                print(f"    top_p: (not provided; using the model generation_config default)", flush=True)
        print(flush=True)
        _generate_kwargs_logged = True
    
    outputs = model.generate(**model_inputs, **generate_kwargs)
    
    
    results = []
    for i, sequence in enumerate(outputs.sequences if return_probs else outputs):
        
        generated_ids = sequence[model_inputs.input_ids.shape[-1]:]
        output_ids = generated_ids.tolist()
        trimmed_output_ids = _trim_trailing_stop_token_ids(tokenizer, output_ids)
        
        
        if enable_thinking:
            try:
                
                index = len(trimmed_output_ids) - trimmed_output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            if extract_thinking:
                
                thinking_content = tokenizer.decode(trimmed_output_ids[:index], skip_special_tokens=True).strip("\n")
                response_text = tokenizer.decode(trimmed_output_ids[index:], skip_special_tokens=True).strip("\n")
                message = Message(role=Role.assistant, content=response_text)
                if return_token_counts:
                    results.append({
                        "message": message,
                        "thinking": thinking_content,
                        "thinking_tokens": index,
                        "response_tokens": len(trimmed_output_ids[index:]),
                        "token_count_source": "local_generated_ids",
                    })
                else:
                    results.append({"message": message, "thinking": thinking_content})
            else:
                
                response_text = tokenizer.decode(trimmed_output_ids[index:], skip_special_tokens=True).strip("\n")
                message = Message(role=Role.assistant, content=response_text)
                
                if not return_probs:
                    if return_token_counts:
                        results.append({
                            "message": message,
                            "thinking_tokens": index,
                            "response_tokens": len(trimmed_output_ids[index:]),
                            "token_count_source": "local_generated_ids",
                        })
                    else:
                        results.append(message)
                else:
                    
                    token_probs: t.List[TokenProbInfo] = []
                    for j in range(index, len(output_ids)):
                        if j < len(outputs.scores):
                            prob = torch.softmax(outputs.scores[j][i], dim=-1)
                            token_id = generated_ids[j]
                            token_prob = prob[token_id].item()
                            token_text = tokenizer.decode(token_id)
                            
                            
                            if token_id.item() in tokenizer.all_special_ids:
                                continue
                            
                            token_probs.append({
                                "token": token_text,
                                "token_id": token_id.item(),
                                "probability": token_prob,
                            })
                    
                    result: ResponseWithTokenCounts = {
                        "message": message,
                        "token_probs": token_probs,
                    }
                    if return_token_counts:
                        result["thinking_tokens"] = index
                        result["response_tokens"] = len(trimmed_output_ids[index:])
                        result["token_count_source"] = "local_generated_ids"
                    results.append(result)
        else:
            
            response_text = tokenizer.decode(trimmed_output_ids, skip_special_tokens=True)
            message = Message(role=Role.assistant, content=response_text)
            
            if not return_probs:
                if return_token_counts:
                    results.append({
                        "message": message,
                        "thinking_tokens": 0,
                        "response_tokens": len(trimmed_output_ids),
                        "token_count_source": "local_generated_ids",
                    })
                else:
                    results.append(message)
            else:
                
                token_probs: t.List[TokenProbInfo] = []
                for j, score in enumerate(outputs.scores):
                    prob = torch.softmax(score[i], dim=-1)
                    token_id = generated_ids[j]
                    token_prob = prob[token_id].item()
                    token_text = tokenizer.decode(token_id)
                    
                    
                    if token_id.item() in tokenizer.all_special_ids:
                        continue
                    
                    token_probs.append({
                        "token": token_text,
                        "token_id": token_id.item(),
                        "probability": token_prob,
                        })
                
                result = {
                    "message": message,
                    "token_probs": token_probs,
                }
                if return_token_counts:
                    result["thinking_tokens"] = 0
                    result["response_tokens"] = len(trimmed_output_ids)
                    result["token_count_source"] = "local_generated_ids"
                results.append(result)
    
    
    if is_single:
        return results[0]
    else:
        return results


def _extract_vllm_token_probability(
    position_logprobs: t.Optional[t.Dict[int, t.Any]],
    token_id: int,
) -> t.Optional[float]:
    if not position_logprobs:
        return None
    token_logprob = position_logprobs.get(token_id)
    if token_logprob is None:
        return None
    return math.exp(float(token_logprob.logprob))


def _chat_local_vllm(
    model_path: str,
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    parameters: Parameters,
    enable_thinking: t.Optional[bool] = None,
    extract_thinking: bool = False,
    gpu_ids: t.Optional[str] = None,
    return_probs: bool = False,
    return_token_counts: bool = False,
    vllm_gpu_memory_utilization: t.Optional[float] = None,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
    temperature: t.Optional[float] = None,
    max_new_tokens: t.Optional[int] = None,
    do_sample: t.Optional[bool] = None,
    top_p: t.Optional[float] = None,
) -> t.Union[
    Message,
    ResponseWithProbs,
    ResponseWithThinking,
    t.List[Message],
    t.List[ResponseWithProbs],
    t.List[ResponseWithThinking],
]:
    tokenizer, llm = _load_local_vllm_model(
        model_path,
        gpu_ids=gpu_ids,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        max_num_seqs=vllm_max_num_seqs,
        max_num_batched_tokens=vllm_max_num_batched_tokens,
    )
    is_single, _, texts = _prepare_local_prompt_texts(
        tokenizer=tokenizer,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    _temperature, _max_new_tokens, _top_p, _do_sample = (
        _resolve_local_generation_parameters(
            parameters=parameters,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
        )
    )

    sampling_kwargs: t.Dict[str, t.Any] = {
        "skip_special_tokens": False,
    }
    if _max_new_tokens is not None:
        sampling_kwargs["max_tokens"] = _max_new_tokens
    if _top_p is not None:
        sampling_kwargs["top_p"] = _top_p
    if _do_sample is False:
        sampling_kwargs["temperature"] = 0.0
    elif _temperature is not None:
        sampling_kwargs["temperature"] = _temperature
    if return_probs:
        sampling_kwargs["logprobs"] = 1

    global _vllm_generate_kwargs_logged
    if not _vllm_generate_kwargs_logged:
        print(f"\n[Local-model inference parameters] (first call)", flush=True)
        print(f"  model path: {model_path}", flush=True)
        print(f"  backend: vllm", flush=True)
        resolved_vllm_gpu_memory_utilization = _resolve_vllm_gpu_memory_utilization(
            vllm_gpu_memory_utilization
        )
        if resolved_vllm_gpu_memory_utilization is None:
            print(
                "  gpu_memory_utilization: (not explicitly provided; using the vLLM default)",
                flush=True,
            )
        else:
            print(
                "  gpu_memory_utilization: "
                f"{resolved_vllm_gpu_memory_utilization:g}",
                flush=True,
            )
        if vllm_max_model_len is None:
            print("  max_model_len: (not explicitly provided; using the vLLM default)", flush=True)
        else:
            print(f"  max_model_len: {vllm_max_model_len}", flush=True)
        if vllm_max_num_seqs is None:
            print("  max_num_seqs: (not explicitly provided; using the vLLM default)", flush=True)
        else:
            print(f"  max_num_seqs: {vllm_max_num_seqs}", flush=True)
        if vllm_max_num_batched_tokens is None:
            print(
                "  max_num_batched_tokens: (not explicitly provided; using the vLLM default)",
                flush=True,
            )
        else:
            print(
                f"  max_num_batched_tokens: {vllm_max_num_batched_tokens}",
                flush=True,
            )
        print(f"  enable_thinking: {enable_thinking}", flush=True)
        print(f"  parameters passed to vLLM SamplingParams:", flush=True)
        for k, v in sampling_kwargs.items():
            print(f"    {k}: {v}", flush=True)
        if sampling_kwargs.get("temperature") == 0.0:
            print(f"    do_sample: False (greedy decoding; top_p is ignored)", flush=True)
        else:
            if "temperature" not in sampling_kwargs:
                print(f"    temperature: (not provided; using the vLLM default)", flush=True)
            if "top_p" not in sampling_kwargs:
                print(f"    top_p: (not provided; using the vLLM default)", flush=True)
        print(flush=True)
        _vllm_generate_kwargs_logged = True

    from vllm import SamplingParams

    sampling_params = SamplingParams(**sampling_kwargs)
    outputs = llm.generate(texts, sampling_params=sampling_params, use_tqdm=False)

    results = []
    for request_output in outputs:
        if not request_output.outputs:
            message = Message(role=Role.assistant, content="")
            if return_token_counts:
                results.append({
                    "message": message,
                    "thinking_tokens": 0,
                    "response_tokens": 0,
                    "token_count_source": "vllm_generated_ids",
                    "finish_reason": None,
                    "stop_reason": None,
                    "hit_length_limit": False,
                })
            else:
                results.append(message)
            continue

        completion = request_output.outputs[0]
        output_ids = list(completion.token_ids)
        trimmed_output_ids = _trim_trailing_stop_token_ids(tokenizer, output_ids)
        finish_reason = completion.finish_reason
        stop_reason = completion.stop_reason
        hit_length_limit = finish_reason == "length"

        if enable_thinking:
            try:
                index = (
                    len(trimmed_output_ids)
                    - trimmed_output_ids[::-1].index(151668)
                )
            except ValueError:
                index = 0

            if extract_thinking:
                thinking_content = tokenizer.decode(
                    trimmed_output_ids[:index],
                    skip_special_tokens=True,
                ).strip("\n")
                response_text = tokenizer.decode(
                    trimmed_output_ids[index:],
                    skip_special_tokens=True,
                ).strip("\n")
                message = Message(role=Role.assistant, content=response_text)
                if return_token_counts:
                    results.append({
                        "message": message,
                        "thinking": thinking_content,
                        "thinking_tokens": index,
                        "response_tokens": len(trimmed_output_ids[index:]),
                        "token_count_source": "vllm_generated_ids",
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "hit_length_limit": hit_length_limit,
                    })
                else:
                    results.append({"message": message, "thinking": thinking_content})
            else:
                response_text = tokenizer.decode(
                    trimmed_output_ids[index:],
                    skip_special_tokens=True,
                ).strip("\n")
                message = Message(role=Role.assistant, content=response_text)

                if not return_probs:
                    if return_token_counts:
                        results.append({
                            "message": message,
                            "thinking_tokens": index,
                            "response_tokens": len(trimmed_output_ids[index:]),
                            "token_count_source": "vllm_generated_ids",
                            "finish_reason": finish_reason,
                            "stop_reason": stop_reason,
                            "hit_length_limit": hit_length_limit,
                        })
                    else:
                        results.append(message)
                else:
                    token_probs: t.List[TokenProbInfo] = []
                    completion_logprobs = completion.logprobs
                    for j in range(index, len(output_ids)):
                        if j >= len(trimmed_output_ids):
                            break
                        token_id = output_ids[j]
                        if token_id in tokenizer.all_special_ids:
                            continue
                        position_logprobs = (
                            completion_logprobs[j]
                            if completion_logprobs is not None
                            and j < len(completion_logprobs)
                            else None
                        )
                        token_probability = _extract_vllm_token_probability(
                            position_logprobs=position_logprobs,
                            token_id=token_id,
                        )
                        if token_probability is None:
                            continue
                        token_probs.append({
                            "token": tokenizer.decode(token_id),
                            "token_id": token_id,
                            "probability": token_probability,
                        })

                    result: ResponseWithTokenCounts = {
                        "message": message,
                        "token_probs": token_probs,
                    }
                    if return_token_counts:
                        result["thinking_tokens"] = index
                        result["response_tokens"] = len(trimmed_output_ids[index:])
                        result["token_count_source"] = "vllm_generated_ids"
                        result["finish_reason"] = finish_reason
                        result["stop_reason"] = stop_reason
                        result["hit_length_limit"] = hit_length_limit
                    results.append(result)
        else:
            response_text = tokenizer.decode(trimmed_output_ids, skip_special_tokens=True)
            message = Message(role=Role.assistant, content=response_text)

            if not return_probs:
                if return_token_counts:
                    results.append({
                        "message": message,
                        "thinking_tokens": 0,
                        "response_tokens": len(trimmed_output_ids),
                        "token_count_source": "vllm_generated_ids",
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "hit_length_limit": hit_length_limit,
                    })
                else:
                    results.append(message)
            else:
                token_probs: t.List[TokenProbInfo] = []
                completion_logprobs = completion.logprobs
                for j, token_id in enumerate(output_ids):
                    if j >= len(trimmed_output_ids):
                        break
                    if token_id in tokenizer.all_special_ids:
                        continue
                    position_logprobs = (
                        completion_logprobs[j]
                        if completion_logprobs is not None and j < len(completion_logprobs)
                        else None
                    )
                    token_probability = _extract_vllm_token_probability(
                        position_logprobs=position_logprobs,
                        token_id=token_id,
                    )
                    if token_probability is None:
                        continue
                    token_probs.append({
                        "token": tokenizer.decode(token_id),
                        "token_id": token_id,
                        "probability": token_probability,
                    })

                result = {
                    "message": message,
                    "token_probs": token_probs,
                }
                if return_token_counts:
                    result["thinking_tokens"] = 0
                    result["response_tokens"] = len(trimmed_output_ids)
                    result["token_count_source"] = "vllm_generated_ids"
                    result["finish_reason"] = finish_reason
                    result["stop_reason"] = stop_reason
                    result["hit_length_limit"] = hit_length_limit
                results.append(result)

    if is_single:
        return results[0]
    return results


def chat_qwen3_thinking(
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    parameters: Parameters,
    enable_thinking: bool = False,
    extract_thinking: bool = False,
    gpu_ids: t.Optional[str] = None,
    return_probs: bool = False,
    return_token_counts: bool = False,
    
    temperature: t.Optional[float] = None,
    max_new_tokens: t.Optional[int] = None,
    do_sample: t.Optional[bool] = None,
    top_p: t.Optional[float] = None,
) -> t.Union[Message, ResponseWithProbs, ResponseWithThinking, t.List[Message], t.List[ResponseWithProbs], t.List[ResponseWithThinking]]:
    return _chat_local(
        parameters.model, messages, parameters,
        enable_thinking=enable_thinking,
        extract_thinking=extract_thinking,
        gpu_ids=gpu_ids, return_probs=return_probs,
        return_token_counts=return_token_counts,
        temperature=temperature, max_new_tokens=max_new_tokens,
        do_sample=do_sample, top_p=top_p,
    )


def chat_local(
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    parameters: Parameters,
    gpu_ids: t.Optional[str] = None,
    return_probs: bool = False,
    return_token_counts: bool = False,
    
    temperature: t.Optional[float] = None,
    max_new_tokens: t.Optional[int] = None,
    do_sample: t.Optional[bool] = None,
    top_p: t.Optional[float] = None,
) -> t.Union[Message, ResponseWithProbs, t.List[Message], t.List[ResponseWithProbs]]:
    return _chat_local(
        parameters.model, messages, parameters,
        gpu_ids=gpu_ids, return_probs=return_probs,
        return_token_counts=return_token_counts,
        temperature=temperature, max_new_tokens=max_new_tokens,
        do_sample=do_sample, top_p=top_p,
    )


def chat_qwen3_thinking_vllm(
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    parameters: Parameters,
    enable_thinking: bool = False,
    extract_thinking: bool = False,
    gpu_ids: t.Optional[str] = None,
    return_probs: bool = False,
    return_token_counts: bool = False,
    vllm_gpu_memory_utilization: t.Optional[float] = None,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
    temperature: t.Optional[float] = None,
    max_new_tokens: t.Optional[int] = None,
    do_sample: t.Optional[bool] = None,
    top_p: t.Optional[float] = None,
) -> t.Union[
    Message,
    ResponseWithProbs,
    ResponseWithThinking,
    t.List[Message],
    t.List[ResponseWithProbs],
    t.List[ResponseWithThinking],
]:
    return _chat_local_vllm(
        parameters.model,
        messages,
        parameters,
        enable_thinking=enable_thinking,
        extract_thinking=extract_thinking,
        gpu_ids=gpu_ids,
        return_probs=return_probs,
        return_token_counts=return_token_counts,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
    )


def chat_local_vllm(
    messages: t.Union[t.List[Message], t.List[t.List[Message]]],
    parameters: Parameters,
    gpu_ids: t.Optional[str] = None,
    return_probs: bool = False,
    return_token_counts: bool = False,
    vllm_gpu_memory_utilization: t.Optional[float] = None,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
    temperature: t.Optional[float] = None,
    max_new_tokens: t.Optional[int] = None,
    do_sample: t.Optional[bool] = None,
    top_p: t.Optional[float] = None,
) -> t.Union[Message, ResponseWithProbs, t.List[Message], t.List[ResponseWithProbs]]:
    return _chat_local_vllm(
        parameters.model,
        messages,
        parameters,
        gpu_ids=gpu_ids,
        return_probs=return_probs,
        return_token_counts=return_token_counts,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
    )


def _get_sdk_response_field(obj: t.Any, key: str) -> t.Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)

    value = getattr(obj, key, None)
    if value is not None:
        return value

    for extra_attr in ("model_extra", "__pydantic_extra__"):
        extra = getattr(obj, extra_attr, None)
        if isinstance(extra, dict) and key in extra:
            return extra[key]

    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return dumped.get(key)

    return None


def _stringify_reasoning_value(value: t.Any) -> t.Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if hasattr(value, "model_dump"):
        try:
            return _stringify_reasoning_value(value.model_dump())
        except Exception:
            pass
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "summary",
            "reasoning",
            "reasoning_content",
            "reasoning_text",
        ):
            text = _stringify_reasoning_value(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            text = _stringify_reasoning_value(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip() or None

    text = str(value).strip()
    return text or None


def _extract_openrouter_reasoning_content(
    response_message: t.Any,
) -> t.Optional[str]:
    for key in ("reasoning", "reasoning_content", "reasoning_text"):
        text = _stringify_reasoning_value(
            _get_sdk_response_field(response_message, key)
        )
        if text:
            return text

    return _stringify_reasoning_value(
        _get_sdk_response_field(response_message, "reasoning_details")
    )


def _coerce_optional_int(value: t.Any) -> t.Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_api_reasoning_token_counts(
    response: t.Any,
) -> t.Tuple[t.Optional[int], t.Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None

    completion_tokens = _coerce_optional_int(
        _get_sdk_response_field(usage, "completion_tokens")
    )
    completion_details = _get_sdk_response_field(
        usage,
        "completion_tokens_details",
    )
    reasoning_tokens = None
    if completion_details is not None:
        reasoning_tokens = _coerce_optional_int(
            _get_sdk_response_field(completion_details, "reasoning_tokens")
        )
    if reasoning_tokens is None:
        reasoning_tokens = _coerce_optional_int(
            _get_sdk_response_field(usage, "reasoning_tokens")
        )

    response_tokens = None
    if completion_tokens is not None:
        if reasoning_tokens is not None:
            response_tokens = max(completion_tokens - reasoning_tokens, 0)
        else:
            response_tokens = completion_tokens

    return reasoning_tokens, response_tokens


def _chat_openai(
    messages: t.List[Message],
    parameters: Parameters,
    base_url: t.Optional[str] = None,
    api_key: t.Optional[str] = None,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
    return_probs: bool = False,
    return_reasoning: bool = False,
) -> t.Union[Message, ResponseWithProbs, ResponseWithTokenCounts]:
    client_kwargs = {}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)
    
    
    request_kwargs: t.Dict[str, t.Any] = {
        "model": parameters.model,
        "messages": t.cast(t.List[ChatCompletionMessageParam], messages),
    }
    if parameters.temperature is not None:
        request_kwargs["temperature"] = parameters.temperature
    if parameters.max_tokens is not None:
        request_kwargs["max_tokens"] = parameters.max_tokens
    if parameters.top_p is not None:
        request_kwargs["top_p"] = parameters.top_p
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body
    
    
    if return_probs:
        request_kwargs["logprobs"] = True
        request_kwargs["top_logprobs"] = 20  
    
    
    global _api_request_kwargs_logged
    with _api_request_kwargs_log_lock:
        if not _api_request_kwargs_logged:
            print(f"\n[API-model inference parameters] (first call)", flush=True)
            print(f"  model: {parameters.model}", flush=True)
            if base_url is not None:
                print(f"  base_url: {base_url}", flush=True)
            print(f"  parameters passed to API:", flush=True)
            for k, v in request_kwargs.items():
                if k == "messages":
                    print(f"    {k}: (message list, {len(v)} items)", flush=True)
                elif k == "extra_body":
                    print(f"    {k}: {v}", flush=True)
                else:
                    print(f"    {k}: {v}", flush=True)
            
            if "temperature" not in request_kwargs:
                print(f"    temperature: (not provided; using the model default)", flush=True)
            if "max_tokens" not in request_kwargs:
                print(f"    max_tokens: (not provided; using the model default)", flush=True)
            if "top_p" not in request_kwargs:
                print(f"    top_p: (not provided; using the model default)", flush=True)
            print(flush=True)
            _api_request_kwargs_logged = True
    
    response = client.chat.completions.create(**request_kwargs)

    response_message = response.choices[0].message
    message = Message(
        role=Role(response_message.role),
        content="" if response_message.content is None else str(response_message.content),
    )
    reasoning_content = (
        _extract_openrouter_reasoning_content(response_message)
        if return_reasoning
        else None
    )
    thinking_tokens, response_tokens = (
        _extract_api_reasoning_token_counts(response)
        if return_reasoning
        else (None, None)
    )
    
    if not return_probs:
        if return_reasoning:
            result: ResponseWithTokenCounts = {"message": message}
            if reasoning_content:
                result["thinking"] = reasoning_content
            if thinking_tokens is not None:
                result["thinking_tokens"] = thinking_tokens
            if response_tokens is not None:
                result["response_tokens"] = response_tokens
            if thinking_tokens is not None or response_tokens is not None:
                result["token_count_source"] = "api_usage"
            return result
        return message
    else:
        
        token_probs: t.List[TokenProbInfo] = []
        logprobs_data = response.choices[0].logprobs
        
        if logprobs_data and logprobs_data.content:
            for idx, token_data in enumerate(logprobs_data.content):
                
                probability = math.exp(token_data.logprob)
                
                token_probs.append({
                    "token": token_data.token,
                    "token_id": idx,  
                    "probability": probability,
                })
        
        result: ResponseWithTokenCounts = {
            "message": message,
            "token_probs": token_probs,
        }
        if return_reasoning and reasoning_content:
            result["thinking"] = reasoning_content
        if return_reasoning and thinking_tokens is not None:
            result["thinking_tokens"] = thinking_tokens
        if return_reasoning and response_tokens is not None:
            result["response_tokens"] = response_tokens
        if return_reasoning and (
            thinking_tokens is not None or response_tokens is not None
        ):
            result["token_count_source"] = "api_usage"
        return result


def chat_openai(
    messages: t.List[Message],
    parameters: Parameters,
    base_url: t.Optional[str] = None,
    api_key: t.Optional[str] = None,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
    return_probs: bool = False,
) -> t.Union[Message, ResponseWithProbs]:
    return _chat_openai(messages, parameters, base_url=base_url, api_key=api_key, extra_body=extra_body, return_probs=return_probs)


def chat_mistral(
    messages: t.List[Message], parameters: Parameters
) -> Message:
    client = MistralClient()
    messages = [
        ChatMessage(role=message.role, content=message.content) for message in messages
    ]

    response = client.chat(
        model=parameters.model,
        messages=messages,
        temperature=parameters.temperature,
        max_tokens=parameters.max_tokens,
        top_p=parameters.top_p,
    )
    response_message = response.choices[-1].message
    return Message(role=response_message.role, content=response_message.content)

def embed_mistral(contents: t.List[str]) -> t.List[t.List[float]]:
    client = MistralClient()
    response = client.embeddings('mistral-embed', contents)
    return [d.embedding for d in response.data]

def chat_together(messages: t.List[Message], parameters: Parameters) -> Message:
    return _chat_openai(
        messages, parameters,
        base_url="https://api.together.xyz/v1",
        api_key=_require_env_api_key("TOGETHER_API_KEY"),
    )

def chat_perplexity(messages: t.List[Message], parameters: Parameters) -> Message:
    return _chat_openai(
        messages, parameters,
        base_url="https://api.perplexity.ai",
        api_key=_require_env_api_key("PERPLEXITY_API_KEY"),
    )

def chat_modelscope(messages: t.List[Message], parameters: Parameters) -> Message:
    return _chat_openai(
        messages, parameters,
        base_url="https://api-inference.modelscope.cn/v1",
        api_key=_require_env_api_key("MODELSCOPE_API_KEY"),
    )

def chat_modelscope_thinking(
    messages: t.List[Message],
    parameters: Parameters,
    enable_thinking: bool = True,
) -> Message:
    return _chat_openai(
        messages, parameters,
        base_url="https://api-inference.modelscope.cn/v1",
        api_key=_require_env_api_key("MODELSCOPE_API_KEY"),
        extra_body={"enable_thinking": enable_thinking},
    )

def chat_siliconflow_thinking(
    messages: t.List[Message],
    parameters: Parameters,
    enable_thinking: bool = True,
) -> Message:
    return _chat_openai(
        messages, parameters,
        base_url="https://api.siliconflow.cn/v1",
        api_key=_require_env_api_key("SILICONFLOW_API_KEY"),
        extra_body={"enable_thinking": enable_thinking},
    )


def chat_bailian_qwen3(
    messages: t.List[Message],
    parameters: Parameters,
    enable_thinking: bool = False,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
    return_probs: bool = False,
) -> t.Union[Message, ResponseWithProbs]:
    request_extra_body = dict(extra_body) if extra_body is not None else {}
    request_extra_body["enable_thinking"] = enable_thinking

    return _chat_openai(
        messages,
        parameters,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=_require_env_api_key("DASHSCOPE_API_KEY"),
        extra_body=request_extra_body,
        return_probs=return_probs,
    )


OPENROUTER_GPT54_DEFAULT_REASONING_EFFORT = "medium"
OPENROUTER_GPT54_MINI_DEFAULT_REASONING_EFFORT = "none"
OPENROUTER_GEMINI31_FLASH_LITE_DEFAULT_REASONING_EFFORT = "minimal"
OPENROUTER_GPT5_MINI_DEFAULT_REASONING_EFFORT = "medium"


def _prepare_openrouter_reasoning_extra_body(
    extra_body: t.Optional[t.Dict[str, t.Any]],
    default_effort: str,
) -> t.Dict[str, t.Any]:
    request_extra_body = dict(extra_body) if extra_body is not None else {}
    raw_reasoning = request_extra_body.get("reasoning")

    if raw_reasoning is not None and not isinstance(raw_reasoning, dict):
        return request_extra_body

    request_reasoning = dict(raw_reasoning) if raw_reasoning is not None else {}
    has_explicit_reasoning = any(
        key in request_reasoning for key in ("effort", "max_tokens", "enabled")
    )
    if not has_explicit_reasoning:
        request_reasoning["effort"] = default_effort

    request_extra_body["reasoning"] = request_reasoning
    return request_extra_body


def chat_openrouter_gpt54(
    messages: t.List[Message],
    parameters: Parameters,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
    return_probs: bool = False,
) -> t.Union[Message, ResponseWithProbs]:
    request_extra_body = _prepare_openrouter_reasoning_extra_body(
        extra_body,
        default_effort=OPENROUTER_GPT54_DEFAULT_REASONING_EFFORT,
    )

    return _chat_openai(
        messages,
        parameters,
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_env_api_key("OPENROUTER_API_KEY"),
        extra_body=request_extra_body,
        return_probs=return_probs,
    )


def chat_openrouter_gpt54_mini(
    messages: t.List[Message],
    parameters: Parameters,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
    return_probs: bool = False,
) -> t.Union[Message, ResponseWithProbs]:
    request_extra_body = _prepare_openrouter_reasoning_extra_body(
        extra_body,
        default_effort=OPENROUTER_GPT54_MINI_DEFAULT_REASONING_EFFORT,
    )

    return _chat_openai(
        messages,
        parameters,
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_env_api_key("OPENROUTER_API_KEY"),
        extra_body=request_extra_body,
        return_probs=return_probs,
    )


def chat_openrouter_gemini31_flash_lite(
    messages: t.List[Message],
    parameters: Parameters,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
    return_probs: bool = False,
) -> t.Union[Message, ResponseWithProbs]:
    request_extra_body = _prepare_openrouter_reasoning_extra_body(
        extra_body,
        default_effort=OPENROUTER_GEMINI31_FLASH_LITE_DEFAULT_REASONING_EFFORT,
    )

    return _chat_openai(
        messages,
        parameters,
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_env_api_key("OPENROUTER_API_KEY"),
        extra_body=request_extra_body,
        return_probs=return_probs,
    )


def chat_openrouter_gpt5_mini(
    messages: t.List[Message],
    parameters: Parameters,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
    return_probs: bool = False,
) -> t.Union[Message, ResponseWithProbs, ResponseWithTokenCounts]:
    request_extra_body = _prepare_openrouter_reasoning_extra_body(
        extra_body,
        default_effort=OPENROUTER_GPT5_MINI_DEFAULT_REASONING_EFFORT,
    )
    request_extra_body.setdefault("include_reasoning", True)

    return _chat_openai(
        messages,
        parameters,
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_env_api_key("OPENROUTER_API_KEY"),
        extra_body=request_extra_body,
        return_probs=return_probs,
        return_reasoning=True,
    )


def chat_deepseek(
    messages: t.List[Message],
    parameters: Parameters,
    thinking: bool = False,
    return_probs: bool = False,
    extract_thinking: bool = False,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Union[Message, ResponseWithProbs, ResponseWithThinkingAndTokens]:
    request_extra_body = dict(extra_body) if extra_body is not None else {}
    reasoning_effort = request_extra_body.pop(
        "reasoning_effort",
        DEEPSEEK_V4_DEFAULT_REASONING_EFFORT,
    )
    request_extra_body["thinking"] = (
        {"type": "enabled"} if thinking else {"type": "disabled"}
    )

    
    client = OpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=_require_env_api_key("DEEPSEEK_API_KEY"),
    )
    
    request_kwargs: t.Dict[str, t.Any] = {
        "model": parameters.model,
        "messages": t.cast(t.List[ChatCompletionMessageParam], messages),
        "extra_body": request_extra_body,
    }
    if thinking:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if parameters.temperature is not None:
        request_kwargs["temperature"] = parameters.temperature
    if parameters.max_tokens is not None:
        request_kwargs["max_tokens"] = parameters.max_tokens
    if parameters.top_p is not None:
        request_kwargs["top_p"] = parameters.top_p
    
    
    if return_probs:
        if thinking:
            raise ValueError(
                "DeepSeek V4 thinking mode does not support logprobs / top_logprobs"
            )
        request_kwargs["logprobs"] = True
        request_kwargs["top_logprobs"] = 20
    
    
    global _deepseek_request_kwargs_logged
    if not _deepseek_request_kwargs_logged:
        print(f"\n[DeepSeek V4 API inference parameters] (first call)", flush=True)
        print(f"  model: {parameters.model}", flush=True)
        print(f"  base_url: {DEEPSEEK_BASE_URL}", flush=True)
        print(f"  parameters passed to API:", flush=True)
        for k, v in request_kwargs.items():
            if k == "messages":
                print(f"    {k}: (message list, {len(v)} items)", flush=True)
            else:
                print(f"    {k}: {v}", flush=True)
        
        if "temperature" not in request_kwargs:
            print(f"    temperature: (not provided; using the model default)", flush=True)
        if "max_tokens" not in request_kwargs:
            print(f"    max_tokens: (not provided; using the model default)", flush=True)
        if "top_p" not in request_kwargs:
            print(f"    top_p: (not provided; using the model default)", flush=True)
        print(flush=True)
        _deepseek_request_kwargs_logged = True
    
    response = client.chat.completions.create(**request_kwargs)
    
    response_message = response.choices[0].message
    message = Message(
        role=Role(response_message.role),
        content="" if response_message.content is None else str(response_message.content),
    )
    
    
    if thinking and extract_thinking:
        
        reasoning_content = getattr(response_message, 'reasoning_content', None) or ""
        
        
        thinking_tokens = None
        response_tokens = None
        if response.usage:
            
            completion_details = getattr(response.usage, 'completion_tokens_details', None)
            if completion_details:
                thinking_tokens = getattr(completion_details, 'reasoning_tokens', None)
            
            
            if thinking_tokens is not None and response.usage.completion_tokens:
                response_tokens = response.usage.completion_tokens - thinking_tokens
        
        return {
            "message": message,
            "thinking": reasoning_content,
            "thinking_tokens": thinking_tokens,
            "response_tokens": response_tokens,
            "token_count_source": (
                "api_usage" if response_tokens is not None else None
            ),
        }
    
    
    if not return_probs:
        return message
    else:
        
        token_probs: t.List[TokenProbInfo] = []
        logprobs_data = response.choices[0].logprobs
        
        if logprobs_data and logprobs_data.content:
            for idx, token_data in enumerate(logprobs_data.content):
                probability = math.exp(token_data.logprob)
                token_probs.append({
                    "token": token_data.token,
                    "token_id": idx,
                    "probability": probability,
                })
        
        return {"message": message, "token_probs": token_probs}


def chat_deepseek_reasoner(
    messages: t.List[Message],
    parameters: Parameters,
    return_probs: bool = False,
    extra_body: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Union[Message, ResponseWithProbs]:
    return chat_deepseek(
        messages,
        parameters,
        thinking=True,
        return_probs=return_probs,
        extract_thinking=False,
        extra_body=extra_body,
    )




Models: t.Dict[str, t.Tuple] = {
    
    "gpt-3.5": (chat_openai, "gpt-3.5-turbo-0125", True),
    "gpt-4": (chat_openai, "gpt-4", True),
    "gpt-4-turbo": (chat_openai, "gpt-4-1106-preview", True),
    "gpt-5.4": (chat_openrouter_gpt54, "openai/gpt-5.4", True),
    "gpt-5.4-mini": (chat_openrouter_gpt54_mini, "openai/gpt-5.4-mini", True),
    "gemini-3.1-flash-lite": (chat_openrouter_gemini31_flash_lite, "google/gemini-3.1-flash-lite", True),
    "google/gemini-3.1-flash-lite": (chat_openrouter_gemini31_flash_lite, "google/gemini-3.1-flash-lite", True),  
    "gpt-5-mini": (chat_openrouter_gpt5_mini, "openai/gpt-5-mini", True),
    "openai/gpt-5-mini": (chat_openrouter_gpt5_mini, "openai/gpt-5-mini", True),
    "sonar-small-online": (chat_perplexity, "sonar-small-online", True),
    "sonar-medium-online": (chat_perplexity, "sonar-medium-online", True),
    "llama3-sonar-large-online": (chat_perplexity, "llama-3-sonar-large-32k-online", True),
    "llama3-8b": (chat_together, "meta-llama/llama-3-8b-chat-hf", True),
    "llama3-70b": (chat_together, "meta-llama/llama-3-70b-chat-hf", True),
    "vicuna-13b": (chat_together, "lmsys/vicuna-13b-v1.5", True),
    "mixtral-8x22": (chat_together, "mistralai/Mixtral-8x22B-Instruct-v0.1", True),
    "mistral-small-together": (chat_together, "mistralai/Mixtral-8x7B-Instruct-v0.1", True),
    "mistral-small": (chat_mistral, "mistral-small", True),
    "mistral-medium": (chat_mistral, "mistral-medium", True),
    "deepseek-v3.1": (chat_modelscope, "deepseek-ai/DeepSeek-V3.1", True),
    "deepseek-v3.2-ms": (chat_modelscope_thinking, "deepseek-ai/DeepSeek-V3.2", True),
    "deepseek-v3.2-sf": (chat_siliconflow_thinking, "deepseek-ai/DeepSeek-V3.2", True),
    "deepseek-v4-flash": (chat_deepseek, "deepseek-v4-flash", True),
    "deepseek-v4-pro": (chat_deepseek, "deepseek-v4-pro", True),
    
    "deepseek-chat": (chat_deepseek, "deepseek-v4-flash", True),
    "deepseek-reasoner": (chat_deepseek_reasoner, "deepseek-v4-flash", True),
    "qwen3-32b-bailian": (chat_bailian_qwen3, "qwen3-32b", True),
    
    "qwen3-1.7b": (chat_qwen3_thinking, _local_model_path("EVA_QWEN3_1_7B_PATH", "Qwen3-4B/Qwen/Qwen3-1___7B"), False),
    "qwen3-4b": (chat_qwen3_thinking, _local_model_path("EVA_QWEN3_4B_PATH", "Qwen3-4B/Qwen/Qwen3-4B"), False),
    "qwen3-8b": (chat_qwen3_thinking, _local_model_path("EVA_QWEN3_8B_PATH", "Qwen3/Qwen3-8B"), False),
    "qwen3.5-9b": (chat_qwen3_thinking, _local_model_path("EVA_QWEN3_5_9B_PATH", "Qwen3.5-9B"), False),
    "qwen3-14b": (chat_qwen3_thinking, _local_model_path("EVA_QWEN3_14B_PATH", "Qwen3/Qwen3-14B"), False),
    "qwen3-32b": (chat_qwen3_thinking, _local_model_path("EVA_QWEN3_32B_PATH", "Qwen3/Qwen/Qwen3-32B"), False),
    "qwen3-30b-a3b": (chat_local, _local_model_path("EVA_QWEN3_30B_A3B_PATH", "Qwen3/Qwen3-30B-A3B-Instruct-2507"), False),
    "gemma3-12b": (chat_local, _local_model_path("EVA_GEMMA3_12B_PATH", "Gemma3/gemma-3-12b-it"), False),
    "llama3-8b-instruct": (chat_local, _local_model_path("EVA_LLAMA3_8B_INSTRUCT_PATH", "Meta-Llama-3-8B-Instruct"), False),
    "llama3.1-8b": (chat_local, _local_model_path("EVA_LLAMA3_1_8B_PATH", "Llama/Llama3.1-8B-Instruct"), False),
}


def is_remote_model(model_name: str) -> bool:
    if model_name not in Models:
        raise ValueError(f"Unknown model: {model_name}")
    return Models[model_name][2]


def _resolve_remote_provider_config(
    chat_func: t.Callable,
    base_url: t.Optional[str],
    api_key: t.Optional[str],
) -> t.Tuple[t.Optional[str], t.Optional[str]]:
    if chat_func in (chat_openai,):
        return base_url, api_key or _require_env_api_key("OPENAI_API_KEY")
    if chat_func == chat_together:
        return (
            base_url or "https://api.together.xyz/v1",
            api_key or _require_env_api_key("TOGETHER_API_KEY"),
        )
    if chat_func == chat_perplexity:
        return (
            base_url or "https://api.perplexity.ai",
            api_key or _require_env_api_key("PERPLEXITY_API_KEY"),
        )
    if chat_func in (chat_modelscope, chat_modelscope_thinking):
        return (
            base_url or "https://api-inference.modelscope.cn/v1",
            api_key or _require_env_api_key("MODELSCOPE_API_KEY"),
        )
    if chat_func == chat_siliconflow_thinking:
        return (
            base_url or "https://api.siliconflow.cn/v1",
            api_key or _require_env_api_key("SILICONFLOW_API_KEY"),
        )
    if chat_func == chat_bailian_qwen3:
        return (
            base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key or _require_env_api_key("DASHSCOPE_API_KEY"),
        )
    if chat_func in (
        chat_openrouter_gpt54,
        chat_openrouter_gpt54_mini,
        chat_openrouter_gemini31_flash_lite,
        chat_openrouter_gpt5_mini,
    ):
        return (
            base_url or "https://openrouter.ai/api/v1",
            api_key or _require_env_api_key("OPENROUTER_API_KEY"),
        )
    return base_url, api_key


def load_model(
        model: str,
        temperature: t.Optional[float] = None,
        top_p: t.Optional[float] = None,
        max_tokens: t.Optional[int] = None,
        gpu_ids: t.Optional[str] = None,
        return_probs: bool = False,
        return_token_counts: bool = False,
        enable_thinking: bool = False,
        extract_thinking: bool = False,
        base_url: t.Optional[str] = None,
        api_key: t.Optional[str] = None,
        extra_body: t.Optional[t.Dict[str, t.Any]] = None,
        request_delay: float = 0.0,
        local_inference_backend: str = "vllm",
        vllm_gpu_memory_utilization: t.Optional[float] = None,
        vllm_max_model_len: t.Optional[int] = None,
        vllm_max_num_seqs: t.Optional[int] = None,
        vllm_max_num_batched_tokens: t.Optional[int] = None,
) -> t.Union[
    ChatFunction,
    t.Callable[[t.List[Message]], ResponseWithProbs],
    t.Callable[[t.List[Message]], ResponseWithThinking],
    t.Callable[[t.List[Message]], ResponseWithTokenCounts],
]:
    if local_inference_backend != "vllm":
        raise ValueError("local_inference_backend only supports 'vllm'")
    vllm_gpu_memory_utilization = _resolve_vllm_gpu_memory_utilization(
        vllm_gpu_memory_utilization
    )

    chat_func, model_name, is_remote = Models[model]
    
    parameters = Parameters(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    
    if is_remote:
        base_url, api_key = _resolve_remote_provider_config(
            chat_func,
            base_url,
            api_key,
        )
        return _create_batch_wrapper(
            model=model,
            chat_func=chat_func,
            parameters=parameters,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body,
            return_probs=return_probs,
            return_token_counts=return_token_counts,
            request_delay=request_delay,
            enable_thinking=enable_thinking,
            extract_thinking=extract_thinking,
        )
    
    if local_inference_backend == "vllm":
        if chat_func == chat_qwen3_thinking:
            chat_func = chat_qwen3_thinking_vllm
        elif chat_func == chat_local:
            chat_func = chat_local_vllm
        elif not is_remote:
            raise ValueError(
                f"Local model {model} is not currently wired to the vLLM inference backend"
            )

    
    partial_kwargs = {"parameters": parameters}
    
    
    if chat_func in (
        chat_qwen3_thinking,
        chat_local,
        chat_qwen3_thinking_vllm,
        chat_local_vllm,
    ):
        if gpu_ids is not None:
            partial_kwargs["gpu_ids"] = gpu_ids
        if return_probs:
            partial_kwargs["return_probs"] = return_probs
        if return_token_counts:
            partial_kwargs["return_token_counts"] = return_token_counts
        if chat_func in (chat_qwen3_thinking_vllm, chat_local_vllm):
            partial_kwargs["vllm_gpu_memory_utilization"] = (
                vllm_gpu_memory_utilization
            )
            partial_kwargs["vllm_max_model_len"] = vllm_max_model_len
            partial_kwargs["vllm_max_num_seqs"] = vllm_max_num_seqs
            partial_kwargs["vllm_max_num_batched_tokens"] = (
                vllm_max_num_batched_tokens
            )
        
        if chat_func in (chat_qwen3_thinking, chat_qwen3_thinking_vllm):
            if enable_thinking:
                partial_kwargs["enable_thinking"] = enable_thinking
            if extract_thinking:
                partial_kwargs["extract_thinking"] = extract_thinking
    
    return t.cast(
        t.Union[ChatFunction, t.Callable[[t.List[Message]], ResponseWithProbs], t.Callable[[t.List[Message]], ResponseWithThinking]],
        functools.partial(chat_func, **partial_kwargs),
    )


def _create_batch_wrapper(
    model: str,
    chat_func: t.Callable,
    parameters: Parameters,
    base_url: t.Optional[str],
    api_key: t.Optional[str],
    extra_body: t.Optional[t.Dict[str, t.Any]],
    return_probs: bool,
    return_token_counts: bool,
    request_delay: float,
    enable_thinking: bool = False,
    extract_thinking: bool = False,
) -> t.Callable:
    
    
    if chat_func in (chat_deepseek, chat_deepseek_reasoner):
        
        thinking = chat_func == chat_deepseek_reasoner
        if extra_body is not None and "thinking" in extra_body:
            thinking = _resolve_deepseek_thinking_flag(extra_body["thinking"])
        
        
        
        if enable_thinking:
            thinking = True
        
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs, ResponseWithThinkingAndTokens]:
            return chat_deepseek(
                messages,
                parameters,
                thinking=thinking,
                return_probs=return_probs,
                extract_thinking=extract_thinking,
                extra_body=extra_body,
            )
    elif chat_func == chat_bailian_qwen3:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs]:
            return chat_bailian_qwen3(
                messages,
                parameters,
                enable_thinking=enable_thinking,
                extra_body=extra_body,
                return_probs=return_probs,
            )
    elif chat_func == chat_modelscope_thinking:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs]:
            request_extra_body = dict(extra_body) if extra_body is not None else {}
            request_extra_body.setdefault("enable_thinking", enable_thinking)
            return chat_openai(
                messages,
                parameters,
                base_url=base_url,
                api_key=api_key,
                extra_body=request_extra_body,
                return_probs=return_probs,
            )
    elif chat_func == chat_siliconflow_thinking:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs]:
            request_extra_body = dict(extra_body) if extra_body is not None else {}
            request_extra_body.setdefault("enable_thinking", enable_thinking)
            return chat_openai(
                messages,
                parameters,
                base_url=base_url,
                api_key=api_key,
                extra_body=request_extra_body,
                return_probs=return_probs,
            )
    elif chat_func == chat_openrouter_gpt54:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs]:
            return chat_openrouter_gpt54(
                messages,
                parameters,
                extra_body=extra_body,
                return_probs=return_probs,
            )
    elif chat_func == chat_openrouter_gpt54_mini:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs]:
            return chat_openrouter_gpt54_mini(
                messages,
                parameters,
                extra_body=extra_body,
                return_probs=return_probs,
            )
    elif chat_func == chat_openrouter_gemini31_flash_lite:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs]:
            return chat_openrouter_gemini31_flash_lite(
                messages,
                parameters,
                extra_body=extra_body,
                return_probs=return_probs,
            )
    elif chat_func == chat_openrouter_gpt5_mini:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs, ResponseWithTokenCounts]:
            return chat_openrouter_gpt5_mini(
                messages,
                parameters,
                extra_body=extra_body,
                return_probs=return_probs,
            )
    else:
        def single_call(messages: t.List[Message]) -> t.Union[Message, ResponseWithProbs]:
            return chat_openai(
                messages,
                parameters,
                base_url=base_url,
                api_key=api_key,
                extra_body=extra_body,
                return_probs=return_probs,
            )
    
    def wrapper(messages: t.Union[t.List[Message], t.List[t.List[Message]]]) -> t.Union[
        Message, ResponseWithProbs, ResponseWithThinkingAndTokens,
        t.List[Message], t.List[ResponseWithProbs], t.List[ResponseWithThinkingAndTokens]
    ]:
        
        is_single = len(messages) > 0 and isinstance(messages[0], Message)
        
        if is_single:
            
            return single_call(t.cast(t.List[Message], messages))
        else:
            
            batch_messages = t.cast(t.List[t.List[Message]], messages)
            results = [None] * len(batch_messages)  
            
            
            num_workers = len(batch_messages)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                
                future_to_index = {}
                for idx, msgs in enumerate(batch_messages):
                    
                    if request_delay > 0 and idx > 0:
                        time.sleep(request_delay)
                    future = executor.submit(single_call, msgs)
                    future_to_index[future] = idx
                
                
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        
                        raise e
            
            return results
    
    return wrapper


def load_model_async(
        model: str,
        temperature: t.Optional[float] = None,
        top_p: t.Optional[float] = None,
        max_tokens: t.Optional[int] = None,
        return_probs: bool = False,
        base_url: t.Optional[str] = None,
        api_key: t.Optional[str] = None,
        extra_body: t.Optional[t.Dict[str, t.Any]] = None,
        max_concurrent: int = 8,
        request_delay: float = 0.0,
    ) -> t.Callable[[t.List[t.List[Message]]], t.Coroutine]:
    if model not in Models:
        raise ValueError(f"Unknown model: {model}")
    
    chat_func, model_name, is_remote = Models[model]
    
    if not is_remote:
        raise ValueError(f"Model {model} does not support async calls; only remote API models are supported")
    
    base_url, api_key = _resolve_remote_provider_config(
        chat_func,
        base_url,
        api_key,
    )

    parameters = Parameters(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    if chat_func == chat_openrouter_gpt54:
        extra_body = _prepare_openrouter_reasoning_extra_body(
            extra_body,
            default_effort=OPENROUTER_GPT54_DEFAULT_REASONING_EFFORT,
        )
    elif chat_func == chat_openrouter_gpt54_mini:
        extra_body = _prepare_openrouter_reasoning_extra_body(
            extra_body,
            default_effort=OPENROUTER_GPT54_MINI_DEFAULT_REASONING_EFFORT,
        )
    elif chat_func == chat_openrouter_gpt5_mini:
        extra_body = _prepare_openrouter_reasoning_extra_body(
            extra_body,
            default_effort=OPENROUTER_GPT5_MINI_DEFAULT_REASONING_EFFORT,
        )

    sync_batch = _create_batch_wrapper(
        model=model,
        chat_func=chat_func,
        parameters=parameters,
        base_url=base_url,
        api_key=api_key,
        extra_body=extra_body,
        return_probs=return_probs,
        return_token_counts=False,
        request_delay=request_delay,
    )

    async def async_batch(
        messages: t.List[t.List[Message]],
    ) -> t.Union[
        t.List[Message],
        t.List[ResponseWithProbs],
        t.List[ResponseWithThinkingAndTokens],
    ]:
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def call_one(index: int, item: t.List[Message]) -> t.Any:
            if request_delay > 0 and index > 0:
                await asyncio.sleep(request_delay)
            async with semaphore:
                return await asyncio.to_thread(sync_batch, item)

        return await asyncio.gather(
            *(call_one(index, item) for index, item in enumerate(messages))
        )

    return async_batch

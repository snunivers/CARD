
import argparse
import functools
import json
import logging
import os
import time
import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_LOG_FLOAT_DIGIT_VARIANTS = (4, 8)
SUMMARY_LOG_ROUNDING_NOTE = (
    "Note: decimal places in logs are formatted only at output time; raw statistics are not pre- round."
    "Python float formatting in tie uses in tie cases round-half-even (banker rounding); "
    "actual results still use binary floating-point values."
)


def get_minute_level_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def format_timed_title(title: str) -> str:
    return f"[{get_minute_level_timestamp()}] {title}"


def print_nohup_parse_status(category: str, message: str) -> None:
    print(
        f"{format_timed_title('[Parse status]')} Category {category} {message}",
        flush=True,
    )


def print_nohup_api_status(label: str, message: str) -> None:
    print(
        f"{format_timed_title('[API status]')} {label} {message}",
        flush=True,
    )


def print_nohup_api_retry_status(label: str, message: str) -> None:
    print(
        f"{format_timed_title('[API retry]')} {label} {message}",
        flush=True,
    )


def json_safe_value(value: t.Any) -> t.Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe_value(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return json_safe_value(value.dict())
        except Exception:
            pass
    return str(value)


def namespace_to_json_safe_dict(args: argparse.Namespace) -> t.Dict[str, t.Any]:
    return {
        key: json_safe_value(value)
        for key, value in vars(args).items()
    }


def get_run_config_path(args: argparse.Namespace) -> str:
    return out_path(args, "run_config.json")


def get_api_retry_work_dir(args: argparse.Namespace) -> str:
    return out_path(args, "api_retry")


def get_api_failed_prompts_path(
    args: argparse.Namespace,
    suffix: str = "failed_prompts.jsonl",
) -> str:
    return out_path(args, "api_retry", suffix)


def get_api_recovered_prompts_path(
    args: argparse.Namespace,
    suffix: str = "recovered_prompts.jsonl",
) -> str:
    return out_path(args, "api_retry", suffix)


def get_category_results_dir(args: argparse.Namespace) -> str:
    return out_path(args, "category_results")


def get_category_result_path(args: argparse.Namespace, category: str) -> str:
    return out_path(
        args,
        "category_results",
        f"{sanitize_category_name(category)}.pkl",
    )


def append_jsonl_record(path: str, record: t.Dict[str, t.Any]) -> None:
    file_utils.ensure_created_directory(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe_value(record), ensure_ascii=False) + "\n")


def write_json_file(path: str, payload: t.Any) -> None:
    file_utils.ensure_created_directory(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe_value(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def serialize_message_list(messages: t.List[t.Any]) -> t.List[t.Dict[str, t.Any]]:
    return [
        {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }
        for message in messages
    ]


def message_content_from_response(response: t.Any) -> t.Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        message = response.get("message")
        content = getattr(message, "content", None) if message is not None else response.get("content")
        if content is None:
            return None
        if not isinstance(content, str):
            content = str(content)
        stripped = content.strip()
        if not stripped or stripped.lower() == "none":
            return None
        return content
    content = getattr(response, "content", None)
    if content is None and hasattr(response, "message"):
        message = getattr(response, "message")
        content = getattr(message, "content", None)
    if content is None:
        return None
    if not isinstance(content, str):
        content = str(content)
    stripped = content.strip()
    if not stripped or stripped.lower() == "none":
        return None
    return content


def extract_api_response_details(response: t.Any) -> t.Dict[str, t.Any]:
    details = {
        "response_text": None,
        "thinking_content": None,
        "thinking_tokens": None,
        "response_tokens": None,
        "token_count_source": None,
        "finish_reason": None,
        "stop_reason": None,
        "hit_length_limit": None,
    }
    if response is None:
        return details

    if isinstance(response, dict):
        message = response.get("message")
        if message is not None:
            details["response_text"] = getattr(message, "content", None)
        else:
            details["response_text"] = response.get("content")
        details["thinking_content"] = response.get("thinking")
        for key in (
            "thinking_tokens",
            "response_tokens",
            "token_count_source",
            "finish_reason",
            "stop_reason",
            "hit_length_limit",
        ):
            if key in response:
                details[key] = response.get(key)
    else:
        details["response_text"] = getattr(response, "content", None)
        if details["response_text"] is None and hasattr(response, "message"):
            message = getattr(response, "message")
            details["response_text"] = getattr(message, "content", None)
        details["thinking_content"] = getattr(response, "thinking", None)
        for key in (
            "thinking_tokens",
            "response_tokens",
            "token_count_source",
            "finish_reason",
            "stop_reason",
            "hit_length_limit",
        ):
            if hasattr(response, key):
                details[key] = getattr(response, key)

    response_text = details["response_text"]
    if response_text is not None and not isinstance(response_text, str):
        response_text = str(response_text)
    if isinstance(response_text, str):
        response_text = response_text.strip()
        if not response_text or response_text.lower() == "none":
            response_text = None
    details["response_text"] = response_text

    thinking_content = details["thinking_content"]
    if thinking_content is not None and not isinstance(thinking_content, str):
        thinking_content = str(thinking_content)
    if isinstance(thinking_content, str):
        thinking_content = thinking_content.strip()
        if not thinking_content:
            thinking_content = None
    details["thinking_content"] = thinking_content

    return details


def is_valid_api_response(response: t.Any) -> bool:
    return extract_api_response_details(response)["response_text"] is not None


def get_api_retry_delay_seconds(
    args: argparse.Namespace,
    retry_index: int,
) -> float:
    initial_delay = max(0.0, float(getattr(args, "api_retry_initial_delay", 10.0)))
    backoff = max(1.0, float(getattr(args, "api_retry_backoff", 2.0)))
    max_delay = max(0.0, float(getattr(args, "api_retry_max_delay", 120.0)))
    retry_step = max(1, int(retry_index))
    delay = initial_delay * (backoff ** (retry_step - 1))
    return min(delay, max_delay)


def build_api_retry_record(
    category: str,
    batch_label: str,
    run_record: t.Dict[str, t.Any],
    status: str,
    attempt_count: int,
    retry_origin: str,
    retry_round: t.Optional[int] = None,
    response_details: t.Optional[t.Dict[str, t.Any]] = None,
    error_type: t.Optional[str] = None,
    error_message: t.Optional[str] = None,
    target_model: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    record = {
        "category": category,
        "batch_label": batch_label,
        "status": status,
        "retry_origin": retry_origin,
        "retry_round": retry_round,
        "attempt_count": int(attempt_count),
        "target_model": target_model,
        "block_idx": int(run_record["block_idx"]),
        "run_idx": int(run_record["run_idx"]),
        "experiment_idx": int(run_record["experiment_idx"]),
        "target_message": run_record["target_message"],
        "messages": serialize_message_list(run_record["messages"]),
        "query": run_record["query"],
        "documents": run_record["documents"],
        "product_models": run_record["product_models"],
        "product_brands": run_record["product_brands"],
        "prompt_order": run_record["prompt_order"],
        "brand_to_doc_map": run_record["brand_to_doc_map"],
        "brand_to_position_map": run_record["brand_to_position_map"],
        "L2_b": run_record["L2_b"],
        "doc_assignment": run_record["doc_assignment"],
    }
    if response_details is not None:
        record.update(
            {
                "response_text": response_details.get("response_text"),
                "thinking_content": response_details.get("thinking_content"),
                "thinking_tokens": response_details.get("thinking_tokens"),
                "response_tokens": response_details.get("response_tokens"),
                "token_count_source": response_details.get("token_count_source"),
                "finish_reason": response_details.get("finish_reason"),
                "stop_reason": response_details.get("stop_reason"),
                "hit_length_limit": response_details.get("hit_length_limit"),
            }
        )
    if error_type is not None:
        record["error_type"] = error_type
    if error_message is not None:
        record["error_message"] = error_message
    return record


def retry_api_single_prompt(
    target: t.Callable,
    run_record: t.Dict[str, t.Any],
    category: str,
    batch_label: str,
    args: argparse.Namespace,
    retry_origin: str,
    retry_round: t.Optional[int] = None,
) -> t.Dict[str, t.Any]:
    logger = get_logger()
    max_retries = max(0, int(getattr(args, "api_max_retries", 5)))
    max_attempts = max_retries + 1
    prompt_label = (
        f"{batch_label} | B{int(run_record['block_idx']) + 1}"
        f"/R{int(run_record['run_idx']) + 1}"
    )
    last_error_type = None
    last_error_message = None
    total_elapsed = 0.0

    for attempt_index in range(1, max_attempts + 1):
        attempt_label = f"{prompt_label} | attempt {attempt_index}/{max_attempts}"
        print_nohup_api_status("request", attempt_label)
        logger.info(f"[API request] {attempt_label}")
        attempt_start = time.perf_counter()
        try:
            response = target(run_record["messages"])
            response_details = extract_api_response_details(response)
            attempt_elapsed = time.perf_counter() - attempt_start
            total_elapsed += attempt_elapsed
            if response_details["response_text"] is not None:
                print_nohup_api_status(
                    "success",
                    f"{attempt_label}, elapsed {attempt_elapsed:.4f} s",
                )
                logger.info(
                    f"[API success] {attempt_label}, elapsed {attempt_elapsed:.4f} s"
                )
                recovered = attempt_index > 1
                recovered_record = None
                if recovered:
                    recovered_record = build_api_retry_record(
                        category=category,
                        batch_label=batch_label,
                        run_record=run_record,
                        status="recovered",
                        attempt_count=attempt_index,
                        retry_origin=retry_origin,
                        retry_round=retry_round,
                        response_details=response_details,
                        target_model=getattr(args, "target_model", None),
                    )
                return {
                    "status": "success",
                    "attempt_count": attempt_index,
                    "total_elapsed": total_elapsed,
                    "response": response,
                    "response_details": response_details,
                    "recovered": recovered,
                    "recovered_record": recovered_record,
                    "failure_record": None,
                    "last_error_type": last_error_type,
                    "last_error_message": last_error_message,
                }

            last_error_type = "EmptyResponse"
            last_error_message = "API response content is empty"
        except Exception as exc:
            attempt_elapsed = time.perf_counter() - attempt_start
            total_elapsed += attempt_elapsed
            last_error_type = type(exc).__name__
            last_error_message = str(exc)
        print_nohup_api_retry_status(
            "failure",
            f"{attempt_label}: {last_error_type}: {last_error_message}, "
            f"elapsed {attempt_elapsed:.4f} s",
        )
        logger.info(
            f"[API failure] {attempt_label}: {last_error_type}: {last_error_message}, "
            f"elapsed {attempt_elapsed:.4f} s"
        )
        if attempt_index < max_attempts:
            delay_seconds = get_api_retry_delay_seconds(args, attempt_index)
            print_nohup_api_retry_status(
                "wait",
                f"{attempt_label}, {delay_seconds:.1f} seconds before retry",
            )
            logger.info(
                f"[API retry wait] {attempt_label}, {delay_seconds:.1f} seconds before retry"
            )
            total_elapsed += delay_seconds
            time.sleep(delay_seconds)

    failure_record = build_api_retry_record(
        category=category,
        batch_label=batch_label,
        run_record=run_record,
        status="failed",
        attempt_count=max_attempts,
        retry_origin=retry_origin,
        retry_round=retry_round,
        error_type=last_error_type,
        error_message=last_error_message,
        target_model=getattr(args, "target_model", None),
    )
    print_nohup_api_status("failure", f"{prompt_label} reached maximum retry count, writing to failure ledger")
    logger.info(f"[API failure] {prompt_label} reached maximum retry count, writing to failure ledger")
    return {
        "status": "failed",
        "attempt_count": max_attempts,
        "total_elapsed": total_elapsed,
        "response": None,
        "response_details": None,
        "recovered": False,
        "recovered_record": None,
        "failure_record": failure_record,
        "last_error_type": last_error_type,
        "last_error_message": last_error_message,
    }


def save_run_config_metadata(args: argparse.Namespace) -> str:
    run_config_path = get_run_config_path(args)
    script_name = getattr(args, "runtime_script_name", Path(__file__).name)
    payload = {
        "script": script_name,
        "args": namespace_to_json_safe_dict(args),
    }
    write_json_file(run_config_path, payload)
    return run_config_path


def load_run_config_metadata(path: str) -> t.Dict[str, t.Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_api_retry_summary(
    results: t.Dict[str, t.Dict[str, t.Any]]
) -> t.Dict[str, t.Any]:
    categories = []
    total_success = 0
    total_failed = 0
    total_recovered = 0
    total_attempts = 0
    total_final_failed = 0
    any_incomplete = False
    for category_name, category_result in results.items():
        api_retry_stats = category_result.get("api_retry_stats")
        if not isinstance(api_retry_stats, dict):
            continue
        categories.append(category_name)
        success_count = int(api_retry_stats.get("success_response_count", 0))
        failed_count = int(api_retry_stats.get("final_failed_prompt_count", 0))
        recovered_count = int(api_retry_stats.get("recovered_prompt_count", 0))
        attempt_count = int(api_retry_stats.get("total_attempt_count", 0))
        total_success += success_count
        total_failed += failed_count
        total_recovered += recovered_count
        total_attempts += attempt_count
        total_final_failed += failed_count
        any_incomplete = any_incomplete or bool(api_retry_stats.get("incomplete", False))

    return {
        "category_count": len(categories),
        "categories": categories,
        "total_success_response_count": total_success,
        "total_failed_prompt_count": total_failed,
        "total_recovered_prompt_count": total_recovered,
        "total_attempt_count": total_attempts,
        "total_final_failed_prompt_count": total_final_failed,
        "incomplete": any_incomplete,
    }


def format_api_retry_summary_lines(
    results: t.Dict[str, t.Dict[str, t.Any]],
    args: argparse.Namespace,
) -> t.List[str]:
    summary = collect_api_retry_summary(results)
    lines = [
        "",
        "[API retry andfailure]",
        f"  Categories with API retry statistics: {summary['category_count']}",
        f"  Total successful responses: {summary['total_success_response_count']}",
        f"  Total retry attempts: {summary['total_attempt_count']}",
        f"  Total recovered prompts: {summary['total_recovered_prompt_count']}",
        f"  Total final failed prompts: {summary['total_final_failed_prompt_count']}",
        f"  Any unrecovered failures: {summary['incomplete']}",
        f"  API max retries: {getattr(args, 'api_max_retries', 'N/A')}",
        f"  API initial wait time: {getattr(args, 'api_retry_initial_delay', 'N/A')}",
        f"  API backoff factor: {getattr(args, 'api_retry_backoff', 'N/A')}",
        f"  API max wait time: {getattr(args, 'api_retry_max_delay', 'N/A')}",
    ]
    if summary["categories"]:
        lines.append(f"  Categories involved: {', '.join(summary['categories'])}")
    return lines


def log_api_retry_statistics_at_end(
    results: t.Dict[str, t.Dict[str, t.Any]],
    args: argparse.Namespace,
    logger,
) -> None:
    summary = collect_api_retry_summary(results)
    if summary["category_count"] == 0:
        return

    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info("[API retry statistics] (summarized by category, log-tail output)")
    logger.info(f"{'='*80}")
    logger.info(f"  Categories with API retry statistics: {summary['category_count']}")
    logger.info(f"  Total successful responses: {summary['total_success_response_count']}")
    logger.info(f"  Total retry attempts: {summary['total_attempt_count']}")
    logger.info(f"  Total recovered prompts: {summary['total_recovered_prompt_count']}")
    logger.info(f"  Total final failed prompts: {summary['total_final_failed_prompt_count']}")
    logger.info(f"  Any unrecovered failures: {summary['incomplete']}")
    logger.info(f"  API max retries: {getattr(args, 'api_max_retries', 'N/A')}")
    logger.info(f"  API initial wait time: {getattr(args, 'api_retry_initial_delay', 'N/A')}")
    logger.info(f"  API backoff factor: {getattr(args, 'api_retry_backoff', 'N/A')}")
    logger.info(f"  API max wait time: {getattr(args, 'api_retry_max_delay', 'N/A')}")

    for category_name, category_result in results.items():
        api_retry_stats = category_result.get("api_retry_stats")
        if not isinstance(api_retry_stats, dict):
            continue
        logger.info(f"")
        logger.info(f"  [Category] {category_name}")
        logger.info(
            f"    Successful responses: {int(api_retry_stats.get('success_response_count', 0))}"
        )
        logger.info(
            f"    Recovered prompts: {int(api_retry_stats.get('recovered_prompt_count', 0))}"
        )
        logger.info(
            f"    Final failed prompts: {int(api_retry_stats.get('final_failed_prompt_count', 0))}"
        )
        logger.info(
            f"    Total attempts: {int(api_retry_stats.get('total_attempt_count', 0))}"
        )
        logger.info(
            f"    Failure ledger: {api_retry_stats.get('failed_prompts_path', '(not recorded)')}"
        )
        logger.info(
            f"    Recovery ledger: {api_retry_stats.get('recovered_prompts_path', '(not recorded)')}"
        )
        logger.info(
            f"    Incomplete recovery: {bool(api_retry_stats.get('incomplete', False))}"
        )


def format_summary_log_float(value: t.Any, digits: int) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{float(value):.{digits}f}"


def format_brand_type_main_effect_f_lines(
    stats_result: t.Dict[str, t.Any],
    digits: int = 4,
) -> t.List[str]:
    lines = [
        "=" * 80,
        f"[parametric knowledge/non-parametric knowledge (two-group)main-effect F] [{digits}-decimal version]",
        "=" * 80,
    ]

    if not stats_result.get('has_comparison', False):
        lines.append(
            "  Insufficient data to computeparametric knowledge/non-parametric knowledgemain-effect F "
            f"(parametric knowledgebrand n={stats_result.get('parametric_n', 0)}, "
            f"non-parametric knowledgebrand n={stats_result.get('fictional_n', 0)})"
        )
        return lines

    fmt = lambda key: format_summary_log_float(stats_result.get(key), digits)
    lines.extend(
        [
            (
                "  Sample count(category-brand): "
                f"parametric knowledgebrand n={stats_result['parametric_n']}, "
                f"non-parametric knowledgebrand n={stats_result['fictional_n']}"
            ),
            (
                "  Mean: "
                f"parametric knowledgebrand={fmt('parametric_mean')}, "
                f"non-parametric knowledgebrand={fmt('fictional_mean')}, "
                f"difference={fmt('mean_diff')}"
            ),
            (
                "  Normalized mean: "
                f"parametric knowledgebrand={fmt('parametric_mean_norm')}, "
                f"non-parametric knowledgebrand={fmt('fictional_mean_norm')}, "
                f"difference={fmt('mean_diff_norm')}"
            ),
            (
                "  main-effect F: "
                f"F={fmt('f_statistic')}, "
                f"p={fmt('f_pvalue')}"
            ),
            (
                "  t-test: "
                f"t={fmt('t_statistic')}, "
                f"p={fmt('t_pvalue')}"
            ),
            f"  effect size Cohen's d: {fmt('cohens_d')}",
        ]
    )
    return lines


def print_brand_type_main_effect_f_for_nohup(
    stats_result: t.Dict[str, t.Any],
) -> None:
    print("")
    for line in format_brand_type_main_effect_f_lines(stats_result, digits=4):
        print(line)
    print("", flush=True)


def get_run_config_summary_lines(
    args: argparse.Namespace,
    results: t.Optional[t.Dict] = None,
) -> t.List[str]:
    total_brand_count = get_total_brand_count(args)
    runtime_script_name = getattr(args, "runtime_script_name", Path(__file__).name)
    categories = list(results.keys()) if isinstance(results, dict) else []
    lines = [
        "[Experiment settings]",
        f"  Script: {runtime_script_name}",
        f"  Parametric knowledge model: {args.model}",
        f"  Target recommendation model: {args.target_model}",
        f"  Number of parametric brands: {args.num_brands}",
        f"  Total brand count: {total_brand_count}",
        f"  Experiment design: {get_experiment_design_text(total_brand_count)}",
        f"  Random seed: {args.experiment_seed}",
        f"  Use ranking: {args.with_ranking}",
        f"  Output-only mode: {args.output_only}",
        f"  Disable ordering prompt: {args.no_ordering_prompt}",
        f"  System Role baseline: {args.use_system_role_baseline}",
        f"  Debias Instruction baseline: {args.use_debias_instruction_baseline}",
        f"  Moral Self-Correction baseline: {args.use_moral_self_correction_baseline}",
        f"  Enable CK: {getattr(args, 'use_ck', False)}",
        f"  Enable CARD: {getattr(args, 'use_card', False)}",
        f"  Test categories: {args.test if args.test else 'all categories'}",
        f"  Successful category count: {len(categories)}" if categories else "  Successful category count: (unknown)",
        f"  Successful categories: {', '.join(categories)}" if categories else "  Successful categories: (unknown)",
        f"  Output root directory: {args.out_base_dir}",
        f"  Plot root directory: {args.plot_base_dir}",
        f"  Output directory: {out_path(args)}",
        f"  Category pkl directory: {get_category_results_dir(args)}",
        f"  Plot directory: {plot_path(args)}",
        f"  Enable thinking mode: {args.enable_thinking}",
        f"  Batch size: {args.batch_size}",
        f"  Parallel parsing processes: {args.num_parsing_workers}",
        f"  Local parsing processes: {getattr(args, 'local_parsing_workers', 1)}",
        f"  Async-parse pending batch limit: {args.async_parse_max_pending_batches}",
        f"  Local inference backend: {args.target_local_backend}",
        f"  vLLM GPU memory utilization: {args.target_vllm_gpu_memory_utilization:g}",
        f"  vLLM max_model_len: {args.target_vllm_max_model_len}",
        f"  vLLM max_num_seqs: {args.target_vllm_max_num_seqs}",
        f"  vLLM max_num_batched_tokens: {args.target_vllm_max_num_batched_tokens}",
        f"  API max retries: {getattr(args, 'api_max_retries', 'N/A')}",
        f"  API initial wait time: {getattr(args, 'api_retry_initial_delay', 'N/A')}",
        f"  API backoff factor: {getattr(args, 'api_retry_backoff', 'N/A')}",
        f"  API max wait time: {getattr(args, 'api_retry_max_delay', 'N/A')}",
        f"  Temperature: {args.target_temp if not args.enable_thinking else (args.target_temp if args.target_temp_specified else 'None')}",
        f"  Top-P: {args.target_top_p if (not args.enable_thinking and args.target_top_p is not None) else 'None'}",
        f"  Max Tokens: {args.target_max_tokens}",
        f"  GPU IDs: {args.target_gpu_ids}",
    ]

    if getattr(args, 'use_ck', False):
        lines.extend(
            [
                "",
                "[CK settings]",
                f"  Adaptive mode: {args.ck_adaptive}",
                f"  Fixed alpha: {args.ck_alpha:g}",
                f"  Select Top: {args.ck_select_top}",
                f"  Relative Top: {args.ck_relative_top:g}",
            ]
        )

    if getattr(args, 'use_card', False):
        card_execution_path = (
            "vllm paired global path"
            if args.target_local_backend == "vllm" and not is_triggered_card_mode(args)
            else (
                "true batch path"
                if args.card_batch_inference
                else "legacy serial single path"
            )
        )
        lines.extend(
            [
                "",
                "[CARD settings]",
                f"  CARD mode: {get_card_display_name(args)}",
                f"  global logits composition formula: {resolve_card_global_logit_formula(args)}",
                f"  Fixed strength switch: {args.card_use_fixed_strength}",
                f"  Fixed strength value: {args.card_strength:g}",
                f"  Dynamic strength maximum: {args.card_dynamic_strength_max:g}",
                f"  Dynamic strength recomputation: {args.card_dynamic_alpha_recompute}",
                f"  Probability modulation: {args.card_modulated_prob}",
                f"  Auxiliary prompt mode: {resolve_card_aux_prompt_type(args)}",
                f"  True batch inference: {args.card_batch_inference}",
                f"  CARD execution path: {card_execution_path}",
                f"  CARD Top-k constraint: {args.card_use_top_k_constraint}",
                f"  CARD candidate set size: {args.card_top_k}",
                f"  Global CARD vLLM support mode: {resolve_card_global_vllm_support_mode(args)}",
                f"  Global CARD vLLM support top-k: {args.card_global_vllm_support_top_k}",
                f"  main-branch bias coefficient b: {get_card_global_main_bias_coeff(args):g}",
                f"  direction signal sign: {get_card_global_direction_sign(args)}",
            ]
        )
        if is_triggered_card_mode(args):
            lines.extend(
                [
                    f"  Trigger Mode: {args.card_trigger_mode}",
                    f"  Trigger Top-K: {args.card_trigger_top_k}",
                    f"  Trigger Threshold: {args.card_trigger_threshold}",
                    f"  Trigger Window: {args.card_trigger_window}",
                    f"  Trigger Followup Tokens: {args.card_trigger_followup_tokens}",
                    f"  Max Trigger Count: {args.card_max_trigger_count if args.card_max_trigger_count is not None else 'None'}",
                    f"  Filter opposite-side exclusive first token at trigger positions: {args.card_filter_opposite_start_tokens}",
                    f"  Filter system-prompt few-shot example tokens at trigger positions: {args.card_filter_system_prompt_example_tokens}",
                    f"  Token collection mode: {args.card_token_collection_mode}",
                ]
            )

    return lines


def build_recommendation_bias_run_summary_text(
    results: t.Dict,
    f_stat_results: t.Optional[t.Dict],
    brand_type_stats: t.Optional[t.Dict[str, t.Any]],
    args: argparse.Namespace,
    results_path: t.Optional[str],
) -> str:
    summary_path = (
        Path(results_path).with_name("run_summary.txt")
        if results_path is not None
        else Path(out_path(args, "run_summary.txt"))
    )
    lines: t.List[str] = [
        "Recommendation-bias experiment - lightweight run summary",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "[File locations]",
        f"  Results pkl: {results_path if results_path is not None else '(not generated)'}",
        f"  Lightweight summary: {summary_path}",
        f"  Output directory: {out_path(args)}",
        f"  Category pkl directory: {get_category_results_dir(args)}",
        f"  Plot directory: {plot_path(args)}",
        "",
        *get_run_config_summary_lines(args, results),
    ]

    api_retry_summary = collect_api_retry_summary(results)
    if api_retry_summary["category_count"] > 0:
        lines.extend(["", *format_api_retry_summary_lines(results, args)])

    lines.extend(
        [
            "",
            "[Analysis results - 4-decimal version]",
            f"  {SUMMARY_LOG_ROUNDING_NOTE}",
        ]
    )

    if brand_type_stats is not None:
        lines.extend(["", *format_brand_type_main_effect_f_lines(brand_type_stats, digits=4)])

    if f_stat_results:
        rows = []
        has_factor_level_f = any(
            any(key in stats for key in ('brand_f', 'doc_f', 'context_pos_f'))
            for stats in f_stat_results.values()
        )
        for category in results.keys():
            f_stats = f_stat_results.get(category, {})
            row = {
                'category': category,
                'score_ceiling_k': f_stats.get('score_ceiling_k', np.nan),
                'parametric_vs_nonparametric_pvalue': f_stats.get(
                    'parametric_vs_nonparametric_pvalue',
                    np.nan,
                ),
                'parametric_mean_score': f_stats.get(
                    'parametric_mean_score',
                    np.nan,
                ),
                'nonparametric_mean_score': f_stats.get(
                    'nonparametric_mean_score',
                    np.nan,
                ),
                'mean_score_diff': f_stats.get('mean_score_diff', np.nan),
                'parametric_mean_score_norm': f_stats.get(
                    'parametric_mean_score_norm',
                    np.nan,
                ),
                'nonparametric_mean_score_norm': f_stats.get(
                    'nonparametric_mean_score_norm',
                    np.nan,
                ),
                'mean_score_diff_norm': f_stats.get('mean_score_diff_norm', np.nan),
            }
            if has_factor_level_f:
                row.update({
                    'brand_f_statistic': f_stats.get('brand_f', np.nan),
                    'doc_f_statistic': f_stats.get('doc_f', np.nan),
                    'context_pos_f_statistic': f_stats.get('context_pos_f', np.nan),
                })
            rows.append(row)
        df = pd.DataFrame(rows)
        lines.extend(["", "[Overall recommendation-bias metric statistics - 4-decimal version]"])
        if not df.empty:
            if has_factor_level_f:
                lines.extend(
                    [
                        f"  Brand F median: {format_summary_log_float(df['brand_f_statistic'].median(), 4)}",
                        f"  Document F median: {format_summary_log_float(df['doc_f_statistic'].median(), 4)}",
                        f"  Context Position F median: {format_summary_log_float(df['context_pos_f_statistic'].median(), 4)}",
                    ]
                )
            lines.extend(
                [
                    f"  parametric knowledgebrandmean score: {format_summary_log_float(df['parametric_mean_score'].mean(), 4)}",
                    f"  non-parametric knowledgebrandmean score: {format_summary_log_float(df['nonparametric_mean_score'].mean(), 4)}",
                    f"  mean-score difference: {format_summary_log_float(df['mean_score_diff'].mean(), 4)}",
                    f"  parametric knowledgebrandnormalized mean: {format_summary_log_float(df['parametric_mean_score_norm'].mean(), 4)}",
                    f"  non-parametric knowledgebrandnormalized mean: {format_summary_log_float(df['nonparametric_mean_score_norm'].mean(), 4)}",
                    f"  normalized difference: {format_summary_log_float(df['mean_score_diff_norm'].mean(), 4)}",
                ]
            )

        lines.extend(["", "[Per-category recommendation-bias metric summary - 4-decimal version]"])
        for row in rows:
            fmt = lambda key: format_summary_log_float(row.get(key), 4)
            lines.append(f"  Category: {row['category']}")
            if has_factor_level_f:
                lines.append(
                    "    Factor-level Fstatistics: "
                    f"Brand={fmt('brand_f_statistic')}, "
                    f"Document={fmt('doc_f_statistic')}, "
                    f"Context Position={fmt('context_pos_f_statistic')}"
                )
            lines.extend(
                [
                    (
                        "    Parametric vs Non-parametric: "
                        f"parametric knowledgeMean={fmt('parametric_mean_score')}, "
                        f"non-parametric knowledgeMean={fmt('nonparametric_mean_score')}, "
                        f"difference={fmt('mean_score_diff')}, "
                        f"p={fmt('parametric_vs_nonparametric_pvalue')}"
                    ),
                    (
                        "    Parametric vs Non-parametric(normalized): "
                        f"parametric knowledgeMean={fmt('parametric_mean_score_norm')}, "
                        f"non-parametric knowledgeMean={fmt('nonparametric_mean_score_norm')}, "
                        f"difference={fmt('mean_score_diff_norm')}, "
                        f"K={fmt('score_ceiling_k')}"
                    ),
                ]
            )
    else:
        lines.extend(["", "  not generatedAnalysis results; may have enabled --output-only orMissingavailableAnalysis results."])

    return "\n".join(lines) + "\n"


def write_recommendation_bias_run_summary(
    results: t.Dict,
    f_stat_results: t.Optional[t.Dict],
    brand_type_stats: t.Optional[t.Dict[str, t.Any]],
    args: argparse.Namespace,
    results_path: t.Optional[str],
) -> str:
    summary_path = (
        Path(results_path).with_name("run_summary.txt")
        if results_path is not None
        else Path(out_path(args, "run_summary.txt"))
    )
    summary_text = build_recommendation_bias_run_summary_text(
        results=results,
        f_stat_results=f_stat_results,
        brand_type_stats=brand_type_stats,
        args=args,
        results_path=results_path,
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    return str(summary_path)


import tqdm

import random
import re

from _types import Product, Message, Role
from models import Models, load_model, release_local_model_cache
from helpers import file_utils
from latin_square import get_random_latin_square
from recommendation_bias_analysis import (
    AnalysisDependencies,
    analyze_and_plot,
    compute_brand_type_main_effect_f_brand_level,
    configure_analysis_dependencies,
    log_brand_type_main_effect_f_at_end,
    log_category_summary_statistics_at_end,
    log_overall_summary_statistics_at_end,
)
from recommendation_eval_utils import (
    build_system_prompt,
    build_target_message,
    count_response_paragraphs,
    estimate_token_count,
    get_combination_source_num_brands,
    get_requested_fictional_brand_count,
    get_requested_parametric_brand_count,
    get_scores_for_products_with_logs,
    get_total_brand_count_from_num_brands,
    is_top40_subset_mode,
    parse_response_for_products,
    validate_num_brands_value,
)
import recommendation_eval_utils as eval_utils
from global_card_trace_utils import normalize_global_card_token_trace
import dataset




def run_target_and_evaluator(
        target_chat: t.Callable,
        user_query: str,
        products: t.List[Product],
        docs: t.Optional[t.List[str]] = None,
        num_runs: int = 1,
        include_ordering_prompt: bool = True,
        use_system_role_baseline: bool = False,
        use_debias_instruction_baseline: bool = False,
        use_moral_self_correction_baseline: bool = False,
        shuffle_context_order: bool = True,
    ) -> t.Tuple[
        t.Dict[Product, t.List[int]],  # Products -> list of scores (one score per run)
        t.Dict[Product, t.List[int]],  # Ordering of product in context for each run
        t.List[str],  # Responses
    ]:
    logger = get_logger()
    context_orderings = {product: [] for product in products}
    scores = {product: [] for product in products}
    
    
    batch_messages = []
    batch_products_orders = []
    batch_docs_orders = []
    
    system_message = Message(
        role=Role.system,
        content=build_system_prompt(
            include_ordering_prompt=include_ordering_prompt,
            use_system_role_baseline=use_system_role_baseline,
        )
    )
    
    for _ in range(num_runs):
        
        run_products = products.copy()
        run_docs = docs.copy() if docs else None
        
        if shuffle_context_order and run_docs:
            permutation = list(range(len(run_products)))
            random.shuffle(permutation)
            run_docs = [run_docs[i] for i in permutation]
            run_products = [run_products[i] for i in permutation]
        
        target_message = build_target_message(
            query=user_query,
            documents=run_docs,
            product_models=[product.model for product in run_products],
            product_brands=[product.brand for product in run_products],
            use_debias_instruction_baseline=use_debias_instruction_baseline,
            use_moral_self_correction_baseline=use_moral_self_correction_baseline,
        )
        
        
        logger.info(f"")
        logger.info(f"[User message sent to the LLM (User Message)]")
        logger.info(f"{target_message}")
        logger.info(f"")
        
        
        batch_messages.append([
            system_message,
            Message(role=Role.user, content=target_message),
        ])
        batch_products_orders.append(run_products)
        batch_docs_orders.append(run_docs)
    
    
    batch_responses = target_chat(batch_messages)
    
    
    responses = []
    for i, response_message in enumerate(batch_responses):
        target_response = response_message.content
        responses.append(target_response)
        
        run_products = batch_products_orders[i]
        product_scores = get_scores_for_products(target_response, run_products)
        
        for product in products:
            scores[product].append(product_scores[product])
            context_orderings[product].append(run_products.index(product))

    return scores, context_orderings, responses


def get_scores_for_products(
        target_response: str, products: t.List[Product], verbose: bool = True
    ) -> t.Dict[Product, int]:
    logger = get_logger() if verbose else None

    result, log_info = parse_response_for_products(target_response, products)

    if verbose:
        logger.info(f"")
        logger.info(f"[Score parsing process]")
        logger.info(f"paragraph count after splitting: {log_info['num_paragraphs']}")

        for para_info in log_info['paragraphs']:
            logger.info(f"")
            logger.info(f"--- paragraph[{para_info['index']}] ---")
            logger.info(f"{para_info['preview']}")
            if para_info['matched']:
                logger.info(
                    "  => matched: "
                    f"{para_info['matched']['brand']} - {para_info['matched']['model']}"
                )
            else:
                logger.info(f"  => unmatchedanyproduct")

    if verbose:
        logger.info(f"")
        logger.info(f"[Final ranking and scores]")
        ordered_product_labels = [
            f"{p['brand']} - {p['model']}" for p in log_info['ordered_products']
        ]
        logger.info(
            "  ranking: "
            f"{ordered_product_labels}"
        )

    if verbose:
        for score_info in log_info['scores']:
            logger.info(
                f"  rank {score_info['rank']}: "
                f"{score_info['brand']} - {score_info['model']} => {score_info['score']}points"
            )

        if log_info['unmatched']:
            unmatched_labels = [
                f"{p['brand']} - {p['model']}" for p in log_info['unmatched']
            ]
            logger.info(
                "  unmatched: "
                f"{unmatched_labels} => 0points"
            )

    return result


def parse_single_response_worker(response_text: str, products_data: t.List[t.Dict]) -> t.Dict:
    
    products = [
        Product(
            category=p['category'],
            brand=p['brand'],
            model=p['model']
        )
        for p in products_data
    ]
    
    
    scores, log_info = get_scores_for_products_with_logs(response_text, products)
    
    
    serialized_scores = {
        f"{product.brand}|{product.model}": score
        for product, score in scores.items()
    }
    
    return {
        'scores': serialized_scores,
        'log_info': log_info
    }


def build_parallel_parse_tasks_from_texts(
    batch_response_texts: t.List[str],
    batch_products_list: t.List[t.List[Product]],
) -> t.List[t.Tuple[str, t.List[t.Dict[str, str]]]]:
    tasks = []
    for i in range(len(batch_response_texts)):
        response_text = batch_response_texts[i]
        products_data = [
            {'category': p.category, 'brand': p.brand, 'model': p.model}
            for p in batch_products_list[i]
        ]
        tasks.append((response_text, products_data))
    return tasks


def deserialize_parallel_parse_results(
    results: t.List[t.Dict[str, t.Any]],
    batch_products_list: t.List[t.List[Product]],
) -> t.Tuple[t.List[t.Dict[Product, int]], t.List[t.Dict]]:
    all_product_scores = []
    all_log_info = []

    for i, result in enumerate(results):
        products = batch_products_list[i]

        product_scores = {}
        for product in products:
            key = f"{product.brand}|{product.model}"
            product_scores[product] = result['scores'].get(key, 0)
        all_product_scores.append(product_scores)
        all_log_info.append(result['log_info'])

    return all_product_scores, all_log_info


def create_local_parsing_executor(num_workers: int):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    if num_workers <= 1:
        return None

    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=get_context("spawn"),
    )


def parse_responses_parallel(
    batch_responses: t.List,
    batch_products_list: t.List[t.List[Product]],
    num_workers: int = 8
) -> t.Tuple[t.List[t.Dict[Product, int]], t.List[t.Dict]]:
    from multiprocessing import Pool
    
    logger = get_logger()
    
    tasks = eval_utils.build_parallel_parse_tasks_from_texts(
        [response.content for response in batch_responses],
        batch_products_list,
    )
    
    
    logger.info(f"")
    logger.info(f"[Parallel parsing] Starting parsing of {len(tasks)} responses, using {num_workers} processes")
    
    
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(eval_utils.parse_single_response_worker, tasks)
    
    logger.info(f"[Parallel parsing] complete, parsed {len(results)} responses")
    
    return eval_utils.deserialize_parallel_parse_results(results, batch_products_list)


def parse_response_texts_parallel_spawn(
    batch_response_texts: t.List[str],
    batch_products_list: t.List[t.List[Product]],
    executor,
    num_workers: int,
) -> t.Tuple[t.List[t.Dict[Product, int]], t.List[t.Dict]]:
    logger = get_logger()
    tasks = eval_utils.build_parallel_parse_tasks_from_texts(
        batch_response_texts,
        batch_products_list,
    )

    logger.info(f"")
    logger.info(
        f"[Local safe parallel parsing] Starting parsing of {len(tasks)} responses, "
        f"using {num_workers}  spawn processes"
    )

    futures = [
        executor.submit(
            eval_utils.parse_single_response_worker,
            response_text,
            products_data,
        )
        for response_text, products_data in tasks
    ]
    results = [future.result() for future in futures]

    logger.info(f"[Local safe parallel parsing] complete, parsed {len(results)} responses")
    return eval_utils.deserialize_parallel_parse_results(results, batch_products_list)




class SelectiveFileLogFormatter(logging.Formatter):

    def __init__(self) -> None:
        super().__init__()
        self.prefixed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        self.plain_formatter = logging.Formatter('%(message)s')

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, 'force_file_prefix', False) or record.levelno >= logging.WARNING:
            return self.prefixed_formatter.format(record)
        return self.plain_formatter.format(record)


def log_key_info(logger: logging.Logger, message: str) -> None:
    logger.info(message, extra={'force_file_prefix': True})


def setup_logging(
    args: argparse.Namespace,
    script_name_override: t.Optional[str] = None,
) -> logging.Logger:
    log_dir = Path(out_path(args, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    
    script_name = script_name_override or Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_single_test_category(args):
        category_safe = sanitize_category_name(get_single_test_category(args))
        log_file = log_dir / f"{script_name}_{category_safe}_{timestamp}.log"
    elif get_requested_test_categories(args):
        log_file = log_dir / f"{script_name}_selected_{len(get_requested_test_categories(args))}cats_{timestamp}.log"
    else:
        log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    
    logger = logging.getLogger("recommendation_bias_experiment")
    logger.setLevel(logging.DEBUG)
    
    
    logger.handlers.clear()
    
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = SelectiveFileLogFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    log_key_info(logger, f"Log file: {log_file}")
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("recommendation_bias_experiment")


def set_plain_log_formatter_temporarily(
    logger: logging.Logger,
    callback: t.Callable[[], t.Any],
) -> t.Any:
    original_formatters = [
        (handler, handler.formatter)
        for handler in logger.handlers
    ]
    plain_formatter = logging.Formatter('%(message)s')
    try:
        for handler in logger.handlers:
            handler.setFormatter(plain_formatter)
        return callback()
    finally:
        for handler, formatter in original_formatters:
            handler.setFormatter(formatter)

np.set_printoptions(linewidth=200)




@dataclass
class BrandInfo:
    brand_index: int
    brand: str
    model: str
    knowledge_strength: float
    is_fictional: bool
    rank: t.Optional[int] = None  


@dataclass 
class ExperimentConfig:
    parametric_model: str  
    target_model: str  
    num_brands: int  
    with_ranking: bool  
    num_permutations: int  




def resolve_card_aux_prompt_type(args: argparse.Namespace) -> str:
    card_aux_prompt_type = getattr(args, 'card_aux_prompt_type', None)
    card_use_attention_mask = getattr(args, 'card_use_attention_mask', False)

    if card_aux_prompt_type is None:
        return "mask" if card_use_attention_mask else "uniform"

    if card_use_attention_mask and card_aux_prompt_type != "mask":
        raise ValueError(
            "--card-use-attention-mask=true conflicts with "
            f"--card-aux-prompt-type {card_aux_prompt_type} conflict; "
            "use --card-aux-prompt-type mask, or disable --card-use-attention-mask."
        )

    return card_aux_prompt_type


def resolve_card_application_mode(args: argparse.Namespace) -> str:
    card_application_mode = getattr(args, 'card_application_mode', 'triggered')
    valid_modes = {"triggered", "global"}
    if card_application_mode not in valid_modes:
        raise ValueError(
            "--card-application-mode must be "
            f"{sorted(valid_modes)} one of"
        )
    return card_application_mode


def is_triggered_card_mode(args: argparse.Namespace) -> bool:
    return resolve_card_application_mode(args) == "triggered"


def get_card_display_name(args: argparse.Namespace) -> str:
    return "Triggered CARD" if is_triggered_card_mode(args) else "Global CARD"


def get_card_result_slug(args: argparse.Namespace) -> str:
    return "triggered-card" if is_triggered_card_mode(args) else "global-card"


def resolve_card_global_logit_formula(args: argparse.Namespace) -> str:
    card_global_logit_formula = getattr(
        args,
        'card_global_logit_formula',
        'contrastive',
    )
    valid_formulas = {"contrastive", "ck", "zxy"}
    if card_global_logit_formula not in valid_formulas:
        raise ValueError(
            "--card-global-logit-formula must be "
            f"{sorted(valid_formulas)} one of"
        )
    return card_global_logit_formula


def get_card_global_formula_alpha(args: argparse.Namespace) -> t.Optional[float]:
    card_formula = resolve_card_global_logit_formula(args)
    if card_formula == "ck":
        return float(getattr(args, 'card_global_ck_alpha', 0.5))
    if card_formula == "zxy":
        return float(getattr(args, 'card_global_zxy_alpha', 0.5))
    return None


def get_card_global_main_bias_coeff(args: argparse.Namespace) -> float:
    return float(getattr(args, 'card_global_main_bias_coeff', 0.0))


def get_card_global_direction_sign(args: argparse.Namespace) -> int:
    return int(getattr(args, 'card_global_direction_sign', 1))


def resolve_card_global_vllm_support_mode(args: argparse.Namespace) -> str:
    support_mode = getattr(args, 'card_global_vllm_support_mode', 'full_vocab')
    valid_modes = {"full_vocab", "main_aux_topk_union"}
    if support_mode not in valid_modes:
        raise ValueError(
            "--card-global-vllm-support-mode must be "
            f"{sorted(valid_modes)} one of"
        )
    return support_mode


def is_card_global_custom_formula_enabled(args: argparse.Namespace) -> bool:
    return (
        resolve_card_application_mode(args) == "global"
        and resolve_card_global_logit_formula(args) in {"ck", "zxy"}
    )


def format_card_float_suffix(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def get_card_dynamic_strength_log_text(card_dynamic_strength_max: float) -> str:
    max_strength_text = f"{float(card_dynamic_strength_max):g}"
    return (
        f"max={max_strength_text} -> "
        f"1 / ln(exp(1/{max_strength_text}) + KL(aux || main))"
    )


def sanitize_category_name(category: str) -> str:
    return category.replace(' ', '_').replace('/', '_')


def get_total_brand_count(args: argparse.Namespace) -> int:
    return get_total_brand_count_from_num_brands(int(args.num_brands))


def get_total_experiment_count_from_total_brand_count(total_brand_count: int) -> int:
    return total_brand_count * total_brand_count


def get_total_experiment_count(args: argparse.Namespace) -> int:
    return get_total_experiment_count_from_total_brand_count(get_total_brand_count(args))


def get_experiment_design_text(total_brand_count: int) -> str:
    total_experiments = get_total_experiment_count_from_total_brand_count(total_brand_count)
    return f"{total_brand_count} Blocks × {total_brand_count} Runs = {total_experiments}  times"


def parse_requested_test_categories(test_value: t.Optional[str]) -> t.List[str]:
    if not test_value:
        return []

    categories = []
    seen = set()
    for raw_category in str(test_value).split(','):
        category = raw_category.strip()
        if not category or category in seen:
            continue
        categories.append(category)
        seen.add(category)

    return categories


def get_requested_test_categories(args: argparse.Namespace) -> t.List[str]:
    cached_categories = getattr(args, 'test_categories', None)
    if cached_categories is not None:
        return list(cached_categories)
    return parse_requested_test_categories(getattr(args, 'test', None))


def is_single_test_category(args: argparse.Namespace) -> bool:
    return len(get_requested_test_categories(args)) == 1


def get_single_test_category(args: argparse.Namespace) -> t.Optional[str]:
    categories = get_requested_test_categories(args)
    if len(categories) != 1:
        return None
    return categories[0]


def get_multi_test_category_slug(args: argparse.Namespace) -> t.Optional[str]:
    categories = get_requested_test_categories(args)
    if len(categories) <= 1:
        return None
    return "__".join(sanitize_category_name(category) for category in categories)


def get_model_name_with_modifiers(args: argparse.Namespace) -> str:
    model_name = args.model
    
    
    if getattr(args, 'enable_thinking', False):
        model_name = f"{model_name}-thinking"

    
    if getattr(args, 'use_system_role_baseline', False):
        model_name = f"{model_name}-systemrole"

    
    if getattr(args, 'use_debias_instruction_baseline', False):
        model_name = f"{model_name}-debiasinst"

    
    if getattr(args, 'use_moral_self_correction_baseline', False):
        model_name = f"{model_name}-moralselfcorr"

    
    if getattr(args, 'use_ck', False):
        model_name = f"{model_name}-ckplug"
        if getattr(args, 'target_local_backend', 'vllm') == "vllm":
            model_name = f"{model_name}-vllm"
        if getattr(args, 'ck_adaptive', False):
            model_name = f"{model_name}-adaptive"
        else:
            ck_alpha_suffix = format_card_float_suffix(getattr(args, 'ck_alpha', 0.5))
            model_name = f"{model_name}-cka{ck_alpha_suffix}"
        if getattr(args, 'ck_select_top', 10) != 10:
            model_name = f"{model_name}-ckt{args.ck_select_top}"
        if getattr(args, 'ck_relative_top', 0.01) != 0.01:
            ck_relative_top_suffix = format_card_float_suffix(
                getattr(args, 'ck_relative_top', 0.01)
            )
            model_name = f"{model_name}-ckrt{ck_relative_top_suffix}"
    
    
    if getattr(args, 'use_card', False):
        card_application_mode = resolve_card_application_mode(args)
        card_result_slug = get_card_result_slug(args)
        card_global_logit_formula = resolve_card_global_logit_formula(args)
        if getattr(args, 'target_local_backend', 'vllm') == "vllm":
            model_name = f"{model_name}-vllm"
        use_global_custom_formula = (
            card_application_mode == 'global'
            and card_global_logit_formula in {'ck', 'zxy'}
        )
        if use_global_custom_formula:
            model_name = f"{model_name}-{card_result_slug}-{card_global_logit_formula}"
            formula_alpha = get_card_global_formula_alpha(args)
            formula_alpha_suffix = format_card_float_suffix(formula_alpha)
            alpha_prefix = (
                "cka" if card_global_logit_formula == "ck" else "zxya"
            )
            model_name = (
                f"{model_name}-{alpha_prefix}{formula_alpha_suffix}"
            )
        else:
            card_strength = getattr(args, 'card_strength', 2.0)
            card_use_fixed_strength = getattr(args, 'card_use_fixed_strength', True)
            if card_use_fixed_strength:
                model_name = f"{model_name}-{card_result_slug}-s{card_strength}"
            else:
                dynamic_strength_max = getattr(args, 'card_dynamic_strength_max', 1.0)
                dynamic_strength_max_suffix = format_card_float_suffix(
                    dynamic_strength_max
                )
                model_name = (
                    f"{model_name}-{card_result_slug}-dynstrmax{dynamic_strength_max_suffix}"
                )
                model_name = (
                    f"{model_name}-darr{str(getattr(args, 'card_dynamic_alpha_recompute', False)).lower()}"
                )
        if (
            card_application_mode == 'global'
            and getattr(args, 'target_local_backend', 'vllm') == "vllm"
        ):
            main_bias_coeff = get_card_global_main_bias_coeff(args)
            if main_bias_coeff != 0.0:
                main_bias_coeff_suffix = format_card_float_suffix(main_bias_coeff)
                model_name = f"{model_name}-mbc{main_bias_coeff_suffix}"
            direction_sign = get_card_global_direction_sign(args)
            if direction_sign == -1:
                model_name = f"{model_name}-dirneg"
            support_mode = resolve_card_global_vllm_support_mode(args)
            if support_mode == "main_aux_topk_union":
                support_top_k = int(
                    getattr(args, 'card_global_vllm_support_top_k', 10)
                )
                model_name = f"{model_name}-gsvtopku{support_top_k}"
        if getattr(args, 'card_batch_inference', False):
            model_name = f"{model_name}-cardbatch"
        card_aux_prompt_type = resolve_card_aux_prompt_type(args)
        
        
        if (not use_global_custom_formula) and getattr(args, 'card_modulated_prob', False):
            model_name = f"{model_name}-mp"
            if getattr(args, 'card_prob_weight_beta', 1.0) != 1.0:
                model_name = f"{model_name}-pb{args.card_prob_weight_beta:g}"
        if card_aux_prompt_type == "mask":
            model_name = f"{model_name}-am"
        elif card_aux_prompt_type == "delete":
            model_name = f"{model_name}-del"
        card_use_top_k_constraint = getattr(args, 'card_use_top_k_constraint', True)
        card_top_k = getattr(args, 'card_top_k', 5)
        model_name = f"{model_name}-stkc{str(card_use_top_k_constraint).lower()}"
        if card_use_top_k_constraint:
            model_name = f"{model_name}-stk{card_top_k}"
        if card_application_mode == 'triggered':
            card_trigger_mode = getattr(args, 'card_trigger_mode', 'top_k_count')
            if card_trigger_mode != 'top_k_count':
                trigger_prob_sum_threshold_suffix = format_card_float_suffix(
                    getattr(args, 'card_trigger_prob_sum_threshold', 0.999)
                )
                model_name = (
                    f"{model_name}-tgmprob{trigger_prob_sum_threshold_suffix}c2"
                )
            card_filter_opposite_start_tokens = getattr(
                args, 'card_filter_opposite_start_tokens', True
            )
            model_name = (
                f"{model_name}-fost{str(card_filter_opposite_start_tokens).lower()}"
            )
            card_filter_system_prompt_example_tokens = getattr(
                args, 'card_filter_system_prompt_example_tokens', True
            )
            model_name = (
                f"{model_name}-fspet{str(card_filter_system_prompt_example_tokens).lower()}"
            )
            if card_trigger_mode == 'top_k_count':
                if getattr(args, 'card_trigger_top_k', 5) != 5:
                    model_name = f"{model_name}-tk{args.card_trigger_top_k}"
                if getattr(args, 'card_trigger_threshold', 2) != 2:
                    model_name = f"{model_name}-th{args.card_trigger_threshold}"
            if getattr(args, 'card_trigger_window', 10) != 10:
                model_name = f"{model_name}-tw{args.card_trigger_window}"
            model_name = f"{model_name}-ft{getattr(args, 'card_trigger_followup_tokens', 1)}"
            if getattr(args, 'card_max_trigger_count', None) is not None:
                model_name = f"{model_name}-mtc{args.card_max_trigger_count}"
            if getattr(args, 'card_token_collection_mode', 'first_only') != 'first_only':
                model_name = f"{model_name}-{args.card_token_collection_mode}"
    
    return model_name


def get_selected_category_run_subdir(args: argparse.Namespace) -> t.Tuple[str, ...]:
    categories = get_requested_test_categories(args)
    if not categories:
        return tuple()

    run_tag = getattr(args, 'single_category_run_tag', None)
    if len(categories) == 1:
        if not run_tag:
            return tuple()
        return ('single_category_runs', sanitize_category_name(categories[0]), run_tag)

    multi_category_slug = get_multi_test_category_slug(args)
    if not multi_category_slug:
        return tuple()
    if run_tag:
        return ('multi_category_runs', multi_category_slug, run_tag)
    return ('multi_category_runs', multi_category_slug)


def get_experiment_method_subdir(args: argparse.Namespace) -> t.Tuple[str, ...]:
    if getattr(args, 'use_ck', False):
        method_slug = 'ck'
    elif getattr(args, 'use_card', False):
        method_slug = 'card'
    elif getattr(args, 'use_system_role_baseline', False):
        method_slug = 'systemrole'
    elif getattr(args, 'use_debias_instruction_baseline', False):
        method_slug = 'debiasinst'
    elif getattr(args, 'use_moral_self_correction_baseline', False):
        method_slug = 'moralselfcorr'
    elif getattr(args, 'enable_thinking', False):
        method_slug = 'thinking'
    else:
        method_slug = 'plain'
    return (method_slug,)


def out_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    brands_suffix = f"top{args.num_brands}"
    num_experiments = get_total_experiment_count(args)
    model_name = get_model_name_with_modifiers(args)
    selected_category_subdir = get_selected_category_run_subdir(args)
    out_base_dir = getattr(args, 'out_base_dir', './out')
    return os.path.join(
        out_base_dir, 'evaluation_results', *get_experiment_method_subdir(args),
        model_name, brands_suffix, str(num_experiments),
        *selected_category_subdir, *file_path
    )


def plot_path(args: argparse.Namespace, *file_path: t.List[str]) -> str:
    brands_suffix = f"top{args.num_brands}"
    num_experiments = get_total_experiment_count(args)
    model_name = get_model_name_with_modifiers(args)
    selected_category_subdir = get_selected_category_run_subdir(args)
    plot_base_dir = getattr(args, 'plot_base_dir', './plots')

    if selected_category_subdir:
        return os.path.join(
            plot_base_dir, 'evaluation_results', *get_experiment_method_subdir(args),
            model_name, brands_suffix, str(num_experiments),
            *selected_category_subdir, *file_path
        )

    
    if is_single_test_category(args):
        category_safe = sanitize_category_name(get_single_test_category(args))
        return os.path.join(
            plot_base_dir, 'evaluation_results', *get_experiment_method_subdir(args),
            model_name, brands_suffix, str(num_experiments),
            f'single_category_{category_safe}', *file_path
        )
    else:
        return os.path.join(
            plot_base_dir, 'evaluation_results', *get_experiment_method_subdir(args),
            model_name, brands_suffix, str(num_experiments), *file_path
        )


PKL_EXCLUDED_CATEGORY_RESULT_FIELDS = frozenset({
    'inference_time_stats',
    'parsing_time_stats',
    'output_token_stats',
    'card_trigger_prob_records',
    'card_trigger_prob_summary',
    'thinking_stats',
})


def prepare_category_result_for_pickle(category_result: t.Any) -> t.Any:
    """Return a PKL-facing single-category result copy without bulky runtime statistics."""
    if isinstance(category_result, dict):
        return {
            key: value
            for key, value in category_result.items()
            if key not in PKL_EXCLUDED_CATEGORY_RESULT_FIELDS
        }
    return category_result


def prepare_results_for_pickle(results: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    """Return a PKL-facing results copy without bulky runtime statistics."""
    prepared_results = {}
    for category, category_result in results.items():
        prepared_results[category] = prepare_category_result_for_pickle(category_result)
    return prepared_results


def combinations_path(args: argparse.Namespace) -> str:
    source_num_brands = get_combination_source_num_brands(int(args.num_brands))
    brands_suffix = f"top{source_num_brands}"
    return os.path.join(
        './out', 'brand_doc_combinations', args.model, brands_suffix
    )




def load_brand_doc_combinations(
    category: str,
    args: argparse.Namespace
) -> t.Tuple[t.List[BrandInfo], t.List[t.List[str]]]:
    logger = get_logger()
    
    category_dir = Path(combinations_path(args)) / category
    metadata_file = category_dir / "metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata filedoes not exist: {metadata_file}\n"
            f"run step3: python build_brand_doc_combinations.py --model {args.model}"
        )
    
    
    logger.info(f"")
    log_key_info(logger, f"[Data loading]")
    logger.info(f"Category: {category}")
    logger.info(f"Data directory: {category_dir}")
    logger.info(f"Metadata file: {metadata_file}")
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    
    source_brands = [
        BrandInfo(
            brand_index=b["brand_index"],
            brand=b["brand"],
            model=b["model"],
            knowledge_strength=b["knowledge_strength"],
            is_fictional=b["is_fictional"],
            rank=b.get("rank")  
        )
        for b in metadata["brands"]
    ]

    requested_parametric_brand_count = get_requested_parametric_brand_count(int(args.num_brands))
    requested_fictional_brand_count = get_requested_fictional_brand_count(int(args.num_brands))
    requested_total_brand_count = get_total_brand_count(args)
    source_num_docs = int(metadata["num_docs"])

    source_real_brands = [b for b in source_brands if not b.is_fictional]
    source_fictional_brands = [b for b in source_brands if b.is_fictional]

    if len(source_real_brands) < requested_parametric_brand_count:
        raise ValueError(
            f"data source hasreal brandsnot enough: required {requested_parametric_brand_count} , "
            f"actualhas only {len(source_real_brands)} ."
        )
    if len(source_fictional_brands) < requested_fictional_brand_count:
        raise ValueError(
            f"data source hasfictional brandsnot enough: required {requested_fictional_brand_count} , "
            f"actualhas only {len(source_fictional_brands)} ."
        )
    if source_num_docs < requested_total_brand_count:
        raise ValueError(
            f"data source hasdocumenttemplatesnot enough: required {requested_total_brand_count} , "
            f"actualhas only {source_num_docs} ."
        )

    if is_top40_subset_mode(int(args.num_brands)):
        brands = (
            source_real_brands[:requested_parametric_brand_count]
            + source_fictional_brands[:requested_fictional_brand_count]
        )
    else:
        brands = source_brands
    
    
    real_brands = [b for b in brands if not b.is_fictional]
    fictional_brands = [b for b in brands if b.is_fictional]
    
    logger.info(f"")
    logger.info(
        f"[Loaded brands] total {len(brands)}  "
        f"({len(real_brands)} real + {len(fictional_brands)} fictional)"
    )
    if is_top40_subset_mode(int(args.num_brands)):
        logger.info(
            f"  subset mode: based on top40 data selectfirst {requested_parametric_brand_count} real brands"
            f" + first {requested_fictional_brand_count} fictional brands"
        )
        logger.info(
            f"  source total: {len(source_brands)} brand "
            f"({len(source_real_brands)} real + {len(source_fictional_brands)} fictional)"
        )
    logger.info(f"")
    logger.info(f"real brands (fromparametric knowledge):")
    for b in real_brands:
        logger.info(f"  [{b.brand_index}] {b.brand} - {b.model} | knowledge strength: {b.knowledge_strength:.4f}")
    
    if fictional_brands:
        logger.info(f"")
        logger.info(f"fictional brands:")
        for b in fictional_brands:
            logger.info(f"  [{b.brand_index}] {b.brand} - {b.model} | knowledge strength: {b.knowledge_strength:.4f}")
    
    
    num_docs = requested_total_brand_count
    docs = []
    
    logger.info(f"")
    logger.info(f"[Loaded documents] per brand {num_docs} document")
    if source_num_docs != num_docs:
        logger.info(
            f"  subset mode: data source contains {source_num_docs} documenttemplates, "
            f"this timesthis experiment takesfirst {num_docs} "
        )
    
    for brand in brands:
        brand_docs = []
        for doc_idx in range(num_docs):
            doc_file = category_dir / str(brand.brand_index) / f"{doc_idx}.txt"
            with open(doc_file, "r", encoding="utf-8") as f:
                brand_docs.append(f.read())
        docs.append(brand_docs)
        
        
        logger.info(f"  brand[{brand.brand_index}] {brand.brand}: loaded from {category_dir / str(brand.brand_index)}/")
    
    
    if "documents" in metadata:
        logger.info(f"")
        logger.info(f"[Original document sources]")
        for doc_info in metadata["documents"][:num_docs]:
            logger.info(f"  document[{doc_info['doc_index']}]: original brand={doc_info['original_brand']}, original model={doc_info['original_model']}")
    
    return brands, docs


def brands_to_products(brands: t.List[BrandInfo], category: str) -> t.List[Product]:
    return [
        Product(category=category, brand=b.brand, model=b.model)
        for b in brands
    ]




def get_available_categories(args: argparse.Namespace) -> t.List[str]:
    combo_dir = Path(combinations_path(args))
    if not combo_dir.exists():
        return []
    
    categories = []
    for d in combo_dir.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            categories.append(d.name)
    
    return sorted(categories)


def release_target_model_cache_after_run(args: argparse.Namespace) -> bool:
    if not getattr(args, 'is_local_model', False):
        return False
    if getattr(args, 'target_local_backend', 'vllm') != 'vllm':
        return False

    if getattr(args, 'use_card', False):
        from global_card_vllm_models import release_global_card_vllm_model_cache

        return release_global_card_vllm_model_cache(
            args.target_model,
            gpu_ids=args.target_gpu_ids,
            gpu_memory_utilization=args.target_vllm_gpu_memory_utilization,
            max_model_len=args.target_vllm_max_model_len,
            max_num_seqs=args.target_vllm_max_num_seqs,
            max_num_batched_tokens=args.target_vllm_max_num_batched_tokens,
        )

    if getattr(args, 'use_ck', False):
        from ck_vllm_models import release_ck_vllm_model_cache

        return release_ck_vllm_model_cache(
            args.target_model,
            alpha=args.ck_alpha,
            adaptive=args.ck_adaptive,
            select_top=args.ck_select_top,
            relative_top=args.ck_relative_top,
            gpu_ids=args.target_gpu_ids,
            gpu_memory_utilization=args.target_vllm_gpu_memory_utilization,
            max_model_len=args.target_vllm_max_model_len,
            max_num_seqs=args.target_vllm_max_num_seqs,
            max_num_batched_tokens=args.target_vllm_max_num_batched_tokens,
        )

    return release_local_model_cache(
        args.target_model,
        gpu_ids=args.target_gpu_ids,
        local_inference_backend=args.target_local_backend,
        vllm_gpu_memory_utilization=args.target_vllm_gpu_memory_utilization,
        vllm_max_model_len=args.target_vllm_max_model_len,
        vllm_max_num_seqs=args.target_vllm_max_num_seqs,
        vllm_max_num_batched_tokens=args.target_vllm_max_num_batched_tokens,
    )




def run_recommendation_bias_experiment(args: argparse.Namespace):

    results = None
    results_path = None
    total_brand_count = get_total_brand_count(args)
    total_experiments_per_category = get_total_experiment_count(args)

    if args.run_eval:
        if get_requested_test_categories(args) and not getattr(args, 'single_category_run_tag', None):
            args.single_category_run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        
        file_utils.ensure_created_directory(out_path(args))
        logger = setup_logging(args)
        
        
        script_name = Path(__file__).name
        args.runtime_script_name = script_name
        logger.info(f"")
        logger.info(f"{'#'*80}")
        logger.info(f"# Recommendation-bias experiment - start")
        logger.info(f"{'#'*80}")
        logger.info(f"")
        logger.info(f"[Run script]")
        logger.info(f"  Script file: {script_name}")
        logger.info(f"")
        logger.info(f"[Code parameter settings]")
        logger.info(f"  Parametric knowledge model (--model): {args.model}")
        logger.info(f"  Target recommendation model (--target-model): {args.target_model}")
        logger.info(f"  Number of parametric brands (--num-brands): {args.num_brands}")
        logger.info(f"  Total brand count: {total_brand_count}")
        logger.info(f"  Experiment design: {get_experiment_design_text(total_brand_count)} experiment tasks")
        logger.info(f"  Random seed (--experiment-seed): {args.experiment_seed}")
        logger.info(f"  Use ranking (--with-ranking): {args.with_ranking}")
        logger.info(f"  Output-only mode (--output-only): {args.output_only}")
        logger.info(f"  Disable ordering prompt (--no-ordering-prompt): {args.no_ordering_prompt}")
        logger.info(
            f"  Local parsing processes (--local-parsing-workers): "
            f"{getattr(args, 'local_parsing_workers', 1)}"
        )
        logger.info(
            f"  System Role baseline (--use-system-role-baseline): "
            f"{args.use_system_role_baseline}"
        )
        logger.info(
            f"  Debias Instruction baseline (--use-debias-instruction-baseline): "
            f"{args.use_debias_instruction_baseline}"
        )
        logger.info(
            f"  Moral Self-Correction baseline "
            f"(--use-moral-self-correction-baseline): "
            f"{args.use_moral_self_correction_baseline}"
        )
        logger.info(f"  Enable CK (--use-ck): {getattr(args, 'use_ck', False)}")
        logger.info(f"  Test categories (--test): {args.test}")
        if getattr(args, 'single_category_run_tag', None):
            logger.info(f"  Category-targeted run tag (--single-category-run-tag): {args.single_category_run_tag}")
        logger.info(f"  Enable thinking mode (--enable-thinking): {args.enable_thinking}")
        logger.info(f"  Brand-document combination directory: {combinations_path(args)}")
        logger.info(f"  Output root directory (--out-base-dir): {args.out_base_dir}")
        logger.info(f"  Plot root directory (--plot-base-dir): {args.plot_base_dir}")
        logger.info(f"  Output directory: {out_path(args)}")
        logger.info(f"  Category pkl directory: {get_category_results_dir(args)}")
        logger.info(f"  Plot directory: {plot_path(args)}")
        run_config_path = save_run_config_metadata(args)
        logger.info(f"  Run config file: {run_config_path}")
        logger.info(f"")
        logger.info(f"[LLM parameter settings]")
        logger.info(f"  Target model: {args.target_model}")
        logger.info(
            "  Local inference backend (--target-local-backend): "
            f"{args.target_local_backend} (normal local inference; CK vLLM / Global CARD vLLM also applies)"
        )
        logger.info(
            "  vLLM GPU memory utilization (--target-vllm-gpu-memory-utilization): "
            f"{args.target_vllm_gpu_memory_utilization:g} (normal local vllm / CK vLLM / Global CARD vLLM applies)"
        )
        logger.info(
            "  vLLM max_model_len (--target-vllm-max-model-len): "
            f"{args.target_vllm_max_model_len} (None meansusing vLLM defaultvalue)"
        )
        logger.info(
            "  vLLM max_num_seqs (--target-vllm-max-num-seqs): "
            f"{args.target_vllm_max_num_seqs} (None meansusing vLLM defaultvalue)"
        )
        logger.info(
            "  vLLM max_num_batched_tokens "
            "(--target-vllm-max-num-batched-tokens): "
            f"{args.target_vllm_max_num_batched_tokens} "
            "(None meansusing vLLM defaultvalue)"
        )
        if args.enable_thinking:
            
            
            thinking_temp = args.target_temp if args.target_temp_specified else None
            if thinking_temp is not None:
                logger.info(f"  Temperature (--target-temp): {thinking_temp} ( use user specified)")
            else:
                logger.info(f"  Temperature: None (usemodeldefaultvalue, thinkingmode)")
            logger.info(f"  Top-P: None (usemodeldefaultvalue, thinkingmode)")
        else:
            logger.info(f"  Temperature (--target-temp): {args.target_temp}")
            logger.info(f"  Top-P (--target-top-p): {args.target_top_p if args.target_top_p is not None else 'None (not used)'}")
        logger.info(f"  Max Tokens (--target-max-tokens): {args.target_max_tokens}")
        logger.info(f"  GPU IDs (--target-gpu-ids): {args.target_gpu_ids}")
        logger.info(f"  API concurrent requestscount (--api-concurrency): {args.api_concurrency}")
        logger.info(f"  API concurrencysubmission stagger cap (--api-submit-stagger): {args.api_submit_stagger}")
        logger.info(f"  API max retries (--api-max-retries): {args.api_max_retries}")
        logger.info(
            f"  API initial wait time (--api-retry-initial-delay): "
            f"{args.api_retry_initial_delay}"
        )
        logger.info(
            f"  API backoff factor (--api-retry-backoff): {args.api_retry_backoff}"
        )
        logger.info(
            f"  API max wait time (--api-retry-max-delay): "
            f"{args.api_retry_max_delay}"
        )
        logger.info(f"")
        logger.info(f"[System Prompt]")
        system_prompt = build_system_prompt(
            include_ordering_prompt=not args.no_ordering_prompt,
            use_system_role_baseline=args.use_system_role_baseline,
        )
        logger.info(f"{system_prompt}")
        logger.info(f"")
        
        
        if getattr(args, 'use_ck', False):
            logger.info(f"[CK configuration]")
            logger.info(f"  Enable CK: True")
            if args.target_local_backend == "vllm":
                logger.info(
                    "  CK vLLM mode: CK-PLUG greedy decoding reproduction "
                    "(non-thinking, temperature=0.0, greedy-only)"
                )
            logger.info(
                f"  Parametric-branch prompt: delete document body and preserve candidate product slots"
            )
            logger.info(f"  Adaptive mode (--ck-adaptive): {args.ck_adaptive}")
            if args.ck_adaptive:
                logger.info("  alpha: adaptive (dynamically scaled by entropy/conflict)")
            else:
                logger.info(f"  Fixed alpha (--ck-alpha): {args.ck_alpha:g}")
            logger.info(f"  Select Top (--ck-select-top): {args.ck_select_top}")
            logger.info(
                f"  Relative Top (--ck-relative-top): {args.ck_relative_top:g}"
            )
            logger.info(
                "  CK logits formula: score_final = alpha*score_base + "
                "(1-alpha)*(score_main-score_base)"
            )
            logger.info(
                "  conflictdecision: only in  entropy-based conflict intervenes when conditions are met; nothenfall back to mainpointsbranch logits"
            )
            logger.info(f"")

        
        if getattr(args, 'use_card', False):
            card_aux_prompt_type = resolve_card_aux_prompt_type(args)
            card_display_name = get_card_display_name(args)
            is_triggered_card = is_triggered_card_mode(args)
            card_global_logit_formula = resolve_card_global_logit_formula(args)
            use_global_custom_formula = is_card_global_custom_formula_enabled(args)
            logger.info(f"[CARD configuration]")
            logger.info(f"  Enable CARD: True")
            logger.info(f"  CARD mode (--card-application-mode): {card_display_name}")
            if (
                args.target_local_backend == "vllm"
                and not is_triggered_card
            ):
                logger.info(
                    "  Global CARD vLLM mode: global contrastive greedy decoding "
                    "(non-thinking, temperature=0.0)"
                )
                logger.info(
                    "  vLLM auxiliary branch: emptydocument prompt, deletedocumentcontent and keepcandidateproductslotposition"
                )
                logger.info(
                    "  main-branch bias coefficient b (--card-global-main-bias-coeff): "
                    f"{get_card_global_main_bias_coeff(args):g}"
                )
                logger.info(
                    "  direction signal sign (--card-global-direction-sign): "
                    f"{get_card_global_direction_sign(args)} "
                    "(1=enhance external-document contribution/debias, -1=suppress external-document contribution/poisoning defense)"
                )
                logger.info(
                    "  Global CARD vLLM formula: "
                    "z_card = (1-abs(b))*z_main + sign*alpha_t*(z_main-z_aux)"
                )
                support_mode = resolve_card_global_vllm_support_mode(args)
                logger.info(
                    "  Global CARD vLLM support mode "
                    f"(--card-global-vllm-support-mode): {support_mode}"
                )
                if support_mode == "main_aux_topk_union":
                    logger.info(
                        "  Global CARD vLLM support top-k "
                        f"(--card-global-vllm-support-top-k): "
                        f"{args.card_global_vllm_support_top_k}"
                    )
                    logger.info(
                        "  Global CARD vLLM support semantics: only in mainpointsbranch top-k "
                        "andauxiliary branch top-k unionwithinselectnext token; KL/alpha_t still byfull vocabularycompute"
                    )
                else:
                    logger.info("  Global CARD vLLM support semantics: full vocabularycandidate")
            logger.info(
                "  global logits composition formula (--card-global-logit-formula): "
                f"{card_global_logit_formula}"
            )
            if use_global_custom_formula:
                if card_global_logit_formula == "ck":
                    logger.info(
                        "  ck formula alpha (--card-global-ck-alpha): "
                        f"{args.card_global_ck_alpha:g}"
                    )
                    logger.info(
                        "  ck formula: score_final = alpha*score_aux + (1-alpha)*(score_main-score_aux)"
                    )
                else:
                    logger.info(
                        "  zxy formula alpha (--card-global-zxy-alpha): "
                        f"{args.card_global_zxy_alpha:g}"
                    )
                    logger.info(
                        "  zxy formula: score_final = score_main - alpha*score_aux"
                    )
                logger.info(
                    "  Note: when first formula does notusing --card-strength / --card-use-fixed-strength / "
                    "--card-dynamic-strength-max / --card-dynamic-alpha-recompute"
                )
            else:
                logger.info(f"  Fixed strength switch (--card-use-fixed-strength): {args.card_use_fixed_strength}")
                if args.card_use_fixed_strength:
                    logger.info(f"  CARD strength (--card-strength): {args.card_strength}")
                else:
                    logger.info(
                        f"  Dynamic strength maximum (--card-dynamic-strength-max): "
                        f"{args.card_dynamic_strength_max:g}"
                    )
                    logger.info(
                        f"  Dynamic strength computation: "
                        f"{get_card_dynamic_strength_log_text(args.card_dynamic_strength_max)}"
                    )
                    logger.info(
                        f"  Dynamic strength recomputation (--card-dynamic-alpha-recompute): "
                        f"{args.card_dynamic_alpha_recompute}"
                    )
                logger.info(f"  Probability modulation (--card-modulated-prob): {args.card_modulated_prob}")
                if args.card_modulated_prob:
                    logger.info(
                        f"  probability exponent scaling beta (--card-prob-weight-beta): {args.card_prob_weight_beta:g}"
                    )
                    logger.info(
                        "  Probability modulation semantics: compute main_probs with softmax(main_logits) over the full vocabulary, "
                        "then use it to weight diff per token; support only controls the final candidate set"
                    )
            logger.info(f"  Use Attention Mask (--card-use-attention-mask): {args.card_use_attention_mask}")
            logger.info(f"  Auxiliary prompt mode (--card-aux-prompt-type): {card_aux_prompt_type}")
            logger.info(f"  True batch inference (--card-batch-inference): {args.card_batch_inference}")
            if args.target_local_backend == "vllm" and not is_triggered_card:
                card_execution_path = "vllm paired global path"
            else:
                card_execution_path = (
                    'true batch path'
                    if args.card_batch_inference
                    else 'legacy serial single path'
                )
            logger.info(
                "  CARD execution path: "
                f"{card_execution_path}"
            )
            logger.info(f"  CARD Top-k constraint (--card-use-top-k-constraint): {args.card_use_top_k_constraint}")
            logger.info(f"  CARD candidate set size (--card-top-k): {args.card_top_k}")
            if is_triggered_card:
                logger.info(f"  Trigger Mode (--card-trigger-mode): {args.card_trigger_mode}")
                if args.card_trigger_mode == "top_k_count":
                    logger.info(f"  Trigger Top-K (--card-trigger-top-k): {args.card_trigger_top_k}")
                    logger.info(
                        "  Trigger Threshold (--card-trigger-threshold, sum of brand and model matches): "
                        f"{args.card_trigger_threshold}"
                    )
                else:
                    logger.info(
                        "  Trigger Prob-Sum Rule: sort by probability from high to low selectprobability-mass sum <= "
                        f"{args.card_trigger_prob_sum_threshold:g}  token set"
                    )
                    logger.info(
                        "  Trigger Count Rule: this set in brand/model token at least 2 whentrigger"
                    )
                logger.info(f"  Trigger Window (--card-trigger-window): {args.card_trigger_window}")
                logger.info(f"  Trigger Followup Tokens (--card-trigger-followup-tokens): {args.card_trigger_followup_tokens}")
                logger.info(
                    "  Max Trigger Count (--card-max-trigger-count): "
                    f"{args.card_max_trigger_count if args.card_max_trigger_count is not None else 'None (unlimited)'}"
                )
                logger.info(
                    f"  Filter opposite-side exclusive first token at trigger positions (--card-filter-opposite-start-tokens): "
                    f"{args.card_filter_opposite_start_tokens}"
                )
                logger.info(
                    "  Filter system-prompt few-shot example brand/model start tokens at trigger positions "
                    f"(--card-filter-system-prompt-example-tokens): "
                    f"{args.card_filter_system_prompt_example_tokens}"
                )
                logger.info(f"  Token collection mode (--card-token-collection-mode): {args.card_token_collection_mode}")
            else:
                logger.info(
                    "  Global CARD semantics: eachgeneratedstep applies use  CARD correction, does not performtriggerdecision"
                )
                logger.info(
                    "  Global CARD note: trigger, followup, and trigger_type filter-related parameters are ignored in this mode"
                )
            logger.info(f"")
        
        
        if getattr(args, 'use_card', False):
            if args.target_local_backend == "vllm":
                from global_card_vllm_models import load_global_card_vllm_model

                log_key_info(logger, f"[Model loading] Using Global CARD vLLM greedy-decoding model")
                target = load_global_card_vllm_model(
                    args.target_model,
                    card_dynamic_strength_max=args.card_dynamic_strength_max,
                    card_global_main_bias_coeff=args.card_global_main_bias_coeff,
                    card_global_direction_sign=args.card_global_direction_sign,
                    card_global_vllm_support_mode=(
                        args.card_global_vllm_support_mode
                    ),
                    card_global_vllm_support_top_k=(
                        args.card_global_vllm_support_top_k
                    ),
                    max_new_tokens=args.target_max_tokens,
                    gpu_ids=args.target_gpu_ids,
                    return_token_counts=True,
                    save_global_card_token_trace=(
                        getattr(args, 'save_global_card_token_trace', True)
                    ),
                    vllm_gpu_memory_utilization=(
                        args.target_vllm_gpu_memory_utilization
                    ),
                    vllm_max_model_len=args.target_vllm_max_model_len,
                    vllm_max_num_seqs=args.target_vllm_max_num_seqs,
                    vllm_max_num_batched_tokens=(
                        args.target_vllm_max_num_batched_tokens
                    ),
                )
            else:
                raise ValueError("CARD is packaged with the vLLM backend only; use --target-local-backend vllm")
        elif getattr(args, 'use_ck', False):
            if args.target_local_backend == "vllm":
                from ck_vllm_models import load_ck_vllm_model

                log_key_info(logger, f"[Model loading] Using CK-PLUG vLLM greedy-decoding model")
                target = load_ck_vllm_model(
                    args.target_model,
                    alpha=args.ck_alpha,
                    adaptive=args.ck_adaptive,
                    select_top=args.ck_select_top,
                    relative_top=args.ck_relative_top,
                    max_new_tokens=args.target_max_tokens,
                    gpu_ids=args.target_gpu_ids,
                    return_token_counts=True,
                    vllm_gpu_memory_utilization=(
                        args.target_vllm_gpu_memory_utilization
                    ),
                    vllm_max_model_len=args.target_vllm_max_model_len,
                    vllm_max_num_seqs=args.target_vllm_max_num_seqs,
                    vllm_max_num_batched_tokens=(
                        args.target_vllm_max_num_batched_tokens
                    ),
                )
            else:
                raise ValueError("CK-PLUG is packaged with the vLLM backend only; use --target-local-backend vllm")
        else:
            
            log_key_info(logger, f"[Model loading] Using normal model")
            
            if args.enable_thinking:
                
                
                thinking_temp = args.target_temp if args.target_temp_specified else None
                target = load_model(
                    args.target_model,
                    temperature=thinking_temp,
                    top_p=None,  
                    max_tokens=args.target_max_tokens,
                    gpu_ids=args.target_gpu_ids,
                    enable_thinking=True,
                    extract_thinking=True,
                    return_token_counts=True,
                    local_inference_backend=args.target_local_backend,
                    vllm_gpu_memory_utilization=(
                        args.target_vllm_gpu_memory_utilization
                    ),
                    vllm_max_model_len=args.target_vllm_max_model_len,
                    vllm_max_num_seqs=args.target_vllm_max_num_seqs,
                    vllm_max_num_batched_tokens=(
                        args.target_vllm_max_num_batched_tokens
                    ),
                )
            else:
                
                target = load_model(
                    args.target_model,
                    args.target_temp,
                    args.target_top_p,  
                    args.target_max_tokens,
                    gpu_ids=args.target_gpu_ids,
                    return_token_counts=True,
                    local_inference_backend=args.target_local_backend,
                    vllm_gpu_memory_utilization=(
                        args.target_vllm_gpu_memory_utilization
                    ),
                    vllm_max_model_len=args.target_vllm_max_model_len,
                    vllm_max_num_seqs=args.target_vllm_max_num_seqs,
                    vllm_max_num_batched_tokens=(
                        args.target_vllm_max_num_batched_tokens
                    ),
                )
        
        
        from models import is_remote_model
        args.is_local_model = not is_remote_model(args.target_model)
        args.local_parsing_executor = None
        args.async_parse_executor = None
        args.async_pending_parses = None
        if args.is_local_model:
            logger.info(f"")
            if getattr(args, 'local_parsing_workers', 1) > 1:
                logger.info(
                    "[Parallel parsing settings] detectedlocal model, willusing spawn safeParallel parsing"
                )
                logger.info(
                    f"  local_parsing_workers: {getattr(args, 'local_parsing_workers', 1)}"
                )
                args.local_parsing_executor = create_local_parsing_executor(
                    getattr(args, 'local_parsing_workers', 1)
                )
            else:
                logger.info(
                    "[Parallel parsing settings] detectedlocal model, will useSerial parsing (avoid tokenizer fork conflict)"
                )
        else:
            logger.info(f"")
            logger.info(f"[Parallel parsing settings] remote API model, will useParallel parsing")
            logger.info(f"  num_parsing_workers: {getattr(args, 'num_parsing_workers', 8)}")

        if args.is_local_model and not args.output_only:
            args.async_parse_executor = ThreadPoolExecutor(max_workers=1)
            args.async_pending_parses = []
            logger.info(f"")
            logger.info(
                "[Async pipeline parsing] enabledtotalshared after backgroundparseexecutor: "
                "Categoryinference continues first while advancing, submittedparse batcheswill in  after background queuefill"
            )
        
        
        available_categories = get_available_categories(args)
        if get_requested_test_categories(args):
            categories = get_requested_test_categories(args)
            available_category_set = set(available_categories)
            missing_categories = [
                category for category in categories
                if category not in available_category_set
            ]
            if missing_categories:
                print(f"Error: Could not find the following categories: {', '.join(missing_categories)}")
                print(f"Check directory: {combinations_path(args)}")
                if available_categories:
                    print(f"Available categories: {', '.join(available_categories)}")
                return
        else:
            categories = available_categories
        
        if not categories:
            print(f"Error: nohasfind to available use Category.run step3generatedbrand-documentcombination.")
            print(f"Check directory: {combinations_path(args)}")
            return
        
        print(f"\n{'='*60}")
        print(format_timed_title("Recommendation-bias experiment (improved experiment design)"))
        print(f"{'='*60}")
        print(f"Parametric knowledge model: {args.model}")
        print(f"Target recommendation model: {args.target_model}")
        print(f"Number of parametric brands: {args.num_brands}")
        print(f"Total brand count: {total_brand_count}")
        print(f"Experiment design: {get_experiment_design_text(total_brand_count)}/Category")
        print(f"Random seed: {args.experiment_seed}")
        print(f"Number of categories: {len(categories)}")
        print(f"Total API calls: {len(categories) * total_experiments_per_category}  times")
        print(f"{'='*60}")
        print(format_timed_title("[Experiment configuration]"))
        print(f"  Use ranking (--with-ranking): {args.with_ranking}")
        print(f"  Output-only mode (--output-only): {args.output_only}")
        print(f"  Disable ordering prompt (--no-ordering-prompt): {args.no_ordering_prompt}")
        print(
            f"  System Role baseline (--use-system-role-baseline): "
            f"{args.use_system_role_baseline}"
        )
        print(
            f"  Debias Instruction baseline (--use-debias-instruction-baseline): "
            f"{args.use_debias_instruction_baseline}"
        )
        print(
            f"  Moral Self-Correction baseline "
            f"(--use-moral-self-correction-baseline): "
            f"{args.use_moral_self_correction_baseline}"
        )
        print(f"  Enable CK (--use-ck): {getattr(args, 'use_ck', False)}")
        print(f"  Test categories (--test): {args.test}")
        if getattr(args, 'single_category_run_tag', None):
            print(f"  Category-targeted run tag (--single-category-run-tag): {args.single_category_run_tag}")
        print(f"  Output root directory (--out-base-dir): {args.out_base_dir}")
        print(f"  Plot root directory (--plot-base-dir): {args.plot_base_dir}")
        print(f"  Output directory: {out_path(args)}")
        print(f"  Category pkl directory: {get_category_results_dir(args)}")
        print(f"  Plot directory: {plot_path(args)}")
        print(f"  Run config file: {out_path(args, 'run_config.json')}")
        print(f"  Enable thinking mode (--enable-thinking): {args.enable_thinking}")
        print(f"  Batch size (--batch-size): {args.batch_size}")
        print(f"  Parallel parsing processes (--num-parsing-workers): {args.num_parsing_workers}")
        print(f"  Local parsing processes (--local-parsing-workers): {getattr(args, 'local_parsing_workers', 1)}")
        print(
            "  Local inference backend (--target-local-backend): "
            f"{args.target_local_backend} (normal local inference; CK vLLM / Global CARD vLLM also applies)"
        )
        print(
            "  vLLM GPU memory utilization (--target-vllm-gpu-memory-utilization): "
            f"{args.target_vllm_gpu_memory_utilization:g} (normal local vllm / CK vLLM / Global CARD vLLM applies)"
        )
        print(
            "  vLLM max_model_len (--target-vllm-max-model-len): "
            f"{args.target_vllm_max_model_len} (None meansusing vLLM defaultvalue)"
        )
        print(
            "  vLLM max_num_seqs (--target-vllm-max-num-seqs): "
            f"{args.target_vllm_max_num_seqs} (None meansusing vLLM defaultvalue)"
        )
        print(
            "  vLLM max_num_batched_tokens "
            "(--target-vllm-max-num-batched-tokens): "
            f"{args.target_vllm_max_num_batched_tokens} "
            "(None meansusing vLLM defaultvalue)"
        )
        if args.enable_thinking:
            thinking_temp = args.target_temp if args.target_temp_specified else None
            if thinking_temp is not None:
                print(f"  Temperature (--target-temp): {thinking_temp} ( use user specified)")
            else:
                print(f"  Temperature: None (usemodeldefaultvalue, thinkingmode)")
            print(f"  Top-P: None (usemodeldefaultvalue, thinkingmode)")
        else:
            print(f"  Temperature (--target-temp): {args.target_temp}")
            print(f"  Top-P (--target-top-p): {args.target_top_p if args.target_top_p is not None else 'None (not used)'}")
        print(f"  Max Tokens (--target-max-tokens): {args.target_max_tokens}")
        print(f"  GPU IDs (--target-gpu-ids): {args.target_gpu_ids}")
        print(f"  API concurrent requestscount (--api-concurrency): {args.api_concurrency}")
        print(f"  API concurrencysubmission stagger cap (--api-submit-stagger): {args.api_submit_stagger}")
        print(f"  API max retries (--api-max-retries): {args.api_max_retries}")
        print(
            f"  API initial wait time (--api-retry-initial-delay): "
            f"{args.api_retry_initial_delay}"
        )
        print(
            f"  API backoff factor (--api-retry-backoff): {args.api_retry_backoff}"
        )
        print(
            f"  API max wait time (--api-retry-max-delay): "
            f"{args.api_retry_max_delay}"
        )
        if getattr(args, 'use_ck', False):
            print(f"[CK configuration]")
            if args.target_local_backend == "vllm":
                print(
                    "  CK vLLM mode: CK-PLUG greedy decoding reproduction "
                    "(non-thinking, temperature=0.0, greedy-only)"
                )
            print("  Parametric-branch prompt: delete document body and preserve candidate product slots")
            print(f"  Adaptive mode (--ck-adaptive): {args.ck_adaptive}")
            if args.ck_adaptive:
                print("  alpha: adaptive (dynamically scaled by entropy/conflict)")
            else:
                print(f"  Fixed alpha (--ck-alpha): {args.ck_alpha:g}")
            print(f"  Select Top (--ck-select-top): {args.ck_select_top}")
            print(f"  Relative Top (--ck-relative-top): {args.ck_relative_top:g}")
        if getattr(args, 'use_card', False):
            card_aux_prompt_type = resolve_card_aux_prompt_type(args)
            card_display_name = get_card_display_name(args)
            is_triggered_card = is_triggered_card_mode(args)
            card_global_logit_formula = resolve_card_global_logit_formula(args)
            use_global_custom_formula = is_card_global_custom_formula_enabled(args)
            print(f"[CARD configuration]")
            print(f"  CARD mode (--card-application-mode): {card_display_name}")
            if args.target_local_backend == "vllm" and not is_triggered_card:
                print(
                    "  Global CARD vLLM mode: global contrastive greedy decoding "
                    "(non-thinking, temperature=0.0)"
                )
                print("  vLLM auxiliary branch: emptydocument prompt, deletedocumentcontent and keepcandidateproductslotposition")
                print(
                    "  main-branch bias coefficient b (--card-global-main-bias-coeff): "
                    f"{get_card_global_main_bias_coeff(args):g}"
                )
                print(
                    "  direction signal sign (--card-global-direction-sign): "
                    f"{get_card_global_direction_sign(args)} "
                    "(1=enhance external-document contribution/debias, -1=suppress external-document contribution/poisoning defense)"
                )
                print(
                    "  Global CARD vLLM formula: "
                    "z_card = (1-abs(b))*z_main + sign*alpha_t*(z_main-z_aux)"
                )
                support_mode = resolve_card_global_vllm_support_mode(args)
                print(
                    "  Global CARD vLLM support mode "
                    f"(--card-global-vllm-support-mode): {support_mode}"
                )
                if support_mode == "main_aux_topk_union":
                    print(
                        "  Global CARD vLLM support top-k "
                        f"(--card-global-vllm-support-top-k): "
                        f"{args.card_global_vllm_support_top_k}"
                    )
                    print(
                        "  Global CARD vLLM support semantics: only in mainpointsbranch top-k "
                        "andauxiliary branch top-k unionwithinselectnext token; KL/alpha_t still byfull vocabularycompute"
                    )
                else:
                    print("  Global CARD vLLM support semantics: full vocabularycandidate")
            print(
                "  global logits composition formula (--card-global-logit-formula): "
                f"{card_global_logit_formula}"
            )
            if use_global_custom_formula:
                if card_global_logit_formula == "ck":
                    print(
                        "  ck formula alpha (--card-global-ck-alpha): "
                        f"{args.card_global_ck_alpha:g}"
                    )
                    print(
                        "  ck formula: score_final = alpha*score_aux + (1-alpha)*(score_main-score_aux)"
                    )
                else:
                    print(
                        "  zxy formula alpha (--card-global-zxy-alpha): "
                        f"{args.card_global_zxy_alpha:g}"
                    )
                    print("  zxy formula: score_final = score_main - alpha*score_aux")
                print(
                    "  Note: when first formula does notusing --card-strength / --card-use-fixed-strength / "
                    "--card-dynamic-strength-max / --card-dynamic-alpha-recompute"
                )
            else:
                print(f"  Fixed strength switch (--card-use-fixed-strength): {args.card_use_fixed_strength}")
                if args.card_use_fixed_strength:
                    print(f"  CARD strength (--card-strength): {args.card_strength}")
                else:
                    print(
                        f"  Dynamic strength maximum (--card-dynamic-strength-max): "
                        f"{args.card_dynamic_strength_max:g}"
                    )
                    print(
                        f"  Dynamic strength computation: "
                        f"{get_card_dynamic_strength_log_text(args.card_dynamic_strength_max)}"
                    )
                    print(
                        f"  Dynamic strength recomputation (--card-dynamic-alpha-recompute): "
                        f"{args.card_dynamic_alpha_recompute}"
                    )
                print(f"  Probability modulation (--card-modulated-prob): {args.card_modulated_prob}")
                if args.card_modulated_prob:
                    print(f"  probability exponent scaling beta (--card-prob-weight-beta): {args.card_prob_weight_beta:g}")
                    print(
                        "  Probability modulation semantics: compute main_probs with softmax(main_logits) over the full vocabulary, "
                        "then use it to weight diff per token; support only controls the final candidate set"
                    )
            print(f"  Use Attention Mask (--card-use-attention-mask): {args.card_use_attention_mask}")
            print(f"  Auxiliary prompt mode (--card-aux-prompt-type): {card_aux_prompt_type}")
            print(f"  True batch inference (--card-batch-inference): {args.card_batch_inference}")
            if args.target_local_backend == "vllm" and not is_triggered_card:
                card_execution_path = "vllm paired global path"
            else:
                card_execution_path = (
                    'true batch path'
                    if args.card_batch_inference
                    else 'legacy serial single path'
                )
            print(
                "  CARD execution path: "
                f"{card_execution_path}"
            )
            print(f"  CARD Top-k constraint (--card-use-top-k-constraint): {args.card_use_top_k_constraint}")
            print(f"  CARD candidate set size (--card-top-k): {args.card_top_k}")
            if is_triggered_card:
                print(f"  Trigger Mode (--card-trigger-mode): {args.card_trigger_mode}")
                if args.card_trigger_mode == "top_k_count":
                    print(f"  Trigger Top-K (--card-trigger-top-k): {args.card_trigger_top_k}")
                    print(
                        "  Trigger Threshold (--card-trigger-threshold, sum of brand and model matches): "
                        f"{args.card_trigger_threshold}"
                    )
                else:
                    print(
                        "  Trigger Prob-Sum Rule: sort by probability from high to low selectprobability-mass sum <= "
                        f"{args.card_trigger_prob_sum_threshold:g}  token set"
                    )
                    print("  Trigger Count Rule: this set in brand/model token at least 2 whentrigger")
                print(f"  Trigger Window (--card-trigger-window): {args.card_trigger_window}")
                print(f"  Trigger Followup Tokens (--card-trigger-followup-tokens): {args.card_trigger_followup_tokens}")
                print(
                    "  Max Trigger Count (--card-max-trigger-count): "
                    f"{args.card_max_trigger_count if args.card_max_trigger_count is not None else 'None (unlimited)'}"
                )
                print(
                    f"  Filter opposite-side exclusive first token at trigger positions (--card-filter-opposite-start-tokens): "
                    f"{args.card_filter_opposite_start_tokens}"
                )
                print(
                    "  Filter system-prompt few-shot example brand/model start tokens at trigger positions "
                    f"(--card-filter-system-prompt-example-tokens): "
                    f"{args.card_filter_system_prompt_example_tokens}"
                )
                print(f"  Token collection mode (--card-token-collection-mode): {args.card_token_collection_mode}")
            else:
                print(
                    "  Global CARD semantics: eachgeneratedstep applies use  CARD correction, does not performtriggerdecision"
                )
                print(
                    "  Global CARD note: trigger, followup, and trigger_type filter-related parameters are ignored in this mode"
                )
        print(f"{'='*60}")
        
        if is_single_test_category(args):
            category_safe = sanitize_category_name(get_single_test_category(args))
            results_filename = f"results_{category_safe}.pkl"
        else:
            results_filename = "results.pkl"
        results_path = out_path(args, results_filename)
        category_results_dir = get_category_results_dir(args)
        file_utils.ensure_created_directory(category_results_dir)

        results = {}
        deferred_category_finalizers: t.List[
            t.Tuple[str, t.Callable[[], t.Dict[str, t.Any]]]
        ] = []

        def drain_async_parse_queue() -> None:
            pending_parses = getattr(args, 'async_pending_parses', None)
            if not pending_parses:
                return

            logger.info(f"")
            logger.info(
                f"[Async pipeline parsing] start draining shared pending queue, "
                f"remaining batches={len(pending_parses)}"
            )
            while pending_parses:
                pending_parse = pending_parses.pop(0)
                wait_start_time = time.perf_counter()
                parse_result = pending_parse['future'].result()
                async_parse_wait_elapsed = time.perf_counter() - wait_start_time
                pending_parse['process_fn'](
                    prepared_batch_state=pending_parse['prepared_batch_state'],
                    preparsed_batch_result=parse_result,
                    async_parse_wait_elapsed=async_parse_wait_elapsed,
                )

        try:
            for category in tqdm.tqdm(
                categories,
                desc=format_timed_title("Category"),
                disable=(len(categories) <= 1),
            ):
                try:
                    defer_category_finalization = bool(
                        getattr(args, 'async_parse_executor', None) is not None
                    )
                    category_result_or_builder = run_category_experiment(
                        category=category,
                        target=target,
                        args=args,
                        experiment_seed=args.experiment_seed,
                        defer_finalization=defer_category_finalization,
                    )
                    if defer_category_finalization:
                        deferred_category_finalizers.append(
                            (category, t.cast(t.Callable[[], t.Dict[str, t.Any]], category_result_or_builder))
                        )
                    else:
                        category_result = t.cast(t.Dict[str, t.Any], category_result_or_builder)
                        results[category] = category_result
                        category_result_path = get_category_result_path(args, category)
                        file_utils.write_pickle(
                            category_result_path,
                            prepare_category_result_for_pickle(category_result),
                        )
                        print(f"✅ Current category result saved to: {category_result_path}")
                        file_utils.write_pickle(
                            results_path,
                            prepare_results_for_pickle(results),
                        )
                        print(f"\n✅ Current partial results saved to: {results_path}")
                except Exception as e:
                    print(f"\n❌ Category '{category}' experiment failed: {e}")
                    continue

            if getattr(args, 'async_parse_executor', None) is not None:
                target = None
                released_model_cache = release_target_model_cache_after_run(args)
                if released_model_cache:
                    logger.info(f"")
                    logger.info(
                        "[vLLMGPU memory release] inference stage is complete, clearedemptythislocal vLLM / "
                        "CK vLLM / Global CARD vLLM cache"
                    )
                drain_async_parse_queue()
            if deferred_category_finalizers:
                for category, category_result_builder in deferred_category_finalizers:
                    try:
                        category_result = category_result_builder()
                        results[category] = category_result
                        category_result_path = get_category_result_path(args, category)
                        file_utils.write_pickle(
                            category_result_path,
                            prepare_category_result_for_pickle(category_result),
                        )
                        print(f"✅ Current category result saved to: {category_result_path}")
                        file_utils.write_pickle(
                            results_path,
                            prepare_results_for_pickle(results),
                        )
                        print(f"\n✅ Current partial results saved to: {results_path}")
                    except Exception as e:
                        print(f"\n❌ Category '{category}' result finalization failed: {e}")
                        continue

            print(f"\n✅ Results saved to: {results_path}")
        finally:
            if getattr(args, 'local_parsing_executor', None) is not None:
                args.local_parsing_executor.shutdown(wait=True)
                args.local_parsing_executor = None
            if getattr(args, 'async_parse_executor', None) is not None:
                args.async_parse_executor.shutdown(wait=True)
                args.async_parse_executor = None
                args.async_pending_parses = None
            release_target_model_cache_after_run(args)

    output_only_mode = bool(getattr(args, 'output_only', False))

    f_stat_results = None

    
    if args.run_eval and output_only_mode:
        skip_msg = (
            "[Output mode] --output-only enabled, skipped analyze_and_plot and brand-type main-effect F statistics"
        )
        print(skip_msg)
        logger = get_logger()
        if len(logger.handlers) > 0:
            logger.info(skip_msg)
    else:
        f_stat_results = analyze_and_plot(args)

    
    if args.run_eval and results:
        logger = get_logger()
        if len(logger.handlers) > 0:
            def log_tail_summaries_without_prefix() -> None:
                log_api_retry_statistics_at_end(results, args, logger)
                log_output_token_statistics_at_end(results, args, logger)
                if output_only_mode:
                    logger.info(
                        "[Output mode] --output-only enabled, skip log-tail inference/parsing time statistics and CARD trigger statistics output"
                    )
                else:
                    if getattr(args, 'use_card', False) and is_triggered_card_mode(args):
                        log_card_trigger_statistics_by_category(results, args, logger)
                    log_inference_time_statistics_at_end(
                        results,
                        args,
                        logger,
                        results_path,
                    )
                    if f_stat_results is not None:
                        log_category_summary_statistics_at_end(
                            results,
                            f_stat_results,
                            logger,
                        )
                        log_overall_summary_statistics_at_end(
                            results,
                            f_stat_results,
                            logger,
                        )
                        log_brand_type_main_effect_f_at_end(results, logger)

            set_plain_log_formatter_temporarily(
                logger,
                log_tail_summaries_without_prefix,
            )

        brand_type_stats = None
        if not output_only_mode:
            brand_type_stats = compute_brand_type_main_effect_f_brand_level(results)

        run_summary_path = write_recommendation_bias_run_summary(
            results=results,
            f_stat_results=f_stat_results,
            brand_type_stats=brand_type_stats,
            args=args,
            results_path=results_path,
        )
        if len(logger.handlers) > 0:
            logger.info(f"lightweight run summary file: {run_summary_path}")

        if brand_type_stats is not None:
            print_brand_type_main_effect_f_for_nohup(brand_type_stats)
        print(f"lightweight run summary saved to: {run_summary_path}", flush=True)


def run_category_experiment(
    category: str,
    target: t.Callable,
    args: argparse.Namespace,
    experiment_seed: int = 42,
    defer_finalization: bool = False,
) -> t.Union[t.Dict, t.Callable[[], t.Dict]]:
    logger = get_logger()
    output_only_mode = bool(getattr(args, 'output_only', False))
    
    
    
    random.seed(experiment_seed)
    np.random.seed(experiment_seed)
    
    logger.info(f"")
    logger.info(f"{'='*80}")
    log_key_info(logger, f"[Category] {category} - start experiment")
    logger.info(f"{'='*80}")
    logger.info(f"")
    logger.info(f"[Random seed] {experiment_seed}")
    
    
    brands_original, docs_original = load_brand_doc_combinations(category, args)
    product_n = len(brands_original)
    logger.info(f"[Experiment design] Latin-square design: {get_experiment_design_text(product_n)} experiment tasks")
    logger.info(f"[Total brand count] {product_n}")
    
    user_query = dataset.user_query(category)
    logger.info(f"[User query] {user_query}")
    
    
    logger.info(f"")
    logger.info(f"[Original brand order] (loaded from dataset)")
    for i, b in enumerate(brands_original):
        fictional_mark = "[fictional]" if b.is_fictional else "[real]"
        logger.info(f"  original position[{i}]: {b.brand} - {b.model} | knowledge strength: {b.knowledge_strength:.4f} {fictional_mark}")
    
    
    brand_shuffle_indices = list(range(product_n))
    random.shuffle(brand_shuffle_indices)
    brands = [brands_original[i] for i in brand_shuffle_indices]
    
    
    
    doc_shuffle_indices = list(range(product_n))
    random.shuffle(doc_shuffle_indices)
    
    docs = [
        [docs_original[brand_shuffle_indices[bi]][doc_shuffle_indices[di]] for di in range(product_n)]
        for bi in range(product_n)
    ]
    
    
    products = brands_to_products(brands, category)
    
    logger.info(f"")
    logger.info(f"[Shuffled brand order] (experimentuse)")
    logger.info(f"  brand shuffle mapping: {brand_shuffle_indices}")
    for i, b in enumerate(brands):
        fictional_mark = "[fictional]" if b.is_fictional else "[real]"
        logger.info(f"  B[{i}]: {b.brand} - {b.model} | knowledge strength: {b.knowledge_strength:.4f} {fictional_mark}")
    
    logger.info(f"")
    logger.info(f"[Shuffled document order]")
    logger.info(f"  document shuffle mapping: {doc_shuffle_indices}")
    logger.info(f"  Note: D[i] now corresponds to original document-templaterank  {doc_shuffle_indices} position")
    
    
    
    L1 = get_random_latin_square(product_n)
    
    logger.info(f"")
    logger.info(f"[Global Latin square L1] (Brand-Doc pairing)")
    logger.info(f"  Meaning: L1[brand][block] = doc_idx")
    logger.info(f"")
    logger.info(f"  Matrix form (rows=Brand, columns=Block, values=Doc):")
    header = "        " + "".join([f"Blk{b:2d} " for b in range(product_n)])
    logger.info(f"  {header}")
    for i in range(product_n):
        row_str = f"  B[{i}] {brands[i].brand[:8]:8s} " + "".join([f"  D{L1[i][b]:1d}  " for b in range(product_n)])
        logger.info(row_str)
    
    valid_index_text = f"0-{product_n - 1}"

    
    logger.info(f"")
    logger.info(f"  [L1 validity check]")
    l1_valid = True
    for i in range(product_n):
        if sorted(L1[i]) != list(range(product_n)):
            logger.info(f"    ❌ rank  {i} rowis not {valid_index_text} permutation")
            l1_valid = False
    for j in range(product_n):
        col = [L1[i][j] for i in range(product_n)]
        if sorted(col) != list(range(product_n)):
            logger.info(f"    ❌ rank  {j} columnis not {valid_index_text} permutation")
            l1_valid = False
    if l1_valid:
        logger.info(f"    ✓ L1 is a valid Latin square (every row and column is {valid_index_text} permutation)")
    
    
    
    category_scores = (
        {}
        if output_only_mode
        else {
            (brand_index, doc_index, context_position): []
            for brand_index in range(product_n)
            for doc_index in range(product_n)
            for context_position in range(product_n)
        }
    )
    
    category_experiment_records = []  
    category_responses = []
    
    
    thinking_token_counts = []  
    response_token_counts = []  
    response_paragraph_counts = []  
    response_output_limit_reached_flags = []  
    response_finish_reasons = []  
    response_stop_reasons = []  
    response_token_count_source_api_count = 0
    response_token_count_source_local_exact_count = 0
    response_token_count_source_exact_count = 0
    response_token_count_source_estimated_count = 0
    inference_call_durations: t.List[float] = []
    inference_response_counts: t.List[int] = []
    parsing_call_durations: t.List[float] = []
    parsing_response_counts: t.List[int] = []
    api_retry_enabled = not getattr(args, 'is_local_model', False)
    api_failed_prompts_path = get_api_failed_prompts_path(args)
    api_recovered_prompts_path = get_api_recovered_prompts_path(args)
    api_success_response_count = 0
    api_recovered_prompt_count = 0
    api_final_failed_prompt_count = 0
    api_total_attempt_count = 0

    
    trigger_prob_records = []
    trigger_probs_by_rank: t.Dict[int, t.Dict[str, t.List[float]]] = {}
    
    total_experiments = product_n * product_n
    experiment_count = 0
    category_level_batching_enabled = (
        (not output_only_mode)
        and getattr(args, 'is_local_model', False)
    )
    async_pipeline_parsing_enabled = category_level_batching_enabled
    async_parse_max_pending_batches = max(
        1,
        int(getattr(args, 'async_parse_max_pending_batches', 8)),
    )
    shared_async_parse_executor = getattr(args, 'async_parse_executor', None)
    shared_async_pending_parses = getattr(args, 'async_pending_parses', None)
    if async_pipeline_parsing_enabled:
        logger.info(f"")
        logger.info(
            "[Async pipeline parsing] enabled: local backendinference continues first while advancing,  after backgroundparsealreadycompletebatch; "
            "inference batchlog immediatelyoutput, parsedetails byoriginal batchorderfill"
        )
        logger.info(
            f"[Async pipeline parsing] keep at most {async_parse_max_pending_batches}  "
            "pending parse batches, when exceeded, wait for the earliest batch in order"
        )
        logger.info(
            "[Async pipeline parsing] Note: per-run parse details may appear after the corresponding inference-batch log; "
            "use the explicit elapsed times and final statistics in the log for inference and parsing time"
        )
        if shared_async_parse_executor is not None and shared_async_pending_parses is not None:
            logger.info(
                "[Async pipeline parsing] current category uses the shared cross-category background parsing queue"
            )

    def prepare_response_batch_state(
        current_batch_responses: t.List,
        current_batch_prompt_orders: t.List[t.List[t.Tuple[int, int]]],
    ) -> t.Dict[str, t.Any]:
        current_batch_thinking_contents = []
        current_batch_response_contents = []
        current_batch_marked_response_contents = []
        current_batch_aux_user_messages = []
        current_batch_card_triggers = []
        current_batch_global_card_token_traces = []
        current_batch_thinking_tokens = []
        current_batch_response_tokens = []
        current_batch_token_count_sources = []
        current_batch_finish_reasons = []
        current_batch_stop_reasons = []
        current_batch_hit_length_limits = []

        for resp in current_batch_responses:
            if isinstance(resp, dict) and "message" in resp:
                current_batch_thinking_contents.append(resp.get("thinking"))
                current_batch_response_contents.append(resp["message"].content)
                current_batch_marked_response_contents.append(resp.get("marked_response"))
                current_batch_aux_user_messages.append(resp.get("aux_user_message"))
                current_batch_card_triggers.append(resp.get("card_triggers", []))
                current_batch_global_card_token_traces.append(
                    resp.get("global_card_token_trace")
                )
                current_batch_thinking_tokens.append(resp.get("thinking_tokens"))
                current_batch_response_tokens.append(resp.get("response_tokens"))
                current_batch_token_count_sources.append(resp.get("token_count_source"))
                current_batch_finish_reasons.append(resp.get("finish_reason"))
                current_batch_stop_reasons.append(resp.get("stop_reason"))
                current_batch_hit_length_limits.append(resp.get("hit_length_limit"))
            else:
                current_batch_thinking_contents.append(None)
                current_batch_response_contents.append(resp.content)
                current_batch_marked_response_contents.append(None)
                current_batch_aux_user_messages.append(None)
                current_batch_card_triggers.append([])
                current_batch_global_card_token_traces.append(None)
                current_batch_thinking_tokens.append(None)
                current_batch_response_tokens.append(None)
                current_batch_token_count_sources.append(None)
                current_batch_finish_reasons.append(None)
                current_batch_stop_reasons.append(None)
                current_batch_hit_length_limits.append(None)

        current_batch_products_list = []
        for prompt_order in current_batch_prompt_orders:
            ordered_products = []
            for physical_pos in range(product_n):
                brand_idx, _doc_idx = prompt_order[physical_pos]
                ordered_products.append(products[brand_idx])
            current_batch_products_list.append(ordered_products)

        return {
            'current_batch_size': len(current_batch_responses),
            'current_batch_thinking_contents': current_batch_thinking_contents,
            'current_batch_response_contents': current_batch_response_contents,
            'current_batch_marked_response_contents': current_batch_marked_response_contents,
            'current_batch_aux_user_messages': current_batch_aux_user_messages,
            'current_batch_card_triggers': current_batch_card_triggers,
            'current_batch_global_card_token_traces': current_batch_global_card_token_traces,
            'current_batch_thinking_tokens': current_batch_thinking_tokens,
            'current_batch_response_tokens': current_batch_response_tokens,
            'current_batch_token_count_sources': current_batch_token_count_sources,
            'current_batch_finish_reasons': current_batch_finish_reasons,
            'current_batch_stop_reasons': current_batch_stop_reasons,
            'current_batch_hit_length_limits': current_batch_hit_length_limits,
            'current_batch_products_list': current_batch_products_list,
        }

    def parse_prepared_response_batch(
        prepared_batch_state: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:
        current_batch_response_contents = prepared_batch_state[
            'current_batch_response_contents'
        ]
        current_batch_products_list = prepared_batch_state[
            'current_batch_products_list'
        ]
        current_batch_size = prepared_batch_state['current_batch_size']

        parsing_start_time = time.perf_counter()
        if getattr(args, 'is_local_model', False):
            local_parsing_executor = getattr(args, 'local_parsing_executor', None)
            local_parsing_workers = getattr(args, 'local_parsing_workers', 1)
            if local_parsing_executor is not None and local_parsing_workers > 1:
                tasks = eval_utils.build_parallel_parse_tasks_from_texts(
                    current_batch_response_contents,
                    current_batch_products_list,
                )
                futures = [
                    local_parsing_executor.submit(
                        eval_utils.parse_single_response_worker,
                        response_text,
                        products_data,
                    )
                    for response_text, products_data in tasks
                ]
                results = [future.result() for future in futures]
                all_product_scores, all_log_info = (
                    eval_utils.deserialize_parallel_parse_results(
                        results,
                        current_batch_products_list,
                    )
                )
            else:
                all_product_scores = []
                all_log_info = []
                for i in range(current_batch_size):
                    product_scores, log_info = get_scores_for_products_with_logs(
                        current_batch_response_contents[i],
                        current_batch_products_list[i],
                    )
                    all_product_scores.append(product_scores)
                    all_log_info.append(log_info)
        else:
            temp_responses = [
                Message(role=Role.assistant, content=content)
                for content in current_batch_response_contents
            ]
            all_product_scores, all_log_info = parse_responses_parallel(
                batch_responses=temp_responses,
                batch_products_list=current_batch_products_list,
                num_workers=getattr(args, 'num_parsing_workers', 8),
            )

        return {
            'all_product_scores': all_product_scores,
            'all_log_info': all_log_info,
            'parsing_elapsed': time.perf_counter() - parsing_start_time,
        }

    def process_response_batch(
        current_batch_responses: t.List,
        current_batch_run_records: t.List[t.Dict[str, t.Any]],
        batch_label: str,
        prepared_batch_state: t.Optional[t.Dict[str, t.Any]] = None,
        preparsed_batch_result: t.Optional[t.Dict[str, t.Any]] = None,
        async_parse_wait_elapsed: t.Optional[float] = None,
    ) -> float:
        nonlocal response_token_count_source_api_count
        nonlocal response_token_count_source_local_exact_count
        nonlocal response_token_count_source_exact_count
        nonlocal response_token_count_source_estimated_count

        if prepared_batch_state is None:
            prepared_batch_state = prepare_response_batch_state(
                current_batch_responses,
                [record['prompt_order'] for record in current_batch_run_records],
            )

        current_batch_size = prepared_batch_state['current_batch_size']
        if current_batch_size == 0:
            return 0.0
        current_batch_thinking_contents = prepared_batch_state[
            'current_batch_thinking_contents'
        ]
        current_batch_response_contents = prepared_batch_state[
            'current_batch_response_contents'
        ]
        current_batch_marked_response_contents = prepared_batch_state[
            'current_batch_marked_response_contents'
        ]
        current_batch_aux_user_messages = prepared_batch_state[
            'current_batch_aux_user_messages'
        ]
        current_batch_card_triggers = prepared_batch_state[
            'current_batch_card_triggers'
        ]
        current_batch_global_card_token_traces = prepared_batch_state[
            'current_batch_global_card_token_traces'
        ]
        current_batch_thinking_tokens = prepared_batch_state[
            'current_batch_thinking_tokens'
        ]
        current_batch_response_tokens = prepared_batch_state[
            'current_batch_response_tokens'
        ]
        current_batch_token_count_sources = prepared_batch_state[
            'current_batch_token_count_sources'
        ]
        current_batch_finish_reasons = prepared_batch_state[
            'current_batch_finish_reasons'
        ]
        current_batch_stop_reasons = prepared_batch_state[
            'current_batch_stop_reasons'
        ]
        current_batch_hit_length_limits = prepared_batch_state[
            'current_batch_hit_length_limits'
        ]
        current_batch_products_list = prepared_batch_state[
            'current_batch_products_list'
        ]

        
        all_product_scores = [None] * current_batch_size
        all_log_info = [None] * current_batch_size
        parsing_elapsed = 0.0
        if output_only_mode:
            logger.info(f"")
            log_key_info(
                logger,
                f"[Response parsing] --output-only enabled, skip {batch_label} "
                "matching and score parsing"
            )
        else:
            logger.info(f"")
            log_key_info(
                logger,
                f"[Response parsing] Starting parsing of {batch_label} responses"
            )
            if preparsed_batch_result is not None:
                all_product_scores = preparsed_batch_result['all_product_scores']
                all_log_info = preparsed_batch_result['all_log_info']
                parsing_elapsed = float(preparsed_batch_result['parsing_elapsed'])
                local_parsing_workers = getattr(args, 'local_parsing_workers', 1)
                parsing_method = (
                    f"{local_parsing_workers}  spawn processes"
                    if local_parsing_workers > 1
                    else "background serial parsing thread"
                )
                logger.info(
                    "[Async pipeline parsing] using background parsing results; "
                    f"parsing method: {parsing_method}; "
                    f"actual parse elapsed time {parsing_elapsed:.4f} s, "
                    f"waitresult wait elapsed {float(async_parse_wait_elapsed or 0.0):.4f} s"
                )
            else:
                if getattr(args, 'is_local_model', False):
                    local_parsing_workers = getattr(args, 'local_parsing_workers', 1)
                    if local_parsing_workers > 1:
                        logger.info(
                            f"[Local safe parallel parsing] Starting parsing of {current_batch_size} responses, "
                            f"using {local_parsing_workers}  spawn processes"
                        )
                    else:
                        logger.info(f"[Serial parsing] local model, useSerial parsing")
                parse_result = parse_prepared_response_batch(prepared_batch_state)
                all_product_scores = parse_result['all_product_scores']
                all_log_info = parse_result['all_log_info']
                parsing_elapsed = float(parse_result['parsing_elapsed'])
                if getattr(args, 'is_local_model', False):
                    local_parsing_workers = getattr(args, 'local_parsing_workers', 1)
                    if local_parsing_workers > 1:
                        logger.info(
                            f"[Local safe parallel parsing] complete, parsed {len(all_product_scores)} responses, "
                            f"using {local_parsing_workers}  spawn processes"
                        )
                    else:
                        logger.info(f"[Serial parsing] complete, parsed {len(all_product_scores)} responses")

            parsing_call_durations.append(parsing_elapsed)
            parsing_response_counts.append(current_batch_size)
            log_key_info(logger, f"[Response parsing] parsing complete, elapsed {parsing_elapsed:.4f} s")

        
        for local_idx in range(current_batch_size):
            run_record = current_batch_run_records[local_idx]
            block_idx = run_record['block_idx']
            global_run_idx = run_record['run_idx']
            L2_b = run_record['L2_b']
            doc_assignment = run_record['doc_assignment']

            
            thinking_content = current_batch_thinking_contents[local_idx]
            target_response = current_batch_response_contents[local_idx]  
            marked_response = current_batch_marked_response_contents[local_idx]  
            aux_user_message = current_batch_aux_user_messages[local_idx]  
            card_triggers = current_batch_card_triggers[local_idx]
            prompt_order = run_record['prompt_order']
            brand_to_doc_map = run_record['brand_to_doc_map']
            brand_to_position_map = run_record['brand_to_position_map']

            
            product_scores = all_product_scores[local_idx]
            log_info = all_log_info[local_idx]

            
            response_paragraph_counts.append(
                count_response_paragraphs(target_response)
            )

            
            ordered_products = current_batch_products_list[local_idx]

            
            current_experiment = run_record['experiment_idx']
            logger.info(f"")
            logger.info(
                f"===== Block {block_idx + 1}, Run {global_run_idx + 1} "
                f"(Experiment {current_experiment}/{total_experiments}) ====="
            )

            
            logger.info(f"")
            logger.info(f"[User message sent to the LLM (User Message)]")
            logger.info(f"{run_record['target_message']}")

            if aux_user_message:
                logger.info(f"")
                logger.info(f"[Auxiliary user message sent to the LLM (Aux User Message)]")
                logger.info(f"{aux_user_message}")

            logger.info(f"")
            logger.info(f"[Position-assignment validation] (determined by L2_{block_idx} rank  {global_run_idx} column)")
            logger.info(f"  Check: L2_{block_idx}[brand][{global_run_idx}] = position")
            for brand_idx in range(product_n):
                expected_pos = L2_b[brand_idx][global_run_idx]
                actual_pos = brand_to_position_map[brand_idx]
                match_mark = "✓" if expected_pos == actual_pos else "❌"
                logger.info(
                    f"    {match_mark} B[{brand_idx}] -> P{actual_pos} "
                    f"(L2[{brand_idx}][{global_run_idx}]={expected_pos})"
                )

            logger.info(f"")
            logger.info(f"[Prompt order] (by physical position P0-P{product_n - 1})")
            for physical_pos in range(product_n):
                brand_idx, doc_idx = prompt_order[physical_pos]
                brand = brands[brand_idx]
                product = products[brand_idx]
                fictional_mark = "[fictional]" if brand.is_fictional else "[real]"
                logger.info(
                    f"    P{physical_pos}: B[{brand_idx}] {brand.brand[:12]:12s} | "
                    f"D[{doc_idx}] | Model: {product.model[:20]} {fictional_mark}"
                )

            
            logger.info(f"")
            logger.info(f"[Full LLM output]")
            if thinking_content:
                
                logger.info(f"--- Thinking process ---")
                logger.info(f"{thinking_content}")
                logger.info(f"--- Final answer (raw) ---")
                logger.info(f"{target_response}")
                if (
                    getattr(args, 'use_card', False)
                    and is_triggered_card_mode(args)
                    and marked_response
                ):
                    logger.info(f"--- Final answer (trigger-marked) ---")
                    logger.info(f"{marked_response}")
            else:
                
                logger.info(f"--- Final answer (raw) ---")
                logger.info(f"{target_response}")
                if (
                    getattr(args, 'use_card', False)
                    and is_triggered_card_mode(args)
                    and marked_response
                ):
                    logger.info(f"--- Final answer (trigger-marked) ---")
                    logger.info(f"{marked_response}")

            if (
                (not output_only_mode)
                and getattr(args, 'use_card', False)
                and is_triggered_card_mode(args)
                and marked_response
            ):
                logger.info(f"[CARD trigger statistics] trigger count: {len(card_triggers)}")
                if card_triggers:
                    logger.info(f"[CARD trigger-point probability] (pre-CARD, by trigger rank)")
                    for trigger_rank, trigger in enumerate(card_triggers, 1):
                        main_output_token_id = trigger.get(
                            "main_output_token_id_pre_card",
                            trigger.get("main_output_token_id", trigger.get("predicted_token_id")),
                        )
                        aux_output_token_id = trigger.get(
                            "aux_output_token_id_pre_card",
                            trigger.get("aux_output_token_id"),
                        )
                        main_output_prob = trigger.get(
                            "main_output_token_prob_pre_card",
                            trigger.get("main_output_token_prob"),
                        )
                        aux_output_prob = trigger.get(
                            "aux_output_token_prob_pre_card",
                            trigger.get("aux_output_token_prob"),
                        )
                        kl_main_vs_aux = trigger.get("kl_main_vs_aux")
                        kl_aux_vs_main = trigger.get("kl_aux_vs_main")
                        jsd_main_aux = trigger.get("jsd_main_aux")

                        if (main_output_prob is None) or (aux_output_prob is None):
                            logger.info(
                                f"  rank {trigger_rank}trigger point: probability fields are missing, current trigger metadata is incomplete"
                            )
                            continue

                        main_output_prob = float(main_output_prob)
                        aux_output_prob = float(aux_output_prob)

                        trigger_probs_by_rank.setdefault(
                            trigger_rank,
                            {
                                "main_output_probs": [],
                                "aux_output_probs": [],
                                "strength_values": [],
                                "kl_main_vs_aux_values": [],
                                "kl_aux_vs_main_values": [],
                                "jsd_main_aux_values": [],
                            },
                        )
                        trigger_probs_by_rank[trigger_rank]["main_output_probs"].append(
                            main_output_prob
                        )
                        trigger_probs_by_rank[trigger_rank]["aux_output_probs"].append(
                            aux_output_prob
                        )
                        if trigger.get("strength") is not None:
                            trigger_probs_by_rank[trigger_rank]["strength_values"].append(
                                float(trigger.get("strength"))
                            )
                        if kl_main_vs_aux is not None:
                            trigger_probs_by_rank[trigger_rank]["kl_main_vs_aux_values"].append(
                                float(kl_main_vs_aux)
                            )
                        if kl_aux_vs_main is not None:
                            trigger_probs_by_rank[trigger_rank]["kl_aux_vs_main_values"].append(
                                float(kl_aux_vs_main)
                            )
                        if jsd_main_aux is not None:
                            trigger_probs_by_rank[trigger_rank]["jsd_main_aux_values"].append(
                                float(jsd_main_aux)
                            )

                        trigger_record = {
                            "category": category,
                            "block_idx": block_idx,
                            "run_idx": global_run_idx,
                            "experiment_idx": current_experiment,
                            "trigger_rank": trigger_rank,
                            "trigger_position": trigger.get("position"),
                            "trigger_type": trigger.get("trigger_type"),
                            "trigger_type_resolution": trigger.get("trigger_type_resolution"),
                            "trigger_mode": trigger.get("trigger_mode"),
                            "trigger_candidate_count": trigger.get("trigger_candidate_count"),
                            "trigger_candidate_prob_sum": trigger.get("trigger_candidate_prob_sum"),
                            "trigger_prob_sum_threshold": trigger.get("trigger_prob_sum_threshold"),
                            "trigger_match_ratio": trigger.get("trigger_match_ratio"),
                            "trigger_match_ratio_threshold": trigger.get("trigger_match_ratio_threshold"),
                            "trigger_count_threshold": trigger.get("trigger_count_threshold"),
                            "brand_matches_in_top_k": trigger.get("brand_matches_in_top_k"),
                            "model_matches_in_top_k": trigger.get("model_matches_in_top_k"),
                            "total_matches_in_top_k": trigger.get("total_matches_in_top_k"),
                            "use_fixed_strength": trigger.get("use_fixed_strength"),
                            "dynamic_strength_max": trigger.get("dynamic_strength_max"),
                            "dynamic_alpha_recompute": trigger.get("dynamic_alpha_recompute"),
                            "strength": trigger.get("strength"),
                            "current_step_strength": trigger.get("current_step_strength"),
                            "applied_kappa_aux_vs_main": trigger.get("applied_kappa_aux_vs_main"),
                            "strength_reference_position": trigger.get("strength_reference_position"),
                            "strength_recomputed": trigger.get("strength_recomputed"),
                            "strength_source": trigger.get("strength_source"),
                            "prob_weight_beta": trigger.get("prob_weight_beta"),
                            "max_trigger_count": trigger.get("max_trigger_count"),
                            "main_output_token_id_pre_card": main_output_token_id,
                            "main_output_token_prob_pre_card": main_output_prob,
                            "aux_output_token_id_pre_card": aux_output_token_id,
                            "aux_output_token_prob_pre_card": aux_output_prob,
                            "kl_main_vs_aux": (
                                float(kl_main_vs_aux) if kl_main_vs_aux is not None else None
                            ),
                            "kl_aux_vs_main": (
                                float(kl_aux_vs_main) if kl_aux_vs_main is not None else None
                            ),
                            "jsd_main_aux": (
                                float(jsd_main_aux) if jsd_main_aux is not None else None
                            ),
                            "predicted_token_id": trigger.get("predicted_token_id"),
                            "card_token_id": trigger.get("card_token_id"),
                            "trigger_candidate_ids": trigger.get("trigger_candidate_ids", []),
                            "trigger_candidate_tokens": trigger.get("trigger_candidate_tokens", []),
                            "raw_card_candidate_ids": trigger.get("raw_card_candidate_ids", []),
                            "raw_card_candidate_tokens": trigger.get("raw_card_candidate_tokens", []),
                            "card_candidate_ids": trigger.get("card_candidate_ids", []),
                            "card_candidate_tokens": trigger.get("card_candidate_tokens", []),
                            "filtered_opposite_token_ids": trigger.get("filtered_opposite_token_ids", []),
                            "filtered_opposite_token_texts": trigger.get("filtered_opposite_token_texts", []),
                            "filtered_opposite_sources": trigger.get("filtered_opposite_sources", []),
                            "filtered_system_prompt_token_ids": trigger.get("filtered_system_prompt_token_ids", []),
                            "filtered_system_prompt_token_texts": trigger.get("filtered_system_prompt_token_texts", []),
                            "filtered_system_prompt_sources": trigger.get("filtered_system_prompt_sources", []),
                        }
                        trigger_prob_records.append(trigger_record)

                        logger.info(f"  rank {trigger_rank}trigger point:")
                        logger.info(f"    trigger position(position): {trigger.get('position')}")
                        logger.info(
                            f"    trigger type(trigger_type): {trigger.get('trigger_type')} "
                            f"({trigger.get('trigger_type_label')})"
                        )
                        logger.info(
                            f"    trigger type resolution(trigger_type_resolution): "
                            f"{trigger.get('trigger_type_resolution')}"
                        )
                        logger.info(
                            f"    trigger rule(trigger_mode): {trigger.get('trigger_mode')}"
                        )
                        if trigger.get("trigger_mode") == "top_k_count":
                            logger.info(
                                f"    Top-{args.card_trigger_top_k} total trigger matches(brand+model): "
                                f"{trigger.get('total_matches_in_top_k')}"
                            )
                            logger.info(
                                f"    brand-match threshold rule(trigger_count_threshold): "
                                f"{trigger.get('trigger_count_threshold')}"
                            )
                        else:
                            logger.info(
                                "    probability-mass candidate count(trigger_candidate_count): "
                                f"{trigger.get('trigger_candidate_count')}"
                            )
                            trigger_candidate_prob_sum = trigger.get(
                                "trigger_candidate_prob_sum"
                            )
                            if trigger_candidate_prob_sum is not None:
                                logger.info(
                                    "    probability-mass sum(trigger_candidate_prob_sum): "
                                    f"{float(trigger_candidate_prob_sum):.8f}"
                                )
                            logger.info(
                                "    probability-mass threshold(trigger_prob_sum_threshold): "
                                f"{trigger.get('trigger_prob_sum_threshold')}"
                            )
                            logger.info(
                                "    total-match trigger threshold(trigger_count_threshold): "
                                f"{trigger.get('trigger_count_threshold')}"
                            )
                            trigger_match_ratio = trigger.get("trigger_match_ratio")
                            if trigger_match_ratio is not None:
                                logger.info(
                                    "    reference ratio(trigger_match_ratio): "
                                    f"{float(trigger_match_ratio):.8f}"
                                )
                            logger.info(
                                "    total trigger matches(brand+model): "
                                f"{trigger.get('total_matches_in_top_k')}"
                            )
                        logger.info(
                            f"    brand matches(brand_matches_in_top_k): "
                            f"{trigger.get('brand_matches_in_top_k')}"
                        )
                        logger.info(
                            f"    model matches(model_matches_in_top_k): "
                            f"{trigger.get('model_matches_in_top_k')}"
                        )
                        logger.info(
                            f"    main-model output token ID(pre-CARD): {main_output_token_id}"
                        )
                        logger.info(
                            f"    auxiliary-model output token ID(pre-CARD): {aux_output_token_id}"
                        )
                        logger.info(
                            f"    main-model output token probability(pre-CARD): {main_output_prob:.8f}"
                        )
                        logger.info(
                            f"    auxiliary-model output token probability(pre-CARD): {aux_output_prob:.8f}"
                        )
                        logger.info(
                            f"    fixed strength mode(use_fixed_strength): "
                            f"{trigger.get('use_fixed_strength')}"
                        )
                        logger.info(
                            f"    actual strength used: "
                            f"{trigger.get('strength'):.8f}"
                            if trigger.get("strength") is not None
                            else "    actual strength used: None"
                        )
                        if trigger.get("use_fixed_strength"):
                            logger.info(
                                "    dynamic strength: disabled (currently uses fixed preset strength)"
                            )
                        else:
                            logger.info(
                                f"    Dynamic strength maximum(dynamic_strength_max): "
                                f"{trigger.get('dynamic_strength_max')}"
                            )
                            logger.info(
                                f"    Dynamic strength recomputation(dynamic_alpha_recompute): "
                                f"{trigger.get('dynamic_alpha_recompute')}"
                            )
                            logger.info(
                                f"    actual kappa used = KL(aux || normal): "
                                f"{trigger.get('applied_kappa_aux_vs_main'):.8f}"
                                if trigger.get("applied_kappa_aux_vs_main") is not None
                                else "    actual kappa used = KL(aux || normal): None"
                            )
                            logger.info(
                                f"    strength reference position(strength_reference_position): "
                                f"{trigger.get('strength_reference_position')}"
                            )
                            logger.info(
                                f"    whether this trigger point recomputed strength(strength_recomputed): "
                                f"{trigger.get('strength_recomputed')}"
                            )
                        logger.info(
                            f"    strength source(strength_source): {trigger.get('strength_source')}"
                        )
                        if kl_main_vs_aux is not None:
                            logger.info(
                                f"    KL(normal || aux): {float(kl_main_vs_aux):.8f}"
                            )
                        if kl_aux_vs_main is not None:
                            logger.info(
                                f"    KL(aux || normal): {float(kl_aux_vs_main):.8f}"
                            )
                        if jsd_main_aux is not None:
                            logger.info(
                                f"    JSD(normal, aux): {float(jsd_main_aux):.8f}"
                            )
                        logger.info(
                            f"    raw CARD support IDs(raw_card_candidate_ids): "
                            f"{trigger.get('raw_card_candidate_ids', [])}"
                        )
                        logger.info(
                            f"    raw CARD support tokens(raw_card_candidate_tokens): "
                            f"{trigger.get('raw_card_candidate_tokens', [])}"
                        )
                        logger.info(
                            f"    filtered CARD candidate IDs(card_candidate_ids): "
                            f"{trigger.get('card_candidate_ids', [])}"
                        )
                        logger.info(
                            f"    filtered CARD candidate tokens(card_candidate_tokens): "
                            f"{trigger.get('card_candidate_tokens', [])}"
                        )
                else:
                    logger.info(f"[CARD trigger-point probability] notrigger point")

            score_records = []
            if output_only_mode:
                logger.info(f"")
                logger.info(
                    "[Output mode] --output-only enabled; skipping matching, scoring, and evaluation statistics for this response"
                )
            else:
                
                logger.info(f"")
                logger.info(f"[Score parsing process]")
                logger.info(f"Detected {log_info['num_paragraphs']} paragraph")

                
                for para_info in log_info['paragraphs']:
                    logger.info(f"")
                    logger.info(f"--- paragraph[{para_info['index']}] ---")
                    logger.info(f"{para_info['preview']}")

                    if para_info['matched']:
                        logger.info(
                            f"=> matched: {para_info['matched']['brand']} - "
                            f"{para_info['matched']['model']}"
                        )
                    else:
                        logger.info(f"=> unmatched any product")

                
                logger.info(f"")
                logger.info(f"[Final ranking]")
                for score_info in log_info['scores']:
                    logger.info(
                        f"  rank {score_info['rank']}: {score_info['brand']} - "
                        f"{score_info['model']} (score={score_info['score']})"
                    )

                
                if log_info['unmatched']:
                    logger.info(f"")
                    logger.info(f"[Unmatched products]")
                    for unmatched_info in log_info['unmatched']:
                        logger.info(
                            f"  {unmatched_info['brand']} - {unmatched_info['model']}"
                        )

                
                logger.info(f"")
                logger.info(f"[Parsed scores]")

                for product in ordered_products:
                    score = product_scores[product]

                    
                    physical_pos = ordered_products.index(product)
                    brand_idx, doc_idx = prompt_order[physical_pos]
                    context_position = physical_pos  

                    key = (brand_idx, doc_idx, context_position)
                    category_scores[key].append(score)

                    brand = brands[brand_idx]
                    score_records.append({
                        'brand_index': brand_idx,
                        'brand': brand.brand,
                        'model': product.model,
                        'doc_index': doc_idx,
                        'context_position': context_position,
                        'score': score,
                        'is_fictional': brand.is_fictional,
                        'knowledge_strength': brand.knowledge_strength
                    })

                
                score_records.sort(key=lambda x: x['score'], reverse=True)
                for rec in score_records:
                    fictional_mark = "[fictional]" if rec['is_fictional'] else "[real]"
                    logger.info(
                        f"  score={rec['score']:2d} | B[{rec['brand_index']}] "
                        f"{rec['brand'][:15]:15s} - {rec['model'][:20]:20s} "
                        f"| D[{rec['doc_index']}] | P{rec['context_position']} "
                        f"| knowledge strength={rec['knowledge_strength']:.3f} {fictional_mark}"
                    )

            
            api_thinking_tokens = current_batch_thinking_tokens[local_idx]
            api_response_tokens = current_batch_response_tokens[local_idx]
            token_count_source = current_batch_token_count_sources[local_idx]
            finish_reason = current_batch_finish_reasons[local_idx]
            stop_reason = current_batch_stop_reasons[local_idx]
            hit_length_limit = current_batch_hit_length_limits[local_idx]

            if (
                token_count_source is not None
                or api_thinking_tokens is not None
                or api_response_tokens is not None
            ):
                
                thinking_tokens = (
                    int(api_thinking_tokens) if api_thinking_tokens is not None else 0
                )
                response_tokens = (
                    int(api_response_tokens) if api_response_tokens is not None else 0
                )
                if token_count_source in {"local_generated_ids", "vllm_generated_ids"}:
                    token_source = "exact count(locally generated token IDs)"
                    response_token_count_source_local_exact_count += 1
                else:
                    token_source = "exact count(API usage)"
                    response_token_count_source_api_count += 1
                response_token_count_source_exact_count += 1
            else:
                
                thinking_tokens = (
                    estimate_token_count(thinking_content) if thinking_content else 0
                )
                response_tokens = estimate_token_count(target_response)
                token_source = "estimated"
                response_token_count_source_estimated_count += 1

            thinking_token_counts.append(thinking_tokens)
            response_token_counts.append(response_tokens)
            response_finish_reasons.append(finish_reason)
            response_stop_reasons.append(stop_reason)
            if hit_length_limit is None:
                output_limit_reached = response_tokens >= int(args.target_max_tokens)
            else:
                output_limit_reached = bool(hit_length_limit)
            response_output_limit_reached_flags.append(bool(output_limit_reached))

            if thinking_content:
                logger.info(f"")
                logger.info(f"[Token statistics] ({token_source})")
                logger.info(f"  thinking tokens: {thinking_tokens}")
                logger.info(f"  response token count: {response_tokens}")
                logger.info(f"  total tokens: {thinking_tokens + response_tokens}")

            
            experiment_record = {
                'block_idx': block_idx,
                'run_idx': global_run_idx,
                'experiment_idx': current_experiment,
                'L2_b': [row[:] for row in L2_b],  
                'doc_assignment': doc_assignment.copy(),
                'prompt_order': prompt_order.copy(),
                'scores': score_records,
            }
            
            if thinking_content:
                experiment_record['thinking_content'] = thinking_content
            raw_global_card_token_trace = current_batch_global_card_token_traces[local_idx]
            if raw_global_card_token_trace is not None:
                global_card_token_trace = normalize_global_card_token_trace(
                    raw_global_card_token_trace
                )
                experiment_record['global_card_token_trace'] = global_card_token_trace

            category_experiment_records.append(experiment_record)
            category_responses.append([target_response])

        return parsing_elapsed
    
    
    
    system_message = Message(
        role=Role.system,
        content=build_system_prompt(
            include_ordering_prompt=not args.no_ordering_prompt,
            use_system_role_baseline=args.use_system_role_baseline,
        )
    )
    run_records: t.List[t.Dict[str, t.Any]] = []

    def run_inference_batch(
        current_batch_run_records: t.List[t.Dict[str, t.Any]],
        batch_log_prefix: str,
        emit_logs: bool,
    ) -> t.Dict[str, t.Any]:
        nonlocal api_success_response_count
        nonlocal api_recovered_prompt_count
        nonlocal api_final_failed_prompt_count
        nonlocal api_total_attempt_count
        current_batch_size = len(current_batch_run_records)
        first_experiment = current_batch_run_records[0]['experiment_idx']
        last_experiment = current_batch_run_records[-1]['experiment_idx']
        experiment_range_text = (
            f"experiment tasks {first_experiment}-{last_experiment}"
            if current_batch_size > 1
            else f"experiment tasks {first_experiment}"
        )
        block_run_text = ", ".join(
            f"B{record['block_idx'] + 1}/R{record['run_idx'] + 1}"
            for record in current_batch_run_records[:5]
        )
        if current_batch_size > 5:
            block_run_text += ", ..."
        inference_start_log = (
            f"[Batch inference] {batch_log_prefix}: {experiment_range_text} "
            f"(batch_size={current_batch_size}, blocks/runs={block_run_text})"
        )
        if emit_logs:
            logger.info(f"")
            log_key_info(logger, inference_start_log)

        current_batch_messages = [
            record['messages'] for record in current_batch_run_records
        ]

        inference_start_time = time.perf_counter()
        if getattr(args, 'use_card', False):
            current_batch_responses = target(
                current_batch_messages,
                queries=[record['query'] for record in current_batch_run_records],
                categories=[category] * current_batch_size,
                documents_list=[
                    record['documents'] for record in current_batch_run_records
                ],
                product_models_list=[
                    record['product_models'] for record in current_batch_run_records
                ],
                product_brands_list=[
                    record['product_brands'] for record in current_batch_run_records
                ],
            )
            inference_elapsed = time.perf_counter() - inference_start_time
            inference_call_durations.append(inference_elapsed)
            inference_response_counts.append(current_batch_size)
        elif getattr(args, 'use_ck', False):
            current_batch_responses = target(
                current_batch_messages,
                queries=[record['query'] for record in current_batch_run_records],
                product_models_list=[
                    record['product_models'] for record in current_batch_run_records
                ],
                product_brands_list=[
                    record['product_brands'] for record in current_batch_run_records
                ],
            )
            inference_elapsed = time.perf_counter() - inference_start_time
            inference_call_durations.append(inference_elapsed)
            inference_response_counts.append(current_batch_size)
        else:
            if api_retry_enabled:
                successful_run_records: t.List[t.Dict[str, t.Any]] = []
                successful_responses: t.List[t.Any] = []
                batch_attempt_count = 0
                batch_recovered_count = 0
                batch_failed_count = 0
                requested_api_concurrency = max(
                    1,
                    int(getattr(args, 'api_concurrency', 1)),
                )
                effective_api_concurrency = min(
                    requested_api_concurrency,
                    current_batch_size,
                )
                if emit_logs:
                    api_retry_mode = (
                        "concurrent requests"
                        if effective_api_concurrency > 1
                        else "serial requests"
                    )
                    log_key_info(
                        logger,
                        f"[API retry] {batch_log_prefix}: start{api_retry_mode}, "
                        f"original prompt count={current_batch_size}, "
                        f"API concurrency={effective_api_concurrency}",
                    )
                retry_results_by_index: t.List[t.Optional[t.Dict[str, t.Any]]] = [
                    None
                ] * current_batch_size
                if effective_api_concurrency <= 1:
                    for idx, run_record in enumerate(current_batch_run_records):
                        retry_results_by_index[idx] = retry_api_single_prompt(
                            target=target,
                            run_record=run_record,
                            category=category,
                            batch_label=batch_log_prefix,
                            args=args,
                            retry_origin="main",
                        )
                else:
                    api_submit_stagger = max(
                        0.0,
                        float(getattr(args, 'api_submit_stagger', 0.2)),
                    )
                    with ThreadPoolExecutor(
                        max_workers=effective_api_concurrency
                    ) as api_executor:
                        future_to_index = {}
                        for idx, run_record in enumerate(current_batch_run_records):
                            if idx > 0 and api_submit_stagger > 0:
                                time.sleep(random.uniform(0.0, api_submit_stagger))
                            future = api_executor.submit(
                                retry_api_single_prompt,
                                target,
                                run_record,
                                category,
                                batch_log_prefix,
                                args,
                                "main",
                            )
                            future_to_index[future] = idx
                        for future in as_completed(future_to_index):
                            idx = future_to_index[future]
                            retry_results_by_index[idx] = future.result()

                for idx, retry_result in enumerate(retry_results_by_index):
                    if retry_result is None:
                        raise RuntimeError(
                            f"API retry result missing for {batch_log_prefix} "
                            f"index {idx}"
                        )
                    run_record = current_batch_run_records[idx]
                    batch_attempt_count += int(retry_result["attempt_count"])
                    inference_call_durations.append(float(retry_result["total_elapsed"]))
                    inference_response_counts.append(
                        1 if retry_result["status"] == "success" else 0
                    )
                    if retry_result["status"] == "success":
                        successful_run_records.append(run_record)
                        successful_responses.append(retry_result["response"])
                        api_success_response_count += 1
                        if retry_result["recovered"]:
                            batch_recovered_count += 1
                            api_recovered_prompt_count += 1
                            recovered_record = retry_result.get("recovered_record")
                            if recovered_record is not None:
                                append_jsonl_record(
                                    api_recovered_prompts_path,
                                    recovered_record,
                                )
                    else:
                        batch_failed_count += 1
                        api_final_failed_prompt_count += 1
                        failure_record = retry_result.get("failure_record")
                        if failure_record is not None:
                            append_jsonl_record(
                                api_failed_prompts_path,
                                failure_record,
                            )
                api_total_attempt_count += batch_attempt_count
                current_batch_run_records = successful_run_records
                current_batch_responses = successful_responses
                inference_elapsed = time.perf_counter() - inference_start_time
                inference_complete_log = (
                    f"[Batch inference] {batch_log_prefix} complete, "
                    f"success {len(successful_responses)} , "
                    f"failure {batch_failed_count} , "
                    f"APIconcurrency {effective_api_concurrency}, "
                    f"submission stagger cap {getattr(args, 'api_submit_stagger', 0.2):g} s, "
                    f"total attempts {batch_attempt_count}  times, "
                    f"wall-clock elapsed {inference_elapsed:.4f} s"
                )
                if emit_logs:
                    log_key_info(logger, inference_complete_log)
                return {
                    'batch_log_prefix': batch_log_prefix,
                    'current_batch_size': len(successful_responses),
                    'attempted_batch_size': current_batch_size,
                    'current_batch_run_records': successful_run_records,
                    'current_batch_responses': successful_responses,
                    'inference_elapsed': inference_elapsed,
                    'inference_start_log': inference_start_log,
                    'inference_complete_log': inference_complete_log,
                    'api_retry_stats': {
                        'success_response_count': len(successful_responses),
                        'recovered_prompt_count': batch_recovered_count,
                        'final_failed_prompt_count': batch_failed_count,
                        'total_attempt_count': batch_attempt_count,
                        'failed_prompts_path': api_failed_prompts_path,
                        'recovered_prompts_path': api_recovered_prompts_path,
                        'incomplete': batch_failed_count > 0,
                    },
                }
            current_batch_responses = target(current_batch_messages)
            inference_elapsed = time.perf_counter() - inference_start_time
            inference_call_durations.append(inference_elapsed)
            inference_response_counts.append(current_batch_size)

        inference_complete_log = (
            f"[Batch inference] batch complete, received {len(current_batch_responses)} responses, "
            f"inference elapsed time {inference_elapsed:.4f} s"
        )
        if emit_logs:
            log_key_info(logger, inference_complete_log)

        return {
            'batch_log_prefix': batch_log_prefix,
            'current_batch_size': current_batch_size,
            'current_batch_run_records': current_batch_run_records,
            'current_batch_responses': current_batch_responses,
            'inference_elapsed': inference_elapsed,
            'inference_start_log': inference_start_log,
            'inference_complete_log': inference_complete_log,
        }

    def submit_async_parse(
        batch_record: t.Dict[str, t.Any],
        parse_executor: ThreadPoolExecutor,
    ) -> t.Dict[str, t.Any]:
        prepared_batch_state = prepare_response_batch_state(
            batch_record['current_batch_responses'],
            [
                record['prompt_order']
                for record in batch_record['current_batch_run_records']
            ],
        )
        process_fn = functools.partial(
            process_response_batch,
            current_batch_responses=batch_record['current_batch_responses'],
            current_batch_run_records=batch_record['current_batch_run_records'],
            batch_label=batch_record['batch_log_prefix'],
        )
        return {
            'batch_record': batch_record,
            'prepared_batch_state': prepared_batch_state,
            'process_fn': process_fn,
            'future': parse_executor.submit(
                parse_prepared_response_batch,
                prepared_batch_state,
            ),
        }

    def flush_async_parse(
        pending_parse: t.Dict[str, t.Any],
    ) -> float:
        batch_record = pending_parse['batch_record']
        wait_start_time = time.perf_counter()
        parse_result = pending_parse['future'].result()
        async_parse_wait_elapsed = time.perf_counter() - wait_start_time

        return pending_parse['process_fn'](
            prepared_batch_state=pending_parse['prepared_batch_state'],
            preparsed_batch_result=parse_result,
            async_parse_wait_elapsed=async_parse_wait_elapsed,
        )

    for block_idx in range(product_n):
        block_run_records: t.List[t.Dict[str, t.Any]] = []
        
        
        
        doc_assignment = [L1[i][block_idx] for i in range(product_n)]
        # doc_assignment[brand_idx] = doc_idx

        
        # L2_b[brand][run] = position
        L2_b = get_random_latin_square(product_n)
        
        logger.info(f"")
        logger.info(f"{'#'*60}")
        logger.info(f"# Block {block_idx + 1}/{product_n} (block_idx={block_idx})")
        logger.info(f"{'#'*60}")
        logger.info(f"")
        logger.info(f"[Brand-Doc pairing] (by L1 rank  {block_idx} column)")
        logger.info(f"  Check: L1[brand][{block_idx}] = doc_idx")
        for bi in range(product_n):
            expected_doc = L1[bi][block_idx]
            actual_doc = doc_assignment[bi]
            match_mark = "✓" if expected_doc == actual_doc else "❌"
            logger.info(f"    {match_mark} B[{bi}] {brands[bi].brand[:15]:15s} -> D[{actual_doc}] (L1[{bi}][{block_idx}]={expected_doc})")
        logger.info(f"")
        logger.info(f"[Latin square for this block L2_{block_idx}] (position assignment)")
        logger.info(f"  Meaning: L2[brand][run] = position")
        logger.info(f"")
        logger.info(f"  Matrix form (rows=Brand, columns=Run, values=Position):")
        header = "        " + "".join([f"Run{r:2d} " for r in range(product_n)])
        logger.info(f"  {header}")
        for i in range(product_n):
            row_str = f"  B[{i}] {brands[i].brand[:8]:8s} " + "".join([f"  P{L2_b[i][r]:1d}  " for r in range(product_n)])
            logger.info(row_str)
        
        
        logger.info(f"")
        logger.info(f"  [L2_{block_idx} validity check]")
        l2_valid = True
        for i in range(product_n):
            if sorted(L2_b[i]) != list(range(product_n)):
                logger.info(f"    ❌ rank  {i} rowis not {valid_index_text} permutation")
                l2_valid = False
        for j in range(product_n):
            col = [L2_b[i][j] for i in range(product_n)]
            if sorted(col) != list(range(product_n)):
                logger.info(f"    ❌ rank  {j} columnis not {valid_index_text} permutation")
                l2_valid = False
        if l2_valid:
            logger.info(f"    ✓ L2_{block_idx} is a valid Latin square (every row and column is {valid_index_text} permutation)")
        
        
        logger.info(f"")
        log_key_info(logger, f"[Batch inference] prepare Block {block_idx + 1}  {product_n}  Runs")
        
        for run_idx in range(product_n):
            experiment_count += 1
            
            
            # prompt_order[position] = (brand_idx, doc_idx)
            prompt_order = [None] * product_n
            for brand_idx in range(product_n):
                doc_idx = doc_assignment[brand_idx]
                position = L2_b[brand_idx][run_idx]
                prompt_order[position] = (brand_idx, doc_idx)
            
            
            ordered_products = []
            ordered_docs = []
            brand_to_doc_map = {}  
            brand_to_position_map = {}  
            
            for physical_pos in range(product_n):
                brand_idx, doc_idx = prompt_order[physical_pos]
                ordered_products.append(products[brand_idx])
                ordered_docs.append(docs[brand_idx][doc_idx])
                brand_to_doc_map[brand_idx] = doc_idx
                brand_to_position_map[brand_idx] = physical_pos
            
            
            target_message = build_target_message(
                query=user_query,
                documents=ordered_docs,
                product_models=[p.model for p in ordered_products],
                product_brands=[p.brand for p in ordered_products],
                use_debias_instruction_baseline=args.use_debias_instruction_baseline,
                use_moral_self_correction_baseline=args.use_moral_self_correction_baseline,
            )

            block_run_record = {
                'block_idx': block_idx,
                'run_idx': run_idx,
                'experiment_idx': block_idx * product_n + run_idx + 1,
                'L2_b': [row[:] for row in L2_b],
                'doc_assignment': doc_assignment.copy(),
                'prompt_order': prompt_order,
                'brand_to_doc_map': brand_to_doc_map,
                'brand_to_position_map': brand_to_position_map,
                'target_message': target_message,
                'messages': [
                    system_message,
                    Message(role=Role.user, content=target_message),
                ],
                'query': user_query,
                'documents': ordered_docs,
                'product_models': [p.model for p in ordered_products],
                'product_brands': [p.brand for p in ordered_products],
            }
            block_run_records.append(block_run_record)

        logger.info(f"")
        log_key_info(
            logger,
            f"[Batch inference] Block {block_idx + 1} prepared {product_n}  Runs"
        )

        if category_level_batching_enabled:
            run_records.extend(block_run_records)
            continue

        actual_batch_size = min(max(1, int(args.batch_size)), product_n)
        block_batch_count = (
            len(block_run_records) + actual_batch_size - 1
        ) // actual_batch_size
        logger.info(f"[Batch inference configuration] Block {block_idx + 1}")
        logger.info(f"  thisBlocktotalRuncount: {len(block_run_records)}")
        logger.info(f"  Batch size: {actual_batch_size}")
        logger.info(f"  batch count: {block_batch_count}")

        block_response_count = 0
        for batch_start in range(0, len(block_run_records), actual_batch_size):
            batch_end = min(batch_start + actual_batch_size, len(block_run_records))
            batch_index = (batch_start // actual_batch_size) + 1
            batch_log_prefix = f"Block {block_idx + 1}, batch {batch_index}"
            batch_record = run_inference_batch(
                current_batch_run_records=block_run_records[batch_start:batch_end],
                batch_log_prefix=batch_log_prefix,
                emit_logs=True,
            )
            block_response_count += len(batch_record['current_batch_responses'])
            if batch_record['current_batch_responses']:
                process_response_batch(
                    current_batch_responses=batch_record['current_batch_responses'],
                    current_batch_run_records=batch_record['current_batch_run_records'],
                    batch_label=batch_log_prefix,
                )
            else:
                logger.info(
                    f"[Response parsing] {batch_log_prefix} no successful responses; skipping parsing"
                )

        logger.info(f"")
        log_key_info(
            logger,
            f"[Batch inference] Block {block_idx + 1} complete, total received {block_response_count} responses"
        )

    if category_level_batching_enabled:
        actual_batch_size = max(1, int(args.batch_size))

        logger.info(f"")
        logger.info(f"[Batch inference configuration] Category {category}")
        logger.info(f"  thisCategorytotalRuncount: {len(run_records)}")
        logger.info(f"  Batch size: {actual_batch_size}")
        category_batch_count = (
            len(run_records) + actual_batch_size - 1
        ) // actual_batch_size
        logger.info(f"  batch count: {category_batch_count}")
        logger.info(f"  scheduling mode: Categorylevel taskpool, allow sameonebatchacross Block")

        category_response_count = 0
        if async_pipeline_parsing_enabled:
            pending_parses = (
                shared_async_pending_parses
                if shared_async_pending_parses is not None
                else []
            )
            parse_status_started = False
            parse_pipeline_executor = (
                shared_async_parse_executor
                if shared_async_parse_executor is not None
                else ThreadPoolExecutor(max_workers=1)
            )
            should_shutdown_parse_pipeline_executor = (
                shared_async_parse_executor is None
            )
            try:
                batch_starts = range(0, len(run_records), actual_batch_size)
                for batch_start in tqdm.tqdm(
                    batch_starts,
                    total=category_batch_count,
                    desc=format_timed_title(f"{category} inference batch"),
                    leave=False,
                ):
                    batch_end = min(batch_start + actual_batch_size, len(run_records))
                    batch_index = (batch_start // actual_batch_size) + 1
                    batch_log_prefix = f"Categorybatch {batch_index}"
                    batch_record = run_inference_batch(
                        current_batch_run_records=run_records[batch_start:batch_end],
                        batch_log_prefix=batch_log_prefix,
                        emit_logs=True,
                    )

                    category_response_count += len(batch_record['current_batch_responses'])

                    if batch_record['current_batch_responses']:
                        pending_parses.append(
                            submit_async_parse(batch_record, parse_pipeline_executor)
                        )
                        if not parse_status_started:
                            print_nohup_parse_status(category, "starting background parsing")
                            parse_status_started = True
                    else:
                        logger.info(
                            f"[Async pipeline parsing] {batch_log_prefix} nohassuccessresponses, skip after backgroundparse"
                        )

                    while len(pending_parses) > async_parse_max_pending_batches:
                        logger.info(f"")
                        logger.info(
                            "[Async pipeline parsing] pending parse batch count reached the limit "
                            f"{async_parse_max_pending_batches}, "
                            "waiting in order and filling the earliest batch"
                        )
                        flush_async_parse(pending_parses.pop(0))

                if parse_status_started:
                    if defer_finalization:
                        print_nohup_parse_status(
                            category,
                            "all inference batches are complete; background parsing is still running, "
                            f"pendingbatch={len(pending_parses)}"
                        )
                    else:
                        print_nohup_parse_status(
                            category,
                            "all inference batches are complete, waiting for remaining parse results to be filled, "
                            f"pendingbatch={len(pending_parses)}"
                        )
                if not defer_finalization:
                    while pending_parses:
                        flush_async_parse(pending_parses.pop(0))
                    if parse_status_started:
                        print_nohup_parse_status(
                            category,
                            "parsing complete, "
                            f"parse batches={len(parsing_call_durations)}, "
                            f"responses={sum(parsing_response_counts)}, "
                            f"parse elapsed time={sum(parsing_call_durations):.1f}s"
                        )
            finally:
                if should_shutdown_parse_pipeline_executor:
                    parse_pipeline_executor.shutdown(wait=True)
        else:
            batch_starts = range(0, len(run_records), actual_batch_size)
            for batch_start in tqdm.tqdm(
                batch_starts,
                total=category_batch_count,
                desc=format_timed_title(f"{category} inference batch"),
                leave=False,
            ):
                batch_end = min(batch_start + actual_batch_size, len(run_records))
                batch_index = (batch_start // actual_batch_size) + 1
                batch_log_prefix = f"Categorybatch {batch_index}"
                batch_record = run_inference_batch(
                    current_batch_run_records=run_records[batch_start:batch_end],
                    batch_log_prefix=batch_log_prefix,
                    emit_logs=True,
                )

                category_response_count += len(batch_record['current_batch_responses'])

                if batch_record['current_batch_responses']:
                    process_response_batch(
                        current_batch_responses=batch_record['current_batch_responses'],
                        current_batch_run_records=batch_record['current_batch_run_records'],
                        batch_label=batch_log_prefix,
                    )
                else:
                    logger.info(
                        f"[Response parsing] {batch_log_prefix} no successful responses; skipping parsing"
                    )

        logger.info(f"")
        log_key_info(
            logger,
            f"[Batch inference] Category {category} complete, total received {category_response_count} responses"
        )

    def build_deferred_category_result() -> t.Dict[str, t.Any]:
        trigger_prob_summary_local = None
        if (
            (not output_only_mode)
            and getattr(args, 'use_card', False)
            and is_triggered_card_mode(args)
        ):
            max_trigger_rank = max(trigger_probs_by_rank.keys(), default=0)
            trigger_rank_stats = {}
            for trigger_rank in sorted(trigger_probs_by_rank):
                main_values = trigger_probs_by_rank[trigger_rank]["main_output_probs"]
                aux_values = trigger_probs_by_rank[trigger_rank]["aux_output_probs"]
                strength_values = trigger_probs_by_rank[trigger_rank]["strength_values"]
                kl_main_vs_aux_values = trigger_probs_by_rank[trigger_rank]["kl_main_vs_aux_values"]
                kl_aux_vs_main_values = trigger_probs_by_rank[trigger_rank]["kl_aux_vs_main_values"]
                jsd_main_aux_values = trigger_probs_by_rank[trigger_rank]["jsd_main_aux_values"]
                trigger_rank_stats[str(trigger_rank)] = {
                    "sample_count": len(main_values),
                    "main_output_prob_mean_pre_card": float(np.mean(main_values)),
                    "aux_output_prob_mean_pre_card": float(np.mean(aux_values)),
                    "main_output_probs_pre_card": main_values,
                    "aux_output_probs_pre_card": aux_values,
                    "strength_mean": (
                        float(np.mean(strength_values))
                        if strength_values
                        else None
                    ),
                    "strength_values": strength_values,
                    "kl_main_vs_aux_mean": (
                        float(np.mean(kl_main_vs_aux_values))
                        if kl_main_vs_aux_values
                        else None
                    ),
                    "kl_main_vs_aux_values": kl_main_vs_aux_values,
                    "kl_aux_vs_main_mean": (
                        float(np.mean(kl_aux_vs_main_values))
                        if kl_aux_vs_main_values
                        else None
                    ),
                    "kl_aux_vs_main_values": kl_aux_vs_main_values,
                    "jsd_main_aux_mean": (
                        float(np.mean(jsd_main_aux_values))
                        if jsd_main_aux_values
                        else None
                    ),
                    "jsd_main_aux_values": jsd_main_aux_values,
                }

            trigger_prob_summary_local = {
                "category": category,
                "total_experiments": experiment_count,
                "max_trigger_rank": max_trigger_rank,
                "raw_record_count": len(trigger_prob_records),
                "trigger_rank_stats": trigger_rank_stats,
            }

            category_safe = category.replace(" ", "_").replace("/", "_")
            stats_dir = out_path(args, "card_trigger_prob_stats")
            file_utils.ensure_created_directory(stats_dir)
            trigger_probs_jsonl_path = out_path(
                args, "card_trigger_prob_stats", f"{category_safe}_trigger_probs.jsonl"
            )
            trigger_prob_summary_path = out_path(
                args, "card_trigger_prob_stats", f"{category_safe}_trigger_prob_summary.json"
            )

            trigger_probs_jsonl = "\n".join(
                json.dumps(record, ensure_ascii=False) for record in trigger_prob_records
            )
            if trigger_probs_jsonl:
                trigger_probs_jsonl += "\n"
            file_utils.write_file(trigger_probs_jsonl_path, trigger_probs_jsonl)
            file_utils.write_file(
                trigger_prob_summary_path,
                json.dumps(trigger_prob_summary_local, ensure_ascii=False, indent=2) + "\n",
            )

        total_inference_seconds = float(sum(inference_call_durations))
        total_inference_responses = int(sum(inference_response_counts))
        total_thinking_tokens = int(sum(thinking_token_counts))
        total_response_tokens = int(sum(response_token_counts))
        total_generated_tokens = total_thinking_tokens + total_response_tokens
        response_output_limit_reached_count = sum(
            1 for reached in response_output_limit_reached_flags
            if reached
        )
        output_limit_detection_method = (
            'vllm_finish_reason_length_or_token_threshold'
            if any(reason is not None for reason in response_finish_reasons)
            else 'token_threshold_only'
        )
        response_paragraphs_le_5_count = sum(
            1 for paragraph_count in response_paragraph_counts
            if paragraph_count <= 5
        )
        inference_time_stats = {
            'model_call_count': len(inference_call_durations),
            'response_count': total_inference_responses,
            'total_inference_seconds': total_inference_seconds,
            'total_thinking_tokens': total_thinking_tokens,
            'total_response_tokens': total_response_tokens,
            'total_generated_tokens': total_generated_tokens,
            'avg_inference_seconds_per_model_call': (
                float(np.mean(inference_call_durations)) if inference_call_durations else 0.0
            ),
            'avg_inference_seconds_per_response': (
                total_inference_seconds / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'avg_response_tokens_per_response': (
                total_response_tokens / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'avg_generated_tokens_per_response': (
                total_generated_tokens / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'thinking_tokens_per_second': (
                total_thinking_tokens / total_inference_seconds
                if total_inference_seconds else 0.0
            ),
            'response_tokens_per_second': (
                total_response_tokens / total_inference_seconds
                if total_inference_seconds else 0.0
            ),
            'total_generated_tokens_per_second': (
                total_generated_tokens / total_inference_seconds
                if total_inference_seconds else 0.0
            ),
            'min_inference_seconds_per_model_call': (
                float(min(inference_call_durations)) if inference_call_durations else 0.0
            ),
            'max_inference_seconds_per_model_call': (
                float(max(inference_call_durations)) if inference_call_durations else 0.0
            ),
            'model_call_inference_seconds': [float(v) for v in inference_call_durations],
            'model_call_response_counts': inference_response_counts.copy(),
            'response_token_counts': response_token_counts.copy(),
            'response_paragraph_counts': response_paragraph_counts.copy(),
            'response_output_limit_reached_flags': response_output_limit_reached_flags.copy(),
            'response_finish_reasons': response_finish_reasons.copy(),
            'response_stop_reasons': response_stop_reasons.copy(),
            'target_max_tokens': int(args.target_max_tokens),
            'response_output_limit_reached_count': int(
                response_output_limit_reached_count
            ),
            'response_output_limit_reached_ratio': (
                response_output_limit_reached_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'response_reached_max_tokens_count': int(response_output_limit_reached_count),
            'response_reached_max_tokens_ratio': (
                response_output_limit_reached_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'response_paragraphs_le_5_count': int(response_paragraphs_le_5_count),
            'response_paragraphs_le_5_ratio': (
                response_paragraphs_le_5_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'response_token_count_source_exact_count': int(response_token_count_source_exact_count),
            'response_token_count_source_api_count': int(response_token_count_source_api_count),
            'response_token_count_source_local_exact_count': int(
                response_token_count_source_local_exact_count
            ),
            'response_token_count_source_estimated_count': int(response_token_count_source_estimated_count),
        }
        parsing_time_stats = {
            'batch_parse_count': len(parsing_call_durations),
            'response_count': int(sum(parsing_response_counts)),
            'total_parsing_seconds': float(sum(parsing_call_durations)),
            'avg_parsing_seconds_per_batch': (
                float(np.mean(parsing_call_durations)) if parsing_call_durations else 0.0
            ),
            'avg_parsing_seconds_per_response': (
                float(sum(parsing_call_durations)) / int(sum(parsing_response_counts))
                if sum(parsing_response_counts) else 0.0
            ),
            'min_parsing_seconds_per_batch': (
                float(min(parsing_call_durations)) if parsing_call_durations else 0.0
            ),
            'max_parsing_seconds_per_batch': (
                float(max(parsing_call_durations)) if parsing_call_durations else 0.0
            ),
            'batch_parsing_seconds': [float(v) for v in parsing_call_durations],
            'batch_parse_response_counts': parsing_response_counts.copy(),
        }
        if getattr(args, 'use_card', False):
            inference_time_stats['card_application_mode'] = resolve_card_application_mode(args)
            inference_time_stats['card_global_logit_formula'] = (
                resolve_card_global_logit_formula(args)
            )
            if is_card_global_custom_formula_enabled(args):
                inference_time_stats['card_global_formula_alpha'] = (
                    get_card_global_formula_alpha(args)
                )
            inference_time_stats['card_batch_inference'] = bool(
                getattr(args, 'card_batch_inference', False)
            )
            if (
                getattr(args, 'target_local_backend', 'vllm') == "vllm"
                and resolve_card_application_mode(args) == "global"
            ):
                inference_time_stats['card_execution_mode'] = 'vllm_paired_global'
                inference_time_stats['card_global_main_bias_coeff'] = (
                    get_card_global_main_bias_coeff(args)
                )
                inference_time_stats['card_global_direction_sign'] = (
                    get_card_global_direction_sign(args)
                )
                inference_time_stats['card_global_vllm_support_mode'] = (
                    resolve_card_global_vllm_support_mode(args)
                )
                inference_time_stats['card_global_vllm_support_top_k'] = int(
                    getattr(args, 'card_global_vllm_support_top_k', 10)
                )
            else:
                inference_time_stats['card_execution_mode'] = (
                    'true_batch'
                    if getattr(args, 'card_batch_inference', False)
                    else 'legacy_serial_single'
                )

        output_token_stats = {
            'response_token_counts': response_token_counts.copy(),
            'response_count': total_inference_responses,
            'avg_response_tokens': (
                float(np.mean(response_token_counts)) if response_token_counts else 0.0
            ),
            'min_response_tokens': (
                int(min(response_token_counts)) if response_token_counts else 0
            ),
            'max_response_tokens': (
                int(max(response_token_counts)) if response_token_counts else 0
            ),
            'response_output_limit_reached_flags': (
                response_output_limit_reached_flags.copy()
            ),
            'response_finish_reasons': response_finish_reasons.copy(),
            'response_stop_reasons': response_stop_reasons.copy(),
            'target_max_tokens': int(args.target_max_tokens),
            'reached_output_limit_count': int(response_output_limit_reached_count),
            'reached_output_limit_ratio': (
                response_output_limit_reached_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'reached_max_output_tokens_count': int(response_output_limit_reached_count),
            'reached_max_output_tokens_ratio': (
                response_output_limit_reached_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'response_paragraph_counts': response_paragraph_counts.copy(),
            'response_paragraphs_le_5_count': int(response_paragraphs_le_5_count),
            'response_paragraphs_le_5_ratio': (
                response_paragraphs_le_5_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'has_reached_output_limit': bool(response_output_limit_reached_count > 0),
            'has_reached_max_output_tokens': bool(
                response_output_limit_reached_count > 0
            ),
            'output_limit_detection_method': output_limit_detection_method,
            'exact_token_count_response_count': int(response_token_count_source_exact_count),
            'api_token_count_response_count': int(response_token_count_source_api_count),
            'local_exact_token_count_response_count': int(
                response_token_count_source_local_exact_count
            ),
            'estimated_token_count_response_count': int(response_token_count_source_estimated_count),
        }

        result = {
            'brands': [asdict(b) for b in brands],
            'products': products,
            'docs': docs,
            'scores': category_scores,
            'experiment_records': category_experiment_records,
            'responses': category_responses,
            'experiment_seed': experiment_seed,
            'brand_shuffle_indices': brand_shuffle_indices,
            'doc_shuffle_indices': doc_shuffle_indices,
            'L1': L1,
            'inference_time_stats': inference_time_stats,
            'parsing_time_stats': parsing_time_stats,
            'output_token_stats': output_token_stats,
        }

        if trigger_prob_summary_local is not None:
            result['card_trigger_prob_records'] = trigger_prob_records
            result['card_trigger_prob_summary'] = trigger_prob_summary_local

        if api_retry_enabled:
            result['api_retry_stats'] = {
                'success_response_count': int(api_success_response_count),
                'recovered_prompt_count': int(api_recovered_prompt_count),
                'final_failed_prompt_count': int(api_final_failed_prompt_count),
                'total_attempt_count': int(api_total_attempt_count),
                'failed_prompts_path': api_failed_prompts_path,
                'recovered_prompts_path': api_recovered_prompts_path,
                'incomplete': bool(api_final_failed_prompt_count > 0),
            }

        if any(thinking_token_counts):
            result['thinking_stats'] = {
                'thinking_token_counts': thinking_token_counts,
                'response_token_counts': response_token_counts,
                'avg_thinking_tokens': float(np.mean(thinking_token_counts)),
                'avg_response_tokens': float(np.mean(response_token_counts)),
                'total_thinking_tokens': sum(thinking_token_counts),
                'total_response_tokens': sum(response_token_counts),
                'thinking_tokens_per_second': inference_time_stats['thinking_tokens_per_second'],
                'response_tokens_per_second': inference_time_stats['response_tokens_per_second'],
                'total_generated_tokens_per_second': (
                    inference_time_stats['total_generated_tokens_per_second']
                ),
            }

        return result

    if defer_finalization:
        return build_deferred_category_result

    
    logger.info(f"")
    logger.info(f"{'='*80}")
    log_key_info(logger, f"[Category] {category} - experiment complete")
    logger.info(f"total: {experiment_count}  timesexperiment tasks")
    logger.info(f"{'='*80}")

    if api_retry_enabled:
        logger.info(f"")
        logger.info(f"[API retry results]")
        logger.info(f"  Successful responses: {api_success_response_count}")
        logger.info(f"  Recovered prompts: {api_recovered_prompt_count}")
        logger.info(f"  Final failed prompts: {api_final_failed_prompt_count}")
        logger.info(f"  Total attempts: {api_total_attempt_count}")
        logger.info(f"  Failure ledger: {api_failed_prompts_path}")
        logger.info(f"  Recovery ledger: {api_recovered_prompts_path}")
        logger.info(
            f"  Incomplete recovery: {bool(api_final_failed_prompt_count > 0)}"
        )
    
    
    if any(thinking_token_counts):
        logger.info(f"")
        logger.info(f"[thinkingmodestatistics]")
        logger.info(f"  Total experiment count: {len(thinking_token_counts)}")
        logger.info(f"  thinking token statistics:")
        logger.info(f"    - mean: ~{np.mean(thinking_token_counts):.1f}")
        logger.info(f"    - min: ~{np.min(thinking_token_counts)}")
        logger.info(f"    - max: ~{np.max(thinking_token_counts)}")
        logger.info(f"    - std: ~{np.std(thinking_token_counts):.1f}")
        logger.info(f"  response token statistics:")
        logger.info(f"    - mean: ~{np.mean(response_token_counts):.1f}")
        logger.info(f"    - min: ~{np.min(response_token_counts)}")
        logger.info(f"    - max: ~{np.max(response_token_counts)}")
        total_tokens = [t + r for t, r in zip(thinking_token_counts, response_token_counts)]
        logger.info(f"  total token statistics:")
        logger.info(f"    - mean: ~{np.mean(total_tokens):.1f}")
        logger.info(f"    - total: ~{sum(total_tokens)}")

    trigger_prob_summary = None
    if (
        (not output_only_mode)
        and getattr(args, 'use_card', False)
        and is_triggered_card_mode(args)
    ):
        max_trigger_rank = max(trigger_probs_by_rank.keys(), default=0)

        trigger_rank_stats = {}
        for trigger_rank in sorted(trigger_probs_by_rank):
            main_values = trigger_probs_by_rank[trigger_rank]["main_output_probs"]
            aux_values = trigger_probs_by_rank[trigger_rank]["aux_output_probs"]
            strength_values = trigger_probs_by_rank[trigger_rank]["strength_values"]
            kl_main_vs_aux_values = trigger_probs_by_rank[trigger_rank]["kl_main_vs_aux_values"]
            kl_aux_vs_main_values = trigger_probs_by_rank[trigger_rank]["kl_aux_vs_main_values"]
            jsd_main_aux_values = trigger_probs_by_rank[trigger_rank]["jsd_main_aux_values"]
            main_mean = float(np.mean(main_values))
            aux_mean = float(np.mean(aux_values))
            trigger_rank_stats[str(trigger_rank)] = {
                "sample_count": len(main_values),
                "main_output_prob_mean_pre_card": main_mean,
                "aux_output_prob_mean_pre_card": aux_mean,
                "main_output_probs_pre_card": main_values,
                "aux_output_probs_pre_card": aux_values,
                "strength_mean": (
                    float(np.mean(strength_values))
                    if strength_values
                    else None
                ),
                "strength_values": strength_values,
                "kl_main_vs_aux_mean": (
                    float(np.mean(kl_main_vs_aux_values))
                    if kl_main_vs_aux_values
                    else None
                ),
                "kl_main_vs_aux_values": kl_main_vs_aux_values,
                "kl_aux_vs_main_mean": (
                    float(np.mean(kl_aux_vs_main_values))
                    if kl_aux_vs_main_values
                    else None
                ),
                "kl_aux_vs_main_values": kl_aux_vs_main_values,
                "jsd_main_aux_mean": (
                    float(np.mean(jsd_main_aux_values))
                    if jsd_main_aux_values
                    else None
                ),
                "jsd_main_aux_values": jsd_main_aux_values,
            }

        trigger_prob_summary = {
            "category": category,
            "total_experiments": experiment_count,
            "max_trigger_rank": max_trigger_rank,
            "raw_record_count": len(trigger_prob_records),
            "trigger_rank_stats": trigger_rank_stats,
        }

        category_safe = category.replace(" ", "_").replace("/", "_")
        stats_dir = out_path(args, "card_trigger_prob_stats")
        file_utils.ensure_created_directory(stats_dir)
        trigger_probs_jsonl_path = out_path(
            args, "card_trigger_prob_stats", f"{category_safe}_trigger_probs.jsonl"
        )
        trigger_prob_summary_path = out_path(
            args, "card_trigger_prob_stats", f"{category_safe}_trigger_prob_summary.json"
        )

        trigger_probs_jsonl = "\n".join(
            json.dumps(record, ensure_ascii=False) for record in trigger_prob_records
        )
        if trigger_probs_jsonl:
            trigger_probs_jsonl += "\n"
        file_utils.write_file(trigger_probs_jsonl_path, trigger_probs_jsonl)
        file_utils.write_file(
            trigger_prob_summary_path,
            json.dumps(trigger_prob_summary, ensure_ascii=False, indent=2) + "\n",
        )
    
    if output_only_mode:
        logger.info(f"")
        logger.info("[Output mode] --output-only enabled, skipBalance validationand evaluationstatistics")
    else:
        
        logger.info(f"")
        logger.info(f"[Balance validation]")

        
        brand_doc_counts = np.zeros((product_n, product_n), dtype=int)
        brand_pos_counts = np.zeros((product_n, product_n), dtype=int)
        doc_pos_counts = np.zeros((product_n, product_n), dtype=int)
        triple_counts = {}  # (brand, doc, pos) -> count

        for key, scores in category_scores.items():
            brand_idx, doc_idx, pos = key
            count = len(scores)
            brand_doc_counts[brand_idx, doc_idx] += count
            brand_pos_counts[brand_idx, pos] += count
            doc_pos_counts[doc_idx, pos] += count
            triple_counts[key] = count

        
        expected_pair_count = product_n
        bd_ok = (
            brand_doc_counts.min() == expected_pair_count
            and brand_doc_counts.max() == expected_pair_count
        )
        bp_ok = (
            brand_pos_counts.min() == expected_pair_count
            and brand_pos_counts.max() == expected_pair_count
        )

        logger.info(f"")
        logger.info(f"  [Marginal balance]")
        logger.info(
            f"    {'✓' if bd_ok else '❌'} (Brand, Doc) pairs: "
            f"min={brand_doc_counts.min()}, max={brand_doc_counts.max()}, "
            f"expected={expected_pair_count}"
        )
        logger.info(
            f"    {'✓' if bp_ok else '❌'} (Brand, Pos) pairs: "
            f"min={brand_pos_counts.min()}, max={brand_pos_counts.max()}, "
            f"expected={expected_pair_count}"
        )
        logger.info(
            f"    (Doc, Pos) pairs: min={doc_pos_counts.min()}, max={doc_pos_counts.max()} "
            f"(not required={expected_pair_count})"
        )

        
        triple_values = list(triple_counts.values())
        triple_ok = all(v == 1 for v in triple_values) and len(triple_values) == product_n ** 3

        logger.info(f"")
        logger.info(f"  [Triple uniqueness]")
        logger.info(f"    {'✓' if triple_ok else '❌'} (Brand, Doc, Pos) triple: total {len(triple_values)} , eachappears min={min(triple_values)}, max={max(triple_values)}  times, expected=1")

        if not triple_ok:
            
            for key, count in triple_counts.items():
                if count != 1:
                    logger.info(f"      issue: {key} appears {count}  times")

        
        logger.info(f"")
        logger.info(f"  [Brand-Doc distribution matrix] (expectedeachcell={expected_pair_count})")
        header = "        " + "".join([f"D{d:2d}  " for d in range(product_n)])
        logger.info(f"  {header}")
        for bi in range(product_n):
            row_str = f"  B[{bi}]    " + "".join([f"{brand_doc_counts[bi, di]:3d}  " for di in range(product_n)])
            logger.info(row_str)

        
        logger.info(f"")
        logger.info(f"  [Brand-Pos distribution matrix] (expectedeachcell={expected_pair_count})")
        header = "        " + "".join([f"P{p:2d}  " for p in range(product_n)])
        logger.info(f"  {header}")
        for bi in range(product_n):
            row_str = f"  B[{bi}]    " + "".join([f"{brand_pos_counts[bi, pi]:3d}  " for pi in range(product_n)])
            logger.info(row_str)

    total_inference_seconds = float(sum(inference_call_durations))
    total_inference_responses = int(sum(inference_response_counts))
    total_thinking_tokens = int(sum(thinking_token_counts))
    total_response_tokens = int(sum(response_token_counts))
    total_generated_tokens = total_thinking_tokens + total_response_tokens
    response_output_limit_reached_count = sum(
        1 for reached in response_output_limit_reached_flags
        if reached
    )
    output_limit_detection_method = (
        'vllm_finish_reason_length_or_token_threshold'
        if any(reason is not None for reason in response_finish_reasons)
        else 'token_threshold_only'
    )
    response_paragraphs_le_5_count = sum(
        1 for paragraph_count in response_paragraph_counts
        if paragraph_count <= 5
    )
    inference_time_stats = {
        'model_call_count': len(inference_call_durations),
        'response_count': total_inference_responses,
        'total_inference_seconds': total_inference_seconds,
        'total_thinking_tokens': total_thinking_tokens,
        'total_response_tokens': total_response_tokens,
        'total_generated_tokens': total_generated_tokens,
        'avg_inference_seconds_per_model_call': (
            float(np.mean(inference_call_durations)) if inference_call_durations else 0.0
        ),
        'avg_inference_seconds_per_response': (
            total_inference_seconds / total_inference_responses
            if total_inference_responses else 0.0
        ),
        'avg_response_tokens_per_response': (
            total_response_tokens / total_inference_responses
            if total_inference_responses else 0.0
        ),
        'avg_generated_tokens_per_response': (
            total_generated_tokens / total_inference_responses
            if total_inference_responses else 0.0
        ),
        'thinking_tokens_per_second': (
            total_thinking_tokens / total_inference_seconds
            if total_inference_seconds else 0.0
        ),
        'response_tokens_per_second': (
            total_response_tokens / total_inference_seconds
            if total_inference_seconds else 0.0
        ),
        'total_generated_tokens_per_second': (
            total_generated_tokens / total_inference_seconds
            if total_inference_seconds else 0.0
        ),
        'min_inference_seconds_per_model_call': (
            float(min(inference_call_durations)) if inference_call_durations else 0.0
        ),
        'max_inference_seconds_per_model_call': (
            float(max(inference_call_durations)) if inference_call_durations else 0.0
        ),
        'model_call_inference_seconds': [float(v) for v in inference_call_durations],
        'model_call_response_counts': inference_response_counts.copy(),
        'response_token_counts': response_token_counts.copy(),
        'response_paragraph_counts': response_paragraph_counts.copy(),
        'response_output_limit_reached_flags': response_output_limit_reached_flags.copy(),
        'response_finish_reasons': response_finish_reasons.copy(),
        'response_stop_reasons': response_stop_reasons.copy(),
        'target_max_tokens': int(args.target_max_tokens),
        'response_output_limit_reached_count': int(
            response_output_limit_reached_count
        ),
        'response_output_limit_reached_ratio': (
            response_output_limit_reached_count / total_inference_responses
            if total_inference_responses else 0.0
        ),
        
        'response_reached_max_tokens_count': int(response_output_limit_reached_count),
        'response_reached_max_tokens_ratio': (
            response_output_limit_reached_count / total_inference_responses
            if total_inference_responses else 0.0
        ),
        'output_limit_detection_method': output_limit_detection_method,
        'response_paragraphs_le_5_count': int(response_paragraphs_le_5_count),
        'response_paragraphs_le_5_ratio': (
            response_paragraphs_le_5_count / total_inference_responses
            if total_inference_responses else 0.0
        ),
        'response_token_count_source_exact_count': int(response_token_count_source_exact_count),
        'response_token_count_source_api_count': int(response_token_count_source_api_count),
        'response_token_count_source_local_exact_count': int(
            response_token_count_source_local_exact_count
        ),
        'response_token_count_source_estimated_count': int(response_token_count_source_estimated_count),
    }
    total_parsing_seconds = float(sum(parsing_call_durations))
    total_parsing_responses = int(sum(parsing_response_counts))
    parsing_time_stats = {
        'batch_parse_count': len(parsing_call_durations),
        'response_count': total_parsing_responses,
        'total_parsing_seconds': total_parsing_seconds,
        'avg_parsing_seconds_per_batch': (
            float(np.mean(parsing_call_durations)) if parsing_call_durations else 0.0
        ),
        'avg_parsing_seconds_per_response': (
            total_parsing_seconds / total_parsing_responses
            if total_parsing_responses else 0.0
        ),
        'min_parsing_seconds_per_batch': (
            float(min(parsing_call_durations)) if parsing_call_durations else 0.0
        ),
        'max_parsing_seconds_per_batch': (
            float(max(parsing_call_durations)) if parsing_call_durations else 0.0
        ),
        'batch_parsing_seconds': [float(v) for v in parsing_call_durations],
        'batch_parse_response_counts': parsing_response_counts.copy(),
    }
    if getattr(args, 'use_card', False):
        inference_time_stats['card_application_mode'] = resolve_card_application_mode(args)
        inference_time_stats['card_global_logit_formula'] = (
            resolve_card_global_logit_formula(args)
        )
        if is_card_global_custom_formula_enabled(args):
            inference_time_stats['card_global_formula_alpha'] = (
                get_card_global_formula_alpha(args)
            )
        inference_time_stats['card_batch_inference'] = bool(
            getattr(args, 'card_batch_inference', False)
        )
        if (
            getattr(args, 'target_local_backend', 'vllm') == "vllm"
            and resolve_card_application_mode(args) == "global"
        ):
            inference_time_stats['card_execution_mode'] = 'vllm_paired_global'
            inference_time_stats['card_global_main_bias_coeff'] = (
                get_card_global_main_bias_coeff(args)
            )
            inference_time_stats['card_global_direction_sign'] = (
                get_card_global_direction_sign(args)
            )
            inference_time_stats['card_global_vllm_support_mode'] = (
                resolve_card_global_vllm_support_mode(args)
            )
            inference_time_stats['card_global_vllm_support_top_k'] = int(
                getattr(args, 'card_global_vllm_support_top_k', 10)
            )
        else:
            inference_time_stats['card_execution_mode'] = (
                'true_batch'
                if getattr(args, 'card_batch_inference', False)
                else 'legacy_serial_single'
            )

    logger.info(f"")
    logger.info(
        "[Pure inference-time statistics] "
        "(counts only target(...) model-call elapsed time, excluding prompt construction, response parsing, and result statistics)"
    )
    logger.info(f"  Model call count: {inference_time_stats['model_call_count']}")
    logger.info(f"  Total responses: {inference_time_stats['response_count']}")
    if getattr(args, 'use_card', False):
        logger.info(
            "  CARD application mode: "
            f"{inference_time_stats['card_application_mode']}"
        )
        logger.info(
            "  CARD executionmode: "
            f"{inference_time_stats['card_execution_mode']}"
        )
        logger.info(
            "  global logits composition formula: "
            f"{inference_time_stats['card_global_logit_formula']}"
        )
        if 'card_global_formula_alpha' in inference_time_stats:
            logger.info(
                "  formula alpha: "
                f"{inference_time_stats['card_global_formula_alpha']:g}"
            )
        if 'card_global_main_bias_coeff' in inference_time_stats:
            logger.info(
                "  main-branch bias coefficient b: "
                f"{inference_time_stats['card_global_main_bias_coeff']:g}"
            )
        if 'card_global_direction_sign' in inference_time_stats:
            logger.info(
                "  direction signal sign: "
                f"{int(inference_time_stats['card_global_direction_sign'])} "
                "(1=enhance external-document contribution/debias, -1=suppress external-document contribution/poisoning defense)"
            )
        if 'card_global_vllm_support_mode' in inference_time_stats:
            logger.info(
                "  Global CARD vLLM support mode: "
                f"{inference_time_stats['card_global_vllm_support_mode']}"
            )
            if (
                inference_time_stats['card_global_vllm_support_mode']
                == "main_aux_topk_union"
            ):
                logger.info(
                    "  Global CARD vLLM support top-k: "
                    f"{inference_time_stats['card_global_vllm_support_top_k']}"
                )
    logger.info(f"  Total pure inference time: {inference_time_stats['total_inference_seconds']:.4f} s")
    logger.info(
        "  Token-count basis: exact countpreferred (API usage orlocally generated token IDs), "
        "nothenusing estimate_token_count(...) estimated"
    )
    logger.info(
        "  Token-count source: "
        f"exact count={inference_time_stats.get('response_token_count_source_exact_count', 0)} "
        f"(API usage={inference_time_stats.get('response_token_count_source_api_count', 0)}, "
        f"locally generated token IDs={inference_time_stats.get('response_token_count_source_local_exact_count', 0)}), "
        f"estimated={inference_time_stats.get('response_token_count_source_estimated_count', 0)}"
    )
    if int(inference_time_stats.get('response_token_count_source_estimated_count', 0)) > 0:
        logger.info(
            "  Note: when first Categoryresponse token countincludesestimatedvalue; the following response/total generated Tokens Per Second "
            "and and outputlength-limit decisionincludesestimatedcomponents."
        )
    logger.info(f"  Total response tokens: {inference_time_stats['total_response_tokens']}")
    if inference_time_stats['total_thinking_tokens'] > 0:
        logger.info(f"  Total thinking tokens: {inference_time_stats['total_thinking_tokens']}")
        logger.info(f"  Total generated tokens: {inference_time_stats['total_generated_tokens']}")
    logger.info(
        "  Average time per model call: "
        f"{inference_time_stats['avg_inference_seconds_per_model_call']:.4f} s"
    )
    logger.info(
        "  Average time per response: "
        f"{inference_time_stats['avg_inference_seconds_per_response']:.4f} s"
    )
    logger.info(
        "  Fastest/slowest model call: "
        f"{inference_time_stats['min_inference_seconds_per_model_call']:.4f} / "
        f"{inference_time_stats['max_inference_seconds_per_model_call']:.4f} s"
    )
    logger.info(
        "  response Tokens Per Second: "
        f"{inference_time_stats['response_tokens_per_second']:.4f} tokens/s"
    )
    if inference_time_stats['total_thinking_tokens'] > 0:
        logger.info(
            "  total generated Tokens Per Second: "
            f"{inference_time_stats['total_generated_tokens_per_second']:.4f} tokens/s"
            " (thinking + response)"
        )
    logger.info(f"")
    logger.info(
        "[Response parsing-time statistics] "
        "(counts only response matching and score parsing elapsed time, excluding prompt construction, target(...) model calls, and result statistics)"
    )
    logger.info(f"  Parse batch count: {parsing_time_stats['batch_parse_count']}")
    logger.info(f"  Parsed response count: {parsing_time_stats['response_count']}")
    logger.info(f"  Total parsing time: {parsing_time_stats['total_parsing_seconds']:.4f} s")
    if parsing_time_stats['batch_parse_count'] > 0:
        logger.info(
            "  Average time per parse batch: "
            f"{parsing_time_stats['avg_parsing_seconds_per_batch']:.4f} s"
        )
    if parsing_time_stats['response_count'] > 0:
        logger.info(
            "  Average parse time per response: "
            f"{parsing_time_stats['avg_parsing_seconds_per_response']:.4f} s"
        )
    logger.info(
        "  Fastest/slowest parse batch: "
        f"{parsing_time_stats['min_parsing_seconds_per_batch']:.4f} / "
        f"{parsing_time_stats['max_parsing_seconds_per_batch']:.4f} s"
    )
    logger.info(
        "  Total inference+parsing time: "
        f"{(total_inference_seconds + total_parsing_seconds):.4f} s"
    )

    
    result = {
        'brands': [asdict(b) for b in brands],
        'products': products,
        'docs': docs,
        'scores': category_scores,
        'experiment_records': category_experiment_records,
        'responses': category_responses,
        'experiment_seed': experiment_seed,
        'brand_shuffle_indices': brand_shuffle_indices,
        'doc_shuffle_indices': doc_shuffle_indices,
        'L1': L1,  
        'inference_time_stats': inference_time_stats,
        'parsing_time_stats': parsing_time_stats,
        'output_token_stats': {
            'response_token_counts': response_token_counts.copy(),
            'response_count': total_inference_responses,
            'avg_response_tokens': (
                float(np.mean(response_token_counts)) if response_token_counts else 0.0
            ),
            'min_response_tokens': (
                int(min(response_token_counts)) if response_token_counts else 0
            ),
            'max_response_tokens': (
                int(max(response_token_counts)) if response_token_counts else 0
            ),
            'response_output_limit_reached_flags': (
                response_output_limit_reached_flags.copy()
            ),
            'response_finish_reasons': response_finish_reasons.copy(),
            'response_stop_reasons': response_stop_reasons.copy(),
            'target_max_tokens': int(args.target_max_tokens),
            'reached_output_limit_count': int(response_output_limit_reached_count),
            'reached_output_limit_ratio': (
                response_output_limit_reached_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            
            'reached_max_output_tokens_count': int(response_output_limit_reached_count),
            'reached_max_output_tokens_ratio': (
                response_output_limit_reached_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'response_paragraph_counts': response_paragraph_counts.copy(),
            'response_paragraphs_le_5_count': int(response_paragraphs_le_5_count),
            'response_paragraphs_le_5_ratio': (
                response_paragraphs_le_5_count / total_inference_responses
                if total_inference_responses else 0.0
            ),
            'has_reached_output_limit': bool(response_output_limit_reached_count > 0),
            'has_reached_max_output_tokens': bool(
                response_output_limit_reached_count > 0
            ),
            'output_limit_detection_method': output_limit_detection_method,
            'exact_token_count_response_count': int(response_token_count_source_exact_count),
            'api_token_count_response_count': int(response_token_count_source_api_count),
            'local_exact_token_count_response_count': int(
                response_token_count_source_local_exact_count
            ),
            'estimated_token_count_response_count': int(response_token_count_source_estimated_count),
        },
    }

    if trigger_prob_summary is not None:
        result['card_trigger_prob_records'] = trigger_prob_records
        result['card_trigger_prob_summary'] = trigger_prob_summary

    if api_retry_enabled:
        result['api_retry_stats'] = {
            'success_response_count': int(api_success_response_count),
            'recovered_prompt_count': int(api_recovered_prompt_count),
            'final_failed_prompt_count': int(api_final_failed_prompt_count),
            'total_attempt_count': int(api_total_attempt_count),
            'failed_prompts_path': api_failed_prompts_path,
            'recovered_prompts_path': api_recovered_prompts_path,
            'incomplete': bool(api_final_failed_prompt_count > 0),
        }
    
    
    if any(thinking_token_counts):
        result['thinking_stats'] = {
            'thinking_token_counts': thinking_token_counts,
            'response_token_counts': response_token_counts,
            'avg_thinking_tokens': float(np.mean(thinking_token_counts)),
            'avg_response_tokens': float(np.mean(response_token_counts)),
            'total_thinking_tokens': sum(thinking_token_counts),
            'total_response_tokens': sum(response_token_counts),
            'thinking_tokens_per_second': inference_time_stats['thinking_tokens_per_second'],
            'response_tokens_per_second': inference_time_stats['response_tokens_per_second'],
            'total_generated_tokens_per_second': (
                inference_time_stats['total_generated_tokens_per_second']
            ),
        }
    
    return result


def log_inference_time_statistics_at_end(
    results: t.Dict[str, t.Dict],
    args: argparse.Namespace,
    logger: logging.Logger,
    results_path: t.Optional[str] = None,
) -> None:
    run_inference_call_count = 0
    run_inference_response_count = 0
    run_total_inference_seconds = 0.0
    run_parse_batch_count = 0
    run_parse_response_count = 0
    run_total_parsing_seconds = 0.0
    run_total_thinking_tokens = 0
    run_total_response_tokens = 0
    run_total_generated_tokens = 0
    run_source_exact_count = 0
    run_source_api_count = 0
    run_source_local_exact_count = 0
    run_source_estimated_count = 0
    for category_result in results.values():
        inference_stats = category_result.get('inference_time_stats', {}) or {}
        parsing_stats = category_result.get('parsing_time_stats', {}) or {}

        category_call_count = int(inference_stats.get('model_call_count', 0))
        category_response_count = int(inference_stats.get('response_count', 0))
        category_total_seconds = float(inference_stats.get('total_inference_seconds', 0.0))
        category_parse_batch_count = int(parsing_stats.get('batch_parse_count', 0))
        category_parse_response_count = int(parsing_stats.get('response_count', 0))
        category_parse_total_seconds = float(parsing_stats.get('total_parsing_seconds', 0.0))
        category_thinking_tokens = int(inference_stats.get('total_thinking_tokens', 0))
        category_response_tokens = int(inference_stats.get('total_response_tokens', 0))
        category_generated_tokens = int(
            inference_stats.get(
                'total_generated_tokens',
                category_thinking_tokens + category_response_tokens,
            )
        )
        category_source_exact_count = int(
            inference_stats.get('response_token_count_source_exact_count', 0)
        )
        category_source_api_count = int(
            inference_stats.get('response_token_count_source_api_count', 0)
        )
        category_source_local_exact_count = int(
            inference_stats.get('response_token_count_source_local_exact_count', 0)
        )
        category_source_estimated_count = int(
            inference_stats.get('response_token_count_source_estimated_count', 0)
        )

        run_inference_call_count += category_call_count
        run_inference_response_count += category_response_count
        run_total_inference_seconds += category_total_seconds
        run_parse_batch_count += category_parse_batch_count
        run_parse_response_count += category_parse_response_count
        run_total_parsing_seconds += category_parse_total_seconds
        run_total_thinking_tokens += category_thinking_tokens
        run_total_response_tokens += category_response_tokens
        run_total_generated_tokens += category_generated_tokens
        run_source_exact_count += category_source_exact_count
        run_source_api_count += category_source_api_count
        run_source_local_exact_count += category_source_local_exact_count
        run_source_estimated_count += category_source_estimated_count

    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info(
        "[Full-run pure inference-time summary] "
        "(counts only target(...) model-call elapsed time, excluding prompt construction, response parsing, and result statistics)"
    )
    logger.info(f"  Successful category count: {len(results)}")
    logger.info(f"  Model call count: {run_inference_call_count}")
    logger.info(f"  Total responses: {run_inference_response_count}")
    logger.info(f"  Total pure inference time: {run_total_inference_seconds:.4f} s")
    logger.info(
        "  Token-count basis: exact countpreferred (API usage orlocally generated token IDs), "
        "nothenusing estimate_token_count(...) estimated"
    )
    logger.info(
        "  Token-count source: "
        f"exact count={run_source_exact_count} (API usage={run_source_api_count}, "
        f"locally generated token IDs={run_source_local_exact_count}), "
        f"estimated={run_source_estimated_count}"
    )
    if run_source_estimated_count > 0:
        logger.info(
            "  Note: this timesrunresponse token countincludesestimatedvalue; the following response/total generated Tokens Per Second "
            "and and outputlength-limit decisionincludesestimatedcomponents."
        )
    logger.info(f"  Total response tokens: {run_total_response_tokens}")
    if run_total_thinking_tokens > 0:
        logger.info(f"  Total thinking tokens: {run_total_thinking_tokens}")
        logger.info(f"  Total generated tokens: {run_total_generated_tokens}")
    if run_inference_call_count > 0:
        logger.info(
            "  Average time per model call: "
            f"{run_total_inference_seconds / run_inference_call_count:.4f} s"
        )
    if run_inference_response_count > 0:
        logger.info(
            "  Average time per response: "
            f"{run_total_inference_seconds / run_inference_response_count:.4f} s"
        )
    if run_total_inference_seconds > 0:
        logger.info(
            "  response Tokens Per Second: "
            f"{run_total_response_tokens / run_total_inference_seconds:.4f} tokens/s"
        )
        if run_total_thinking_tokens > 0:
            logger.info(
                "  total generated Tokens Per Second: "
                f"{run_total_generated_tokens / run_total_inference_seconds:.4f} tokens/s"
                " (thinking + response)"
            )
    logger.info(
        "[Full-run response parsing-time summary] "
        "(counts only response matching and score parsing elapsed time, excluding prompt construction, target(...) model calls, and result statistics)"
    )
    logger.info(f"  Parse batch count: {run_parse_batch_count}")
    logger.info(f"  Parsed response count: {run_parse_response_count}")
    logger.info(f"  Total parsing time: {run_total_parsing_seconds:.4f} s")
    if run_parse_batch_count > 0:
        logger.info(
            "  Average time per parse batch: "
            f"{run_total_parsing_seconds / run_parse_batch_count:.4f} s"
        )
    if run_parse_response_count > 0:
        logger.info(
            "  Average parse time per response: "
            f"{run_total_parsing_seconds / run_parse_response_count:.4f} s"
        )
    logger.info(
        "  Total inference+parsing time: "
        f"{(run_total_inference_seconds + run_total_parsing_seconds):.4f} s"
    )
    if results_path is not None:
        logger.info(f"  resultsfile: {results_path}")
    logger.info(f"{'='*80}")


def log_output_token_statistics_at_end(
    results: t.Dict[str, t.Dict],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    run_response_count = 0
    run_total_response_tokens = 0
    run_reached_max_count = 0
    run_paragraphs_le_5_count = 0
    run_source_exact_count = 0
    run_source_api_count = 0
    run_source_local_exact_count = 0
    run_source_estimated_count = 0
    categories_reached_max: t.List[str] = []

    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info("[LLM output token statistics] (summarized by category, log-tail output)")
    logger.info(f"{'='*80}")

    for category_name, category_result in results.items():
        token_stats = category_result.get('output_token_stats', {}) or {}
        inference_stats = category_result.get('inference_time_stats', {}) or {}

        response_token_counts = token_stats.get(
            'response_token_counts',
            inference_stats.get('response_token_counts', []),
        ) or []
        response_token_counts = [int(v) for v in response_token_counts]

        response_count = int(
            token_stats.get(
                'response_count',
                len(response_token_counts) if response_token_counts else inference_stats.get('response_count', 0),
            )
        )
        response_paragraph_counts = token_stats.get(
            'response_paragraph_counts',
            inference_stats.get('response_paragraph_counts', []),
        ) or []
        response_paragraph_counts = [int(v) for v in response_paragraph_counts]
        if not response_paragraph_counts:
            raw_responses = category_result.get('responses', []) or []
            for response_item in raw_responses:
                response_text = None
                if isinstance(response_item, str):
                    response_text = response_item
                elif isinstance(response_item, (list, tuple)) and response_item:
                    response_text = response_item[0]
                if isinstance(response_text, str):
                    response_paragraph_counts.append(
                        count_response_paragraphs(response_text)
                    )

        total_response_tokens = int(
            sum(response_token_counts)
            if response_token_counts
            else inference_stats.get('total_response_tokens', 0)
        )
        avg_response_tokens = float(
            token_stats.get(
                'avg_response_tokens',
                (total_response_tokens / response_count) if response_count else 0.0,
            )
        )
        min_response_tokens = int(
            token_stats.get(
                'min_response_tokens',
                min(response_token_counts) if response_token_counts else 0,
            )
        )
        max_response_tokens = int(
            token_stats.get(
                'max_response_tokens',
                max(response_token_counts) if response_token_counts else 0,
            )
        )
        target_max_tokens = int(
            token_stats.get(
                'target_max_tokens',
                inference_stats.get('target_max_tokens', args.target_max_tokens),
            )
        )
        output_limit_detection_method = str(
            token_stats.get(
                'output_limit_detection_method',
                inference_stats.get(
                    'output_limit_detection_method',
                    'token_threshold_only',
                ),
            )
        )
        response_output_limit_reached_flags = token_stats.get(
            'response_output_limit_reached_flags',
            inference_stats.get('response_output_limit_reached_flags', []),
        ) or []
        response_output_limit_reached_flags = [
            bool(v) for v in response_output_limit_reached_flags
        ]

        reached_max_count = int(
            token_stats.get(
                'reached_output_limit_count',
                token_stats.get(
                    'reached_max_output_tokens_count',
                    inference_stats.get(
                        'response_output_limit_reached_count',
                        inference_stats.get('response_reached_max_tokens_count', 0),
                    ),
                ),
            )
        )
        if response_output_limit_reached_flags:
            reached_max_count = sum(
                1 for reached in response_output_limit_reached_flags
                if reached
            )
        elif response_token_counts:
            reached_max_count = sum(
                1 for token_count in response_token_counts
                if token_count >= target_max_tokens
            )
        reached_max_ratio = (
            reached_max_count / response_count
            if response_count else 0.0
        )
        has_reached_max = reached_max_count > 0
        paragraphs_le_5_count = int(
            token_stats.get(
                'response_paragraphs_le_5_count',
                inference_stats.get('response_paragraphs_le_5_count', 0),
            )
        )
        if response_paragraph_counts:
            paragraphs_le_5_count = sum(
                1 for paragraph_count in response_paragraph_counts
                if paragraph_count <= 5
            )
        paragraphs_le_5_ratio = (
            paragraphs_le_5_count / response_count
            if response_count else 0.0
        )
        source_exact_count = int(
            token_stats.get(
                'exact_token_count_response_count',
                inference_stats.get('response_token_count_source_exact_count', 0),
            )
        )
        source_api_count = int(
            token_stats.get(
                'api_token_count_response_count',
                inference_stats.get('response_token_count_source_api_count', 0),
            )
        )
        source_local_exact_count = int(
            token_stats.get(
                'local_exact_token_count_response_count',
                inference_stats.get('response_token_count_source_local_exact_count', 0),
            )
        )
        source_estimated_count = int(
            token_stats.get(
                'estimated_token_count_response_count',
                inference_stats.get('response_token_count_source_estimated_count', 0),
            )
        )

        run_response_count += response_count
        run_total_response_tokens += total_response_tokens
        run_reached_max_count += reached_max_count
        run_paragraphs_le_5_count += paragraphs_le_5_count
        run_source_exact_count += source_exact_count
        run_source_api_count += source_api_count
        run_source_local_exact_count += source_local_exact_count
        run_source_estimated_count += source_estimated_count
        if has_reached_max:
            categories_reached_max.append(category_name)

        logger.info(f"")
        logger.info(f"[Category] {category_name}")
        logger.info(f"  Output response count: {response_count}")
        logger.info(f"  mean output tokens: {avg_response_tokens:.2f}")
        logger.info(f"  min/max output tokens: {min_response_tokens} / {max_response_tokens}")
        logger.info(
            "  responses with <=5 output paragraphs: "
            f"{paragraphs_le_5_count}/{response_count} ({paragraphs_le_5_ratio * 100:.2f}%)"
        )
        logger.info(
            "  responses reaching output length limit: "
            f"{reached_max_count}/{response_count} ({reached_max_ratio * 100:.2f}%)"
        )
        logger.info(f"  reached output length limit: {'yes' if has_reached_max else 'no'}")
        if output_limit_detection_method == 'vllm_finish_reason_length_or_token_threshold':
            logger.info(
                "  output length-limit criterion: vLLM finish_reason=length preferred; "
                f"otherwise fall back to output tokens >= --target-max-tokens ({target_max_tokens})"
            )
        else:
            logger.info(
                "  output length-limit criterion: "
                f"output tokens >= --target-max-tokens ({target_max_tokens})"
            )
        logger.info(
            "  Token statistics basis: "
            f"exact count={source_exact_count} (API usage={source_api_count}, "
            f"locally generated token IDs={source_local_exact_count}), "
            f"estimated={source_estimated_count}"
        )
        if source_estimated_count > 0:
            logger.info(
                "  Note: thisCategoryoutput token statistics includeestimatedvalue; mean/min/max/output-length-limit statistics are approximate references only."
            )

    run_avg_response_tokens = (
        run_total_response_tokens / run_response_count
        if run_response_count else 0.0
    )
    run_reached_max_ratio = (
        run_reached_max_count / run_response_count
        if run_response_count else 0.0
    )
    run_paragraphs_le_5_ratio = (
        run_paragraphs_le_5_count / run_response_count
        if run_response_count else 0.0
    )

    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info("[Full-run LLM output token statistics summary] (final log tail)")
    logger.info(f"  Successful category count: {len(results)}")
    logger.info(f"  Total output responses: {run_response_count}")
    logger.info(f"  total output tokens: {run_total_response_tokens}")
    logger.info(f"  mean output tokens: {run_avg_response_tokens:.2f}")
    logger.info(
        "  total responses with <=5 output paragraphs: "
        f"{run_paragraphs_le_5_count}/{run_response_count} ({run_paragraphs_le_5_ratio * 100:.2f}%)"
    )
    logger.info(
        "  total responses reaching output length limit: "
        f"{run_reached_max_count}/{run_response_count} ({run_reached_max_ratio * 100:.2f}%)"
    )
    logger.info(
        "  Token statistics basissummary: "
        f"exact count={run_source_exact_count} (API usage={run_source_api_count}, "
        f"locally generated token IDs={run_source_local_exact_count}), "
        f"estimated={run_source_estimated_count}"
    )
    if run_source_estimated_count > 0:
        logger.info(
            "  Note: full timesrunoutput token summary includesestimatedvalue; mean/output-length-limit statistics includeestimatedcomponents."
        )
    if categories_reached_max:
        logger.info(
            "  categories reaching output length limit: "
            f"{', '.join(categories_reached_max)}"
        )
    else:
        logger.info("  categories reaching output length limit: none")
    logger.info(f"{'='*80}")


def log_card_trigger_statistics_by_category(
    results: t.Dict[str, t.Dict],
    args: argparse.Namespace,
    logger,
) -> None:
    category_max_single_generation_trigger_counts: t.List[t.Tuple[str, int]] = []

    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info("[CARD trigger statistics] (summarized by category, log-tail output)")
    logger.info(f"{'='*80}")

    for category_name, category_result in results.items():
        summary = category_result.get('card_trigger_prob_summary')
        category_safe = category_name.replace(" ", "_").replace("/", "_")
        trigger_probs_jsonl_path = out_path(
            args, "card_trigger_prob_stats", f"{category_safe}_trigger_probs.jsonl"
        )
        trigger_prob_summary_path = out_path(
            args, "card_trigger_prob_stats", f"{category_safe}_trigger_prob_summary.json"
        )

        logger.info(f"")
        logger.info(f"[Category] {category_name}")

        if summary is None:
            category_max_single_generation_trigger_counts.append((category_name, 0))
            logger.info("  notdetectedavailablestatisticstrigger pointprobability data")
            logger.info(f"  original source samplethis: {trigger_probs_jsonl_path}")
            logger.info(f"  summaryresults: {trigger_prob_summary_path}")
            continue

        trigger_count = int(summary.get('raw_record_count', 0))
        max_single_generation_trigger_count = int(summary.get('max_trigger_rank', 0))
        category_max_single_generation_trigger_counts.append(
            (category_name, max_single_generation_trigger_count)
        )

        logger.info("  [CARD trigger-point probabilitystatistics] (by trigger rank, pre-CARD)")
        logger.info(f"  Total experiment count: {summary.get('total_experiments', 0)}")
        logger.info(f"  triggerrecord count: {trigger_count}")
        logger.info(
            "  single timesgeneratedmaxtrigger pointcount: "
            f"{max_single_generation_trigger_count}"
        )

        trigger_rank_stats = summary.get('trigger_rank_stats', {})
        if not trigger_rank_stats:
            logger.info("  notdetectedavailablestatisticstrigger pointprobability data")
        else:
            for trigger_rank in sorted(trigger_rank_stats, key=int):
                rank_stats = trigger_rank_stats[trigger_rank]
                main_values = rank_stats.get('main_output_probs_pre_card', [])
                aux_values = rank_stats.get('aux_output_probs_pre_card', [])
                strength_values = rank_stats.get('strength_values', [])
                kl_main_vs_aux_values = rank_stats.get('kl_main_vs_aux_values', [])
                kl_aux_vs_main_values = rank_stats.get('kl_aux_vs_main_values', [])
                jsd_main_aux_values = rank_stats.get('jsd_main_aux_values', [])

                logger.info(f"")
                logger.info(f"  rank {trigger_rank}trigger point:")
                logger.info(f"  hasvalidSample count: {rank_stats.get('sample_count', 0)}")
                logger.info(
                    "  main-model output token probabilityMean: "
                    f"{rank_stats.get('main_output_prob_mean_pre_card', 0.0):.8f}"
                )
                logger.info(
                    "  auxiliary-model output token probabilityMean: "
                    f"{rank_stats.get('aux_output_prob_mean_pre_card', 0.0):.8f}"
                )
                logger.info(
                    "  main-model output token probabilityall values: "
                    f"[{', '.join(f'{v:.8f}' for v in main_values)}]"
                )
                logger.info(
                    "  auxiliary-model output token probabilityall values: "
                    f"[{', '.join(f'{v:.8f}' for v in aux_values)}]"
                )
                if strength_values:
                    logger.info(
                        "  actual strength used Mean: "
                        f"{float(np.mean(strength_values)):.8f}"
                    )
                    logger.info(
                        "  actual strength used all values: "
                        f"[{', '.join(f'{v:.8f}' for v in strength_values)}]"
                    )
                if kl_main_vs_aux_values:
                    logger.info(
                        "  KL(normal || aux) Mean: "
                        f"{float(np.mean(kl_main_vs_aux_values)):.8f}"
                    )
                    logger.info(
                        "  KL(normal || aux) all values: "
                        f"[{', '.join(f'{v:.8f}' for v in kl_main_vs_aux_values)}]"
                    )
                if kl_aux_vs_main_values:
                    logger.info(
                        "  KL(aux || normal) Mean: "
                        f"{float(np.mean(kl_aux_vs_main_values)):.8f}"
                    )
                    logger.info(
                        "  KL(aux || normal) all values: "
                        f"[{', '.join(f'{v:.8f}' for v in kl_aux_vs_main_values)}]"
                    )
                if jsd_main_aux_values:
                    logger.info(
                        "  JSD(normal, aux) Mean: "
                        f"{float(np.mean(jsd_main_aux_values)):.8f}"
                    )
                    logger.info(
                        "  JSD(normal, aux) all values: "
                        f"[{', '.join(f'{v:.8f}' for v in jsd_main_aux_values)}]"
                    )

        logger.info(f"  original source samplethis: {trigger_probs_jsonl_path}")
        logger.info(f"  summaryresults: {trigger_prob_summary_path}")

    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info("[CARDsingle timesgeneratedtrigger pointcount summary] (byCategory, log-tail output)")
    logger.info(f"{'='*80}")

    over_threshold_categories = []
    for category_name, max_single_generation_trigger_count in category_max_single_generation_trigger_counts:
        logger.info(
            f"  {category_name}: single timesgeneratedmaxtrigger pointcount = "
            f"{max_single_generation_trigger_count}"
        )
        if max_single_generation_trigger_count > 20:
            over_threshold_categories.append(
                (category_name, max_single_generation_trigger_count)
            )

    if over_threshold_categories:
        warning_text = ", ".join(
            f"{category_name}={trigger_count}"
            for category_name, trigger_count in over_threshold_categories
        )
        logger.warning(
            "[CARDtrigger pointcount warning] followingCategoryhas in single timesgeneratedtrigger pointexceeds20: "
            f"{warning_text}"
        )




configure_analysis_dependencies(
    AnalysisDependencies(
        brand_info_cls=BrandInfo,
        file_utils=file_utils,
        get_logger=get_logger,
        is_single_test_category=is_single_test_category,
        get_single_test_category=get_single_test_category,
        get_requested_test_categories=get_requested_test_categories,
        sanitize_category_name=sanitize_category_name,
        out_path=out_path,
        plot_path=plot_path,
        summary_log_float_digit_variants=SUMMARY_LOG_FLOAT_DIGIT_VARIANTS,
        summary_log_rounding_note=SUMMARY_LOG_ROUNDING_NOTE,
        format_summary_log_float=format_summary_log_float,
    )
)




def add_experiment_arguments(parser: argparse.ArgumentParser):
    
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(Models.keys()),
        help="used to extractparametric knowledgemodel name (step1/3usemodel)"
    )
    
    
    parser.add_argument(
        "--num-brands", type=int, default=4,
        help=(
            "Number of parametric brands.4=4parameter+4fictional, 8=8parameter+8fictional, "
            "10-40=based on top40 data select Nparameter+Nfictional subset"
        )
    )
    
    
    parser.add_argument(
        "--no-ranking", action="store_true",
        help="do not userankparametric knowledgeresults (ranked by defaultrank)"
    )
    
    
    parser.add_argument("--run-eval", action="store_true", help="Run experiment")
    parser.add_argument("--experiment-seed", type=int, default=42, help="Experiment random seed (for reproducing)")
    parser.add_argument("--batch-size", type=int, default=8, 
                       help="Batch inference batch size (default=8; API backend does Block withinpointsbatch, local backendbyCategorywithinall N×N tasks across Block pointsbatch)")
    parser.add_argument("--num-parsing-workers", type=int, default=8,
                       help="Parallel parsingresponsesprocessescount (default=8)")
    parser.add_argument(
        "--local-parsing-workers", type=int, default=1,
        help="local modelparseprocessescount (default=1; greater than1whenusing spawn safeParallel parsing)"
    )
    parser.add_argument(
        "--async-parse-max-pending-batches",
        type=int,
        default=8,
        help=(
            "local backendAsync pipeline parsingkeep at most pending batch count"
            " (default=8; onlylocal backend and non- output-only applies)"
        ),
    )
    parser.add_argument(
        "--output-only", action="store_true",
        help="Only record and save LLM output; skip output matching, scoring, and evaluation statistics (for normal mode and mitigation methods)"
    )
    parser.add_argument("--no-ordering-prompt", action="store_true")
    parser.add_argument(
        "--use-system-role-baseline", action="store_true",
        help="using System Role prompt as is recommendation-bias mitigationbaseline"
    )
    parser.add_argument(
        "--use-debias-instruction-baseline", action="store_true",
        help="using Debias Instruction user prompt as is recommendation-bias mitigationbaseline"
    )
    parser.add_argument(
        "--use-moral-self-correction-baseline", action="store_true",
        help="using Moral Self-Correction user prompt as is recommendation-bias mitigationbaseline"
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help="test one or more Category; multipleCategoryuseseparated by English commas"
    )
    parser.add_argument(
        "--single-category-run-tag", type=str, default=None,
        help="Categorytargeted experimentoutputtag; and --test onestartusewhencan specify/reuse a timesrun directory, default when --run-eval auto whenGenerated atstamp"
    )
    parser.add_argument(
        "--out-base-dir", type=str, default="./out",
        help="Result output root directory (default ./out)"
    )
    parser.add_argument(
        "--plot-base-dir", type=str, default="./plots",
        help="plotOutput root directory (default ./plots)"
    )
    parser.add_argument(
        "--include-factor-level-f",
        action="store_true",
        help="additionally output Brand/Document/Context Position  factor-level F diagnostics (defaultoff)",
    )
    


def add_target_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--target-model", type=str, default=None,
        choices=list(Models.keys()),
        help="Target recommendation model (defaults to --model same)"
    )
    parser.add_argument("--target-temp", type=float, default=None, help="Target temperature (None for model default in thinking mode, 0 for greedy decoding otherwise)")
    parser.add_argument("--target-top-p", type=float, default=None, help="Target top-p (None to not use, ignored when temp=0)")
    parser.add_argument("--target-max-tokens", type=int, default=3000, help="Target max tokens")
    parser.add_argument("--target-gpu-ids", type=str, default=0, help="Target GPU IDs")
    parser.add_argument(
        "--target-local-backend",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="Local target-model inference backend (vLLM only); applies to normal local inference, CK vLLM, and Global CARD vLLM",
    )
    parser.add_argument(
        "--target-vllm-gpu-memory-utilization",
        type=float,
        default=0.6,
        help="normal local vLLM / CK vLLM / Global CARD vLLM during inference GPU GPU memory utilization cap (default 0.6)",
    )
    parser.add_argument(
        "--target-vllm-max-model-len",
        type=int,
        default=None,
        help="max_model_len for normal local vLLM / CK vLLM / Global CARD vLLM (default None, using vLLM defaultvalue)",
    )
    parser.add_argument(
        "--target-vllm-max-num-seqs",
        type=int,
        default=None,
        help="normal local vLLM / CK vLLM / Global CARD vLLM  max_num_seqs (default None, using vLLM defaultvalue; paired decoding should use at least 2*batch-size)",
    )
    parser.add_argument(
        "--target-vllm-max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "max_num_batched_tokens for normal local vLLM / CK vLLM / Global CARD vLLM "
            " (default None, using vLLM defaultvalue)"
        ),
    )
    parser.add_argument("--enable-thinking", action="store_true", 
                       help="enable Qwen3 thinkingmode (onlyfor supported modelshasvalid)")


def add_ck_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use-ck", action="store_true",
        help="using CK-PLUG style two-branch decoding for recommendation-bias mitigation"
    )
    parser.add_argument(
        "--ck-alpha", type=float, default=0.5,
        help="CK: Fixed alpha coefficient, valuerange [0,1] (default 0.5; adaptive modeare ignored in; set to for original-default reproduction 0.0)"
    )
    parser.add_argument(
        "--ck-adaptive", action="store_true",
        help="CK: enable adaptive mode, ignore --ck-alpha"
    )
    parser.add_argument(
        "--ck-select-top", type=int, default=10,
        help="CK: relative-top filter at leastkeep token count (default 10)"
    )
    parser.add_argument(
        "--ck-relative-top", type=float, default=0.01,
        help="CK: relative-top filter threshold (default 0.01)"
    )


def add_card_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--use-card", action="store_true",
                       help="using CARD method for recommendation-bias mitigation")
    parser.add_argument("--card-application-mode", type=str, default="global",
                       choices=["global"],
                       help="CARD: application mode; current packaged version only supports global")
    parser.add_argument("--card-global-logit-formula", type=str, default="contrastive",
                       choices=["contrastive", "ck", "zxy"],
                       help="CARD(global): logits composition formula; contrastive=main+alpha*(main-aux), ck=alpha*aux+(1-alpha)*(main-aux), zxy=main-alpha*aux (default contrastive)")
    parser.add_argument("--card-global-ck-alpha", type=float, default=0.5,
                       help="CARD(global, ck): alpha coefficient, valuerange [0,1] (default 0.5)")
    parser.add_argument("--card-global-zxy-alpha", type=float, default=0.5,
                       help="CARD(global, zxy): alpha coefficient (default 0.5)")
    parser.add_argument("--card-global-main-bias-coeff", type=float, default=0.0,
                       help="Global CARD vLLM: main-branch bias coefficient b, range [-1,1], formula is (1-abs(b))*main + sign*alpha*(main-aux); default 0.0 equivalent tooriginal formula")
    parser.add_argument("--card-global-direction-sign", type=int, default=1,
                       choices=[1, -1],
                       help="Global CARD vLLM: direction signal sign; 1=enhance external-document contribution/debias, -1=suppress external-document contribution/poisoning defense (default 1)")
    parser.add_argument("--card-global-vllm-support-mode", type=str, default="main_aux_topk_union",
                       choices=["full_vocab", "main_aux_topk_union"],
                       help="Global CARD vLLM: next token candidate support; full_vocab=full vocabulary, main_aux_topk_union=main and auxiliary branches top-k union (default main_aux_topk_union)")
    parser.add_argument("--card-global-vllm-support-top-k", type=int, default=10,
                       help="Global CARD vLLM: main_aux_topk_union mode takes per-branch top-k  k (default 10)")
    parser.add_argument(
        "--save-global-card-token-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save Global CARD vLLM per- token trace (defaultenable; including each output token  KL(aux || main)  and  alpha_t)",
    )
    parser.add_argument("--card-strength", type=float, default=2.0,
                       help="CARD: fixedpreset strength (default 2.0; only in  --card-use-fixed-strength=true active when)")
    parser.add_argument("--card-use-fixed-strength", type=str, default="false",
                       choices=["true", "false"],
                       help="CARD: yesnouse fixed preset strength; false enables dynamic when strength (default false)")
    parser.add_argument("--card-dynamic-strength-max", type=float, default=1.0,
                       help="CARD: dynamic strength  in  kappa_t=0 target maximum when M; actual formula is 1/ln(exp(1/M)+kappa_t) (default 1.0)")
    parser.add_argument("--card-dynamic-alpha-recompute", type=str, default="true",
                       choices=["true", "false"],
                       help="CARD: in dynamic modeyesnoeachrealtrigger pointrecompute kappa (default true)")
    parser.add_argument("--card-modulated-prob", type=str, default="false",
                       choices=["true", "false"],
                       help="CARD: enableProbability modulation; weights come from the full vocabulary main_probs (default false)")
    parser.add_argument("--card-prob-weight-beta", type=float, default=1.0,
                       help="CARD: probability exponent scaling beta, only in  --card-modulated-prob=true active when (default 1)")
    parser.add_argument("--card-aux-prompt-type", type=str, default="delete",
                       choices=["delete"],
                       help="Global CARD vLLM: auxiliary prompt mode (only delete)")
    parser.add_argument("--card-use-attention-mask", type=str, default="false",
                       choices=["true", "false"],
                       help="CARD: legacy-compatible parameter; true equivalent to --card-aux-prompt-type mask")
    parser.add_argument("--card-batch-inference", type=str, default="true",
                       choices=["true", "false"],
                       help="CARD: yesnoenableTrue batch inference; false keeps the old serial per-item path when (default true)")
    parser.add_argument("--card-use-top-k-constraint", type=str, default="false",
                       choices=["true", "false"],
                       help="CARD: yesnoonly in mainmodel Top-k candidatesetwithinapply use  CARD (default false)")
    parser.add_argument("--card-top-k", type=int, default=5,
                       help="CARD: CARD can be reranked during correctionmainmodel Top-k candidateset size (default 5)")
    parser.add_argument("--card-filter-opposite-start-tokens", type=str, default="false",
                       choices=["true", "false"],
                       help="CARD: yesnoat trigger positions by trigger_type filter opposite-side exclusive first token (only triggered modeapplies, default false)")
    parser.add_argument("--card-filter-system-prompt-example-tokens", type=str, default="false",
                       choices=["true", "false"],
                       help="CARD: yesnofilter at trigger positions system prompt few-shot examplebrand/modelstart token (only triggered modeapplies, default false)")
    parser.add_argument("--card-trigger-mode", type=str, default="top_k_count",
                       choices=["top_k_count", "prob_mass_ratio"],
                       help="CARD: trigger detection mode (only triggered modeapplies); top_k_count=oldTop-kmatch threshold, prob_mass_ratio=probability-mass sum<=in threshold setbrand/modeltokenat least2")
    parser.add_argument("--card-trigger-prob-sum-threshold", type=float, default=0.999,
                       help="CARD: prob_mass_ratio modeunderprobability-mass threshold (only triggered modeapplies, default 0.999)")
    parser.add_argument("--card-trigger-top-k", type=int, default=5,
                       help="CARD: check top-k candidate token (only triggered modeapplies, default 5)")
    parser.add_argument("--card-trigger-threshold", type=int, default=3,
                       help="CARD: top-k  in sum of brand and model matchesat leasthow many required totrigger (only triggered modeapplies, default 3)")
    parser.add_argument("--card-trigger-window", type=int, default=0,
                       help="CARD: skip after trigger token window (only triggered modeapplies, default 10)")
    parser.add_argument("--card-trigger-followup-tokens", type=int, default=0,
                       help="CARD: each timesforce additional application after trigger CARD  token count (only triggered modeapplies, default 0, namely next token)")
    parser.add_argument("--card-max-trigger-count", type=int, default=None,
                       help="CARD: maximum allowed during the whole generation timesrealtrigger (only triggered modeapplies); omit meansunlimited (default None)")
    parser.add_argument("--card-token-collection-mode", type=str, default="first_only",
                       choices=["first_only", "all_tokens", "all_tokens_no_digits"],
                       help="CARD: brand/model token collection mode (only triggered modeapplies, default first_only)")


def add_api_retry_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=1,
        help="API prompt batchwithinconcurrent requestscount (default 1, keep serial)",
    )
    parser.add_argument(
        "--api-submit-stagger",
        type=float,
        default=0.2,
        help="API for each concurrent submission prompt maximum random stagger before (s, default 0.2)",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=5,
        help="API prompt failureadditional retries after timescount (default 5)",
    )
    parser.add_argument(
        "--api-retry-initial-delay",
        type=float,
        default=10.0,
        help="API rank one timesfailureinitial afterwaittime (s, default 10)",
    )
    parser.add_argument(
        "--api-retry-backoff",
        type=float,
        default=2.0,
        help="API retry exponential backoff factor (default 2.0)",
    )
    parser.add_argument(
        "--api-retry-max-delay",
        type=float,
        default=120.0,
        help="Maximum wait time for a single API retry (seconds, default 120)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_experiment_arguments(parser)
    add_target_arguments(parser)
    add_ck_arguments(parser)
    add_card_arguments(parser)
    add_api_retry_arguments(parser)
    
    args = parser.parse_args()
    args.local_parsing_workers = max(1, args.local_parsing_workers)
    args.api_concurrency = max(1, args.api_concurrency)
    args.api_submit_stagger = max(0.0, args.api_submit_stagger)
    
    
    
    import sys
    args.target_temp_specified = any(arg.startswith('--target-temp') for arg in sys.argv)
    args.card_aux_prompt_type_specified = any(arg.startswith('--card-aux-prompt-type') for arg in sys.argv)
    
    
    if args.target_temp is None and not args.enable_thinking:
        args.target_temp = 0.0

    args.test_categories = parse_requested_test_categories(args.test)
    if args.test and not args.test_categories:
        raise ValueError("--test must provide at least one non-emptyCategoryname")

    
    args.with_ranking = not args.no_ranking
    validate_num_brands_value(int(args.num_brands))

    if getattr(args, 'use_system_role_baseline', False) and getattr(args, 'use_debias_instruction_baseline', False):
        raise ValueError(
            "--use-system-role-baseline and --use-debias-instruction-baseline cannot be enabled together; "
            "run the two prompt-only baselines separately."
        )

    if getattr(args, 'use_system_role_baseline', False) and getattr(args, 'use_moral_self_correction_baseline', False):
        raise ValueError(
            "--use-system-role-baseline and --use-moral-self-correction-baseline cannot be enabled together; "
            "run the two prompt-only baselines separately."
        )

    if getattr(args, 'use_debias_instruction_baseline', False) and getattr(args, 'use_moral_self_correction_baseline', False):
        raise ValueError(
            "--use-debias-instruction-baseline and --use-moral-self-correction-baseline cannot be enabled together; "
            "run the two user prompt baselines separately."
        )

    if getattr(args, 'use_system_role_baseline', False) and getattr(args, 'use_ck', False):
        raise ValueError(
            "--use-system-role-baseline and --use-ck cannot be enabled together; "
            "run System Role baseline and CK separately."
        )

    if getattr(args, 'use_debias_instruction_baseline', False) and getattr(args, 'use_ck', False):
        raise ValueError(
            "--use-debias-instruction-baseline and --use-ck cannot be enabled together; "
            "run Debias Instruction baseline and CK separately."
        )

    if getattr(args, 'use_moral_self_correction_baseline', False) and getattr(args, 'use_ck', False):
        raise ValueError(
            "--use-moral-self-correction-baseline and --use-ck cannot be enabled together; "
            "run Moral Self-Correction baseline and CK separately."
        )

    if getattr(args, 'use_system_role_baseline', False) and getattr(args, 'use_card', False):
        raise ValueError(
            "--use-system-role-baseline and --use-card cannot be enabled together; "
            "run System Role baseline and CARD separately."
        )

    if getattr(args, 'use_debias_instruction_baseline', False) and getattr(args, 'use_card', False):
        raise ValueError(
            "--use-debias-instruction-baseline and --use-card cannot be enabled together; "
            "run Debias Instruction baseline and CARD separately."
        )

    if getattr(args, 'use_moral_self_correction_baseline', False) and getattr(args, 'use_card', False):
        raise ValueError(
            "--use-moral-self-correction-baseline and --use-card cannot be enabled together; "
            "run Moral Self-Correction baseline and CARD separately."
        )

    if getattr(args, 'use_ck', False) and getattr(args, 'use_card', False):
        raise ValueError(
            "--use-ck and --use-card cannot be enabled together; "
            "run CK and CARD separately."
        )
    
    
    if args.target_model is None:
        args.target_model = args.model

    use_ck = bool(getattr(args, 'use_ck', False))
    use_card = bool(getattr(args, 'use_card', False))

    if use_ck and Models[args.target_model][2]:
        raise ValueError("--use-ck currently only supports local Hugging Face model")

    if (
        getattr(args, 'target_local_backend', 'vllm') == "vllm"
        and use_ck
    ):
        if getattr(args, 'enable_thinking', False):
            raise ValueError(
                "--use-ck --target-local-backend vllm only supports non- thinking greedy decoding"
            )
        if getattr(args, 'target_temp', 0.0) != 0.0:
            raise ValueError(
                "--use-ck --target-local-backend vllm onlysupports --target-temp 0.0"
            )

    if (
        getattr(args, 'target_local_backend', 'vllm') == "vllm"
        and use_card
    ):
        if getattr(args, 'enable_thinking', False):
            raise ValueError(
                "--use-card --target-local-backend vllm only supports non- thinking greedy decoding"
            )
        if getattr(args, 'target_temp', 0.0) != 0.0:
            raise ValueError(
                "--use-card --target-local-backend vllm onlysupports --target-temp 0.0"
            )
        if getattr(args, 'card_application_mode', 'triggered') != "global":
            raise ValueError(
                "--use-card --target-local-backend vllm when first onlysupports "
                "--card-application-mode global"
            )
        if getattr(args, 'card_global_logit_formula', 'contrastive') != "contrastive":
            raise ValueError(
                "--use-card --target-local-backend vllm when first onlysupports "
                "--card-global-logit-formula contrastive"
            )
        if getattr(args, 'card_use_fixed_strength', 'false') == "true":
            raise ValueError(
                "Global CARD vLLM currently only supports dynamic strength, use "
                "--card-use-fixed-strength false"
            )
        if getattr(args, 'card_modulated_prob', 'false') == "true":
            raise ValueError(
                "Global CARD vLLM currently does not supportProbability modulation, "
                "use --card-modulated-prob false"
            )
        if getattr(args, 'card_use_top_k_constraint', 'false') == "true":
            raise ValueError(
                "--card-use-top-k-constraint yesold Triggered CARD pathmainpointsbranch "
                "top-k constraint, does not apply to Global CARD vLLM; if needed vLLM main/auxiliary branches "
                "top-k union support, use "
                "--card-global-vllm-support-mode main_aux_topk_union"
            )
        if (
            getattr(args, 'card_global_vllm_support_mode', 'full_vocab')
            == "main_aux_topk_union"
            and int(getattr(args, 'card_global_vllm_support_top_k', 10)) <= 0
        ):
            raise ValueError(
                "--card-global-vllm-support-top-k must be greater than 0 "
                " (when --card-global-vllm-support-mode=main_aux_topk_union when)"
            )
        if (
            getattr(args, 'card_use_attention_mask', 'false') == "true"
            or getattr(args, 'card_aux_prompt_type', 'uniform') == "mask"
        ):
            raise ValueError(
                "Global CARD vLLM currently uses an emptydocumentauxiliary branch, "
                "does not support mask/attention-mask auxiliary prompt"
            )
        if (
            getattr(args, 'card_aux_prompt_type_specified', False)
            and getattr(args, 'card_aux_prompt_type', 'uniform') != "delete"
        ):
            raise ValueError(
                "Global CARD vLLM auxiliary branchmustyesemptydocument, "
                "use --card-aux-prompt-type delete"
            )
        args.card_aux_prompt_type = "delete"

    if not (
        0.0 < float(getattr(args, 'target_vllm_gpu_memory_utilization', 0.6)) < 1.0
    ):
        raise ValueError(
            "--target-vllm-gpu-memory-utilization must be in (0, 1) range"
        )

    for arg_name in (
        "target_vllm_max_model_len",
        "target_vllm_max_num_seqs",
        "target_vllm_max_num_batched_tokens",
        "async_parse_max_pending_batches",
    ):
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None and int(arg_value) <= 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be greater than 0")

    # Method-specific runtime validation is active only when the method is enabled.
    if use_ck:
        if hasattr(args, 'ck_alpha') and not (0.0 <= args.ck_alpha <= 1.0):
            raise ValueError("--ck-alpha must be in [0, 1] range")
        if hasattr(args, 'ck_select_top') and args.ck_select_top <= 0:
            raise ValueError("--ck-select-top must be greater than 0")
        if (
            hasattr(args, 'ck_relative_top')
            and not (0.0 < args.ck_relative_top <= 1.0)
        ):
            raise ValueError("--ck-relative-top must be in (0, 1] range")

    if use_card:
        
        if hasattr(args, 'card_use_fixed_strength'):
            args.card_use_fixed_strength = (args.card_use_fixed_strength == "true")
        if hasattr(args, 'card_dynamic_alpha_recompute'):
            args.card_dynamic_alpha_recompute = (
                args.card_dynamic_alpha_recompute == "true"
            )
        if hasattr(args, 'card_modulated_prob'):
            args.card_modulated_prob = (args.card_modulated_prob == "true")
        if hasattr(args, 'card_use_attention_mask'):
            args.card_use_attention_mask = (args.card_use_attention_mask == "true")
        if hasattr(args, 'card_batch_inference'):
            args.card_batch_inference = (args.card_batch_inference == "true")
        if hasattr(args, 'card_aux_prompt_type'):
            args.card_aux_prompt_type = resolve_card_aux_prompt_type(args)
            args.card_use_attention_mask = (args.card_aux_prompt_type == "mask")
        if hasattr(args, 'card_application_mode'):
            args.card_application_mode = resolve_card_application_mode(args)
        if hasattr(args, 'card_global_logit_formula'):
            args.card_global_logit_formula = resolve_card_global_logit_formula(args)
        if hasattr(args, 'card_global_vllm_support_mode'):
            args.card_global_vllm_support_mode = (
                resolve_card_global_vllm_support_mode(args)
            )
        if hasattr(args, 'save_global_card_token_trace'):
            if isinstance(args.save_global_card_token_trace, bool):
                pass
            else:
                args.save_global_card_token_trace = (
                    str(args.save_global_card_token_trace).lower() == "true"
                )
        if hasattr(args, 'card_global_ck_alpha'):
            args.card_global_ck_alpha = float(args.card_global_ck_alpha)
        if hasattr(args, 'card_global_zxy_alpha'):
            args.card_global_zxy_alpha = float(args.card_global_zxy_alpha)
        if hasattr(args, 'card_global_main_bias_coeff'):
            args.card_global_main_bias_coeff = float(
                args.card_global_main_bias_coeff
            )
        if hasattr(args, 'card_global_direction_sign'):
            args.card_global_direction_sign = int(args.card_global_direction_sign)
        if hasattr(args, 'card_use_top_k_constraint'):
            args.card_use_top_k_constraint = (
                args.card_use_top_k_constraint == "true"
            )
        if hasattr(args, 'card_filter_opposite_start_tokens'):
            args.card_filter_opposite_start_tokens = (
                args.card_filter_opposite_start_tokens == "true"
            )
        if hasattr(args, 'card_filter_system_prompt_example_tokens'):
            args.card_filter_system_prompt_example_tokens = (
                args.card_filter_system_prompt_example_tokens == "true"
            )
        if (
            getattr(args, 'card_modulated_prob', False)
            and not (0.0 <= args.card_prob_weight_beta <= 1.0)
        ):
            raise ValueError(
                "--card-prob-weight-beta must be in [0, 1] range"
                " (when --card-modulated-prob=true when)"
            )
        if (
            hasattr(args, 'card_global_ck_alpha')
            and not (0.0 <= args.card_global_ck_alpha <= 1.0)
        ):
            raise ValueError("--card-global-ck-alpha must be in [0, 1] range")
        if (
            hasattr(args, 'card_global_zxy_alpha')
            and not np.isfinite(args.card_global_zxy_alpha)
        ):
            raise ValueError("--card-global-zxy-alpha mustyesfinite float")
        if hasattr(args, 'card_global_main_bias_coeff') and not (
            np.isfinite(args.card_global_main_bias_coeff)
            and -1.0 <= args.card_global_main_bias_coeff <= 1.0
        ):
            raise ValueError(
                "--card-global-main-bias-coeff must be [-1, 1] "
                "finite float in range"
            )
        if (
            hasattr(args, 'card_global_direction_sign')
            and args.card_global_direction_sign not in {1, -1}
        ):
            raise ValueError("--card-global-direction-sign must be 1 or -1")
        if (
            getattr(args, 'card_global_vllm_support_mode', 'full_vocab')
            == "main_aux_topk_union"
            and getattr(args, 'card_global_vllm_support_top_k', 10) <= 0
        ):
            raise ValueError(
                "--card-global-vllm-support-top-k must be greater than 0 "
                " (when --card-global-vllm-support-mode=main_aux_topk_union when)"
            )
        if (
            getattr(args, 'card_global_vllm_support_mode', 'full_vocab')
            != "full_vocab"
            and not (
                getattr(args, 'target_local_backend', 'vllm') == "vllm"
                and getattr(args, 'card_application_mode', 'triggered') == "global"
            )
        ):
            raise ValueError(
                "--card-global-vllm-support-mode non- full_vocab currently only in "
                "--use-card --card-application-mode global --target-local-backend vllm "
                "path"
            )
        if (
            getattr(args, 'card_global_main_bias_coeff', 0.0) != 0.0
            and not (
                getattr(args, 'target_local_backend', 'vllm') == "vllm"
                and getattr(args, 'card_application_mode', 'triggered') == "global"
            )
        ):
            raise ValueError(
                "--card-global-main-bias-coeff currently only in "
                "--use-card --card-application-mode global --target-local-backend vllm "
                "path"
            )
        if (
            getattr(args, 'card_global_direction_sign', 1) != 1
            and not (
                getattr(args, 'target_local_backend', 'vllm') == "vllm"
                and getattr(args, 'card_application_mode', 'triggered') == "global"
            )
        ):
            raise ValueError(
                "--card-global-direction-sign currently only in "
                "--use-card --card-application-mode global --target-local-backend vllm "
                "path"
            )
        if getattr(args, 'card_global_logit_formula', 'contrastive') in {
            'ck',
            'zxy',
        }:
            if getattr(args, 'card_application_mode', 'triggered') != 'global':
                raise ValueError("ck/zxy formula only supports --card-application-mode global")
            if getattr(args, 'card_modulated_prob', False):
                raise ValueError(
                    "ck/zxy formula does not yet support --card-modulated-prob=true, set to false"
                )
        if (
            hasattr(args, 'card_trigger_prob_sum_threshold')
            and not (0.0 < args.card_trigger_prob_sum_threshold <= 1.0)
        ):
            raise ValueError("--card-trigger-prob-sum-threshold must be in (0, 1] range")
        if (
            getattr(args, 'card_use_top_k_constraint', False)
            and getattr(args, 'card_top_k', 0) <= 0
        ):
            raise ValueError(
                "--card-top-k must be greater than 0 (when --card-use-top-k-constraint=true when)"
            )
        if (
            getattr(args, 'card_max_trigger_count', None) is not None
            and args.card_max_trigger_count < 0
        ):
            raise ValueError("--card-max-trigger-count must be a non-negative integer or omitted")

    run_recommendation_bias_experiment(args)
 

#!/usr/bin/env python3

import argparse
import json
import random
import time
import typing as t
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import is_remote_model

import recommendation_bias_experiment as exp


REPO_ROOT = Path(__file__).resolve().parent


def display_path(path: t.Union[str, Path]) -> str:
    """Return a repo-relative or redacted path for logs/errors."""
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return f"<external>/{path_obj.name}"


def read_jsonl_records(path: Path) -> t.List[t.Dict[str, t.Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parsing failed: {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl_records(path: Path, records: t.Iterable[t.Dict[str, t.Any]]) -> None:
    exp.file_utils.ensure_created_directory(str(path.parent))
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(exp.json_safe_value(record), ensure_ascii=False) + "\n")


def load_run_args(run_config_path: Path) -> argparse.Namespace:
    metadata = exp.load_run_config_metadata(str(run_config_path))
    raw_args = metadata.get("args")
    if not isinstance(raw_args, dict):
        raise ValueError(f"run_config.json is missing the args field: {run_config_path}")

    args = argparse.Namespace(**raw_args)
    args.runtime_script_name = Path(__file__).name
    if getattr(args, "target_model", None) is None:
        args.target_model = args.model
    if not hasattr(args, "target_temp_specified"):
        args.target_temp_specified = args.target_temp is not None
    if args.target_temp is None and not getattr(args, "enable_thinking", False):
        args.target_temp = 0.0
    if not hasattr(args, "test_categories"):
        args.test_categories = exp.parse_requested_test_categories(getattr(args, "test", None))
    else:
        args.test_categories = exp.parse_requested_test_categories(getattr(args, "test", None))
    if not hasattr(args, "with_ranking"):
        args.with_ranking = not getattr(args, "no_ranking", False)
    if not hasattr(args, "api_max_retries"):
        args.api_max_retries = 5
    if not hasattr(args, "api_concurrency"):
        args.api_concurrency = 1
    if not hasattr(args, "api_submit_stagger"):
        args.api_submit_stagger = 0.2
    if not hasattr(args, "api_retry_initial_delay"):
        args.api_retry_initial_delay = 10.0
    if not hasattr(args, "api_retry_backoff"):
        args.api_retry_backoff = 2.0
    if not hasattr(args, "api_retry_max_delay"):
        args.api_retry_max_delay = 120.0
    args.local_parsing_executor = None
    args.is_local_model = not is_remote_model(args.target_model)
    return args


def apply_retry_overrides(
    run_args: argparse.Namespace,
    cli_args: argparse.Namespace,
) -> None:
    for attr in (
        "api_max_retries",
        "api_concurrency",
        "api_submit_stagger",
        "api_retry_initial_delay",
        "api_retry_backoff",
        "api_retry_max_delay",
    ):
        value = getattr(cli_args, attr)
        if value is not None:
            setattr(run_args, attr, value)


def get_results_path(args: argparse.Namespace) -> Path:
    if exp.is_single_test_category(args):
        category_safe = exp.sanitize_category_name(exp.get_single_test_category(args))
        return Path(exp.out_path(args, f"results_{category_safe}.pkl"))
    return Path(exp.out_path(args, "results.pkl"))


def next_retry_round(summary_path: Path) -> int:
    if not summary_path.exists():
        return 1
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return 1
    previous_round = payload.get("retry_round", payload.get("last_retry_round", 0))
    try:
        return int(previous_round) + 1
    except Exception:
        return 1


def deserialize_messages(serialized: t.List[t.Dict[str, t.Any]]) -> t.List[exp.Message]:
    messages = []
    for item in serialized:
        role = item.get("role")
        content = item.get("content")
        messages.append(exp.Message(role=exp.Role(role), content=content or ""))
    return messages


def normalize_int_key_map(mapping: t.Any) -> t.Dict[int, int]:
    if not isinstance(mapping, dict):
        return {}
    return {int(key): int(value) for key, value in mapping.items()}


def normalize_prompt_order(prompt_order: t.Any) -> t.List[t.Tuple[int, int]]:
    if not isinstance(prompt_order, list):
        return []
    return [(int(pair[0]), int(pair[1])) for pair in prompt_order]


def build_run_record_from_failure(
    failure_record: t.Dict[str, t.Any],
) -> t.Dict[str, t.Any]:
    return {
        "block_idx": int(failure_record["block_idx"]),
        "run_idx": int(failure_record["run_idx"]),
        "experiment_idx": int(failure_record["experiment_idx"]),
        "L2_b": [
            [int(value) for value in row]
            for row in failure_record.get("L2_b", [])
        ],
        "doc_assignment": [
            int(value) for value in failure_record.get("doc_assignment", [])
        ],
        "prompt_order": normalize_prompt_order(failure_record.get("prompt_order")),
        "brand_to_doc_map": normalize_int_key_map(
            failure_record.get("brand_to_doc_map")
        ),
        "brand_to_position_map": normalize_int_key_map(
            failure_record.get("brand_to_position_map")
        ),
        "target_message": failure_record.get("target_message", ""),
        "messages": deserialize_messages(failure_record.get("messages", [])),
        "query": failure_record.get("query", ""),
        "documents": failure_record.get("documents", []),
        "product_models": failure_record.get("product_models", []),
        "product_brands": failure_record.get("product_brands", []),
    }


def build_products_from_record(record: t.Dict[str, t.Any]) -> t.List[exp.Product]:
    category = record["category"]
    brands = record.get("product_brands", [])
    models = record.get("product_models", [])
    return [
        exp.Product(category=category, brand=brand, model=model)
        for brand, model in zip(brands, models)
    ]


def experiment_already_present(
    category_result: t.Dict[str, t.Any],
    experiment_idx: int,
) -> bool:
    for record in category_result.get("experiment_records", []) or []:
        try:
            if int(record.get("experiment_idx")) == int(experiment_idx):
                return True
        except Exception:
            continue
    return False


def merge_recovered_record(
    results: t.Dict[str, t.Any],
    recovered_record: t.Dict[str, t.Any],
    args: argparse.Namespace,
) -> bool:
    category = recovered_record["category"]
    if category not in results:
        raise KeyError(f"Category {category} was not found in the results")

    category_result = results[category]
    experiment_idx = int(recovered_record["experiment_idx"])
    if experiment_already_present(category_result, experiment_idx):
        return False

    response_text = recovered_record.get("response_text")
    if not response_text:
        raise ValueError(f"Recovered record is missing response_text: experiment_idx={experiment_idx}")

    run_record = build_run_record_from_failure(recovered_record)
    prompt_order = run_record["prompt_order"]
    score_records: t.List[t.Dict[str, t.Any]] = []

    if not getattr(args, "output_only", False):
        ordered_products = build_products_from_record(recovered_record)
        product_scores, _log_info = exp.get_scores_for_products_with_logs(
            response_text,
            ordered_products,
        )
        category_scores = category_result.setdefault("scores", {})
        brands = category_result.get("brands", []) or []

        for physical_pos, product in enumerate(ordered_products):
            score = int(product_scores[product])
            brand_idx, doc_idx = prompt_order[physical_pos]
            key = (int(brand_idx), int(doc_idx), int(physical_pos))
            category_scores.setdefault(key, []).append(score)
            brand_info = brands[int(brand_idx)] if int(brand_idx) < len(brands) else {}
            score_records.append(
                {
                    "brand_index": int(brand_idx),
                    "brand": brand_info.get("brand", product.brand),
                    "model": product.model,
                    "doc_index": int(doc_idx),
                    "context_position": int(physical_pos),
                    "score": score,
                    "is_fictional": bool(brand_info.get("is_fictional", False)),
                    "knowledge_strength": float(
                        brand_info.get("knowledge_strength", 0.0)
                    ),
                }
            )

        score_records.sort(key=lambda item: item["score"], reverse=True)

    experiment_record = {
        "block_idx": run_record["block_idx"],
        "run_idx": run_record["run_idx"],
        "experiment_idx": run_record["experiment_idx"],
        "L2_b": [row[:] for row in run_record["L2_b"]],
        "doc_assignment": run_record["doc_assignment"].copy(),
        "prompt_order": [tuple(pair) for pair in prompt_order],
        "scores": score_records,
    }
    thinking_content = recovered_record.get("thinking_content")
    if thinking_content:
        experiment_record["thinking_content"] = thinking_content

    category_result.setdefault("experiment_records", []).append(experiment_record)
    category_result["experiment_records"].sort(
        key=lambda item: int(item.get("experiment_idx", 0))
    )
    category_result.setdefault("responses", []).append([response_text])
    return True


def refresh_api_retry_stats(
    results: t.Dict[str, t.Any],
    remaining_failures: t.List[t.Dict[str, t.Any]],
    attempts_by_category: t.Dict[str, int],
    merged_by_category: t.Dict[str, int],
    retry_round: int,
    failed_prompts_path: Path,
    recovered_prompts_path: Path,
) -> None:
    remaining_by_category = Counter(
        record.get("category") for record in remaining_failures
    )
    categories = set(results.keys())
    categories.update(attempts_by_category.keys())
    categories.update(remaining_by_category.keys())

    for category in categories:
        category_result = results.get(category)
        if not isinstance(category_result, dict):
            continue
        stats = category_result.setdefault("api_retry_stats", {})
        old_recovered = int(stats.get("recovered_prompt_count", 0))
        old_attempts = int(stats.get("total_attempt_count", 0))
        final_failed_count = int(remaining_by_category.get(category, 0))

        stats.update(
            {
                "success_response_count": len(
                    category_result.get("responses", []) or []
                ),
                "recovered_prompt_count": old_recovered
                + int(merged_by_category.get(category, 0)),
                "final_failed_prompt_count": final_failed_count,
                "total_attempt_count": old_attempts
                + int(attempts_by_category.get(category, 0)),
                "failed_prompts_path": str(failed_prompts_path),
                "recovered_prompts_path": str(recovered_prompts_path),
                "incomplete": final_failed_count > 0,
                "last_offline_retry_round": retry_round,
            }
        )


def load_retry_target(args: argparse.Namespace) -> t.Callable:
    if getattr(args, "use_ck", False) or getattr(args, "use_card", False):
        raise ValueError("The offline API retry script only supports normal API / prompt-only baseline paths")
    if args.is_local_model:
        raise ValueError("The offline API retry script only handles remote API models")

    if getattr(args, "enable_thinking", False):
        thinking_temp = (
            args.target_temp if getattr(args, "target_temp_specified", False) else None
        )
        return exp.load_model(
            args.target_model,
            temperature=thinking_temp,
            top_p=None,
            max_tokens=args.target_max_tokens,
            gpu_ids=args.target_gpu_ids,
            enable_thinking=True,
            extract_thinking=True,
            return_token_counts=True,
            local_inference_backend=args.target_local_backend,
            vllm_gpu_memory_utilization=args.target_vllm_gpu_memory_utilization,
            vllm_max_model_len=args.target_vllm_max_model_len,
            vllm_max_num_seqs=args.target_vllm_max_num_seqs,
            vllm_max_num_batched_tokens=args.target_vllm_max_num_batched_tokens,
        )

    return exp.load_model(
        args.target_model,
        args.target_temp,
        args.target_top_p,
        args.target_max_tokens,
        gpu_ids=args.target_gpu_ids,
        return_token_counts=True,
        local_inference_backend=args.target_local_backend,
        vllm_gpu_memory_utilization=args.target_vllm_gpu_memory_utilization,
        vllm_max_model_len=args.target_vllm_max_model_len,
        vllm_max_num_seqs=args.target_vllm_max_num_seqs,
        vllm_max_num_batched_tokens=args.target_vllm_max_num_batched_tokens,
    )


def write_retry_summary(
    summary_path: Path,
    retry_round: int,
    input_failed_count: int,
    recovered_count: int,
    merged_count: int,
    remaining_failed_count: int,
    results_path: Path,
    failed_prompts_path: Path,
    recovered_prompts_path: Path,
    attempts_by_category: t.Dict[str, int],
    merged_by_category: t.Dict[str, int],
) -> None:
    payload = {
        "retry_round": retry_round,
        "input_failed_prompt_count": input_failed_count,
        "recovered_prompt_count": recovered_count,
        "merged_prompt_count": merged_count,
        "remaining_failed_prompt_count": remaining_failed_count,
        "results_path": str(results_path),
        "failed_prompts_path": str(failed_prompts_path),
        "recovered_prompts_path": str(recovered_prompts_path),
        "attempts_by_category": dict(attempts_by_category),
        "merged_by_category": dict(merged_by_category),
    }
    exp.write_json_file(str(summary_path), payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline retry of API-failed prompts saved by recommendation_bias_experiment.py"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Main experiment output directory containing run_config.json and api_retry/",
    )
    parser.add_argument(
        "--failed-prompts-path",
        default=None,
        help="Failed queue to retry; defaults to <run-dir>/api_retry/failed_prompts.jsonl",
    )
    parser.add_argument("--api-max-retries", type=int, default=None)
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=None,
        help="API concurrency for offline retry (default: 1, keep serial)",
    )
    parser.add_argument(
        "--api-submit-stagger",
        type=float,
        default=None,
        help="Maximum random stagger before each prompt when submitting concurrently (default: 0.2s)",
    )
    parser.add_argument("--api-retry-initial-delay", type=float, default=None)
    parser.add_argument("--api-retry-backoff", type=float, default=None)
    parser.add_argument("--api-retry-max-delay", type=float, default=None)
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only merge results; do not regenerate summary_statistics.csv / plots / run_summary.txt",
    )
    return parser.parse_args()


def main() -> int:
    cli_args = parse_args()
    run_dir = Path(cli_args.run_dir).expanduser()
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"Could not find run_config.json: {display_path(run_config_path)}"
        )

    args = load_run_args(run_config_path)
    apply_retry_overrides(args, cli_args)

    derived_run_dir = Path(exp.out_path(args))
    if derived_run_dir.resolve() != run_dir.resolve():
        raise ValueError(
            "The output directory derived from run_config.json does not match --run-dir:\n"
            f"  run_config: {display_path(derived_run_dir)}\n"
            f"  --run-dir:   {display_path(run_dir)}\n"
            "Stopping to avoid overwriting the wrong directory."
        )

    results_path = get_results_path(args)
    if not results_path.exists():
        raise FileNotFoundError(f"Could not find results pkl: {display_path(results_path)}")

    api_retry_dir = run_dir / "api_retry"
    failed_prompts_path = (
        Path(cli_args.failed_prompts_path).expanduser()
        if cli_args.failed_prompts_path
        else api_retry_dir / "failed_prompts.jsonl"
    )
    recovered_prompts_path = api_retry_dir / "recovered_prompts.jsonl"
    retry_summary_path = api_retry_dir / "api_retry_summary.json"
    retry_round = next_retry_round(retry_summary_path)

    logger = exp.setup_logging(
        args,
        script_name_override=Path(__file__).stem,
    )
    logger.info("[API offline retry] Start")
    logger.info(f"  run_dir: {display_path(run_dir)}")
    logger.info(f"  results_path: {display_path(results_path)}")
    logger.info(f"  failed_prompts_path: {display_path(failed_prompts_path)}")
    logger.info(f"  recovered_prompts_path: {display_path(recovered_prompts_path)}")
    logger.info(f"  retry_round: {retry_round}")
    logger.info(
        f"  api_concurrency: {int(getattr(cli_args, 'api_concurrency', 1) or 1)}"
    )
    logger.info(
        f"  api_submit_stagger: "
        f"{float(getattr(cli_args, 'api_submit_stagger', 0.2) or 0.0):g}"
    )

    failure_records = read_jsonl_records(failed_prompts_path)
    if not failure_records:
        exp.print_nohup_api_status("retry", "No failed prompts to retry; exiting")
        logger.info("[API offline retry] No failed prompts to retry")
        return 0

    exp.print_nohup_api_status(
        "retry",
        f"Starting round {retry_round} with {len(failure_records)} prompts to retry",
    )
    results = exp.file_utils.read_pickle(str(results_path))
    target = load_retry_target(args)

    recovered_records: t.List[t.Dict[str, t.Any]] = []
    remaining_failures: t.List[t.Dict[str, t.Any]] = []
    attempts_by_category: t.Dict[str, int] = defaultdict(int)
    merged_by_category: t.Dict[str, int] = defaultdict(int)
    requested_api_concurrency = max(
        1, int(getattr(cli_args, "api_concurrency", 1) or 1)
    )
    api_submit_stagger = max(
        0.0, float(getattr(cli_args, "api_submit_stagger", 0.2) or 0.0)
    )

    retry_results_by_index: t.List[t.Optional[t.Dict[str, t.Any]]] = [
        None
    ] * len(failure_records)

    if requested_api_concurrency <= 1:
        for index, failure_record in enumerate(failure_records, 1):
            category = failure_record.get("category", "")
            batch_label = f"offline retry round {retry_round} {index}/{len(failure_records)}"
            run_record = build_run_record_from_failure(failure_record)
            retry_results_by_index[index - 1] = exp.retry_api_single_prompt(
                target=target,
                run_record=run_record,
                category=category,
                batch_label=batch_label,
                args=args,
                retry_origin="offline_retry",
                retry_round=retry_round,
            )
    else:
        with ThreadPoolExecutor(max_workers=requested_api_concurrency) as executor:
            future_to_index = {}
            for index, failure_record in enumerate(failure_records, 1):
                if index > 1 and api_submit_stagger > 0:
                    time.sleep(random.uniform(0.0, api_submit_stagger))
                category = failure_record.get("category", "")
                batch_label = f"offline retry round {retry_round} {index}/{len(failure_records)}"
                run_record = build_run_record_from_failure(failure_record)
                future = executor.submit(
                    exp.retry_api_single_prompt,
                    target,
                    run_record,
                    category,
                    batch_label,
                    args,
                    "offline_retry",
                    retry_round,
                )
                future_to_index[future] = index - 1
            for future in as_completed(future_to_index):
                retry_results_by_index[future_to_index[future]] = future.result()

    for index, failure_record in enumerate(failure_records, 1):
        retry_result = retry_results_by_index[index - 1]
        if retry_result is None:
            raise RuntimeError(f"Offline retry result missing: {index}/{len(failure_records)}")

        category = failure_record.get("category", "")
        batch_label = f"offline retry round {retry_round} {index}/{len(failure_records)}"
        run_record = build_run_record_from_failure(failure_record)
        attempts_by_category[category] += int(retry_result["attempt_count"])

        if retry_result["status"] == "success":
            recovered_record = retry_result.get("recovered_record")
            if recovered_record is None:
                recovered_record = exp.build_api_retry_record(
                    category=category,
                    batch_label=batch_label,
                    run_record=run_record,
                    status="recovered",
                    attempt_count=int(retry_result["attempt_count"]),
                    retry_origin="offline_retry",
                    retry_round=retry_round,
                    response_details=retry_result["response_details"],
                    target_model=getattr(args, "target_model", None),
                )
            exp.append_jsonl_record(str(recovered_prompts_path), recovered_record)
            recovered_records.append(recovered_record)
            merged = merge_recovered_record(results, recovered_record, args)
            if merged:
                merged_by_category[category] += 1
            continue

        failure_record_new = retry_result.get("failure_record")
        if failure_record_new is None:
            failure_record_new = failure_record
        failure_record_new["previous_failure_attempt_count"] = failure_record.get(
            "attempt_count"
        )
        failure_record_new["previous_retry_origin"] = failure_record.get(
            "retry_origin"
        )
        remaining_failures.append(failure_record_new)

    write_jsonl_records(failed_prompts_path, remaining_failures)
    refresh_api_retry_stats(
        results=results,
        remaining_failures=remaining_failures,
        attempts_by_category=attempts_by_category,
        merged_by_category=merged_by_category,
        retry_round=retry_round,
        failed_prompts_path=failed_prompts_path,
        recovered_prompts_path=recovered_prompts_path,
    )
    exp.file_utils.write_pickle(
        str(results_path),
        exp.prepare_results_for_pickle(results),
    )

    recovered_count = len(recovered_records)
    merged_count = sum(merged_by_category.values())
    remaining_count = len(remaining_failures)
    write_retry_summary(
        summary_path=retry_summary_path,
        retry_round=retry_round,
        input_failed_count=len(failure_records),
        recovered_count=recovered_count,
        merged_count=merged_count,
        remaining_failed_count=remaining_count,
        results_path=results_path,
        failed_prompts_path=failed_prompts_path,
        recovered_prompts_path=recovered_prompts_path,
        attempts_by_category=attempts_by_category,
        merged_by_category=merged_by_category,
    )

    exp.print_nohup_api_status(
        "retry",
        f"Round {retry_round} complete: recovered {recovered_count}, "
        f"merged {merged_count}, remaining failed {remaining_count}",
    )
    logger.info(
        "[API offline retry] Complete: "
        f"recovered {recovered_count}, merged {merged_count}, "
        f"remaining failed {remaining_count}"
    )

    if not cli_args.skip_analysis:
        if getattr(args, "output_only", False):
            logger.info("[Output mode] --output-only is enabled; skipping metric analysis")
        else:
            f_stat_results = exp.analyze_and_plot(args)
            brand_type_stats = exp.compute_brand_type_main_effect_f_brand_level(
                results
            )
            summary_path = exp.write_recommendation_bias_run_summary(
                results=results,
                f_stat_results=f_stat_results,
                brand_type_stats=brand_type_stats,
                args=args,
                results_path=str(results_path),
            )
            logger.info(f"Lightweight run summary file: {summary_path}")

    if remaining_count > 0:
        exp.print_nohup_api_status(
            "retry",
            f"{remaining_count} prompts still failed; continue with: {failed_prompts_path}",
        )
    else:
        exp.print_nohup_api_status("retry", "All failed prompts have been recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

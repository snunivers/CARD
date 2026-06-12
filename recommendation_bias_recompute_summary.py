#!/usr/bin/env python3

import argparse
import ast
from pathlib import Path
import typing as t

import numpy as np
from scipy import stats

from models import is_remote_model

import recommendation_bias_analysis as analysis
import recommendation_bias_experiment as exp


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
    args.test_categories = exp.parse_requested_test_categories(
        getattr(args, "test", None)
    )
    if not hasattr(args, "with_ranking"):
        args.with_ranking = not getattr(args, "no_ranking", False)
    if not hasattr(args, "is_local_model"):
        args.is_local_model = not is_remote_model(str(args.target_model))
    return args


def get_results_path(args: argparse.Namespace) -> Path:
    if exp.is_single_test_category(args):
        category_safe = exp.sanitize_category_name(exp.get_single_test_category(args))
        return Path(exp.out_path(args, f"results_{category_safe}.pkl"))
    return Path(exp.out_path(args, "results.pkl"))


def compute_fstat_results(
    results: t.Dict[str, t.Any],
    args: argparse.Namespace,
) -> t.Dict[str, t.Dict[str, t.Any]]:
    category_f_stats: t.Dict[str, t.Dict[str, t.Any]] = {}
    skip_knowledge_strength = analysis.should_skip_knowledge_strength_analysis(args)

    for category, category_results in results.items():
        brands = [analysis.BrandInfo(**b) for b in category_results["brands"]]
        category_scores = category_results["scores"]
        category_n = len(brands)
        score_ceiling = float(category_n) if category_n > 0 else np.nan

        brand_scores = [[] for _ in range(category_n)]
        doc_scores = [[] for _ in range(category_n)]
        context_pos_scores = [[] for _ in range(category_n)]
        fictional_scores = {True: [], False: []}
        knowledge_strength_data = []  # (strength, score) pairs

        for (brand_index, doc_index, context_pos), scores in category_scores.items():
            brand_scores[brand_index].extend(scores)
            doc_scores[doc_index].extend(scores)
            context_pos_scores[context_pos].extend(scores)

            brand = brands[brand_index]
            fictional_scores[brand.is_fictional].extend(scores)

            if not skip_knowledge_strength:
                for score in scores:
                    knowledge_strength_data.append((brand.knowledge_strength, score))

        brand_f = analysis.get_f_statistic(brand_scores)
        doc_f = analysis.get_f_statistic(doc_scores)
        context_pos_f = analysis.get_f_statistic(context_pos_scores)

        if fictional_scores[True] and fictional_scores[False]:
            t_stat, t_pvalue = stats.ttest_ind(
                fictional_scores[False],
                fictional_scores[True],
            )
        else:
            t_stat, t_pvalue = np.nan, np.nan

        if skip_knowledge_strength:
            correlation, corr_pvalue = np.nan, np.nan
        else:
            real_brand_data = [(ks, s) for ks, s in knowledge_strength_data if ks > 0]
            if real_brand_data:
                ks_values, score_values = zip(*real_brand_data)
                correlation, corr_pvalue = stats.pearsonr(ks_values, score_values)
            else:
                correlation, corr_pvalue = np.nan, np.nan

        real_mean = (
            float(np.mean(fictional_scores[False]))
            if fictional_scores[False]
            else np.nan
        )
        fictional_mean = (
            float(np.mean(fictional_scores[True]))
            if fictional_scores[True]
            else np.nan
        )
        real_mean_norm = (
            real_mean / score_ceiling
            if score_ceiling > 0 and not np.isnan(real_mean)
            else np.nan
        )
        fictional_mean_norm = (
            fictional_mean / score_ceiling
            if score_ceiling > 0 and not np.isnan(fictional_mean)
            else np.nan
        )

        category_f_stats[category] = {
            "score_ceiling_k": score_ceiling,
            "brand_f": brand_f,
            "doc_f": doc_f,
            "context_pos_f": context_pos_f,
            "fictional_t_stat": t_stat,
            "fictional_t_pvalue": t_pvalue,
            "fictional_mean_diff": (
                real_mean - fictional_mean
                if fictional_scores[True] and fictional_scores[False]
                else np.nan
            ),
            "fictional_mean_diff_norm": (
                real_mean_norm - fictional_mean_norm
                if not np.isnan(real_mean_norm) and not np.isnan(fictional_mean_norm)
                else np.nan
            ),
            "real_mean": real_mean,
            "fictional_mean": fictional_mean,
            "real_mean_norm": real_mean_norm,
            "fictional_mean_norm": fictional_mean_norm,
            "knowledge_correlation": correlation,
            "knowledge_corr_pvalue": corr_pvalue,
        }

    return category_f_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute summary metrics for an existing recommendation-bias run and write run_summary.txt"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Experiment output directory containing run_config.json and results.pkl",
    )
    return parser.parse_args()


def main() -> int:
    cli_args = parse_args()
    run_dir = Path(cli_args.run_dir).expanduser()
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Could not find run_config.json: {run_config_path}")

    args = load_run_args(run_config_path)
    results_path = get_results_path(args)
    if not results_path.exists():
        raise FileNotFoundError(f"Could not find results pkl: {results_path}")

    results = exp.file_utils.read_pickle(str(results_path))
    f_stat_results = compute_fstat_results(results, args)
    brand_type_stats = exp.compute_brand_type_main_effect_f_brand_level(results)
    summary_path = exp.write_recommendation_bias_run_summary(
        results=results,
        f_stat_results=f_stat_results,
        brand_type_stats=brand_type_stats,
        args=args,
        results_path=str(results_path),
    )

    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Analysis, plotting, and summary-statistics helpers for recommendation-bias runs.

The experiment entrypoint owns runtime-specific path, logging, and configuration
rules. This module keeps the post-run analysis code separate while receiving
those runtime dependencies explicitly from ``recommendation_bias_experiment.py``.
"""

import argparse
import ast
import json
import logging
import os
import typing as t
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


class _UnconfiguredBrandInfo:
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        raise RuntimeError("recommendation_bias_analysis dependencies have not been configured")


class _UnconfiguredFileUtils:
    def __getattr__(self, name: str) -> t.Any:
        raise RuntimeError("recommendation_bias_analysis dependencies have not been configured")


def _missing_dependency(*args: t.Any, **kwargs: t.Any) -> t.Any:
    raise RuntimeError("recommendation_bias_analysis dependencies have not been configured")


BrandInfo: t.Any = _UnconfiguredBrandInfo
file_utils: t.Any = _UnconfiguredFileUtils()
get_logger: t.Callable[[], logging.Logger] = _missing_dependency
is_single_test_category: t.Callable[[argparse.Namespace], bool] = _missing_dependency
get_single_test_category: t.Callable[[argparse.Namespace], t.Optional[str]] = _missing_dependency
get_requested_test_categories: t.Callable[[argparse.Namespace], t.List[str]] = _missing_dependency
sanitize_category_name: t.Callable[[str], str] = _missing_dependency
out_path: t.Callable[..., str] = _missing_dependency
plot_path: t.Callable[..., str] = _missing_dependency
SUMMARY_LOG_FLOAT_DIGIT_VARIANTS: t.Tuple[int, ...] = (4, 8)
SUMMARY_LOG_ROUNDING_NOTE = ""
format_summary_log_float: t.Callable[[t.Any, int], str] = _missing_dependency


@dataclass(frozen=True)
class AnalysisDependencies:
    brand_info_cls: t.Any
    file_utils: t.Any
    get_logger: t.Callable[[], logging.Logger]
    is_single_test_category: t.Callable[[argparse.Namespace], bool]
    get_single_test_category: t.Callable[[argparse.Namespace], t.Optional[str]]
    get_requested_test_categories: t.Callable[[argparse.Namespace], t.List[str]]
    sanitize_category_name: t.Callable[[str], str]
    out_path: t.Callable[..., str]
    plot_path: t.Callable[..., str]
    summary_log_float_digit_variants: t.Tuple[int, ...]
    summary_log_rounding_note: str
    format_summary_log_float: t.Callable[[t.Any, int], str]


def should_include_factor_level_f(args: argparse.Namespace) -> bool:
    """Whether to emit the legacy Brand/Document/Context-position F diagnostics."""
    return bool(getattr(args, "include_factor_level_f", False))


def configure_analysis_dependencies(deps: AnalysisDependencies) -> None:
    """Install entrypoint-owned helpers used by analysis functions."""
    global BrandInfo, file_utils, get_logger, is_single_test_category
    global get_single_test_category, get_requested_test_categories
    global sanitize_category_name, out_path, plot_path
    global SUMMARY_LOG_FLOAT_DIGIT_VARIANTS, SUMMARY_LOG_ROUNDING_NOTE
    global format_summary_log_float

    BrandInfo = deps.brand_info_cls
    file_utils = deps.file_utils
    get_logger = deps.get_logger
    is_single_test_category = deps.is_single_test_category
    get_single_test_category = deps.get_single_test_category
    get_requested_test_categories = deps.get_requested_test_categories
    sanitize_category_name = deps.sanitize_category_name
    out_path = deps.out_path
    plot_path = deps.plot_path
    SUMMARY_LOG_FLOAT_DIGIT_VARIANTS = deps.summary_log_float_digit_variants
    SUMMARY_LOG_ROUNDING_NOTE = deps.summary_log_rounding_note
    format_summary_log_float = deps.format_summary_log_float




def analyze_and_plot(args: argparse.Namespace) -> t.Optional[t.Dict]:
    logger = get_logger()
    has_logger = len(logger.handlers) > 0

    def emit(message: str):
        print(message)
        if has_logger:
            logger.info(message)

    
    if is_single_test_category(args):
        category_safe = sanitize_category_name(get_single_test_category(args))
        results_filename = f'results_{category_safe}.pkl'
    else:
        results_filename = 'results.pkl'
    
    results_file = out_path(args, results_filename)
    
    if not os.path.exists(results_file):
        emit(f"Result file does not exist: {results_file}")
        if is_single_test_category(args) and not getattr(args, 'single_category_run_tag', None):
            emit("Hint: newer single-category runs are saved under single_category_runs/<category>/<timestamp>/ by default; use --single-category-run-tag to select a run tag")
        elif len(get_requested_test_categories(args)) > 1 and not getattr(args, 'single_category_run_tag', None):
            emit("Hint: targeted multi-category runs are saved under multi_category_runs/<category1__category2...>/<timestamp>/ by default; use --single-category-run-tag to select a run tag")
        emit("Run the experiment first: python recommendation_bias_experiment.py --run-eval ...")
        return None
    
    results = file_utils.read_pickle(results_file)
    
    emit(f"\n{'='*60}")
    emit(f"Analysis results")
    emit(f"{'='*60}")
    
    
    file_utils.create_empty_directory(plot_path(args))
    file_utils.ensure_created_directory(plot_path(args, 'heatmaps'))
    if should_include_factor_level_f(args):
        file_utils.ensure_created_directory(plot_path(args, 'fstatistics'))
    file_utils.ensure_created_directory(plot_path(args, 'recommendation_bias'))
    
    
    generate_heatmaps(results, args)
    
    
    f_stat_results = generate_fstatistic_analysis(results, args)
    
    
    generate_recommendation_bias_analysis(results, args)
    
    
    save_summary_statistics(results, f_stat_results, args)
    
    emit(f"\n✅ Analysis complete. Plots saved to: {plot_path(args)}")
    return f_stat_results


def generate_heatmaps(results: t.Dict, args: argparse.Namespace):
    print("\nGenerating heatmaps...")
    
    fstat_label_root_lookup = {
        'brand': 'Brand',
        'doc': 'Document', 
        'contextpos': 'Context position'
    }
    
    for category, category_results in results.items():
        brands = [BrandInfo(**b) for b in category_results['brands']]
        category_scores = category_results['scores']
        category_n = len(brands)
        
        
        sum_score = np.zeros((category_n, category_n, category_n))
        score_count = np.zeros((category_n, category_n, category_n))
        
        for (brand_index, doc_index, context_pos), scores in category_scores.items():
            sum_score[brand_index, doc_index, context_pos] += np.sum(scores)
            score_count[brand_index, doc_index, context_pos] += len(scores)
        
        
        all_axes = ['brand', 'doc', 'contextpos']
        for ax1 in all_axes:
            for ax2 in all_axes:
                if ax1 == ax2:
                    continue
                
                plot_heatmap(
                    category=category,
                    brands=brands,
                    sum_score=sum_score,
                    score_count=score_count,
                    axes=[ax1, ax2],
                    sort_by=ax1,
                    args=args,
                    fstat_label_root_lookup=fstat_label_root_lookup
                )


def plot_heatmap(
    category: str,
    brands: t.List[BrandInfo],
    sum_score: np.ndarray,
    score_count: np.ndarray,
    axes: t.List[str],
    sort_by: str,
    args: argparse.Namespace,
    fstat_label_root_lookup: t.Dict[str, str]
):
    file_utils.ensure_created_directory(plot_path(args, 'heatmaps', category))
    
    
    axis_lookup = {'brand': 0, 'doc': 1, 'contextpos': 2}
    missing_axis = [ax for ax in ['brand', 'doc', 'contextpos'] if ax not in axes][0]
    missing_axis_index = axis_lookup[missing_axis]
    
    projected_sum_score = np.sum(sum_score, axis=missing_axis_index)
    projected_score_count = np.sum(score_count, axis=missing_axis_index)
    
    average_score = np.divide(
        projected_sum_score, projected_score_count,
        out=np.zeros_like(projected_sum_score), where=projected_score_count != 0
    )
    
    
    if axis_lookup[axes[0]] > axis_lookup[axes[1]]:
        average_score = np.swapaxes(average_score, 0, 1)
        projected_sum_score = np.swapaxes(projected_sum_score, 0, 1)
        projected_score_count = np.swapaxes(projected_score_count, 0, 1)
    
    nonsort_axis = 1 - axes.index(sort_by)
    
    
    average_score_nonsort_axis = np.divide(
        projected_sum_score.sum(axis=nonsort_axis),
        projected_score_count.sum(axis=nonsort_axis),
        out=np.zeros_like(projected_sum_score.sum(axis=nonsort_axis)),
        where=projected_score_count.sum(axis=nonsort_axis) != 0
    )
    
    sorted_indices = np.argsort(average_score_nonsort_axis)
    simultaneous_sort = 'brand' in axes and 'doc' in axes
    category_n = len(brands)
    
    
    brand_labels = [f"{b.brand[:15]}{'*' if b.is_fictional else ''}" for b in brands]
    ticklabels = {
        'brand': brand_labels,
        'doc': brand_labels,
        'contextpos': [str(i) for i in range(category_n)]
    }
    
    xticklabels, yticklabels = ticklabels[axes[1]], ticklabels[axes[0]]
    
    if axes.index(sort_by) == 0 or simultaneous_sort:
        average_score = average_score[sorted_indices, :]
        yticklabels = [yticklabels[i] for i in sorted_indices]
    if axes.index(sort_by) == 1 or simultaneous_sort:
        average_score = average_score[:, sorted_indices]
        xticklabels = [xticklabels[i] for i in sorted_indices]
    
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.flip(average_score, axis=0), cmap='plasma')
    
    ax.set_xticks(range(len(xticklabels)))
    ax.set_yticks(range(len(yticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_yticklabels(list(reversed(yticklabels)), fontsize=8)
    
    ax.set_xlabel(fstat_label_root_lookup[axes[1]])
    ax.set_ylabel(fstat_label_root_lookup[axes[0]])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Average ranking score', rotation=-90, va="bottom")
    
    filename = f'{axes[0]}_{axes[1]}_sorted_{sort_by}'
    fig_path = plot_path(args, 'heatmaps', category, f'{filename}.pdf')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def generate_fstatistic_analysis(results: t.Dict, args: argparse.Namespace) -> t.Dict:
    print("\nComputing F statistics...")
    
    category_f_stats = {}
    include_factor_level_f = should_include_factor_level_f(args)
    
    for category, category_results in results.items():
        brands = [BrandInfo(**b) for b in category_results['brands']]
        category_scores = category_results['scores']
        category_n = len(brands)
        score_ceiling = float(category_n) if category_n > 0 else np.nan
        
        
        brand_scores = [[] for _ in range(category_n)]
        doc_scores = [[] for _ in range(category_n)]
        context_pos_scores = [[] for _ in range(category_n)]
        nonparametric_scores = {True: [], False: []}
        
        for (brand_index, doc_index, context_pos), scores in category_scores.items():
            if include_factor_level_f:
                brand_scores[brand_index].extend(scores)
                doc_scores[doc_index].extend(scores)
                context_pos_scores[context_pos].extend(scores)
            
            brand = brands[brand_index]
            nonparametric_scores[brand.is_fictional].extend(scores)
        
        
        if nonparametric_scores[True] and nonparametric_scores[False]:
            t_stat, t_pvalue = stats.ttest_ind(
                nonparametric_scores[False], nonparametric_scores[True]
            )
        else:
            t_stat, t_pvalue = np.nan, np.nan

        parametric_mean = (
            float(np.mean(nonparametric_scores[False]))
            if nonparametric_scores[False] else np.nan
        )
        nonparametric_mean = (
            float(np.mean(nonparametric_scores[True]))
            if nonparametric_scores[True] else np.nan
        )
        parametric_mean_norm = (
            parametric_mean / score_ceiling
            if score_ceiling > 0 and not np.isnan(parametric_mean) else np.nan
        )
        nonparametric_mean_norm = (
            nonparametric_mean / score_ceiling
            if score_ceiling > 0 and not np.isnan(nonparametric_mean) else np.nan
        )
        
        category_f_stats[category] = {
            'score_ceiling_k': score_ceiling,
            'parametric_vs_nonparametric_t_stat': t_stat,
            'parametric_vs_nonparametric_pvalue': t_pvalue,
            'mean_score_diff': (
                parametric_mean - nonparametric_mean
                if nonparametric_scores[True] and nonparametric_scores[False] else np.nan
            ),
            'mean_score_diff_norm': (
                parametric_mean_norm - nonparametric_mean_norm
                if not np.isnan(parametric_mean_norm) and not np.isnan(nonparametric_mean_norm)
                else np.nan
            ),
            'parametric_mean_score': parametric_mean,
            'nonparametric_mean_score': nonparametric_mean,
            'parametric_mean_score_norm': parametric_mean_norm,
            'nonparametric_mean_score_norm': nonparametric_mean_norm,
        }

        if include_factor_level_f:
            category_f_stats[category].update({
                'brand_f': get_f_statistic(brand_scores),
                'doc_f': get_f_statistic(doc_scores),
                'context_pos_f': get_f_statistic(context_pos_scores),
            })
    
    if include_factor_level_f:
        plot_fstatistic_scatters(category_f_stats, args)
    
    return category_f_stats


def get_f_statistic(scores: t.List[t.List[float]]) -> float:
    categories = [str(i) for i in range(len(scores))]
    
    X, Y = [], []
    for category, sublist in zip(categories, scores):
        for score in sublist:
            X.append(category)
            Y.append(score)
    
    if not Y or len(set(X)) < 2:
        return np.nan
    
    data = {'X': X, 'Y': Y}
    try:
        model = ols('Y ~ X', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table['F']['X']
    except Exception:
        return np.nan


def compute_brand_type_main_effect_f_brand_level(results: t.Dict) -> t.Dict[str, t.Any]:
    parametric_category_brand_means: t.List[float] = []
    fictional_category_brand_means: t.List[float] = []
    parametric_category_brand_norm_means: t.List[float] = []
    fictional_category_brand_norm_means: t.List[float] = []

    for category_results in results.values():
        brands = [BrandInfo(**b) for b in category_results.get('brands', [])]
        category_scores = category_results.get('scores', {})
        score_ceiling = float(len(brands)) if brands else np.nan

        brand_scores: t.Dict[int, t.List[float]] = {i: [] for i in range(len(brands))}

        for key, scores in category_scores.items():
            parsed_key = key
            if isinstance(parsed_key, str):
                try:
                    parsed_key = ast.literal_eval(parsed_key)
                except (ValueError, SyntaxError):
                    continue

            if not isinstance(parsed_key, (tuple, list)) or len(parsed_key) != 3:
                continue

            brand_index = int(parsed_key[0])
            if brand_index not in brand_scores:
                continue

            brand_scores[brand_index].extend(scores)

        for brand_index, scores in brand_scores.items():
            if not scores:
                continue

            brand_mean = float(np.mean(scores))
            brand_mean_norm = (
                brand_mean / score_ceiling
                if score_ceiling > 0 else np.nan
            )
            if brands[brand_index].is_fictional:
                fictional_category_brand_means.append(brand_mean)
                fictional_category_brand_norm_means.append(brand_mean_norm)
            else:
                parametric_category_brand_means.append(brand_mean)
                parametric_category_brand_norm_means.append(brand_mean_norm)

    result = {
        'has_comparison': False,
        'parametric_n': len(parametric_category_brand_means),
        'fictional_n': len(fictional_category_brand_means),
        'parametric_mean': np.nan,
        'fictional_mean': np.nan,
        'mean_diff': np.nan,
        'parametric_mean_norm': np.nan,
        'fictional_mean_norm': np.nan,
        'mean_diff_norm': np.nan,
        't_statistic': np.nan,
        't_pvalue': np.nan,
        'f_statistic': np.nan,
        'f_pvalue': np.nan,
        'cohens_d': np.nan,
    }

    if not parametric_category_brand_means or not fictional_category_brand_means:
        return result

    param_mean = float(np.mean(parametric_category_brand_means))
    fict_mean = float(np.mean(fictional_category_brand_means))
    param_mean_norm = float(np.mean(parametric_category_brand_norm_means))
    fict_mean_norm = float(np.mean(fictional_category_brand_norm_means))
    param_std = float(np.std(parametric_category_brand_means, ddof=1)) if len(parametric_category_brand_means) > 1 else np.nan
    fict_std = float(np.std(fictional_category_brand_means, ddof=1)) if len(fictional_category_brand_means) > 1 else np.nan

    t_stat, t_pvalue = stats.ttest_ind(
        parametric_category_brand_means,
        fictional_category_brand_means,
    )

    X = ['parametric'] * len(parametric_category_brand_means) + ['fictional'] * len(fictional_category_brand_means)
    Y = parametric_category_brand_means + fictional_category_brand_means

    try:
        model = ols('Y ~ X', data={'X': X, 'Y': Y}).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        f_statistic = float(anova_table.loc['X', 'F'])
        f_pvalue = float(anova_table.loc['X', 'PR(>F)'])
    except Exception:
        f_statistic = np.nan
        f_pvalue = np.nan

    pooled_std = np.nan
    if len(parametric_category_brand_means) > 1 and len(fictional_category_brand_means) > 1:
        pooled_std = np.sqrt(
            (
                (len(parametric_category_brand_means) - 1) * (param_std ** 2)
                + (len(fictional_category_brand_means) - 1) * (fict_std ** 2)
            )
            / (len(parametric_category_brand_means) + len(fictional_category_brand_means) - 2)
        )

    cohens_d = (param_mean - fict_mean) / pooled_std if pooled_std and pooled_std > 0 else np.nan

    result.update({
        'has_comparison': True,
        'parametric_mean': param_mean,
        'fictional_mean': fict_mean,
        'mean_diff': param_mean - fict_mean,
        'parametric_mean_norm': param_mean_norm,
        'fictional_mean_norm': fict_mean_norm,
        'mean_diff_norm': param_mean_norm - fict_mean_norm,
        't_statistic': float(t_stat),
        't_pvalue': float(t_pvalue),
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'cohens_d': float(cohens_d) if not np.isnan(cohens_d) else np.nan,
    })

    return result


def log_brand_type_main_effect_f_at_end(results: t.Dict, logger: logging.Logger) -> None:
    stats_result = compute_brand_type_main_effect_f_brand_level(results)

    logger.info(f"")
    logger.info(f"{'='*80}")
    logger.info("[Parametric vs non-parametric (two-group) main-effect F] (category-brand level, log tail)")
    logger.info(f"{'='*80}")

    if not stats_result.get('has_comparison', False):
        logger.info(
            "  Insufficient data to compute the parametric vs non-parametric main-effect F "
            f"(parametric brands n={stats_result.get('parametric_n', 0)}, "
            f"non-parametric brands n={stats_result.get('fictional_n', 0)})"
        )
        return

    logger.info(
        "  Sample count (category-brand): "
        f"parametric brands n={stats_result['parametric_n']}, "
        f"non-parametric brands n={stats_result['fictional_n']}"
    )
    logger.info(f"  {SUMMARY_LOG_ROUNDING_NOTE}")
    for digits in SUMMARY_LOG_FLOAT_DIGIT_VARIANTS:
        logger.info(f"  [{digits}-decimal version]")
        logger.info(
            "    Mean: "
            f"parametric brands={format_summary_log_float(stats_result['parametric_mean'], digits)}, "
            f"non-parametric brands={format_summary_log_float(stats_result['fictional_mean'], digits)}, "
            f"difference={format_summary_log_float(stats_result['mean_diff'], digits)}"
        )
        logger.info(
            "    Normalized mean: "
            f"parametric brands={format_summary_log_float(stats_result['parametric_mean_norm'], digits)}, "
            f"non-parametric brands={format_summary_log_float(stats_result['fictional_mean_norm'], digits)}, "
            f"difference={format_summary_log_float(stats_result['mean_diff_norm'], digits)}"
        )
        logger.info(
            "    Main-effect F: "
            f"F={format_summary_log_float(stats_result['f_statistic'], digits)}, "
            f"p={format_summary_log_float(stats_result['f_pvalue'], digits)}"
        )
        logger.info(
            "    t-test: "
            f"t={format_summary_log_float(stats_result['t_statistic'], digits)}, "
            f"p={format_summary_log_float(stats_result['t_pvalue'], digits)}"
        )
        logger.info(
            "    Effect size Cohen's d: "
            f"{format_summary_log_float(stats_result['cohens_d'], digits)}"
        )


def plot_fstatistic_scatters(category_f_stats: t.Dict, args: argparse.Namespace):
    fstat_label_lookup = {
        'brand': 'Brand F-statistic',
        'doc': 'Document F-statistic',
        'contextpos': 'Context position F-statistic'
    }
    
    all_axes = ['brand', 'doc', 'contextpos']
    f_key_lookup = {'brand': 'brand_f', 'doc': 'doc_f', 'contextpos': 'context_pos_f'}
    
    for ax1 in all_axes:
        for ax2 in all_axes:
            if ax1 == ax2:
                continue
            
            fig, ax = plt.subplots(figsize=(4, 4))
            
            x_values = [category_f_stats[cat][f_key_lookup[ax1]] for cat in category_f_stats]
            y_values = [category_f_stats[cat][f_key_lookup[ax2]] for cat in category_f_stats]
            
            
            valid_pairs = [(x, y) for x, y in zip(x_values, y_values) 
                          if not np.isnan(x) and not np.isnan(y)]
            if valid_pairs:
                x_values, y_values = zip(*valid_pairs)
                ax.scatter(x_values, y_values, alpha=0.6)
                
                max_val = max(max(x_values), max(y_values))
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            ax.set_xlabel(fstat_label_lookup[ax1])
            ax.set_ylabel(fstat_label_lookup[ax2])
            ax.set_aspect('equal', adjustable='box')
            
            fig_path = plot_path(args, 'fstatistics', f'{ax1}_{ax2}.pdf')
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()


def results_have_parametric_and_nonparametric_brands(results: t.Dict) -> bool:
    for category_results in results.values():
        brands = [BrandInfo(**b) for b in category_results.get('brands', [])]
        has_parametric = any(not brand.is_fictional for brand in brands)
        has_nonparametric = any(brand.is_fictional for brand in brands)
        if has_parametric and has_nonparametric:
            return True
    return False


def generate_recommendation_bias_analysis(results: t.Dict, args: argparse.Namespace):
    print("\nGenerating recommendation-bias analysis...")
    
    if results_have_parametric_and_nonparametric_brands(results):
        plot_parametric_vs_nonparametric(results, args)
    

def plot_parametric_vs_nonparametric(results: t.Dict, args: argparse.Namespace):
    parametric_scores_all = []
    nonparametric_scores_all = []
    
    for category, category_results in results.items():
        brands = [BrandInfo(**b) for b in category_results['brands']]
        category_scores = category_results['scores']
        
        for (brand_index, doc_index, context_pos), scores in category_scores.items():
            brand = brands[brand_index]
            if brand.is_fictional:
                nonparametric_scores_all.extend(scores)
            else:
                parametric_scores_all.extend(scores)
    
    if not parametric_scores_all or not nonparametric_scores_all:
        return
    
    
    fig, ax = plt.subplots(figsize=(4, 5))
    
    data = [parametric_scores_all, nonparametric_scores_all]
    bp = ax.boxplot(
        data,
        labels=['Parametric', 'Non-parametric'],
        patch_artist=True,
    )
    
    colors_list = ['#2196F3', '#FF9800']
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Ranking Score')
    ax.set_title('Parametric vs Non-parametric Knowledge')
    
    
    t_stat, p_value = stats.ttest_ind(
        parametric_scores_all,
        nonparametric_scores_all,
    )
    parametric_mean = np.mean(parametric_scores_all)
    nonparametric_mean = np.mean(nonparametric_scores_all)
    
    stats_text = (
        f'Parametric mean: {parametric_mean:.2f}\n'
        f'Non-parametric mean: {nonparametric_mean:.2f}\n'
        f'p-value: {p_value:.4f}'
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig_path = plot_path(
        args,
        'recommendation_bias',
        'parametric_vs_nonparametric.pdf',
    )
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()




def save_summary_statistics(
    results: t.Dict, 
    f_stat_results: t.Dict,
    args: argparse.Namespace
):
    logger = get_logger()
    has_logger = len(logger.handlers) > 0

    def emit(message: str):
        print(message)
        if has_logger:
            logger.info(message)

    emit("\nSaving summary statistics...")
    has_factor_level_f = should_include_factor_level_f(args)
    
    
    summary_rows = []
    
    for category in results.keys():
        f_stats = f_stat_results.get(category, {})
        row = {
            'category': category,
            'score_ceiling_k': f_stats.get('score_ceiling_k', np.nan),
            'parametric_vs_nonparametric_t_stat': f_stats.get(
                'parametric_vs_nonparametric_t_stat',
                np.nan,
            ),
            'parametric_vs_nonparametric_pvalue': f_stats.get(
                'parametric_vs_nonparametric_pvalue',
                np.nan,
            ),
            'parametric_mean_score': f_stats.get('parametric_mean_score', np.nan),
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
        summary_rows.append(row)
    
    df = pd.DataFrame(summary_rows)
    
    
    csv_path = out_path(args, 'summary_statistics.csv')
    df.to_csv(csv_path, index=False)
    emit(f"  Summary statistics saved: {csv_path}")

    
    with open(out_path(args, 'f_stat_results.json'), 'w') as f:
        
        serializable_results = {}
        for cat, stats in f_stat_results.items():
            serializable_results[cat] = {
                k: float(v) if not np.isnan(v) else None
                for k, v in stats.items()
            }
        json.dump(serializable_results, f, indent=2)
    
    
    
    all_thinking_tokens = []
    all_response_tokens = []
    categories_with_thinking = 0
    
    for category, category_results in results.items():
        if 'thinking_stats' in category_results:
            categories_with_thinking += 1
            stats = category_results['thinking_stats']
            all_thinking_tokens.extend(stats['thinking_token_counts'])
            all_response_tokens.extend(stats['response_token_counts'])
    
    if categories_with_thinking > 0:
        emit(f"\n{'='*80}")
        emit("Thinking-mode statistics summary")
        emit(f"{'='*80}")
        emit(f"\nCategories with thinking statistics: {categories_with_thinking}/{len(results)}")
        emit(f"Total experiment count: {len(all_thinking_tokens)}")
        
        emit(f"\nThinking token statistics:")
        emit(f"  mean: ~{np.mean(all_thinking_tokens):.1f}")
        emit(f"  min: ~{np.min(all_thinking_tokens)}")
        emit(f"  max: ~{np.max(all_thinking_tokens)}")
        emit(f"  std: ~{np.std(all_thinking_tokens):.1f}")
        emit(f"  total: ~{sum(all_thinking_tokens)}")
        
        emit(f"\nResponse token statistics:")
        emit(f"  mean: ~{np.mean(all_response_tokens):.1f}")
        emit(f"  min: ~{np.min(all_response_tokens)}")
        emit(f"  max: ~{np.max(all_response_tokens)}")
        emit(f"  std: ~{np.std(all_response_tokens):.1f}")
        emit(f"  total: ~{sum(all_response_tokens)}")
        
        total_tokens = [t + r for t, r in zip(all_thinking_tokens, all_response_tokens)]
        emit(f"\nTotal token statistics (thinking + response):")
        emit(f"  mean: ~{np.mean(total_tokens):.1f}")
        emit(f"  total: ~{sum(total_tokens)}")
        
        
        thinking_summary = {
            'categories_with_thinking': categories_with_thinking,
            'total_experiments': len(all_thinking_tokens),
            'thinking_tokens': {
                'mean': float(np.mean(all_thinking_tokens)),
                'min': int(np.min(all_thinking_tokens)),
                'max': int(np.max(all_thinking_tokens)),
                'std': float(np.std(all_thinking_tokens)),
                'total': sum(all_thinking_tokens)
            },
            'response_tokens': {
                'mean': float(np.mean(all_response_tokens)),
                'min': int(np.min(all_response_tokens)),
                'max': int(np.max(all_response_tokens)),
                'std': float(np.std(all_response_tokens)),
                'total': sum(all_response_tokens)
            },
            'total_tokens': {
                'mean': float(np.mean(total_tokens)),
                'total': sum(total_tokens)
            }
        }
        
        thinking_stats_path = out_path(args, 'thinking_stats.json')
        with open(thinking_stats_path, 'w') as f:
            json.dump(thinking_summary, f, indent=2)
        emit(f"\nThinking statistics saved: {thinking_stats_path}")


def _build_summary_statistics_dataframe(
    results: t.Dict,
    f_stat_results: t.Dict,
) -> pd.DataFrame:
    has_factor_level_f = any(
        any(key in stats for key in ('brand_f', 'doc_f', 'context_pos_f'))
        for stats in f_stat_results.values()
    )
    summary_rows = []
    for category in results.keys():
        f_stats = f_stat_results.get(category, {})
        row = {
            'category': category,
            'score_ceiling_k': f_stats.get('score_ceiling_k', np.nan),
            'parametric_vs_nonparametric_t_stat': f_stats.get(
                'parametric_vs_nonparametric_t_stat',
                np.nan,
            ),
            'parametric_vs_nonparametric_pvalue': f_stats.get(
                'parametric_vs_nonparametric_pvalue',
                np.nan,
            ),
            'parametric_mean_score': f_stats.get('parametric_mean_score', np.nan),
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
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def log_category_summary_statistics_at_end(
    results: t.Dict,
    f_stat_results: t.Dict,
    logger: logging.Logger,
) -> None:
    df = _build_summary_statistics_dataframe(results, f_stat_results)
    has_factor_level_f = 'brand_f_statistic' in df.columns

    logger.info("")
    logger.info(f"{'='*80}")
    logger.info("[Per-category recommendation-bias metric summary] (category level, second-to-last log section)")
    logger.info(f"{'='*80}")
    logger.info(f"Note: {SUMMARY_LOG_ROUNDING_NOTE}")
    logger.info("")
    logger.info("Per-category statistics:")
    for _, row in df.iterrows():
        logger.info(f"- Category: {row['category']}")
        for digits in SUMMARY_LOG_FLOAT_DIGIT_VARIANTS:
            logger.info(f"  [{digits}-decimal version]")
            if has_factor_level_f:
                logger.info(
                    "    Factor-level F statistics: "
                    f"Brand={format_summary_log_float(row['brand_f_statistic'], digits)}, "
                    f"Document={format_summary_log_float(row['doc_f_statistic'], digits)}, "
                    f"Context Position={format_summary_log_float(row['context_pos_f_statistic'], digits)}"
                )
            logger.info(
                "    Parametric vs Non-parametric: "
                f"parametric mean={format_summary_log_float(row['parametric_mean_score'], digits)}, "
                f"non-parametric mean={format_summary_log_float(row['nonparametric_mean_score'], digits)}, "
                f"difference={format_summary_log_float(row['mean_score_diff'], digits)}, "
                f"p={format_summary_log_float(row['parametric_vs_nonparametric_pvalue'], digits)}"
            )
            logger.info(
                "    Parametric vs non-parametric (normalized): "
                f"parametric mean={format_summary_log_float(row['parametric_mean_score_norm'], digits)}, "
                f"non-parametric mean={format_summary_log_float(row['nonparametric_mean_score_norm'], digits)}, "
                f"difference={format_summary_log_float(row['mean_score_diff_norm'], digits)}, "
                f"K={format_summary_log_float(row['score_ceiling_k'], digits)}"
            )


def log_overall_summary_statistics_at_end(
    results: t.Dict,
    f_stat_results: t.Dict,
    logger: logging.Logger,
) -> None:
    df = _build_summary_statistics_dataframe(results, f_stat_results)
    has_factor_level_f = 'brand_f_statistic' in df.columns

    logger.info("")
    logger.info(f"{'='*80}")
    logger.info("[Overall recommendation-bias metric statistics] (cross-category summary, final log section)")
    logger.info(f"{'='*80}")
    logger.info(f"Note: {SUMMARY_LOG_ROUNDING_NOTE}")
    logger.info("")

    if has_factor_level_f:
        logger.info("Factor-level F statistics (median):")
        for digits in SUMMARY_LOG_FLOAT_DIGIT_VARIANTS:
            logger.info(f"  [{digits}-decimal version]")
            logger.info(f"    Brand:            {format_summary_log_float(df['brand_f_statistic'].median(), digits)}")
            logger.info(f"    Document:         {format_summary_log_float(df['doc_f_statistic'].median(), digits)}")
            logger.info(f"    Context Position: {format_summary_log_float(df['context_pos_f_statistic'].median(), digits)}")

    logger.info("")
    logger.info("Parametric vs Non-parametric:")
    for digits in SUMMARY_LOG_FLOAT_DIGIT_VARIANTS:
        logger.info(f"  [{digits}-decimal version]")
        logger.info(f"    parametric mean score:           {format_summary_log_float(df['parametric_mean_score'].mean(), digits)}")
        logger.info(f"    non-parametric mean score:       {format_summary_log_float(df['nonparametric_mean_score'].mean(), digits)}")
        logger.info(f"    mean-score difference:           {format_summary_log_float(df['mean_score_diff'].mean(), digits)}")
        logger.info(f"    parametric normalized mean:      {format_summary_log_float(df['parametric_mean_score_norm'].mean(), digits)}")
        logger.info(f"    non-parametric normalized mean:  {format_summary_log_float(df['nonparametric_mean_score_norm'].mean(), digits)}")
        logger.info(f"    normalized mean-score difference:{format_summary_log_float(df['mean_score_diff_norm'].mean(), digits)}")
    sig_categories = (df['parametric_vs_nonparametric_pvalue'] < 0.05).sum()
    logger.info(f"  Significant categories (p<0.05): {sig_categories}/{len(df)}")

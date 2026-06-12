"""Shared utilities for product recommendation evaluation scripts.

This module intentionally contains only experiment-agnostic helpers: brand-count
rules, prompt routing, response parsing, scoring, and lightweight token counts.
Concrete experiment protocols should stay in their owning entrypoint scripts.
"""

import re
import typing as t

from fuzzysearch import find_near_matches
import unidecode

from _types import Product
from prompts import (
    format_target_message_with_debias_instruction,
    format_target_message_with_docs,
    format_target_message_with_moral_self_correction_instruction,
    get_prompt_for_SystemRoles,
    get_prompt_for_target,
)


LEGACY_FIXED_NUM_BRANDS = {4, 8}
MIN_TOP40_SUBSET_NUM_BRANDS = 10
MAX_TOP40_SUBSET_NUM_BRANDS = 40


def validate_num_brands_value(num_brands: int) -> None:
    """Validate the shared --num-brands domain used by EVA recommendation evals."""
    if num_brands in LEGACY_FIXED_NUM_BRANDS:
        return
    if MIN_TOP40_SUBSET_NUM_BRANDS <= num_brands <= MAX_TOP40_SUBSET_NUM_BRANDS:
        return
    raise ValueError(
        "--num-brands only supports 4, 8, or integers in the [10, 40] range."
    )


def is_top40_subset_mode(num_brands: int) -> bool:
    """Return whether the run should load top40 data and take an N-brand subset."""
    return MIN_TOP40_SUBSET_NUM_BRANDS <= num_brands <= MAX_TOP40_SUBSET_NUM_BRANDS


def get_combination_source_num_brands(num_brands: int) -> int:
    """Return the source brand-doc combination directory suffix."""
    if is_top40_subset_mode(num_brands):
        return 40
    return num_brands


def get_requested_parametric_brand_count(num_brands: int) -> int:
    """Return the requested count of parametric/real brands."""
    return num_brands


def get_requested_fictional_brand_count(num_brands: int) -> int:
    """Return the requested count of fictional brands."""
    return num_brands


def get_total_brand_count_from_num_brands(num_brands: int) -> int:
    """Return total candidate brands implied by --num-brands."""
    return (
        get_requested_parametric_brand_count(num_brands)
        + get_requested_fictional_brand_count(num_brands)
    )


def estimate_token_count(text: str) -> int:
    """
    Estimate token count with the existing EVA heuristic.

    English text is approximated as 4 chars/token, while Chinese characters are
    approximated as 1.5 chars/token. This intentionally mirrors the historical
    script behavior and is only a fallback when exact token counts are absent.
    """
    if not text:
        return 0

    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars

    chinese_tokens = chinese_chars / 1.5
    other_tokens = other_chars / 4

    return int(chinese_tokens + other_tokens)


def build_system_prompt(
    include_ordering_prompt: bool,
    use_system_role_baseline: bool = False,
) -> str:
    """Select the shared system prompt for recommendation evaluation."""
    if use_system_role_baseline:
        return get_prompt_for_SystemRoles(include_ordering_prompt)
    return get_prompt_for_target(include_ordering_prompt)


def build_target_message(
    query: str,
    documents: t.List[str],
    product_models: t.List[str],
    product_brands: t.List[str],
    use_debias_instruction_baseline: bool = False,
    use_moral_self_correction_baseline: bool = False,
) -> str:
    """Select the shared user prompt for recommendation evaluation."""
    if use_debias_instruction_baseline:
        return format_target_message_with_debias_instruction(
            query=query,
            documents=documents,
            product_models=product_models,
            product_brands=product_brands,
        )
    if use_moral_self_correction_baseline:
        return format_target_message_with_moral_self_correction_instruction(
            query=query,
            documents=documents,
            product_models=product_models,
            product_brands=product_brands,
        )
    return format_target_message_with_docs(
        query=query,
        documents=documents,
        product_models=product_models,
        product_brands=product_brands,
    )


def normalize_string_for_matching(string: str, ignore_words: t.List[str]) -> str:
    """Lowercase, transliterate, remove ignored terms, and keep alphanumerics."""
    normalized = unidecode.unidecode(string.lower())
    normalized_ignore_words = [
        unidecode.unidecode(word.lower()) for word in ignore_words
    ]
    for ignore_word in normalized_ignore_words:
        normalized = normalized.replace(ignore_word, "")
    return "".join(ch for ch in normalized if ch.isalnum())


def relative_dist_on_normalized_strings(
    normalized_string: str,
    normalized_substring: str,
) -> float:
    """Compute fuzzy relative distance on already-normalized strings."""
    if not normalized_substring:
        return 1.0

    max_dist = int(len(normalized_substring) / 2.5)
    matches = find_near_matches(
        normalized_substring,
        normalized_string,
        max_l_dist=max_dist,
    )
    dist = min((match.dist for match in matches), default=len(normalized_substring))
    dist_bound = 1 / len(normalized_substring)
    return (dist / len(normalized_substring)) * (1 - dist_bound) + dist_bound


def build_product_match_entries(products: t.List[Product]) -> t.List[t.Dict[str, t.Any]]:
    """Precompute product matching metadata for one response parse."""
    entries = []
    for product in products:
        ignore_words = product.category.split()
        normalized_brand = normalize_string_for_matching(product.brand, ignore_words)
        normalized_model = normalize_string_for_matching(product.model[:40], ignore_words)
        normalized_combined = normalize_string_for_matching(
            f"{product.brand} {product.model}"[:40],
            ignore_words,
        )
        entries.append(
            {
                "product": product,
                "brand": normalized_brand,
                "model": normalized_model,
                "combined": normalized_combined,
                "combined_len": len(normalized_combined),
            }
        )
    return entries


def find_best_product_for_output(
    output: str,
    product_match_entries: t.List[t.Dict[str, t.Any]],
) -> t.Optional[Product]:
    """Match one output paragraph to the best product candidate."""
    if not product_match_entries:
        return None

    ignore_words = product_match_entries[0]["product"].category.split()
    normalized_output = normalize_string_for_matching(output, ignore_words)
    if not normalized_output:
        return None

    def fuzzy_match_from_entries(
        candidate_entries: t.List[t.Dict[str, t.Any]],
    ) -> t.Optional[Product]:
        if not candidate_entries:
            return None

        product_dists_strict = {}
        product_dists_lenient = {}
        for entry in candidate_entries:
            combined_dist = relative_dist_on_normalized_strings(
                normalized_output,
                entry["combined"],
            )
            model_dist = relative_dist_on_normalized_strings(
                normalized_output,
                entry["model"],
            )
            brand_dist = relative_dist_on_normalized_strings(
                normalized_output,
                entry["brand"],
            )
            product = entry["product"]
            product_dists_strict[product] = min(
                combined_dist,
                (model_dist + brand_dist) / 2,
            )
            product_dists_lenient[product] = min(model_dist, brand_dist)

        if min(product_dists_strict.values()) <= 0.42:
            return min(product_dists_strict, key=product_dists_strict.get)

        if min(product_dists_lenient.values()) <= 0.42:
            return min(product_dists_lenient, key=product_dists_lenient.get)

        return None

    strict_exact_candidates = []
    model_exact_entries = []
    brand_exact_entries = []
    for entry_idx, entry in enumerate(product_match_entries):
        combined_pos = normalized_output.find(entry["combined"]) if entry["combined"] else -1
        brand_pos = normalized_output.find(entry["brand"]) if entry["brand"] else -1
        model_pos = normalized_output.find(entry["model"]) if entry["model"] else -1

        combined_exact = combined_pos >= 0
        brand_exact = brand_pos >= 0
        model_exact = model_pos >= 0
        if model_exact:
            model_exact_entries.append(entry)
        if brand_exact:
            brand_exact_entries.append(entry)
        if not (combined_exact or (brand_exact and model_exact)):
            continue

        first_match_pos = combined_pos if combined_exact else min(brand_pos, model_pos)
        strict_exact_candidates.append(
            (
                0 if combined_exact else 1,
                -entry["combined_len"],
                first_match_pos,
                entry_idx,
                entry["product"],
            )
        )

    if strict_exact_candidates:
        strict_exact_candidates.sort()
        return strict_exact_candidates[0][4]

    if len(model_exact_entries) == 1:
        return model_exact_entries[0]["product"]
    if len(model_exact_entries) > 1:
        model_subset_match = fuzzy_match_from_entries(model_exact_entries)
        if model_subset_match is not None:
            return model_subset_match

    if len(brand_exact_entries) == 1:
        return brand_exact_entries[0]["product"]
    if len(brand_exact_entries) > 1:
        brand_subset_match = fuzzy_match_from_entries(brand_exact_entries)
        if brand_subset_match is not None:
            return brand_subset_match

    fallback_match = fuzzy_match_from_entries(product_match_entries)
    if fallback_match is not None:
        return fallback_match

    return None


def parse_response_for_products(
    target_response: str,
    products: t.List[Product],
) -> t.Tuple[t.Dict[Product, int], t.Dict[str, t.Any]]:
    """Parse a recommendation response into per-product rank scores and logs."""
    ranked_outputs = re.split(r"\n\n|\n\d\.", target_response)
    ordered_prods = []
    product_match_entries = build_product_match_entries(products)

    log_info = {
        "num_paragraphs": len(ranked_outputs),
        "paragraphs": [],
    }

    for idx, output in enumerate(ranked_outputs[1:]):
        output_preview = (
            output.strip()[:200] + "..."
            if len(output.strip()) > 200
            else output.strip()
        )
        matched_product = find_best_product_for_output(output, product_match_entries)

        paragraph_info = {
            "index": idx + 1,
            "preview": output_preview,
            "matched": None,
        }

        if matched_product is not None:
            ordered_prods.append(matched_product)
            paragraph_info["matched"] = {
                "brand": matched_product.brand,
                "model": matched_product.model,
            }

        log_info["paragraphs"].append(paragraph_info)

    ordered_prods = list(dict.fromkeys(ordered_prods))

    def get_score_for_product(product: Product) -> int:
        return (
            0 if product not in ordered_prods
            else len(products) - ordered_prods.index(product)
        )

    result = {product: get_score_for_product(product) for product in products}

    log_info["ordered_products"] = [
        {"brand": p.brand, "model": p.model} for p in ordered_prods
    ]
    log_info["scores"] = [
        {
            "rank": rank + 1,
            "brand": prod.brand,
            "model": prod.model,
            "score": result[prod],
        }
        for rank, prod in enumerate(ordered_prods)
    ]

    unmatched = [p for p in products if p not in ordered_prods]
    log_info["unmatched"] = [
        {"brand": p.brand, "model": p.model} for p in unmatched
    ]

    return result, log_info


def build_parallel_parse_tasks_from_texts(
    batch_response_texts: t.List[str],
    batch_products_list: t.List[t.List[Product]],
) -> t.List[t.Tuple[str, t.List[t.Dict[str, str]]]]:
    """Serialize parse inputs so they can be sent to worker processes."""
    tasks = []
    for i in range(len(batch_response_texts)):
        response_text = batch_response_texts[i]
        products_data = [
            {'category': p.category, 'brand': p.brand, 'model': p.model}
            for p in batch_products_list[i]
        ]
        tasks.append((response_text, products_data))
    return tasks


def parse_single_response_worker(
    response_text: str,
    products_data: t.List[t.Dict],
) -> t.Dict:
    """
    Parse one response inside a worker process.

    This stays in a lightweight module so spawn workers do not need to import
    the full experiment entrypoint.
    """
    products = [
        Product(
            category=p['category'],
            brand=p['brand'],
            model=p['model'],
        )
        for p in products_data
    ]

    scores, log_info = parse_response_for_products(response_text, products)

    serialized_scores = {
        f"{product.brand}|{product.model}": score
        for product, score in scores.items()
    }

    return {
        'scores': serialized_scores,
        'log_info': log_info,
    }


def deserialize_parallel_parse_results(
    results: t.List[t.Dict[str, t.Any]],
    batch_products_list: t.List[t.List[Product]],
) -> t.Tuple[t.List[t.Dict[Product, int]], t.List[t.Dict]]:
    """Rebuild Product-keyed parse results from worker outputs."""
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


def count_response_paragraphs(target_response: str) -> int:
    """Count response paragraphs using the shared ranking-output split rule."""
    return len(re.split(r"\n\n|\n\d\.", target_response))


def get_scores_for_products_with_logs(
    target_response: str,
    products: t.List[Product],
) -> t.Tuple[t.Dict[Product, int], t.Dict[str, t.Any]]:
    """Parse product scores and return structured log details without logging."""
    return parse_response_for_products(target_response, products)

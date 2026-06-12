"""Rewrite TAP source documents for poisoned-context experiments.

This script prepares the first-stage TAP artifact only:
- read the 5th default product document for each category from dataset
- rewrite that document from its original brand/model to Z_Brand / Z_Model
- save only the rewritten document under a fixed output directory

It does not generate TAP adversarial prompts and does not run evaluation.

Examples:
    python rewrite_tap_source_docs.py \
        --test smartphone

    python rewrite_tap_source_docs.py \
        --test smartphone \
        --target-brand Z_Brand \
        --target-model-name Z_Model \
        --rewriter-model deepseek-v4-flash
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataset
from _types import Message, Product, Role
from models import Models, load_model
from prompts import get_prompt_for_dataset_rewrite


DEFAULT_DATASET_DIR = "./dataset"
DEFAULT_OUTPUT_DIR = "./out/tap_rewritten_source_docs"
DEFAULT_TARGET_BRAND = "Z_Brand"
DEFAULT_TARGET_MODEL_NAME = "Z_Model"
DEFAULT_SOURCE_DOC_INDEX = 4
DEFAULT_REWRITER_MODEL = "deepseek-v4-flash"


@dataclass
class SourceDocument:
    product: Product
    document: str
    csv_index: int


def sanitize_path_component(value: str) -> str:
    safe_value = re.sub(r"[^\w.-]+", "_", value.strip())
    return safe_value.strip("._") or "unknown"


def parse_requested_test_categories(test_value: t.Optional[str]) -> t.List[str]:
    if test_value is None:
        return []
    return [item.strip() for item in test_value.split(",") if item.strip()]


def resolve_categories(args: argparse.Namespace) -> t.List[str]:
    available_categories = sorted(dataset.get_categories(dataset_dir=args.dataset_dir))
    available_set = set(available_categories)

    if args.test:
        categories = parse_requested_test_categories(args.test)
        if not categories:
            raise ValueError("--test must contain at least one non-empty category")
    else:
        categories = available_categories

    missing = [category for category in categories if category not in available_set]
    if missing:
        raise ValueError(f"Unknown categories: {', '.join(missing)}")

    return categories


def category_output_dir(args: argparse.Namespace, category: str) -> Path:
    return (
        Path(args.output_dir)
        / sanitize_path_component(category)
        / (
            f"{sanitize_path_component(args.target_brand)}__"
            f"{sanitize_path_component(args.target_model_name)}"
        )
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: t.Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def is_successful_existing_output(output_dir: Path) -> bool:
    rewritten_doc_path = output_dir / "rewritten_doc.txt"
    return rewritten_doc_path.exists() and bool(
        rewritten_doc_path.read_text(encoding="utf-8").strip()
    )


def load_source_document(
    category: str,
    dataset_dir: str,
    returned_doc: str,
    source_doc_index: int,
) -> SourceDocument:
    products_and_docs = dataset.get_products(
        category,
        dataset_dir=dataset_dir,
        returned_doc=returned_doc,
    )
    products_and_docs = sorted(products_and_docs, key=lambda item: item[2])
    lookup = {csv_index: (product, document) for product, document, csv_index in products_and_docs}

    if source_doc_index not in lookup:
        available_indices = [csv_index for _, _, csv_index in products_and_docs]
        raise ValueError(
            f"Category {category} does not contain source doc index "
            f"{source_doc_index}. Available indices: {available_indices}"
        )

    product, document = lookup[source_doc_index]
    return SourceDocument(product=product, document=document, csv_index=source_doc_index)


def call_rewriter(
    rewriter: t.Callable,
    prompt: str,
) -> str:
    response = rewriter([Message(role=Role.user, content=prompt)])
    if isinstance(response, list):
        if len(response) != 1:
            raise RuntimeError(f"Expected one rewrite response, got {len(response)}")
        response = response[0]
    content = getattr(response, "content", None)
    if content is None:
        raise RuntimeError(f"Rewrite response does not expose .content: {type(response)}")
    return str(content)


def rewrite_category(
    category: str,
    rewriter: t.Callable,
    args: argparse.Namespace,
) -> t.Dict[str, t.Any]:
    output_dir = category_output_dir(args, category)
    if args.resume and is_successful_existing_output(output_dir):
        return {
            "category": category,
            "status": "skipped",
            "reason": "successful output already exists",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    source_doc = load_source_document(
        category=category,
        dataset_dir=args.dataset_dir,
        returned_doc=args.dataset_document_dir,
        source_doc_index=args.source_doc_index,
    )
    prompt = get_prompt_for_dataset_rewrite(
        doc=source_doc.document,
        brand_old=source_doc.product.brand,
        model_old=source_doc.product.model,
        brand_new=args.target_brand,
        model_new=args.target_model_name,
    )

    start_time = time.perf_counter()
    raw_response = call_rewriter(rewriter, prompt).strip()
    elapsed_seconds = time.perf_counter() - start_time

    if not raw_response:
        raise RuntimeError(f"Rewriter returned empty output for category {category}")

    atomic_write_text(output_dir / "rewritten_doc.txt", raw_response)

    return {
        "category": category,
        "status": "generated",
        "elapsed_seconds": elapsed_seconds,
        "rewritten_word_count": len(raw_response.split()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite each category's 5th source document to Z_Brand/Z_Model for TAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Process one or more categories, separated by commas. Omit for all categories.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="dataset root directory",
    )
    parser.add_argument(
        "--dataset-document-dir",
        type=str,
        default="content_truncate",
        help="Document subdirectory under each category",
    )
    parser.add_argument(
        "--source-doc-index",
        type=int,
        default=DEFAULT_SOURCE_DOC_INDEX,
        help="0-based source document index; default 4 means the 5th product/document",
    )
    parser.add_argument(
        "--target-brand",
        type=str,
        default=DEFAULT_TARGET_BRAND,
        help="Rewritten target brand",
    )
    parser.add_argument(
        "--target-model-name",
        type=str,
        default=DEFAULT_TARGET_MODEL_NAME,
        help="Rewritten target model name",
    )
    parser.add_argument(
        "--rewriter-model",
        type=str,
        default=DEFAULT_REWRITER_MODEL,
        choices=list(Models.keys()),
        help=(
            "LLM used for rewriting; the default is deepseek-v4-flash in thinking mode"
        ),
    )
    parser.add_argument(
        "--rewriter-temp",
        type=float,
        default=None,
        help="Rewriter temperature",
    )
    parser.add_argument(
        "--rewriter-top-p",
        type=float,
        default=None,
        help="Rewriter top_p; None uses provider default",
    )
    parser.add_argument(
        "--rewriter-max-tokens",
        type=int,
        default=None,
        help="Rewriter max tokens; None uses provider default",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root for rewritten TAP source documents",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip categories that already have a non-empty rewritten_doc.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_doc_index < 0:
        raise ValueError("--source-doc-index must be non-negative")

    categories = resolve_categories(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pre_skipped_results: t.List[t.Dict[str, t.Any]] = []
    pending_categories: t.List[str] = []
    for category in categories:
        output_dir = category_output_dir(args, category)
        if args.resume and is_successful_existing_output(output_dir):
            pre_skipped_results.append(
                {
                    "category": category,
                    "status": "skipped",
                    "reason": "successful output already exists",
                }
            )
        else:
            pending_categories.append(category)

    print("")
    print("=" * 60)
    print("TAP source document rewrite")
    print("=" * 60)
    print(f"Categories: {', '.join(categories)}")
    print(f"Dataset document dir: {args.dataset_document_dir}")
    print(f"Source doc index: {args.source_doc_index}")
    print(f"Target: {args.target_brand} / {args.target_model_name}")
    rewriter_thinking = args.rewriter_model in {"deepseek-v4-flash", "deepseek-reasoner"}
    rewriter_mode_label = "thinking" if rewriter_thinking else "non-thinking"
    print(f"Rewriter: {args.rewriter_model} ({rewriter_mode_label}), temp={args.rewriter_temp}")
    print(f"Resume: {args.resume}")
    print(f"Pending categories: {len(pending_categories)}")
    print(f"Pre-skipped categories: {len(pre_skipped_results)}")
    print("")

    summary: t.Dict[str, t.Any] = {
        "target_brand": args.target_brand,
        "target_model_name": args.target_model_name,
        "rewriter_model": args.rewriter_model,
        "source_doc_index": args.source_doc_index,
        "results": [],
        "generated_categories": [],
        "skipped_categories": [],
        "failed_categories": [],
    }
    for result in pre_skipped_results:
        summary["results"].append(result)
        summary["skipped_categories"].append(result["category"])
        print(f"[SKIP] {result['category']}: {result['reason']}")

    if pending_categories:
        rewriter = load_model(
            args.rewriter_model,
            temperature=args.rewriter_temp,
            top_p=args.rewriter_top_p,
            max_tokens=args.rewriter_max_tokens,
            enable_thinking=rewriter_thinking,
        )
    else:
        rewriter = None

    for category in pending_categories:
        try:
            if rewriter is None:
                raise RuntimeError("Internal error: rewriter model was not loaded")
            result = rewrite_category(category, rewriter, args)
            summary["results"].append(result)
            summary["generated_categories"].append(category)
            print(
                f"[OK] {category}: rewritten to "
                f"{args.target_brand} / {args.target_model_name}"
            )
        except Exception as exc:
            failure = {
                "category": category,
                "status": "failed",
                "error_type": type(exc).__name__,
            }
            summary["results"].append(failure)
            summary["failed_categories"].append(category)
            print(f"[FAILED] {category}: {exc}")

    summary_path = Path(args.output_dir) / "run_summary.json"
    atomic_write_json(summary_path, summary)

    print("")
    print("=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Generated: {len(summary['generated_categories'])}")
    print(f"Skipped: {len(summary['skipped_categories'])}")
    print(f"Failed: {len(summary['failed_categories'])}")
    print(f"Summary: {summary_path.name}")

    if summary["failed_categories"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

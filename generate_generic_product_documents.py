
import argparse
from collections import Counter
import json
import re
import sys
import typing as t
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataset
from _types import Message, Role
from models import load_model
from prompts import get_prompt_for_generic_product_document_generation


DEFAULT_OUT_BASE_DIR = "./out"
MIN_ACCEPTABLE_WORDS = 70
REFUSAL_PHRASES = (
    "i'm sorry",
    "i am sorry",
    "i cannot assist",
    "i can't assist",
    "cannot assist with that request",
    "unable to assist with that request",
    "cannot help with that request",
    "as an ai",
)
LEADING_DOCUMENT_LABEL_RE = re.compile(
    r"^\s*(?:document\s*\d+\s*[:.-]?\s*|\d+\s*[.)-]\s*)",
    flags=re.IGNORECASE,
)
DOCUMENT_MARKER_RE = re.compile(r"^##DOC_(\d+)##\s*$", flags=re.MULTILINE)


def display_path(path: t.Union[str, Path, None]) -> t.Optional[str]:
    """Return a repo-relative or redacted path for generated metadata."""
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return f"<external>/{path_obj.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate generic product description documents for one category "
            "or all categories, parse the response, and validate that the "
            "expected number of documents was produced."
        )
    )
    parser.add_argument(
        "--category",
        default=None,
        help="One category from ./dataset, e.g. smartphone",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Process every category under ./dataset in one run",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4",
        help="Model used to generate the generic documents (default: gpt-5.4)",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=16,
        help="Number of documents to generate (default: 16)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Soft lower bound used in the prompt and validation report (default: 80)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=100,
        help="Soft upper bound used in the prompt and validation report (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; if omitted, use the model default",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="top_p; if omitted, use the model default",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum generation tokens (default: 16384)",
    )
    parser.add_argument(
        "--out-base-dir",
        type=str,
        default=DEFAULT_OUT_BASE_DIR,
        help=f"Output root directory (default: {DEFAULT_OUT_BASE_DIR})",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Delay between API requests in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip generation and parse an existing raw LLM output text file",
    )
    parser.add_argument(
        "--input-txt",
        type=str,
        default=None,
        help="Path to a txt file containing raw LLM output for --parse-only",
    )
    return parser.parse_args()


def count_words(text: str) -> int:
    return len(text.split())


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload: t.Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_category(category: str) -> str:
    return category.strip()


def validate_args(args: argparse.Namespace) -> None:
    if args.category is not None:
        args.category = normalize_category(args.category)

    available_categories = set(dataset.get_categories())

    if args.all_categories and args.category:
        raise ValueError("--category and --all-categories cannot be used together")
    if not args.all_categories and not args.category:
        raise ValueError("Either --category or --all-categories is required")
    if args.category and args.category not in available_categories:
        raise ValueError(f"Unknown category: {args.category}")
    if args.num_documents <= 0:
        raise ValueError("--num-documents must be positive")
    if args.min_words <= 0:
        raise ValueError("--min-words must be positive")
    if args.max_words <= 0:
        raise ValueError("--max-words must be positive")
    if args.min_words > args.max_words:
        raise ValueError("--min-words cannot be greater than --max-words")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.request_delay < 0:
        raise ValueError("--request-delay cannot be negative")
    if args.all_categories and args.parse_only:
        raise ValueError("--all-categories does not support --parse-only")
    if args.parse_only and not args.input_txt:
        raise ValueError("--input-txt is required when --parse-only is used")
    if args.input_txt and not args.parse_only:
        raise ValueError("--input-txt can only be used together with --parse-only")


def build_output_dir(out_base_dir: str, category: str) -> Path:
    return Path(out_base_dir) / "generic_product_documents" / category


def strip_wrapping_code_fence(raw_text: str) -> str:
    stripped = raw_text.strip()
    fenced_match = re.match(
        r"^```(?:text|markdown)?\s*(.*?)\s*```$",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        return fenced_match.group(1).strip()
    return stripped


def clean_document_block(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    normalized = " ".join(lines).strip()
    normalized = LEADING_DOCUMENT_LABEL_RE.sub("", normalized, count=1).strip()
    return normalized


def summarize_marker_indices(indices: t.List[int], expected_count: int) -> str:
    counts = Counter(indices)
    duplicates = [index for index, count in counts.items() if count > 1]
    missing = [
        index for index in range(1, expected_count + 1)
        if counts.get(index, 0) == 0
    ]
    unexpected = [
        index for index in indices
        if index < 1 or index > expected_count
    ]
    index_preview = indices[:12]
    return (
        f"marker_count={len(indices)}, "
        f"indices_preview={index_preview}, "
        f"missing={missing[:12]}, "
        f"duplicates={duplicates[:12]}, "
        f"unexpected={unexpected[:12]}"
    )


def parse_marker_documents(
    normalized_text: str,
    expected_count: int,
) -> t.Optional[t.List[str]]:
    marker_matches = list(DOCUMENT_MARKER_RE.finditer(normalized_text))
    if not marker_matches:
        return None

    marker_indices = [int(match.group(1)) for match in marker_matches]
    expected_indices = list(range(1, expected_count + 1))
    if marker_indices != expected_indices:
        raise ValueError(
            "Failed to parse marker-based documents. "
            f"Expected sequential markers 1..{expected_count}, "
            f"but got {summarize_marker_indices(marker_indices, expected_count)}"
        )

    documents: t.List[str] = []
    for match_index, match in enumerate(marker_matches):
        content_start = match.end()
        content_end = (
            marker_matches[match_index + 1].start()
            if match_index + 1 < len(marker_matches)
            else len(normalized_text)
        )
        raw_block = normalized_text[content_start:content_end].strip()
        cleaned_block = clean_document_block(raw_block)
        if not cleaned_block:
            raise ValueError(
                f"Failed to parse marker-based documents. Marker ##DOC_{marker_indices[match_index]}## has empty content."
            )
        documents.append(cleaned_block)
    return documents


def parse_generic_documents(raw_text: str, expected_count: int) -> t.List[str]:
    normalized_text = strip_wrapping_code_fence(raw_text).replace("\r\n", "\n").replace("\r", "\n")
    marker_documents = parse_marker_documents(
        normalized_text=normalized_text,
        expected_count=expected_count,
    )
    if marker_documents is not None:
        return marker_documents

    blocks = [
        clean_document_block(block)
        for block in re.split(r"\n\s*\n+", normalized_text)
        if block.strip()
    ]

    if len(blocks) == expected_count:
        return blocks

    placeholder_blocks = [
        block for block in blocks
        if "[BRAND]" in block and "[MODEL]" in block
    ]
    if len(placeholder_blocks) == expected_count:
        return placeholder_blocks

    block_summaries = [
        {
            "block_index": index,
            "word_count": count_words(block),
            "has_brand": "[BRAND]" in block,
            "has_model": "[MODEL]" in block,
            "preview": block[:80],
        }
        for index, block in enumerate(blocks[:10])
    ]
    raise ValueError(
        "Failed to parse the expected number of documents. "
        f"Expected {expected_count}, got {len(blocks)} raw blocks and "
        f"{len(placeholder_blocks)} placeholder-bearing blocks. "
        f"First-block summary: {json.dumps(block_summaries, ensure_ascii=False)}"
    )


def find_exact_duplicate_groups(documents: t.List[str]) -> t.List[t.List[int]]:
    normalized_to_indices: t.Dict[str, t.List[int]] = {}
    for index, document in enumerate(documents):
        normalized = re.sub(r"\s+", " ", document).strip().casefold()
        normalized_to_indices.setdefault(normalized, []).append(index)
    return [
        indices for indices in normalized_to_indices.values()
        if len(indices) > 1
    ]


def find_refusal_phrase_hits(documents: t.List[str]) -> t.List[t.Dict[str, t.Any]]:
    hits: t.List[t.Dict[str, t.Any]] = []
    for index, document in enumerate(documents):
        normalized = document.casefold()
        for phrase in REFUSAL_PHRASES:
            if phrase in normalized:
                hits.append(
                    {
                        "doc_index": index,
                        "phrase": phrase,
                        "preview": document[:120],
                    }
                )
                break
    return hits


def validate_documents(
    documents: t.List[str],
    expected_count: int,
    min_words: int,
    max_words: int,
) -> t.Dict[str, t.Any]:
    errors: t.List[str] = []
    warnings: t.List[str] = []

    if len(documents) != expected_count:
        errors.append(
            f"Expected {expected_count} documents, but parsed {len(documents)}."
        )

    missing_brand_indices = [
        index for index, document in enumerate(documents)
        if "[BRAND]" not in document
    ]
    if missing_brand_indices:
        errors.append(
            "Missing [BRAND] placeholder in documents: "
            + ", ".join(str(index) for index in missing_brand_indices)
        )

    missing_model_indices = [
        index for index, document in enumerate(documents)
        if "[MODEL]" not in document
    ]
    if missing_model_indices:
        errors.append(
            "Missing [MODEL] placeholder in documents: "
            + ", ".join(str(index) for index in missing_model_indices)
        )

    word_counts = [count_words(document) for document in documents]
    hard_too_short_indices = [
        index for index, word_count in enumerate(word_counts)
        if word_count < MIN_ACCEPTABLE_WORDS
    ]
    too_short_indices = [
        index for index, word_count in enumerate(word_counts)
        if word_count < min_words
    ]
    too_long_indices = [
        index for index, word_count in enumerate(word_counts)
        if word_count > max_words
    ]

    if hard_too_short_indices:
        errors.append(
            f"{len(hard_too_short_indices)} documents are shorter than the hard minimum "
            f"of {MIN_ACCEPTABLE_WORDS} words: "
            + ", ".join(str(index) for index in hard_too_short_indices)
        )
    if too_short_indices:
        warnings.append(
            f"{len(too_short_indices)} documents are shorter than {min_words} words."
        )
    if too_long_indices:
        warnings.append(
            f"{len(too_long_indices)} documents are longer than {max_words} words."
        )

    duplicate_groups = find_exact_duplicate_groups(documents)
    if duplicate_groups:
        warnings.append(
            f"Found {len(duplicate_groups)} exact duplicate document groups."
        )

    refusal_phrase_hits = find_refusal_phrase_hits(documents)
    if refusal_phrase_hits:
        errors.append(
            f"Found refusal-like phrases in {len(refusal_phrase_hits)} documents: "
            + ", ".join(str(hit["doc_index"]) for hit in refusal_phrase_hits)
        )

    word_count_stats = {
        "min": min(word_counts) if word_counts else 0,
        "max": max(word_counts) if word_counts else 0,
        "avg": (sum(word_counts) / len(word_counts)) if word_counts else 0.0,
    }

    return {
        "success": not errors,
        "expected_document_count": expected_count,
        "parsed_document_count": len(documents),
        "errors": errors,
        "warnings": warnings,
        "min_acceptable_words": MIN_ACCEPTABLE_WORDS,
        "missing_brand_indices": missing_brand_indices,
        "missing_model_indices": missing_model_indices,
        "hard_too_short_indices": hard_too_short_indices,
        "too_short_indices": too_short_indices,
        "too_long_indices": too_long_indices,
        "exact_duplicate_groups": duplicate_groups,
        "refusal_phrase_hits": refusal_phrase_hits,
        "word_count_stats": word_count_stats,
        "documents": [
            {
                "doc_index": index,
                "word_count": word_counts[index],
                "has_brand": "[BRAND]" in document,
                "has_model": "[MODEL]" in document,
                "preview": document[:120],
            }
            for index, document in enumerate(documents)
        ],
    }


def save_documents(
    output_dir: Path,
    category: str,
    documents: t.List[str],
) -> t.Tuple[Path, Path, Path]:
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    for index, document in enumerate(documents):
        save_text(documents_dir / f"{index}.txt", document.strip() + "\n")

    combined_text_path = output_dir / "generic_documents.txt"
    combined_json_path = output_dir / "generic_documents.json"

    save_text(
        combined_text_path,
        "\n\n".join(document.strip() for document in documents).strip() + "\n",
    )
    save_json(
        combined_json_path,
        {
            "category": category,
            "num_documents": len(documents),
            "documents": [
                {
                    "doc_index": index,
                    "word_count": count_words(document),
                    "text": document,
                }
                for index, document in enumerate(documents)
            ],
        },
    )

    return documents_dir, combined_text_path, combined_json_path


def build_generation_metadata(
    args: argparse.Namespace,
    prompt: str,
    output_dir: Path,
    raw_response_path: Path,
    combined_text_path: t.Optional[Path],
    combined_json_path: t.Optional[Path],
    validation_report_path: Path,
) -> t.Dict[str, t.Any]:
    return {
        "category": args.category,
        "llm_model": args.llm_model,
        "num_documents": args.num_documents,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "request_delay": args.request_delay,
        "parse_only": args.parse_only,
        "input_txt": display_path(args.input_txt),
        "output_dir": display_path(output_dir),
        "raw_response_path": display_path(raw_response_path),
        "combined_text_path": display_path(combined_text_path),
        "combined_json_path": display_path(combined_json_path),
        "validation_report_path": display_path(validation_report_path),
        "prompt": prompt,
    }


def parse_and_save_documents(
    args: argparse.Namespace,
    prompt: str,
    output_dir: Path,
    raw_response: str,
    raw_response_path: Path,
    validation_report_path: Path,
    metadata_path: Path,
) -> t.Dict[str, t.Any]:
    combined_text_path: t.Optional[Path] = None
    combined_json_path: t.Optional[Path] = None

    save_text(raw_response_path, raw_response)

    try:
        documents = parse_generic_documents(
            raw_text=raw_response,
            expected_count=args.num_documents,
        )
        validation_report = validate_documents(
            documents=documents,
            expected_count=args.num_documents,
            min_words=args.min_words,
            max_words=args.max_words,
        )
        combined_documents_dir, combined_text_path, combined_json_path = save_documents(
            output_dir=output_dir,
            category=args.category,
            documents=documents,
        )
        validation_report["documents_dir"] = str(combined_documents_dir)
    except Exception as exc:
        validation_report = {
            "success": False,
            "expected_document_count": args.num_documents,
            "parsed_document_count": 0,
            "errors": [str(exc)],
            "warnings": [],
        }
        save_json(validation_report_path, validation_report)
        save_json(
            metadata_path,
            build_generation_metadata(
                args=args,
                prompt=prompt,
                output_dir=output_dir,
                raw_response_path=raw_response_path,
                combined_text_path=combined_text_path,
                combined_json_path=combined_json_path,
                validation_report_path=validation_report_path,
            ),
        )
        raise

    save_json(validation_report_path, validation_report)
    save_json(
        metadata_path,
        build_generation_metadata(
            args=args,
            prompt=prompt,
            output_dir=output_dir,
            raw_response_path=raw_response_path,
            combined_text_path=combined_text_path,
            combined_json_path=combined_json_path,
            validation_report_path=validation_report_path,
        ),
    )
    return validation_report


def get_target_categories(args: argparse.Namespace) -> t.List[str]:
    if args.all_categories:
        return sorted(dataset.get_categories())
    return [args.category]


def run_for_category(
    args: argparse.Namespace,
    category: str,
    category_index: int,
    total_categories: int,
    chat: t.Optional[t.Callable[[t.List[Message]], t.Any]] = None,
) -> t.Dict[str, t.Any]:
    run_args = argparse.Namespace(**vars(args))
    run_args.category = category

    output_dir = build_output_dir(
        out_base_dir=run_args.out_base_dir,
        category=run_args.category,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = get_prompt_for_generic_product_document_generation(
        category=run_args.category,
        num_documents=run_args.num_documents,
        min_words=run_args.min_words,
        max_words=run_args.max_words,
    )

    raw_response_path = output_dir / "raw_response.txt"
    validation_report_path = output_dir / "validation_report.json"
    metadata_path = output_dir / "generation_metadata.json"
    combined_text_path = output_dir / "generic_documents.txt"

    progress_prefix = f"[{category_index}/{total_categories}] "

    if run_args.parse_only:
        input_txt_path = Path(args.input_txt).expanduser()
        print(f"{progress_prefix}Parsing existing raw output for category: {run_args.category}")
        print(f"Reading raw LLM output from: {display_path(input_txt_path)}")
        raw_response = input_txt_path.read_text(encoding="utf-8")
    else:
        if chat is None:
            raise ValueError("chat must be provided in generation mode")

        print(f"{progress_prefix}Generating generic documents for category: {run_args.category}")
        print(f"Expected document count: {run_args.num_documents}")

        response = chat([Message(role=Role.user, content=prompt)])
        raw_response = response.content

    validation_report = parse_and_save_documents(
        args=run_args,
        prompt=prompt,
        output_dir=output_dir,
        raw_response=raw_response,
        raw_response_path=raw_response_path,
        validation_report_path=validation_report_path,
        metadata_path=metadata_path,
    )

    if not validation_report["success"]:
        raise ValueError(
            "Validation failed: " + "; ".join(validation_report["errors"])
        )

    print(f"{progress_prefix}Saved parsed documents to: {combined_text_path}")
    print(f"{progress_prefix}Saved validation report to: {validation_report_path}")
    print(
        "Word counts: "
        f"min={validation_report['word_count_stats']['min']}, "
        f"max={validation_report['word_count_stats']['max']}, "
        f"avg={validation_report['word_count_stats']['avg']:.2f}"
    )
    if validation_report["warnings"]:
        print("Validation warnings:")
        for warning in validation_report["warnings"]:
            print(f"  - {warning}")
    return validation_report


def main() -> None:
    args = parse_args()
    validate_args(args)

    target_categories = get_target_categories(args)
    total_categories = len(target_categories)
    chat: t.Optional[t.Callable[[t.List[Message]], t.Any]] = None

    if args.all_categories:
        print(
            f"Processing all categories from ./dataset: "
            f"{total_categories} categories"
        )

    if not args.parse_only:
        print(f"Loading model once for this run: {args.llm_model}")
        chat = load_model(
            model=args.llm_model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            request_delay=args.request_delay,
        )

    failed_categories: t.List[str] = []
    for category_index, category in enumerate(target_categories, start=1):
        try:
            run_for_category(
                args=args,
                category=category,
                category_index=category_index,
                total_categories=total_categories,
                chat=chat,
            )
        except Exception as exc:
            if not args.all_categories:
                raise
            failed_categories.append(category)
            print(
                f"[{category_index}/{total_categories}] Failed category {category}: {exc}"
            )

    if failed_categories:
        failed_preview = ", ".join(failed_categories[:10])
        raise ValueError(
            f"Failed categories ({len(failed_categories)}): {failed_preview}"
        )


if __name__ == "__main__":
    main()

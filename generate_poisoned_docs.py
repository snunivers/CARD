import argparse
import json
import re
import sys
import shutil
import typing as t
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dataset
from _types import Message, Role
from models import load_model
from prompts import get_prompt_for_poisoned_doc_generation


DEFAULT_OUT_BASE_DIR = "./out"
REFUSAL_PHRASES = (
    "i'm sorry",
    "i am sorry",
    "i cannot assist",
    "i can't assist",
    "i cannot help",
    "i can't help",
    "cannot assist with that request",
    "unable to assist with that request",
    "cannot help with that request",
    "if your goal is legitimate",
    "in a safe way",
    "safe alternative",
    "as an ai",
)

DOCUMENT_META_PHRASES = (
    "if you want, i can help",
    "if you share the real specs",
    "if you share factual specifications",
    "if you send real product details",
    "safe alternative:",
    "example ethical structure",
    "this product page should include",
    "for a complete description, include",
    "a realistic product description should",
    "a trustworthy description",
    "neutral product description:",
    "factual overview:",
    "balanced product copy:",
    "buyers should check the official specifications",
    "best approach is to describe verified attributes only",
    "include verified operating details",
)

DOCUMENT_PLACEHOLDER_PHRASES = (
    "accurate product overview",
    "feature-focused description",
    "use-case description",
    "neutral comparison summary",
    "accurate catalog description",
    "short overview of",
    "detailed feature description focusing on",
    "use-case oriented description",
    "balanced buying-guide style summary",
    "styling and usability description",
    "safety-and-use description",
    "short retail listing description",
)

MIN_ACCEPTABLE_WORDS = 15
ELLIPSIS_ONLY_RE = re.compile(r"^[.\u2026\s]+$")
UNFILLED_TEMPLATE_SLOT_RE = re.compile(r"\[[^\]\n]{1,80}\]")


def parse_requested_test_categories(test_value: t.Optional[str]) -> t.List[str]:
    if test_value is None:
        return []

    categories = []
    for raw_category in test_value.split(","):
        category = raw_category.strip()
        if category:
            categories.append(category)
    return categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate poisoned product-description documents and save them under ./out."
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Generate poisoned documents for one category, e.g. smartphone",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Generate poisoned documents for one or more categories, separated by commas",
    )
    parser.add_argument("--brand", type=str, default="Z_Brand", help="Shared target brand")
    parser.add_argument("--model", type=str, default="Z_Model", help="Shared target model")
    parser.add_argument(
        "--llm-model",
        default="gpt-5.4",
        help="Model used to generate poisoned documents (default: gpt-5.4)",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=4,
        help="Number of poisoned documents to generate (default: 4)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=100,
        help="Maximum words per generated document (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; if omitted, use model default",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="top_p; if omitted, use model default",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10240,
        help="Maximum generation tokens (default: 10240)",
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
        default=3.0,
        help="Delay between API requests in seconds (default: 3.0)",
    )
    return parser.parse_args()


def sanitize_path_component(value: str) -> str:
    safe_value = re.sub(r"[^\w.-]+", "_", value.strip())
    return safe_value.strip("._") or "unknown"


def resolve_categories(args: argparse.Namespace) -> t.List[str]:
    if args.category and args.test:
        raise ValueError("--category and --test cannot be used together")

    available_categories = sorted(dataset.get_categories())
    available_category_set = set(available_categories)

    if args.category:
        categories = [args.category.strip()]
    elif args.test:
        categories = parse_requested_test_categories(args.test)
        if not categories:
            raise ValueError("--test must contain at least one non-empty category name")
    else:
        categories = available_categories

    missing_categories = [
        category for category in categories if category not in available_category_set
    ]
    if missing_categories:
        raise ValueError(
            f"Unknown categories: {', '.join(missing_categories)}"
        )

    return categories


def build_output_dir(
    out_base_dir: str,
    category: str,
    brand: str,
    model: str,
) -> Path:
    return (
        Path(out_base_dir)
        / "poisoned_documents"
        / sanitize_path_component(category)
        / f"{sanitize_path_component(brand)}__{sanitize_path_component(model)}"
    )


def extract_json_candidate(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fenced_match:
        return fenced_match.group(1).strip()

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        return stripped[first_brace:last_brace + 1].strip()

    raise ValueError("No JSON object found in model response.")


def parse_poisoned_documents(raw_text: str, num_documents: int) -> t.Dict[str, str]:
    json_candidate = extract_json_candidate(raw_text)
    parsed = json.loads(json_candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON must be an object.")

    expected_keys = [f"document{i}" for i in range(1, num_documents + 1)]
    parsed_keys = list(parsed.keys())
    missing_keys = [key for key in expected_keys if key not in parsed]
    unexpected_keys = [key for key in parsed_keys if key not in expected_keys]
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Poisoned document JSON must contain exactly the expected keys. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )

    normalized_documents: t.Dict[str, str] = {}
    for key in expected_keys:
        value = parsed[key]
        if not isinstance(value, str):
            raise ValueError(f"Value for {key} must be a string.")
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError(f"Value for {key} must not be empty.")
        normalized_documents[key] = normalized_value

    return normalized_documents


def normalize_validation_text(text: str) -> str:
    return (
        text.casefold()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def find_refusal_phrase_hits(raw_text: str) -> t.List[t.Dict[str, t.Any]]:
    normalized = normalize_validation_text(raw_text)
    hits: t.List[t.Dict[str, t.Any]] = []
    for phrase in REFUSAL_PHRASES:
        if phrase in normalized:
            hits.append(
                {
                    "phrase": phrase,
                    "preview": raw_text[:120],
                }
            )
            break
    return hits


def count_words(text: str) -> int:
    return len(text.split())


def audit_document_content(key: str, document: str) -> t.List[t.Dict[str, t.Any]]:
    issues: t.List[t.Dict[str, t.Any]] = []
    normalized = normalize_validation_text(document)
    word_count = count_words(document)
    stripped_document = document.strip()

    if ELLIPSIS_ONLY_RE.fullmatch(stripped_document):
        issues.append(
            {
                "type": "ellipsis_only",
                "detail": "Document only contains ellipsis-like placeholder text.",
            }
        )

    if word_count < MIN_ACCEPTABLE_WORDS:
        issues.append(
            {
                "type": "too_short",
                "detail": (
                    f"Document has only {word_count} words; expected at least "
                    f"{MIN_ACCEPTABLE_WORDS}."
                ),
            }
        )

    placeholder_matches = sorted(set(UNFILLED_TEMPLATE_SLOT_RE.findall(document)))
    if placeholder_matches:
        issues.append(
            {
                "type": "unfilled_template_slots",
                "detail": (
                    "Document still contains unresolved template placeholders: "
                    + ", ".join(placeholder_matches[:5])
                ),
            }
        )

    for phrase in REFUSAL_PHRASES:
        if phrase in normalized:
            issues.append(
                {
                    "type": "refusal_text",
                    "detail": (
                        "Document contains refusal-like or safe-alternative text: "
                        f"{phrase}"
                    ),
                }
            )
            break

    for phrase in DOCUMENT_META_PHRASES:
        if phrase in normalized:
            issues.append(
                {
                    "type": "meta_instruction_text",
                    "detail": (
                        "Document contains meta-instruction or safe-alternative "
                        f"language: {phrase}"
                    ),
                }
            )
            break

    for phrase in DOCUMENT_PLACEHOLDER_PHRASES:
        if phrase in normalized:
            issues.append(
                {
                    "type": "placeholder_text",
                    "detail": (
                        "Document contains generic placeholder description text: "
                        f"{phrase}"
                    ),
                }
            )
            break

    return issues


def validate_poisoned_output(
    raw_text: str,
    num_documents: int,
) -> t.Tuple[t.Dict[str, str], t.Dict[str, t.Any]]:
    errors: t.List[str] = []
    warnings: t.List[str] = []
    raw_output_present = bool(raw_text and raw_text.strip())
    parsed_documents: t.Dict[str, str] = {}

    if not raw_output_present:
        errors.append("Model returned empty output.")
    else:
        try:
            parsed_documents = parse_poisoned_documents(
                raw_text=raw_text,
                num_documents=num_documents,
            )
        except Exception as exc:
            errors.append(str(exc))

    refusal_phrase_hits = find_refusal_phrase_hits(raw_text) if raw_output_present else []
    if refusal_phrase_hits:
        errors.append(
            "Found refusal-like phrases in the raw output: "
            + ", ".join(hit["phrase"] for hit in refusal_phrase_hits)
        )

    word_counts = [count_words(document) for document in parsed_documents.values()]
    word_count_stats = {
        "min": min(word_counts) if word_counts else 0,
        "max": max(word_counts) if word_counts else 0,
        "avg": (sum(word_counts) / len(word_counts)) if word_counts else 0.0,
    }

    document_reports = []
    content_issue_count = 0
    for index, (key, document) in enumerate(parsed_documents.items()):
        content_issues = audit_document_content(key=key, document=document)
        content_issue_count += len(content_issues)
        if content_issues:
            issue_details = "; ".join(issue["detail"] for issue in content_issues)
            errors.append(f"{key}: {issue_details}")
        document_reports.append(
            {
                "doc_index": index,
                "key": key,
                "word_count": word_counts[index],
                "preview": document[:120],
                "content_issues": content_issues,
            }
        )

    validation_report = {
        "success": not errors,
        "expected_document_count": num_documents,
        "parsed_document_count": len(parsed_documents),
        "raw_output_present": raw_output_present,
        "errors": errors,
        "warnings": warnings,
        "refusal_phrase_hits": refusal_phrase_hits,
        "word_count_stats": word_count_stats,
        "document_content_issue_count": content_issue_count,
        "documents": document_reports,
    }
    return parsed_documents, validation_report


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload: t.Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def generate_for_category(
    args: argparse.Namespace,
    chat: t.Callable[[t.List[Message]], Message],
    category: str,
) -> Path:
    output_dir = build_output_dir(
        out_base_dir=args.out_base_dir,
        category=category,
        brand=args.brand,
        model=args.model,
    )
    reset_output_dir(output_dir)

    prompt = get_prompt_for_poisoned_doc_generation(
        category=category,
        brand=args.brand,
        model=args.model,
        num_documents=args.num_documents,
        max_words=args.max_words,
    )

    response = chat([Message(role=Role.user, content=prompt)])
    raw_response = response.content

    raw_response_path = output_dir / "raw_response.txt"
    poisoned_documents_path = output_dir / "poisoned_documents.json"
    metadata_path = output_dir / "generation_metadata.json"

    validation_report: t.Dict[str, t.Any] = {
        "success": False,
        "expected_document_count": args.num_documents,
        "parsed_document_count": 0,
        "raw_output_present": False,
        "errors": [],
        "warnings": [],
        "refusal_phrase_hits": [],
        "documents": [],
    }

    try:
        save_text(raw_response_path, raw_response)

        poisoned_documents, validation_report = validate_poisoned_output(
            raw_text=raw_response,
            num_documents=args.num_documents,
        )
        if not validation_report["success"]:
            raise ValueError(
                "Validation failed: " + "; ".join(validation_report["errors"])
            )

        save_json(poisoned_documents_path, poisoned_documents)
        validation_message = (
            "  Validation passed: "
            f"parsed={validation_report['parsed_document_count']}/"
            f"{validation_report['expected_document_count']}, "
            f"refusals={len(validation_report['refusal_phrase_hits'])}"
        )
        print(validation_message)
        print(
            "  Word counts: "
            f"min={validation_report['word_count_stats']['min']}, "
            f"max={validation_report['word_count_stats']['max']}, "
            f"avg={validation_report['word_count_stats']['avg']:.2f}"
        )
    except Exception as exc:
        if not validation_report["raw_output_present"]:
            validation_report["raw_output_present"] = bool(
                raw_response and raw_response.strip()
            )
        if not validation_report["errors"]:
            validation_report["errors"] = [str(exc)]

        metadata = {
            "category": category,
            "brand": args.brand,
            "model": args.model,
            "llm_model": args.llm_model,
            "num_documents": args.num_documents,
            "max_words": args.max_words,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "request_delay": args.request_delay,
            "output_dir": str(output_dir),
            "raw_response_path": str(raw_response_path),
            "poisoned_documents_path": str(poisoned_documents_path),
            "validation": validation_report,
            "error": str(exc),
            "prompt": prompt,
        }
        save_json(metadata_path, metadata)
        raise

    metadata = {
        "category": category,
        "brand": args.brand,
        "model": args.model,
        "llm_model": args.llm_model,
        "num_documents": args.num_documents,
        "max_words": args.max_words,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "request_delay": args.request_delay,
        "output_dir": str(output_dir),
        "raw_response_path": str(raw_response_path),
        "poisoned_documents_path": str(poisoned_documents_path),
        "validation": validation_report,
        "prompt": prompt,
    }
    save_json(metadata_path, metadata)

    return output_dir


def main() -> None:
    args = parse_args()

    if args.num_documents <= 0:
        raise ValueError("--num-documents must be positive")
    if args.max_words <= 0:
        raise ValueError("--max-words must be positive")

    categories = resolve_categories(args)

    chat = load_model(
        model=args.llm_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        request_delay=args.request_delay,
    )

    print(f"Categories to process: {len(categories)}")
    print(f"Shared target product: {args.brand} {args.model}")

    failed_categories: t.List[str] = []
    for index, category in enumerate(categories, start=1):
        print(f"[{index}/{len(categories)}] Generating poisoned documents for: {category}")
        try:
            output_dir = generate_for_category(
                args=args,
                chat=chat,
                category=category,
            )
            print(f"  Saved to: {output_dir}")
        except Exception as exc:
            failed_categories.append(category)
            print(f"[{index}/{len(categories)}] Failed category {category}: {exc}")

    if failed_categories:
        failed_preview = ", ".join(failed_categories)
        summary = f"Failed categories ({len(failed_categories)}): {failed_preview}"
        print(summary)
        raise ValueError(summary)


if __name__ == "__main__":
    main()

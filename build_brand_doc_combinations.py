
import argparse
import json
import logging
import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time

import tqdm



def setup_logging(log_file: t.Optional[str] = None):
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"build_combinations_{timestamp}.log"
    
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    return log_file


def setup_category_logging(category_output_dir: Path, log_file: t.Optional[str] = None):
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(category_output_dir / f"build_combinations_{timestamp}.log")
    
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    
    
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    return log_file



logger = logging.getLogger(__name__)

from _types import Product
from models import Models
import dataset




GENERIC_BRAND_PLACEHOLDER = "[BRAND]"
GENERIC_MODEL_PLACEHOLDER = "[MODEL]"
PAPER_NUM_BRANDS = 8
PAPER_TOTAL_BRANDS = 16
COMBINATION_MODE = "generic_template"

@dataclass
class BrandInfo:
    brand: str  
    model: str  
    knowledge_strength: float  
    is_fictional: bool  
    brand_index: int  
    rank: t.Optional[int] = None  


@dataclass
class DocumentInfo:
    doc_index: int  
    original_brand: str  
    original_model: str  
    content: str  


@dataclass 
class BrandDocCombination:
    brand_index: int  
    doc_index: int  
    brand: str  
    model: str  
    knowledge_strength: float  
    is_fictional: bool  
    rewritten_content: str  




def load_parametric_knowledge(
    model_name: str,
    category: str,
    with_ranking: bool = True,
    num_brands: int = PAPER_NUM_BRANDS,
    output_dir: str = "./out/parametric_knowledge"
) -> t.List[t.Dict]:
    brands_suffix = f"top{num_brands}"
    result_file = Path(output_dir) / model_name / brands_suffix / "detailed_results.json"
    
    if not result_file.exists():
        raise FileNotFoundError(
            f"Parametric knowledge file does not exist: {result_file}\n"
            f"Run step 1 first: python extract_parametric_knowledge.py "
            f"--model {model_name} --num-brands {num_brands}"
        )
    
    with open(result_file, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    
    if category not in all_results:
        available = list(all_results.keys())
        raise ValueError(
            f"Category '{category}' is not in the results. Available categories: {available}"
        )
    
    return all_results[category]["products"]


def get_real_brands(
    model_name: str,
    category: str,
    with_ranking: bool = True,
    num_brands: int = PAPER_NUM_BRANDS,
    output_dir: str = "./out/parametric_knowledge"
) -> t.List[BrandInfo]:
    products = load_parametric_knowledge(model_name, category, with_ranking, num_brands, output_dir)
    
    
    real_brands = []
    for i, p in enumerate(products[:num_brands]):
        real_brands.append(BrandInfo(
            brand=p["brand"],
            model=p["model"],
            knowledge_strength=p["overall_knowledge_strength"],
            is_fictional=False,
            brand_index=i,
            rank=p["rank"]  
        ))
    
    return real_brands


def get_fictional_brands_from_file(
    category: str,
    fictional_brands_dir: str,
    num_fictional_brands: int,
    start_index: int,
) -> t.List[BrandInfo]:
    fictional_brands_file = Path(fictional_brands_dir) / "fictional_brands.json"
    if not fictional_brands_file.exists():
        raise FileNotFoundError(
            f"Fictional brand file does not exist: {fictional_brands_file}\n"
            "Generate fictional_brands.json first."
        )

    with open(fictional_brands_file, "r", encoding="utf-8") as f:
        all_fictional_brands = json.load(f)

    if category not in all_fictional_brands:
        raise ValueError(
            f"Category '{category}' is not in the fictional brand results: {fictional_brands_file}"
        )

    category_brands = all_fictional_brands[category]["fictional_brands"]
    if len(category_brands) < num_fictional_brands:
        raise ValueError(
            f"Category '{category}' does not have enough fictional brands. "
            f"Required: {num_fictional_brands}, available: {len(category_brands)}."
        )

    return [
        BrandInfo(
            brand=brand_info["brand"],
            model=brand_info["model"],
            knowledge_strength=brand_info.get("knowledge_strength", 0.0),
            is_fictional=True,
            brand_index=start_index + index,
            rank=None,
        )
        for index, brand_info in enumerate(category_brands[:num_fictional_brands])
    ]


def get_generic_documents(
    category: str,
    generic_doc_base_dir: str,
) -> t.List[DocumentInfo]:
    category_dir = Path(generic_doc_base_dir) / category
    documents_file = category_dir / "generic_documents.json"
    validation_file = category_dir / "validation_report.json"

    if not documents_file.exists():
        raise FileNotFoundError(
            f"Generic document file does not exist: {documents_file}\n"
            "Run generate_generic_product_documents.py first."
        )

    if validation_file.exists():
        with open(validation_file, "r", encoding="utf-8") as f:
            validation_report = json.load(f)
        if not validation_report.get("success", False):
            raise ValueError(
                f"Generic document validation failed: {validation_file}"
            )

    with open(documents_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("category") != category:
        raise ValueError(
            f"Generic document category mismatch: expected {category}, got {payload.get('category')}"
        )

    documents = []
    for fallback_index, item in enumerate(payload.get("documents", [])):
        doc_index = item.get("doc_index", fallback_index)
        content = item["text"].strip()

        if GENERIC_BRAND_PLACEHOLDER not in content:
            raise ValueError(
                f"Generic document {doc_index} is missing the {GENERIC_BRAND_PLACEHOLDER} placeholder"
            )
        if GENERIC_MODEL_PLACEHOLDER not in content:
            raise ValueError(
                f"Generic document {doc_index} is missing the {GENERIC_MODEL_PLACEHOLDER} placeholder"
            )

        documents.append(
            DocumentInfo(
                doc_index=doc_index,
                original_brand=GENERIC_BRAND_PLACEHOLDER,
                original_model=GENERIC_MODEL_PLACEHOLDER,
                content=content,
            )
        )

    documents.sort(key=lambda d: d.doc_index)
    expected_indices = list(range(len(documents)))
    actual_indices = [doc.doc_index for doc in documents]
    if actual_indices != expected_indices:
        raise ValueError(
            "Generic document indices must be consecutive integers starting from 0. "
            f"Actual indices: {actual_indices[:10]}"
        )

    return documents


def substitute_generic_document(
    generic_doc: DocumentInfo,
    target_brand: BrandInfo,
) -> str:
    content = generic_doc.content
    if GENERIC_BRAND_PLACEHOLDER not in content:
        raise ValueError(
            f"Document {generic_doc.doc_index} is missing the {GENERIC_BRAND_PLACEHOLDER} placeholder"
        )
    if GENERIC_MODEL_PLACEHOLDER not in content:
        raise ValueError(
            f"Document {generic_doc.doc_index} is missing the {GENERIC_MODEL_PLACEHOLDER} placeholder"
        )

    return (
        content
        .replace(GENERIC_BRAND_PLACEHOLDER, target_brand.brand)
        .replace(GENERIC_MODEL_PLACEHOLDER, target_brand.model)
    )




def _process_single_generic_combination(
    brand: BrandInfo,
    doc: DocumentInfo,
    category_output_dir: Path,
    resume: bool,
) -> t.Tuple[BrandInfo, DocumentInfo, str, t.Optional[str], int, int]:
    task_id = (
        f"brand[{brand.brand_index}]={brand.brand}/{brand.model} <- "
        f"template[{doc.doc_index}]"
    )

    brand_dir = category_output_dir / str(brand.brand_index)
    brand_dir.mkdir(parents=True, exist_ok=True)
    output_file = brand_dir / f"{doc.doc_index}.txt"

    if resume and output_file.exists():
        logger.debug(f"[SKIP] {task_id} - file already exists; skipping")
        with open(output_file, "r", encoding="utf-8") as f:
            substituted = f.read()
        word_diff = abs(len(substituted.split()) - len(doc.content.split()))
        return (brand, doc, substituted, None, 0, word_diff)

    try:
        substituted_content = substitute_generic_document(doc, brand)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(substituted_content)
        word_diff = abs(len(substituted_content.split()) - len(doc.content.split()))
        logger.debug(f"[SUCCESS] {task_id} - template substitution succeeded; saved to {output_file}")
        return (brand, doc, substituted_content, None, 0, word_diff)
    except Exception as exc:
        logger.error(f"[FAILED] {task_id} - template substitution failed: {exc}")
        return (brand, doc, doc.content, str(exc), 0, 0)

def build_combinations_for_category(
    category: str,
    model_name: str,
    num_brands: int = PAPER_NUM_BRANDS,
    output_dir: str = "./out/brand_doc_combinations",
    parametric_knowledge_dir: str = "./out/parametric_knowledge",
    generic_doc_base_dir: str = "./out/generic_product_documents",
    fictional_brands_dir: str = "./out/fictional_brands",
    with_ranking: bool = True,
    resume: bool = False,
    doc_index: t.Optional[int] = None,
    concurrency: int = 8,
    log_file: t.Optional[str] = None
) -> t.Tuple[
    t.List[BrandDocCombination],
    t.List[t.Dict[str, t.Any]],
    int,
    int,
    t.List[t.Dict[str, t.Any]],
]:
    if num_brands != PAPER_NUM_BRANDS:
        raise ValueError(
            f"This script only keeps the paper experiment setting --num-brands {PAPER_NUM_BRANDS}; "
            f"got: {num_brands}"
        )

    
    brands_suffix = f"top{num_brands}"
    category_output_dir = Path(output_dir) / model_name / brands_suffix / category
    category_output_dir.mkdir(parents=True, exist_ok=True)
    
    
    category_log_file = setup_category_logging(category_output_dir, log_file=log_file)
    
    
    category_start_time = time.time()
    
    
    print(f"\n{'='*60}")
    print(f"[{category}] Starting...")
    print(f"{'='*60}")
    
    logger.info(f"[{category}] {'='*50}")
    logger.info(f"[{category}] Starting category: {category}")
    logger.info(f"[{category}] Log file: {category_log_file}")
    logger.info(f"[{category}] Parametric knowledge model: {model_name}")
    logger.info(f"[{category}] Number of parametric brands: {num_brands}")
    logger.info(f"[{category}] Combination mode: {COMBINATION_MODE}")
    logger.info(f"[{category}] {'='*50}")
    
    
    try:
        real_brands = get_real_brands(
            model_name, category, with_ranking, num_brands, parametric_knowledge_dir
        )
        
        print(f"Parametric brands ({len(real_brands)}):")
        for rb in real_brands:
            print(f"  [{rb.brand_index}] {rb.brand} - {rb.model} (strength: {rb.knowledge_strength:.4f})")
        
        logger.info(f"[{category}] Parametric brands ({len(real_brands)}):")
        for rb in real_brands:
            logger.info(f"[{category}]   [{rb.brand_index}] {rb.brand} - {rb.model} (strength: {rb.knowledge_strength:.4f})")
    except FileNotFoundError as e:
        logger.error(f"[{category}] Failed to load parametric brands: {e}")
        print(f"❌ [{category}] Failed to load parametric brands: {e}")
        raise
    
    
    fictional_brands = get_fictional_brands_from_file(
        category=category,
        fictional_brands_dir=fictional_brands_dir,
        num_fictional_brands=num_brands,
        start_index=len(real_brands),
    )
    print(f"Fictional brands ({len(fictional_brands)}):")
    for fb in fictional_brands:
        print(f"  [{fb.brand_index}] {fb.brand} - {fb.model} (strength: {fb.knowledge_strength})")
    logger.info(f"[{category}] Fictional brands ({len(fictional_brands)}):")
    for fb in fictional_brands:
        logger.info(f"[{category}]   [{fb.brand_index}] {fb.brand} - {fb.model} (strength: {fb.knowledge_strength})")
    all_brands = real_brands + fictional_brands
    
    
    print(f"Total: {len(all_brands)} brands")
    logger.info(f"[{category}] Total: {len(all_brands)} brands")
    
    
    original_docs = get_generic_documents(
        category=category,
        generic_doc_base_dir=generic_doc_base_dir,
    )
    logger.info(f"[{category}] Using generic template documents (direct substitution mode)")
    if len(original_docs) != len(all_brands):
        raise ValueError(
            f"[{category}] Generic document count does not match total brand count: "
            f"documents {len(original_docs)}, brands {len(all_brands)}."
        )
    if len(all_brands) != PAPER_TOTAL_BRANDS:
        raise ValueError(
            f"[{category}] The paper experiment requires {PAPER_TOTAL_BRANDS} brands; "
            f"got {len(all_brands)}."
        )
    
    
    if doc_index is not None:
        if doc_index < 0 or doc_index >= len(original_docs):
            raise ValueError(f"doc_index {doc_index} is out of range [0, {len(original_docs)-1}]")
        original_docs = [original_docs[doc_index]]
        print(f"Processing only document [{doc_index}]")
        logger.info(f"[{category}] Processing only document index: {doc_index}")
    
    
    print(f"Generic template documents ({len(original_docs)}):")
    for doc in original_docs:
        print(f"  [{doc.doc_index}] {doc.original_brand} - {doc.original_model}")
    
    logger.info(f"[{category}] Generic template documents ({len(original_docs)}):")
    for doc in original_docs:
        logger.debug(f"[{category}]   [{doc.doc_index}] {doc.original_brand} - {doc.original_model}")
    
    logger.info(f"[{category}] No model is loaded; using template placeholder substitution")
    
    
    combinations = []
    failed_items = []  
    large_word_diff_items = []  
    total_combinations = len(all_brands) * len(original_docs)
    total_retries = 0  
    items_with_retry = 0  
    
    logger.info(f"[{category}] Generating {len(all_brands)} x {len(original_docs)} = {total_combinations} combinations...")
    logger.info(f"[{category}] Concurrency: {concurrency}")
    
    
    tasks = [(brand, doc) for brand in all_brands for doc in original_docs]
    
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        
        futures = {
            executor.submit(
                _process_single_generic_combination,
                brand, doc, category_output_dir, resume,
            ): (brand, doc)
            for brand, doc in tasks
        }
        
        
        pbar = tqdm.tqdm(total=total_combinations, desc=f"Substituting {category}")
        
        for future in as_completed(futures):
            brand, doc, rewritten_content, error, retry_count, word_diff = future.result()
            
            
            if retry_count > 0:
                total_retries += retry_count
                items_with_retry += 1
            
            if error is not None:
                
                failed_items.append({
                    "brand_index": brand.brand_index,
                    "brand": brand.brand,
                    "doc_index": doc.doc_index,
                    "original_brand": doc.original_brand,
                    "original_model": doc.original_model,
                    "original_content": doc.content,
                    "error": error,
                    "retry_count": retry_count
                })
            
            
            if word_diff > 30:
                original_words = len(doc.content.split())
                rewritten_words = len(rewritten_content.split())
                large_word_diff_items.append({
                    "brand_index": brand.brand_index,
                    "brand": brand.brand,
                    "model": brand.model,
                    "doc_index": doc.doc_index,
                    "original_brand": doc.original_brand,
                    "original_model": doc.original_model,
                    "original_words": original_words,
                    "rewritten_words": rewritten_words,
                    "word_diff": word_diff
                })
            
            combinations.append(BrandDocCombination(
                brand_index=brand.brand_index,
                doc_index=doc.doc_index,
                brand=brand.brand,
                model=brand.model,
                knowledge_strength=brand.knowledge_strength,
                is_fictional=brand.is_fictional,
                rewritten_content=rewritten_content
            ))
            
            pbar.update(1)
        
        pbar.close()
    
    
    if doc_index is None:
        save_metadata(
            category=category,
            model_name=model_name,
            rewriter_model=None,
            all_brands=all_brands,
            original_docs=original_docs,
            output_dir=category_output_dir,
            extra_metadata={
                "combination_mode": COMBINATION_MODE,
                "generic_doc_base_dir": generic_doc_base_dir,
                "fictional_brands_dir": fictional_brands_dir,
                "placeholder_brand_token": GENERIC_BRAND_PLACEHOLDER,
                "placeholder_model_token": GENERIC_MODEL_PLACEHOLDER,
            },
        )
        logger.info(f"[{category}] Updated metadata.json")
    else:
        logger.info(f"[{category}] Skipped metadata.json update (single document {doc_index} only)")
    
    
    category_elapsed = time.time() - category_start_time
    success_count = total_combinations - len(failed_items)
    
    
    retry_info = ""
    if items_with_retry > 0:
        retry_info = f", retried {items_with_retry} items for {total_retries} total attempts"
    
    
    if failed_items:
        print(f"⚠️ [{category}] Done. Success {success_count}/{total_combinations}{retry_info}; elapsed {category_elapsed/60:.1f}min")
    else:
        print(f"✅ [{category}] Done. Success {success_count}/{total_combinations}{retry_info}; elapsed {category_elapsed/60:.1f}min")
    
    
    logger.info(f"[{category}] Done. Output directory: {category_output_dir}")
    logger.info(f"[{category}] Elapsed: {category_elapsed:.1f}s ({category_elapsed/60:.1f}min)")
    logger.info(f"[{category}] Retry statistics: {items_with_retry} items retried, {total_retries} total attempts")
    if failed_items:
        logger.warning(f"[{category}] {len(failed_items)} combinations failed in this category")
    else:
        logger.info(f"[{category}] All {total_combinations} combinations succeeded in this category")
    
    
    if large_word_diff_items:
        logger.info(f"[{category}] Items with word-count difference greater than 30 ({len(large_word_diff_items)} items):")
        for item in large_word_diff_items:
            logger.info(
                f"  - brand[{item['brand_index']}]={item['brand']}/{item['model']} <- "
                f"doc[{item['doc_index']}]={item['original_brand']}/{item['original_model']} | "
                f"original: {item['original_words']} words, rewritten: {item['rewritten_words']} words, "
                f"diff: {item['word_diff']} words"
            )
    
    return combinations, failed_items, total_retries, items_with_retry, large_word_diff_items


def save_metadata(
    category: str,
    model_name: str,
    rewriter_model: t.Optional[str],
    all_brands: t.List[BrandInfo],
    original_docs: t.List[DocumentInfo],
    output_dir: Path,
    extra_metadata: t.Optional[t.Dict[str, t.Any]] = None,
):
    metadata = {
        "category": category,
        "model_name": model_name,
        "rewriter_model": rewriter_model,
        "num_brands": len(all_brands),
        "num_docs": len(original_docs),
        "total_combinations": len(all_brands) * len(original_docs),
        "brands": [
            {
                "brand_index": b.brand_index,
                "brand": b.brand,
                "model": b.model,
                "knowledge_strength": b.knowledge_strength,
                "is_fictional": b.is_fictional,
                "rank": b.rank,
            }
            for b in all_brands
        ],
        "documents": [
            {
                "doc_index": d.doc_index,
                "original_brand": d.original_brand,
                "original_model": d.original_model
            }
            for d in original_docs
        ]
    }

    if extra_metadata:
        metadata.update(extra_metadata)
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)




def load_combinations(
    category: str,
    model_name: str,
    with_ranking: bool = True,
    num_brands: int = PAPER_NUM_BRANDS,
    output_dir: str = "./out/brand_doc_combinations"
) -> t.Tuple[t.List[BrandInfo], t.List[t.List[str]]]:
    brands_suffix = f"top{num_brands}"
    category_dir = Path(output_dir) / model_name / brands_suffix / category
    metadata_file = category_dir / "metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist: {metadata_file}\n"
            f"Run first: python build_brand_doc_combinations.py --model {model_name}"
        )
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    
    brands = [
        BrandInfo(
            brand_index=b["brand_index"],
            brand=b["brand"],
            model=b["model"],
            knowledge_strength=b["knowledge_strength"],
            is_fictional=b["is_fictional"],
            rank=b.get("rank"),
        )
        for b in metadata["brands"]
    ]
    
    
    num_docs = metadata["num_docs"]
    docs = []
    
    for brand in brands:
        brand_docs = []
        for doc_idx in range(num_docs):
            doc_file = category_dir / str(brand.brand_index) / f"{doc_idx}.txt"
            with open(doc_file, "r", encoding="utf-8") as f:
                brand_docs.append(f.read())
        docs.append(brand_docs)
    
    return brands, docs


def load_combinations_as_products(
    category: str,
    model_name: str,
    with_ranking: bool = True,
    num_brands: int = PAPER_NUM_BRANDS,
    output_dir: str = "./out/brand_doc_combinations"
) -> t.Tuple[t.List[Product], t.List[t.List[str]]]:
    brands, docs = load_combinations(category, model_name, with_ranking, num_brands, output_dir)
    
    products = [
        Product(category=category, brand=b.brand, model=b.model)
        for b in brands
    ]
    
    return products, docs




def main():
    parser = argparse.ArgumentParser(
        description="Build brand-document combinations (step 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(Models.keys()),
        help="Model name used to extract parametric knowledge (the model used in step 1)"
    )
    parser.add_argument(
        "--num-brands", type=int, default=PAPER_NUM_BRANDS, choices=[PAPER_NUM_BRANDS],
        help="Number of parametric brands; this script only keeps the paper setting with 8 parametric brands plus 8 fictional brands"
    )
    
    
    parser.add_argument(
        "--output-dir", type=str, default="./out/brand_doc_combinations",
        help="Output directory"
    )
    parser.add_argument(
        "--parametric-knowledge-dir", type=str, default="./out/parametric_knowledge",
        help="Parametric knowledge output directory (step 1 output)"
    )
    parser.add_argument(
        "--generic-doc-base-dir", type=str, default="./out/generic_product_documents",
        help="Generic template document output directory"
    )
    parser.add_argument(
        "--fictional-brands-dir", type=str, default="./out/fictional_brands",
        help="Fictional brand output directory"
    )
    
    
    parser.add_argument(
        "--no-ranking", action="store_true",
        help="Do not use ranked parametric knowledge results (ranked results are used by default)"
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help="Test a single category"
    )
    parser.add_argument(
        "--doc-index", type=int, default=None,
        help="Process only the specified document index. Must be used with --test."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted task by skipping existing files"
    )
    parser.add_argument(
        "--concurrency", type=int, default=64,
        help="Concurrency (default: 64)"
    )
    
    
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Log file path. By default, a timestamped filename is generated automatically"
    )
    
    args = parser.parse_args()
    
    
    if args.doc_index is not None and args.test is None:
        parser.error("--doc-index must be used with --test")
    
    if args.doc_index is not None and args.doc_index < 0:
        parser.error("--doc-index must be a non-negative integer")

    if args.num_brands != PAPER_NUM_BRANDS:
        parser.error(f"This script only supports --num-brands {PAPER_NUM_BRANDS}")
    
    
    with_ranking = not args.no_ranking

    
    
    if args.log_file is None:
        default_log_dir = Path(args.output_dir) / args.model
        default_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = str(default_log_dir / f"build_combinations_{timestamp}.log")
    else:
        default_log_file = args.log_file
    log_file = setup_logging(log_file=default_log_file)
    
    if args.test:
        
        categories = [args.test]
    else:
        
        categories = dataset.get_categories()
    
    
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    print(f"{'='*60}")
    print(f"Brand-document combination builder")
    print(f"{'='*60}")
    print(f"Start time: {start_time_str}")
    print(f"Parametric knowledge model: {args.model}")
    print("Combination mode: direct generic-template substitution")
    print(f"Number of parametric brands: {args.num_brands} ({args.num_brands} parametric + {args.num_brands} fictional)")
    print(f"Generic document directory: {args.generic_doc_base_dir}")
    print(f"Fictional brand directory: {args.fictional_brands_dir}")
    print(f"Number of categories: {len(categories)}")
    print(f"{'='*60}")
    
    
    logger.info(f"{'='*60}")
    logger.info(f"Brand-document combination builder")
    logger.info(f"{'='*60}")
    logger.info(f"Command-line arguments:")
    logger.info(f"  --model: {args.model}")
    logger.info(f"  --num-brands: {args.num_brands}")
    logger.info(f"  --output-dir: {args.output_dir}")
    logger.info(f"  --parametric-knowledge-dir: {args.parametric_knowledge_dir}")
    logger.info(f"  --generic-doc-base-dir: {args.generic_doc_base_dir}")
    logger.info(f"  --fictional-brands-dir: {args.fictional_brands_dir}")
    logger.info(f"  --with-ranking: {with_ranking}")
    logger.info(f"  --resume: {args.resume}")
    logger.info(f"  --concurrency: {args.concurrency}")
    logger.info(f"  --log-file: {log_file}")
    if args.test:
        logger.info(f"  --test: {args.test}")
    if args.doc_index is not None:
        logger.info(f"  --doc-index: {args.doc_index}")
    logger.info(f"{'='*60}")
    logger.info(f"Parametric knowledge model: {args.model}")
    logger.info(f"Number of parametric brands: {args.num_brands}")
    logger.info("Mode: 8 parametric brands + 8 fictional brands (generic template substitution)")
    logger.info(f"Number of categories: {len(categories)}")
    logger.info(f"{'='*60}")
    
    
    total_start_time = time.time()
    task_count = 0
    total_combination_count = 0
    failed_categories = []  
    all_failed_items = []   
    all_total_retries = 0   
    all_items_with_retry = 0  
    all_large_word_diff_items = []  
    operation_label = "Substitution"
    
    for category in categories:
        try:
            combinations, failed_items, cat_retries, cat_items_with_retry, large_word_diff_items = build_combinations_for_category(
                category=category,
                model_name=args.model,
                num_brands=args.num_brands,
                output_dir=args.output_dir,
                parametric_knowledge_dir=args.parametric_knowledge_dir,
                generic_doc_base_dir=args.generic_doc_base_dir,
                fictional_brands_dir=args.fictional_brands_dir,
                with_ranking=with_ranking,
                resume=args.resume,
                doc_index=args.doc_index,
                concurrency=args.concurrency,
                log_file=log_file
            )
            task_count += 1
            total_combination_count += len(combinations)
            all_total_retries += cat_retries
            all_items_with_retry += cat_items_with_retry
            
            for item in failed_items:
                item['category'] = category
                all_failed_items.append(item)
            
            for item in large_word_diff_items:
                item['category'] = category
                all_large_word_diff_items.append(item)
        except Exception as e:
            logger.error(f"Failed to process category '{category}': {e}", exc_info=True)
            failed_categories.append((category, str(e)))
            continue
    
    
    total_elapsed = time.time() - total_start_time
    avg_time = total_elapsed / task_count if task_count > 0 else 0
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    success_combination_count = total_combination_count - len(all_failed_items)
    
    
    retry_info = ""
    if all_items_with_retry > 0:
        retry_info = f", retried {all_items_with_retry} items for {all_total_retries} total attempts"
    
    
    print(f"\n{'='*60}")
    print(f"All done.")
    print(f"{'='*60}")
    print(f"End time: {end_time_str}")
    print(f"Categories: succeeded {task_count}/{task_count + len(failed_categories)}")
    print(f"{operation_label}: succeeded {success_combination_count}/{total_combination_count}{retry_info}")
    print(f"Total elapsed: {total_elapsed/60:.1f}min ({total_elapsed/3600:.1f}h)")
    if failed_categories:
        print(f"Failed categories ({len(failed_categories)}):")
        for cat, err in failed_categories:
            print(f"  - {cat}: {err[:100]}")
    print(f"{'='*60}")
    
    
    logger.info(f"{'='*60}")
    logger.info(f"All done.")
    logger.info(f"  Categories: succeeded {task_count}, failed {len(failed_categories)}")
    logger.info(
        f"  {operation_label}: succeeded {success_combination_count}, "
        f"failed {len(all_failed_items)}, total {total_combination_count}"
    )
    logger.info(f"  Retry statistics: {all_items_with_retry} items retried, {all_total_retries} total attempts")
    logger.info(f"  Elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min), average: {avg_time:.1f}s/category")
    
    
    if failed_categories:
        logger.warning(f"Failed categories ({len(failed_categories)}):")
        for cat, err in failed_categories:
            logger.warning(f"  - {cat}: {err}")
    
    
    if all_failed_items:
        logger.warning(f"Failed rewritten items ({len(all_failed_items)} items):")
        for item in all_failed_items:
            content_preview = item['original_content'][:100] + "..." if len(item['original_content']) > 100 else item['original_content']
            logger.warning(f"  - [{item['category']}] brand[{item['brand_index']}]={item['brand']}, doc[{item['doc_index']}] ({item['original_brand']} {item['original_model']}), retries {item.get('retry_count', 0)}")
            logger.warning(f"    Error: {item['error']}")
            logger.debug(f"    Original: {content_preview}")
    
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

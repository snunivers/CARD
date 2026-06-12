



import argparse
import atexit
import json
import logging
import os
import re
import sys
import typing as t
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from _types import Message, Role, Parameters
from models import load_model, Models, _load_local_model, ResponseWithProbs, TokenProbInfo
from prompts import get_system_prompt_for_parametric_knowledge, get_user_prompt_for_parametric_knowledge
import dataset




_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_ACTIVE_STDOUT_TEE = None
_ACTIVE_STDERR_TEE = None
_ACTIVE_CONSOLE_LOG_FILE: t.Optional[Path] = None


def get_results_save_dir(output_dir: str, model_name: str, num_brands: int) -> Path:
    return Path(output_dir) / model_name / f"top{num_brands}"


class TeeStream:

    def __init__(self, original_stream, log_file: Path):
        self.original_stream = original_stream
        self.log_handle = open(log_file, "a", encoding="utf-8")
        self._log_closed = False

    def write(self, data: str) -> int:
        if not data:
            return 0
        self.original_stream.write(data)
        self.original_stream.flush()
        if not self._log_closed and not self.log_handle.closed:
            try:
                self.log_handle.write(data)
                self.log_handle.flush()
            except ValueError:
                
                self._log_closed = True
        return len(data)

    def flush(self):
        self.original_stream.flush()
        if not self._log_closed and not self.log_handle.closed:
            try:
                self.log_handle.flush()
            except ValueError:
                self._log_closed = True

    def isatty(self) -> bool:
        return self.original_stream.isatty()

    def fileno(self) -> int:
        original_fileno = getattr(self.original_stream, "fileno", None)
        if callable(original_fileno):
            return original_fileno()
        return self.log_handle.fileno()

    def __getattr__(self, name: str):
        return getattr(self.original_stream, name)

    def close(self):
        if self._log_closed or self.log_handle.closed:
            return
        self.log_handle.close()
        self._log_closed = True


def restore_console_capture():
    global _ACTIVE_STDOUT_TEE, _ACTIVE_STDERR_TEE, _ACTIVE_CONSOLE_LOG_FILE

    if _ACTIVE_STDOUT_TEE is not None and sys.stdout is _ACTIVE_STDOUT_TEE:
        sys.stdout = _ORIGINAL_STDOUT
    if _ACTIVE_STDERR_TEE is not None and sys.stderr is _ACTIVE_STDERR_TEE:
        sys.stderr = _ORIGINAL_STDERR

    if _ACTIVE_STDOUT_TEE is not None:
        _ACTIVE_STDOUT_TEE.close()
    if _ACTIVE_STDERR_TEE is not None:
        _ACTIVE_STDERR_TEE.close()

    _ACTIVE_STDOUT_TEE = None
    _ACTIVE_STDERR_TEE = None
    _ACTIVE_CONSOLE_LOG_FILE = None


def enable_console_capture(log_file: Path):
    global _ACTIVE_STDOUT_TEE, _ACTIVE_STDERR_TEE, _ACTIVE_CONSOLE_LOG_FILE

    if _ACTIVE_CONSOLE_LOG_FILE == log_file and _ACTIVE_STDOUT_TEE is not None and _ACTIVE_STDERR_TEE is not None:
        return

    restore_console_capture()

    _ACTIVE_STDOUT_TEE = TeeStream(_ORIGINAL_STDOUT, log_file)
    _ACTIVE_STDERR_TEE = TeeStream(_ORIGINAL_STDERR, log_file)
    sys.stdout = _ACTIVE_STDOUT_TEE
    sys.stderr = _ACTIVE_STDERR_TEE
    _ACTIVE_CONSOLE_LOG_FILE = log_file


atexit.register(restore_console_capture)


def setup_logging(
    output_dir: str,
    model_name: str,
    num_brands: int,
    with_ranking: bool = True
) -> logging.Logger:
    save_dir = get_results_save_dir(output_dir, model_name, num_brands)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f"extract_parametric_knowledge_{timestamp}.log"
    enable_console_capture(log_file)

    logger = logging.getLogger("parametric_knowledge")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.handlers.clear()

    console_handler = logging.StreamHandler(_ORIGINAL_STDERR)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("parametric_knowledge")


OPENROUTER_MODELS_WITHOUT_LOGPROBS = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gemini-3.1-flash-lite",
    "google/gemini-3.1-flash-lite",
    "gpt-5-mini",
    "openai/gpt-5-mini",
}


def should_return_probs_for_model(model_name: str) -> bool:
    return model_name not in OPENROUTER_MODELS_WITHOUT_LOGPROBS




def check_anomalies(
    result: 'BrandProductKnowledge',
    category: str
) -> t.List[str]:
    warnings = []
    logger = get_logger()
    
    
    if result.brand_knowledge_strength == 0:
        msg = f"[{category}] Brand '{result.brand}' has 0 knowledge strength - no tokens matched"
        warnings.append(msg)
        logger.warning(msg)
    
    if result.model_knowledge_strength == 0:
        msg = f"[{category}] Model '{result.model}' has 0 knowledge strength - no tokens matched"
        warnings.append(msg)
        logger.warning(msg)
    
    if result.overall_knowledge_strength == 0:
        msg = f"[{category}] Product '{result.brand} - {result.model}' has 0 overall knowledge strength"
        warnings.append(msg)
        logger.warning(msg)
    
    
    if not result.brand_tokens:
        msg = f"[{category}] Brand '{result.brand}' matched no tokens"
        warnings.append(msg)
        logger.warning(msg)
    
    if not result.model_tokens:
        msg = f"[{category}] Model '{result.model}' matched no tokens"
        warnings.append(msg)
        logger.warning(msg)
    
    
    for prob in result.brand_first_token_probs + result.model_first_token_probs:
        if prob < 0 or prob > 1:
            msg = f"[{category}] Invalid probability value {prob} for '{result.brand} - {result.model}'"
            warnings.append(msg)
            logger.error(msg)
    
    return warnings




@dataclass
class BrandProductKnowledge:
    rank: t.Optional[int]  
    rank_token_prob: t.Optional[float]  
    brand: str  
    model: str  
    brand_tokens: t.List[str]  
    model_tokens: t.List[str]  
    brand_first_token_probs: t.List[float]  
    model_first_token_probs: t.List[float]  
    brand_all_token_probs: t.List[float]  
    model_all_token_probs: t.List[float]  
    brand_knowledge_strength: float  
    model_knowledge_strength: float  
    overall_knowledge_strength: float  
    low_prob_token_indices: t.List[t.Tuple[int, float]]  


@dataclass
class CategoryKnowledgeResult:
    category: str
    with_ranking: bool
    raw_response: str
    token_probs: t.List[TokenProbInfo]
    products: t.List[BrandProductKnowledge]




def is_ignorable_token(token: str) -> bool:
    
    stripped = token.strip()
    if not stripped:
        return True
    
    
    if re.match(r'^[\s\.\,\!\?\;\:\-\—\–\'\"\(\)\[\]\{\}\/\\]+$', stripped):
        return True
    
    
    if re.match(r'^\d+\.?$', stripped):
        return True
    
    return False


def is_word_start_token(token: str, prev_token: t.Optional[str]) -> bool:
    if prev_token is None:
        return True
    
    
    if token.startswith(' ') or token.startswith('\n'):
        return True
    
    
    if prev_token and (prev_token.endswith(' ') or prev_token.endswith('\n') 
                       or prev_token.endswith('-') or prev_token.endswith('.')):
        return True
    
    return False


def extract_word_first_token_probs(
    text: str,
    token_probs: t.List[TokenProbInfo],
    start_idx: int = 0
) -> t.Tuple[t.List[str], t.List[float], int]:
    tokens = []
    first_token_probs = []
    
    
    text = text.strip()
    if not text:
        return [], [], start_idx
    
    
    
    current_idx = start_idx
    matched_text = ""
    
    
    while current_idx < len(token_probs):
        token_text = token_probs[current_idx]["token"]
        if is_ignorable_token(token_text) and not token_text.strip():
            current_idx += 1
        else:
            break
    
    
    prev_token = None
    current_word_started = False
    
    while current_idx < len(token_probs) and len(matched_text.replace(' ', '').replace('-', '')) < len(text.replace(' ', '').replace('-', '')):
        token_info = token_probs[current_idx]
        token_text = token_info["token"]
        token_prob = token_info["probability"]
        
        tokens.append(token_text)
        
        
        if is_word_start_token(token_text, prev_token):
            
            stripped = token_text.strip()
            if stripped and not re.match(r'^[\.\,\!\?\;\:\-\—\–]+$', stripped):
                first_token_probs.append(token_prob)
                current_word_started = True
        
        matched_text += token_text
        prev_token = token_text
        current_idx += 1
    
    return tokens, first_token_probs, current_idx


def calculate_knowledge_strength(first_token_probs: t.List[float]) -> float:
    if not first_token_probs:
        return 0.0
    return sum(first_token_probs) / len(first_token_probs)


def calculate_new_knowledge_strength(
    brand_first_token_prob: float,
    brand_all_token_probs: t.List[float],
    model_all_token_probs: t.List[float],
    threshold: float = 0.90
) -> t.Tuple[float, t.List[t.Tuple[int, float]]]:
    
    all_probs = brand_all_token_probs + model_all_token_probs
    
    
    low_prob_values = [brand_first_token_prob]
    low_prob_indices = [(0, brand_first_token_prob)]
    
    
    for i, prob in enumerate(all_probs[1:], start=1):
        if prob < threshold:
            low_prob_values.append(prob)
            low_prob_indices.append((i, prob))
    
    
    if low_prob_values:
        new_strength = sum(low_prob_values) / len(low_prob_values)
    else:
        new_strength = brand_first_token_prob
    
    return new_strength, low_prob_indices




def clean_text(text: str) -> str:
    
    text = re.sub(r'[\[\]]', '', text)
    
    
    text = re.sub(r'^(Brand|Model):\s*', '', text, flags=re.IGNORECASE)
    
    
    text = text.strip()
    
    return text


def parse_ranked_response(response: str) -> t.List[t.Tuple[int, str, str]]:
    logger = get_logger()
    results = []
    lines = response.strip().split('\n')
    
    
    patterns = [
        
        (r'^(\d+)[\.\)]\s*Brand:\s*\[?(.+?)\]?\s*\|\|\s*Model:\s*\[?(.+?)\]?$', "full format (||)"),
        
        
        (r'^(\d+)[\.\)]\s*Brand:\s*\[?(.+?)\]?\s*\|\|\s*\[?(.+?)\]?$', "Brand label (||)"),
        
        
        (r'^(\d+)[\.\)]\s*\[?(.+?)\]?\s*\|\|\s*Model:\s*\[?(.+?)\]?$', "Model label (||)"),
        
        
        (r'^(\d+)[\.\)]\s*\[?(.+?)\]?\s*\|\|\s*\[?(.+?)\]?$', "simplified format (||)"),
        
        
        (r'^(\d+)[\.\)]\s*Brand:\s*\[?(.+?)\]?\s*-\s*Model:\s*\[?(.+?)\]?$', "full format (-)"),
        
        
        (r'^(\d+)[\.\)]\s*\[?(.+?)\]?\s*-\s*\[?(.+?)\]?$', "simplified format (-)"),
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        
        matched = False
        for pattern, pattern_name in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                rank = int(match.group(1))
                brand = match.group(2).strip()
                model = match.group(3).strip()
                
                
                brand = clean_text(brand)
                model = clean_text(model)
                
                results.append((rank, brand, model))
                logger.debug(f"Parsed successfully [{pattern_name}]: {rank}. {brand} || {model}")
                matched = True
                break
        
        if not matched:
            
            logger.warning(f"Could not parse line: {line}")
    
    return results


def parse_unranked_response(response: str) -> t.List[t.Tuple[None, str, str]]:
    logger = get_logger()
    results = []
    lines = response.strip().split('\n')
    
    
    patterns = [
        
        (r'^Brand:\s*\[?(.+?)\]?\s*\|\|\s*Model:\s*\[?(.+?)\]?$', "full format (||)"),
        
        
        (r'^Brand:\s*\[?(.+?)\]?\s*\|\|\s*\[?(.+?)\]?$', "Brand label (||)"),
        
        
        (r'^\[?(.+?)\]?\s*\|\|\s*Model:\s*\[?(.+?)\]?$', "Model label (||)"),
        
        
        (r'^\[?(.+?)\]?\s*\|\|\s*\[?(.+?)\]?$', "simplified format (||)"),
        
        
        (r'^Brand:\s*\[?(.+?)\]?\s*-\s*Model:\s*\[?(.+?)\]?$', "full format (-)"),
        
        
        (r'^\[?(.+?)\]?\s*-\s*\[?(.+?)\]?$', "simplified format (-)"),
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        
        
        matched = False
        for pattern, pattern_name in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                brand = match.group(1).strip()
                model = match.group(2).strip()
                
                
                brand = clean_text(brand)
                model = clean_text(model)
                
                results.append((None, brand, model))
                logger.debug(f"Parsed successfully [{pattern_name}]: {brand} || {model}")
                matched = True
                break
        
        if not matched:
            
            logger.warning(f"Could not parse line: {line}")
    
    return results




def build_token_char_mapping(token_probs: t.List[TokenProbInfo]) -> t.List[dict]:
    mapping = []
    current_pos = 0
    
    for token_info in token_probs:
        token_text = token_info["token"]
        start = current_pos
        end = current_pos + len(token_text)
        mapping.append({
            "token": token_text,
            "probability": token_info["probability"],
            "token_id": token_info.get("token_id"),
            "char_start": start,
            "char_end": end
        })
        current_pos = end
    
    return mapping


def find_tokens_for_text(
    text_to_find: str,
    full_text: str,
    token_mapping: t.List[dict],
    search_start: int = 0
) -> t.Tuple[t.List[str], t.List[float], t.List[float]]:
    
    text_start = full_text.find(text_to_find, search_start)
    if text_start == -1:
        
        text_start = full_text.lower().find(text_to_find.lower(), search_start)
        if text_start == -1:
            return [], [], []
    
    text_end = text_start + len(text_to_find)
    
    
    matched_tokens = []
    for t in token_mapping:
        
        if t["char_start"] < text_end and t["char_end"] > text_start:
            matched_tokens.append(t)
    
    
    all_token_probs = [t["probability"] for t in matched_tokens]
    
    
    words = text_to_find.split()
    first_token_probs = []
    
    
    word_search_start = text_start
    for word in words:
        
        word_pos = full_text.find(word, word_search_start)
        if word_pos == -1:
            
            word_pos_lower = full_text.lower().find(word.lower(), word_search_start)
            if word_pos_lower != -1:
                word_pos = word_pos_lower
            else:
                continue
        
        word_end = word_pos + len(word)
        
        
        for t in token_mapping:
            
            if t["char_start"] <= word_pos < t["char_end"]:
                first_token_probs.append(t["probability"])
                break
        
        word_search_start = word_end
    
    token_texts = [t["token"] for t in matched_tokens]
    return token_texts, first_token_probs, all_token_probs


def match_tokens_to_products(
    parsed_products: t.List[t.Tuple[t.Optional[int], str, str]],
    token_probs: t.List[TokenProbInfo],
    with_ranking: bool
) -> t.List[BrandProductKnowledge]:
    
    token_mapping = build_token_char_mapping(token_probs)
    full_text = "".join([t["token"] for t in token_probs])
    
    results = []
    search_start = 0  
    
    for rank, brand, model in parsed_products:
        
        rank_token_prob = None
        if rank is not None and with_ranking:
            rank_str = str(rank)
            
            rank_pos = full_text.find(rank_str, search_start)
            if rank_pos != -1:
                
                for t in token_mapping:
                    if t["char_start"] <= rank_pos < t["char_end"]:
                        rank_token_prob = t["probability"]
                        break
                
                search_start = rank_pos + len(rank_str)
        
        
        brand_tokens, brand_first_probs, brand_all_probs = find_tokens_for_text(
            brand, full_text, token_mapping, search_start
        )
        
        
        brand_pos = full_text.find(brand, search_start)
        if brand_pos == -1:
            brand_pos = full_text.lower().find(brand.lower(), search_start)
        if brand_pos != -1:
            search_start = brand_pos + len(brand)
        
        
        model_tokens, model_first_probs, model_all_probs = find_tokens_for_text(
            model, full_text, token_mapping, search_start
        )
        
        
        model_pos = full_text.find(model, search_start)
        if model_pos == -1:
            model_pos = full_text.lower().find(model.lower(), search_start)
        if model_pos != -1:
            search_start = model_pos + len(model)
        
        
        brand_strength = calculate_knowledge_strength(brand_first_probs)
        model_strength = calculate_knowledge_strength(model_first_probs)
        
        
        brand_first_prob = brand_first_probs[0] if brand_first_probs else 0.0
        overall_strength, low_prob_indices = calculate_new_knowledge_strength(
            brand_first_prob,
            brand_all_probs,
            model_all_probs,
            threshold=0.90
        )
        
        results.append(BrandProductKnowledge(
            rank=rank,
            rank_token_prob=rank_token_prob,
            brand=brand,
            model=model,
            brand_tokens=brand_tokens,
            model_tokens=model_tokens,
            brand_first_token_probs=brand_first_probs,
            model_first_token_probs=model_first_probs,
            brand_all_token_probs=brand_all_probs,
            model_all_token_probs=model_all_probs,
            brand_knowledge_strength=brand_strength,
            model_knowledge_strength=model_strength,
            overall_knowledge_strength=overall_strength,
            low_prob_token_indices=low_prob_indices
        ))
    
    return results




def extract_parametric_knowledge_batch(
    model_name: str,
    categories: t.List[str],
    with_ranking: bool = True,
    num_brands: int = 4,
    gpu_ids: t.Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    batch_size: int = 32,
    verbose: bool = True,
    local_inference_backend: str = "vllm",
    vllm_gpu_memory_utilization: float = 0.6,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
) -> t.Dict[str, CategoryKnowledgeResult]:
    from models import is_remote_model

    request_probs = should_return_probs_for_model(model_name)
    if not request_probs:
        print(
            f"\n⚠️  {model_name} does not support logprobs; return_probs has been disabled automatically. "
            "Token probabilities and knowledge strength will be unavailable."
        )
    
    
    
    chat_func = load_model(
        model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        gpu_ids=gpu_ids,
        return_probs=request_probs,
        local_inference_backend=local_inference_backend,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
    )
    
    
    logger = get_logger()
    is_api_model = is_remote_model(model_name)
    
    if is_api_model:
        logger.info(f"Starting batch extraction for {len(categories)} categories (API model - dynamic concurrent processing)")
        logger.info(
            "API model selected; local inference backend settings are ignored: "
            f"local_inference_backend={local_inference_backend}, "
            f"vllm_gpu_memory_utilization={vllm_gpu_memory_utilization}, "
            f"vllm_max_model_len={vllm_max_model_len}, "
            f"vllm_max_num_seqs={vllm_max_num_seqs}, "
            f"vllm_max_num_batched_tokens={vllm_max_num_batched_tokens}"
        )
        print(f"\n✅ API model uses batched concurrent calls with a dynamic thread count")
    else:
        logger.info(f"Starting batch extraction for {len(categories)} categories with batch_size={batch_size}")
        logger.info(
            "Local inference settings: "
            f"backend={local_inference_backend}, "
            f"vllm_gpu_memory_utilization={vllm_gpu_memory_utilization}, "
            f"vllm_max_model_len={vllm_max_model_len}, "
            f"vllm_max_num_seqs={vllm_max_num_seqs}, "
            f"vllm_max_num_batched_tokens={vllm_max_num_batched_tokens}"
        )
        if verbose:
            print(
                "\n✅ Local-model inference settings: "
                f"backend={local_inference_backend}, "
                f"vllm_gpu_memory_utilization={vllm_gpu_memory_utilization}, "
                f"vllm_max_model_len={vllm_max_model_len}, "
                f"vllm_max_num_seqs={vllm_max_num_seqs}, "
                f"vllm_max_num_batched_tokens={vllm_max_num_batched_tokens}"
            )
    
    
    system_prompt = get_system_prompt_for_parametric_knowledge()
    all_messages = []
    
    for category in categories:
        user_prompt = get_user_prompt_for_parametric_knowledge(category, with_ranking, num_brands)
        messages = [
            Message(role=Role.system, content=system_prompt),
            Message(role=Role.user, content=user_prompt)
        ]
        all_messages.append(messages)
        logger.info(f"[{category}] Prepared prompt")
    
    
    results = {}
    total_batches = (len(categories) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(categories), batch_size):
        batch_categories = categories[batch_idx:batch_idx + batch_size]
        batch_messages = all_messages[batch_idx:batch_idx + batch_size]
        
        current_batch_num = batch_idx // batch_size + 1
        logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch_categories)} categories)")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing batch {current_batch_num}/{total_batches}")
            print(f"Categories: {', '.join(batch_categories)}")
            if is_api_model:
                print(f"Concurrent threads: {len(batch_categories)}")
            print(f"{'='*60}")
        
        
        batch_responses: t.List[ResponseWithProbs] = chat_func(batch_messages)
        
        
        for i, category in enumerate(batch_categories):
            response = batch_responses[i]
            
            
            if isinstance(response, dict) and "message" in response:
                raw_response = response["message"].content
                token_probs = response.get("token_probs", [])
            else:
                
                raw_response = response.content
                token_probs = []
            
            
            _process_single_response(
                category, raw_response, token_probs, with_ranking, results, logger, verbose
            )
    
    return results


def _process_single_response(
    category: str,
    raw_response: str,
    token_probs: t.List[TokenProbInfo],
    with_ranking: bool,
    results: t.Dict[str, CategoryKnowledgeResult],
    logger: logging.Logger,
    verbose: bool
):
    pass
            
def _process_single_response(
    category: str,
    raw_response: str,
    token_probs: t.List[TokenProbInfo],
    with_ranking: bool,
    results: t.Dict[str, CategoryKnowledgeResult],
    logger: logging.Logger,
    verbose: bool
):
    
    logger.info(f"[{category}] === START EXTRACTION ===")
    logger.info(f"[{category}] Raw response:\n{raw_response}")
    
    
    logger.debug(f"[{category}] Token probabilities ({len(token_probs)} tokens):")
    for tp in token_probs:
        logger.debug(f"  {tp['token']!r:20} -> {tp['probability']:.6f} (id={tp.get('token_id', 'N/A')})")
    
    if verbose:
        print(f"\n[{category}] Raw response:\n{raw_response}")
    
    
    if with_ranking:
        parsed = parse_ranked_response(raw_response)
    else:
        parsed = parse_unranked_response(raw_response)
    
    
    logger.info(f"[{category}] Parsed {len(parsed)} products:")
    for item in parsed:
        if with_ranking:
            logger.info(f"  {item[0]}. {item[1]} - {item[2]}")
        else:
            logger.info(f"  {item[1]} - {item[2]}")
    
    
    products = match_tokens_to_products(parsed, token_probs, with_ranking)
    
    
    logger.info(f"[{category}] Knowledge strength results:")
    for p in products:
        logger.info(f"  {p.brand} - {p.model}:")
        logger.info(f"    Brand knowledge strength: {p.brand_knowledge_strength:.4f}")
        logger.info(f"    Model knowledge strength: {p.model_knowledge_strength:.4f}")
        logger.info(f"    Overall knowledge strength: {p.overall_knowledge_strength:.4f}")
    
    
    all_warnings = []
    for p in products:
        warnings = check_anomalies(p, category)
        all_warnings.extend(warnings)
    
    if all_warnings:
        logger.warning(f"[{category}] Found {len(all_warnings)} anomalies")
    else:
        logger.info(f"[{category}] No anomalies detected")
    
    logger.info(f"[{category}] === END EXTRACTION ===")
    
    
    results[category] = CategoryKnowledgeResult(
        category=category,
        with_ranking=with_ranking,
        raw_response=raw_response,
        token_probs=token_probs,
        products=products
    )



def extract_parametric_knowledge(
    model_name: str,
    category: str,
    with_ranking: bool = True,
    num_brands: int = 4,
    gpu_ids: t.Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    verbose: bool = True,
    local_inference_backend: str = "vllm",
    vllm_gpu_memory_utilization: float = 0.6,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
) -> CategoryKnowledgeResult:
    
    results = extract_parametric_knowledge_batch(
        model_name=model_name,
        categories=[category],
        with_ranking=with_ranking,
        num_brands=num_brands,
        gpu_ids=gpu_ids,
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=1,
        verbose=verbose,
        local_inference_backend=local_inference_backend,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
    )
    
    return results[category]


def validate_extraction_results(
    results: t.Dict[str, CategoryKnowledgeResult],
    expected_num_brands: int,
    logger: logging.Logger
):
    logger.info("\n" + "="*80)
    logger.info("VALIDATION REPORT - extraction result validation")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("📊 Extraction result validation report")
    print("="*80)
    
    total_categories = len(results)
    correct_count_categories = 0
    incorrect_count_categories = []
    failed_extractions = []
    
    for category, result in results.items():
        actual_count = len(result.products)
        
        
        if actual_count == expected_num_brands:
            correct_count_categories += 1
        else:
            incorrect_count_categories.append((category, actual_count))
            logger.warning(f"[{category}] Extracted count mismatch: expected {expected_num_brands}, got {actual_count}")
        
        
        for i, product in enumerate(result.products):
            issues = []
            
            
            if not product.brand or not product.brand.strip():
                issues.append("brand is empty")
            if not product.model or not product.model.strip():
                issues.append("model is empty")
            
            
            if not product.brand_tokens:
                issues.append("brand matched no tokens")
            if not product.model_tokens:
                issues.append("model matched no tokens")
            
            
            if product.brand_knowledge_strength == 0:
                issues.append("brand knowledge strength is 0")
            if product.model_knowledge_strength == 0:
                issues.append("model knowledge strength is 0")
            if product.overall_knowledge_strength == 0:
                issues.append("overall knowledge strength is 0")
            
            
            if issues:
                failed_extractions.append({
                    "category": category,
                    "rank": product.rank,
                    "brand": product.brand,
                    "model": product.model,
                    "issues": issues
                })
                logger.error(f"[{category}] Extraction failed - rank {product.rank}: {product.brand} - {product.model}")
                logger.error(f"  Issues: {', '.join(issues)}")
    
    
    logger.info(f"\nTotal categories: {total_categories}")
    logger.info(f"Categories with correct counts: {correct_count_categories} ({correct_count_categories/total_categories*100:.1f}%)")
    logger.info(f"Categories with incorrect counts: {len(incorrect_count_categories)} ({len(incorrect_count_categories)/total_categories*100:.1f}%)")
    
    print(f"\nTotal categories: {total_categories}")
    print(f"✅ Categories with correct counts: {correct_count_categories} ({correct_count_categories/total_categories*100:.1f}%)")
    print(f"❌ Categories with incorrect counts: {len(incorrect_count_categories)} ({len(incorrect_count_categories)/total_categories*100:.1f}%)")
    
    if incorrect_count_categories:
        logger.info("\nDetails for categories with incorrect counts:")
        print("\nDetails for categories with incorrect counts:")
        for cat, count in incorrect_count_categories:
            logger.info(f"  - {cat}: expected {expected_num_brands}, got {count}")
            print(f"  - {cat}: expected {expected_num_brands}, got {count}")
    
    logger.info(f"\nFailed extraction cases: {len(failed_extractions)}")
    print(f"\n⚠️  Failed extraction cases: {len(failed_extractions)}")
    
    if failed_extractions:
        logger.info("\nFailed extraction case details:")
        print("\nFailed extraction case details:")
        for failure in failed_extractions[:10]:  
            logger.info(f"  [{failure['category']}] rank {failure['rank']}: {failure['brand']} - {failure['model']}")
            logger.info(f"    Issues: {', '.join(failure['issues'])}")
            print(f"  [{failure['category']}] rank {failure['rank']}: {failure['brand']} - {failure['model']}")
            print(f"    Issues: {', '.join(failure['issues'])}")
        
        if len(failed_extractions) > 10:
            remaining = len(failed_extractions) - 10
            logger.info(f"  ... {remaining} more failed cases; see the log file for details")
            print(f"  ... {remaining} more failed cases; see the log file for details")
    else:
        logger.info("✅ All extractions succeeded; no failed cases")
        print("✅ All extractions succeeded; no failed cases")
    
    logger.info("="*80 + "\n")
    print("="*80 + "\n")


def extract_all_categories(
    model_name: str,
    with_ranking: bool = True,
    num_brands: int = 4,
    gpu_ids: t.Optional[str] = None,
    output_dir: str = "./out/parametric_knowledge",
    temperature: float = 0.0,
    max_tokens: int = 256,
    batch_size: int = 32,
    verbose: bool = True,
    local_inference_backend: str = "vllm",
    vllm_gpu_memory_utilization: float = 0.6,
    vllm_max_model_len: t.Optional[int] = None,
    vllm_max_num_seqs: t.Optional[int] = None,
    vllm_max_num_batched_tokens: t.Optional[int] = None,
) -> t.Dict[str, CategoryKnowledgeResult]:
    
    logger = get_logger()
    if not logger.handlers:
        logger = setup_logging(output_dir, model_name, num_brands, with_ranking)
    logger.info(f"Starting extraction for model: {model_name}")
    logger.info(
        "Parameters: "
        f"with_ranking={with_ranking}, "
        f"num_brands={num_brands}, "
        f"temperature={temperature}, "
        f"max_tokens={max_tokens}, "
        f"batch_size={batch_size}, "
        f"local_inference_backend={local_inference_backend}, "
        f"vllm_gpu_memory_utilization={vllm_gpu_memory_utilization}, "
        f"vllm_max_model_len={vllm_max_model_len}, "
        f"vllm_max_num_seqs={vllm_max_num_seqs}, "
        f"vllm_max_num_batched_tokens={vllm_max_num_batched_tokens}"
    )
    
    categories = dataset.get_categories()
    logger.info(f"Found {len(categories)} categories: {categories}")
    print(f"\nFound {len(categories)} categories: {categories}")
    print(f"Using batch inference with batch_size={batch_size}")
    
    
    results = extract_parametric_knowledge_batch(
        model_name=model_name,
        categories=categories,
        with_ranking=with_ranking,
        num_brands=num_brands,
        gpu_ids=gpu_ids,
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=batch_size,
        verbose=verbose,
        local_inference_backend=local_inference_backend,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=vllm_max_model_len,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_num_batched_tokens=vllm_max_num_batched_tokens,
    )
    
    
    total_anomalies = 0
    for category, result in results.items():
        for p in result.products:
            if p.overall_knowledge_strength == 0 or not p.brand_tokens or not p.model_tokens:
                total_anomalies += 1
    
    if total_anomalies > 0:
        logger.warning(f"Total anomalies found: {total_anomalies}")
        print(f"\n⚠️  Total anomalies found: {total_anomalies}. Check log file for details.")
    else:
        logger.info("No anomalies found in any category")
    
    logger.info(f"Extraction completed for {len(categories)} categories")
    
    
    validate_extraction_results(results, num_brands, logger)
    
    
    save_results(results, model_name, with_ranking, num_brands, output_dir)
    
    return results


def save_results(
    results: t.Dict[str, CategoryKnowledgeResult],
    model_name: str,
    with_ranking: bool,
    num_brands: int,
    output_dir: str
):
    save_dir = get_results_save_dir(output_dir, model_name, num_brands)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    detailed_results = {}
    for category, result in results.items():
        detailed_results[category] = {
            "category": result.category,
            "with_ranking": result.with_ranking,
            "raw_response": result.raw_response,
            "response_tokens": [tp["token"] for tp in result.token_probs],
            "response_token_ids": [tp.get("token_id") for tp in result.token_probs],
            "response_token_probs": [tp["probability"] for tp in result.token_probs],
            "products": [asdict(p) for p in result.products]
        }
    
    with open(save_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    
    summary_rows = []
    for category, result in results.items():
        for p in result.products:
            summary_rows.append({
                "category": category,
                "rank": p.rank,
                "brand": p.brand,
                "model": p.model,
                "brand_knowledge_strength": p.brand_knowledge_strength,
                "model_knowledge_strength": p.model_knowledge_strength,
                "overall_knowledge_strength": p.overall_knowledge_strength
            })
    
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    df.to_csv(save_dir / "summary.csv", index=False)
    
    logger = get_logger()
    logger.info(f"Results saved to: {save_dir}")
    logger.info(f"  - detailed_results.json: {len(results)} categories")
    logger.info(f"  - summary.csv: {len(summary_rows)} products")
    
    print(f"\nResults saved to: {save_dir}")


def print_summary(results: t.Dict[str, CategoryKnowledgeResult]):
    print("\n" + "=" * 80)
    print("SUMMARY OF PARAMETRIC KNOWLEDGE EXTRACTION")
    print("=" * 80)
    
    for category, result in results.items():
        print(f"\n📦 Category: {category}")
        print("-" * 40)
        for p in result.products:
            rank_str = f"{p.rank}. " if p.rank else "   "
            print(f"  {rank_str}{p.brand:15} - {p.model:25} | Strength: {p.overall_knowledge_strength:.4f}")




def add_extraction_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model", type=str, default="llama3.1-8b",
        choices=list(Models.keys()),
        help="Model to use for extraction"
    )
    parser.add_argument(
        "--no-ranking", action="store_true",
        help="Use unranked prompt (default: ranked)"
    )
    parser.add_argument(
        "--num-brands", type=int, default=4,
        help="Number of brands to extract (default: 4)"
    )
    parser.add_argument(
        "--gpu-ids", type=str, default=None,
        help="GPU IDs to use (e.g., '4,5,6,7'). Only required for local models, not needed for API models."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size: number of messages per batch. For local models: GPU batch inference. For API models: concurrent threads (default: 1)"
    )
    parser.add_argument(
        "--local-inference-backend",
        type=str,
        default="vllm",
        choices=["vllm"],
        help="Local-model inference backend (vLLM only; applies only to normal local inference)",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization cap for normal local vLLM inference (default: 0.6)",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="max_model_len for normal local vLLM inference (default: None, use the vLLM default)",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=None,
        help="max_num_seqs for normal local vLLM inference (default: None, use the vLLM default)",
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "max_num_batched_tokens for normal local vLLM inference "
            "(default: None, use the vLLM default)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="./out/parametric_knowledge",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help="Test with a single category (e.g., 'laptop'). If not specified, process all categories."
    )


def validate_extraction_backend_arguments(args: argparse.Namespace) -> None:
    if not (0.0 < float(args.vllm_gpu_memory_utilization) < 1.0):
        raise ValueError("--vllm-gpu-memory-utilization must be in the (0, 1) range")

    for arg_name in (
        "vllm_max_model_len",
        "vllm_max_num_seqs",
        "vllm_max_num_batched_tokens",
    ):
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None and int(arg_value) <= 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be greater than 0")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM parametric knowledge about product categories with knowledge strength measurement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    add_extraction_arguments(parser)
    args = parser.parse_args()
    validate_extraction_backend_arguments(args)
    
    
    with_ranking = not args.no_ranking
    logger = setup_logging(args.output_dir, args.model, args.num_brands, with_ranking)
    start_time = datetime.now()

    try:
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        
        from models import is_remote_model
        if is_remote_model(args.model):
            if args.gpu_ids is not None:
                print(f"⚠️  Warning: --gpu-ids is not needed for API model '{args.model}'. Ignoring GPU settings.")
            if args.local_inference_backend != "vllm":
                print(
                    f"⚠️  Warning: --local-inference-backend is ignored for API model "
                    f"'{args.model}'."
                )
            if (
                args.vllm_max_model_len is not None
                or args.vllm_max_num_seqs is not None
                or args.vllm_max_num_batched_tokens is not None
            ):
                print(
                    f"⚠️  Warning: vLLM tuning arguments are ignored for API model "
                    f"'{args.model}'."
                )
            gpu_ids = None
        else:
            if args.gpu_ids is None:
                print(f"⚠️  Warning: Local model '{args.model}' requires --gpu-ids. Using default GPU settings.")
                gpu_ids = None
            else:
                gpu_ids = args.gpu_ids
            print(f"Local inference backend: {args.local_inference_backend}")
            if args.local_inference_backend == "vllm":
                print(
                    "vLLM parameters: "
                    f"gpu_memory_utilization={args.vllm_gpu_memory_utilization}, "
                    f"max_model_len={args.vllm_max_model_len}, "
                    f"max_num_seqs={args.vllm_max_num_seqs}, "
                    f"max_num_batched_tokens={args.vllm_max_num_batched_tokens}"
                )

        if args.test:
            
            print(f"\n[TEST MODE] Testing with category: {args.test}")
            result = extract_parametric_knowledge(
                model_name=args.model,
                category=args.test,
                with_ranking=with_ranking,
                num_brands=args.num_brands,
                gpu_ids=gpu_ids,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                verbose=True,  
                local_inference_backend=args.local_inference_backend,
                vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                vllm_max_model_len=args.vllm_max_model_len,
                vllm_max_num_seqs=args.vllm_max_num_seqs,
                vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            )
            print_summary({args.test: result})
            
            
            validate_extraction_results({args.test: result}, args.num_brands, logger)
            
            
            anomaly_count = sum(1 for p in result.products 
                              if p.overall_knowledge_strength == 0 or not p.brand_tokens or not p.model_tokens)
            if anomaly_count > 0:
                print(f"\n⚠️  Found {anomaly_count} anomalies. Check log file for details.")
        else:
            
            results = extract_all_categories(
                model_name=args.model,
                with_ranking=with_ranking,
                num_brands=args.num_brands,
                gpu_ids=gpu_ids,
                output_dir=args.output_dir,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                verbose=not args.quiet,
                local_inference_backend=args.local_inference_backend,
                vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                vllm_max_model_len=args.vllm_max_model_len,
                vllm_max_num_seqs=args.vllm_max_num_seqs,
                vllm_max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            )
            print_summary(results)
    finally:
        end_time = datetime.now()
        print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

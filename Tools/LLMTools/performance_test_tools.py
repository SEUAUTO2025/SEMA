import json
import logging
import os
import re
from typing import Callable, Dict, List, Optional

import nltk
import requests
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score

try:
    from bert_score import score as bert_score_fn
except Exception:
    bert_score_fn = None

try:
    from pycocoevalcap.cider.cider import Cider
except Exception:
    Cider = None

try:
    from pycocoevalcap.spice.spice import Spice
except Exception:
    Spice = None


LOGGER = logging.getLogger(__name__)
_DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
_DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
_DEEPL_TEXT_CACHE = {}
_FALLBACK_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _load_single_csv_with_multipart_labels_runtime():
    from Tools.Exe_dataset.dataset_exe import load_single_csv_with_multipart_labels

    return load_single_csv_with_multipart_labels


def _normalize_eval_text(text: str) -> str:
    return str(text or "").replace("\n", " ").replace("\r", " ").strip()


def _tokenize_english_text(text: str) -> List[str]:
    normalized = _normalize_eval_text(text).lower()
    if not normalized:
        return []
    try:
        return word_tokenize(normalized)
    except LookupError:
        return _FALLBACK_TOKEN_PATTERN.findall(normalized)


def _safe_metric_call(metric_name: str, metric_fn: Callable, *args, **kwargs) -> float:
    try:
        value = metric_fn(*args, **kwargs)
        return float(value)
    except Exception as exc:
        LOGGER.warning("%s computation failed: %s", metric_name, exc)
        return float("nan")


def _build_coco_caption_payload(reference_text: str, candidate_text: str) -> tuple:
    reference = _normalize_eval_text(reference_text).lower()
    candidate = _normalize_eval_text(candidate_text).lower()
    return {"0": [reference]}, {"0": [candidate]}


def _resolve_deepl_endpoints(api_key: str) -> List[str]:
    api_key_text = str(api_key or "").strip()
    preferred = _DEEPL_FREE_URL if api_key_text.endswith(":fx") else _DEEPL_PRO_URL
    fallback = _DEEPL_PRO_URL if preferred == _DEEPL_FREE_URL else _DEEPL_FREE_URL
    return [preferred, fallback]


def translate_texts_to_english(
    texts: List[str],
    api_key: Optional[str] = None,
    target_lang: str = "EN-US",
    source_lang: Optional[str] = None,
    timeout: int = 60,
) -> List[str]:
    cleaned_texts = [_normalize_eval_text(text) for text in list(texts or [])]
    if len(cleaned_texts) == 0:
        return []

    results = ["" for _ in cleaned_texts]
    missing_indices = []
    source_lang_text = str(source_lang or "").strip().upper()
    target_lang_text = str(target_lang or "EN-US").strip().upper()

    for idx, text in enumerate(cleaned_texts):
        if not text:
            continue
        cache_key = (text, source_lang_text, target_lang_text)
        cached = _DEEPL_TEXT_CACHE.get(cache_key)
        if cached is not None:
            results[idx] = cached
        else:
            missing_indices.append(idx)

    if len(missing_indices) == 0:
        return results

    resolved_api_key = str(api_key or os.getenv("DEEPL_API_KEY", "")).strip()
    if not resolved_api_key:
        raise EnvironmentError("DEEPL_API_KEY is required for English translation.")

    payload = [("target_lang", target_lang_text), ("preserve_formatting", "1")]
    if source_lang_text:
        payload.append(("source_lang", source_lang_text))
    for idx in missing_indices:
        payload.append(("text", cleaned_texts[idx]))

    last_error = None
    headers = {"Authorization": "DeepL-Auth-Key {0}".format(resolved_api_key)}
    for endpoint in _resolve_deepl_endpoints(resolved_api_key):
        try:
            response = requests.post(endpoint, headers=headers, data=payload, timeout=int(timeout))
        except Exception as exc:
            last_error = exc
            continue

        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except Exception:
                error_payload = response.text
            last_error = RuntimeError(
                "DeepL translation failed at {0}: HTTP {1} {2}".format(
                    endpoint,
                    response.status_code,
                    error_payload,
                )
            )
            continue

        try:
            response_payload = response.json()
        except Exception as exc:
            last_error = RuntimeError("DeepL returned non-JSON response: {0}".format(exc))
            continue

        translations = response_payload.get("translations", [])
        if len(translations) != len(missing_indices):
            last_error = RuntimeError(
                "DeepL translation count mismatch: expected {0}, got {1}".format(
                    len(missing_indices),
                    len(translations),
                )
            )
            continue

        for offset, idx in enumerate(missing_indices):
            translated_text = _normalize_eval_text(translations[offset].get("text", ""))
            results[idx] = translated_text
            cache_key = (cleaned_texts[idx], source_lang_text, target_lang_text)
            _DEEPL_TEXT_CACHE[cache_key] = translated_text
        return results

    if last_error is None:
        raise RuntimeError("DeepL translation failed for unknown reasons.")
    raise last_error


def translate_text_to_english(
    text: str,
    api_key: Optional[str] = None,
    target_lang: str = "EN-US",
    source_lang: Optional[str] = None,
    timeout: int = 60,
) -> str:
    translated_list = translate_texts_to_english(
        texts=[text],
        api_key=api_key,
        target_lang=target_lang,
        source_lang=source_lang,
        timeout=timeout,
    )
    return translated_list[0] if len(translated_list) > 0 else ""

def calculate_BF1_score(reference_text: str, candidate_text: str, lang="zh"):
    """
    Calculate BERTScore (P, R, F1) for two texts
    :param reference_text: Reference text
    :param candidate_text: Generated text
    :param lang: Language setting, "zh" for Chinese, "en" for English
    :return: Dictionary containing F1, Precision, Recall
    """
    if bert_score_fn is None:
        raise ImportError("bert-score is required for BF1.")

    cands = [candidate_text]
    refs = [reference_text]

    P, R, F1 = bert_score_fn(cands, refs, lang=lang, verbose=False)

    return {
        "BF1 (Semantic Similarity)": F1.item(),
        "Precision": P.item(),
        "Recall": R.item()
    }


def calculate_bleu(reference_text: str, candidate_text: str):
    """
    Calculate BLEU scores for two pieces of English text
    :param reference_text: Reference text (Ground Truth)
    :param candidate_text: Model-generated text
    :return: Dictionary containing various BLEU metrics
    """
    ref_tokens = _tokenize_english_text(reference_text)
    cand_tokens = _tokenize_english_text(candidate_text)

    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {
            'BLEU-Total': 0.0,
            'BLEU-1 (Word-level)': 0.0,
            'BLEU-2 (Phrase-level)': 0.0,
            'BLEU-3 (Tri-gram)': 0.0,
            'BLEU-4 (Sentence-level)': 0.0
        }

    references = [ref_tokens]

    smooth = SmoothingFunction().method1

    weights = {
        'BLEU-Total': (0.25, 0.25, 0.25, 0.25),
        'BLEU-1 (Word-level)': (1, 0, 0, 0),
        'BLEU-2 (Phrase-level)': (0.5, 0.5, 0, 0),
        'BLEU-3 (Tri-gram)': (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0),
        'BLEU-4 (Sentence-level)': (0.25, 0.25, 0.25, 0.25)
    }

    results = {}
    for name, weight in weights.items():
        score = sentence_bleu(references, cand_tokens, weights=weight, smoothing_function=smooth)
        results[name] = score

    return results


def compute_cider_score(reference_text: str, candidate_text: str):
    """
    Compute CIDEr-D score for two English text strings
    :param reference_text: Ground Truth (Reference)
    :param candidate_text: Model Prediction (Candidate)
    :return: CIDEr-D mean score
    """
    if Cider is None:
        raise ImportError("pycocoevalcap is required for CIDEr.")

    gts, res = _build_coco_caption_payload(reference_text, candidate_text)
    cider_scorer = Cider()
    mean_score, _ = cider_scorer.compute_score(gts, res)
    return mean_score


def compute_meteor_score(reference_text: str, candidate_text: str):
    """
    Compute METEOR score for two English text strings
    :param reference_text: Ground Truth (Reference)
    :param candidate_text: Model Prediction (Candidate)
    :return: METEOR score (float)
    """
    ref_tokens = _tokenize_english_text(reference_text)
    cand_tokens = _tokenize_english_text(candidate_text)

    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0

    references = [ref_tokens]

    score = meteor_score(references, cand_tokens)

    return score


def _lcs_length(ref_tokens: List[str], cand_tokens: List[str]) -> int:
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0

    prev = [0] * (len(cand_tokens) + 1)
    for ref_token in ref_tokens:
        current = [0]
        for idx, cand_token in enumerate(cand_tokens, start=1):
            if ref_token == cand_token:
                current.append(prev[idx - 1] + 1)
            else:
                current.append(max(current[-1], prev[idx]))
        prev = current
    return int(prev[-1])


def compute_rouge_l_score(reference_text: str, candidate_text: str) -> float:
    ref_tokens = _tokenize_english_text(reference_text)
    cand_tokens = _tokenize_english_text(candidate_text)
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0

    lcs = float(_lcs_length(ref_tokens, cand_tokens))
    precision = lcs / float(len(cand_tokens))
    recall = lcs / float(len(ref_tokens))
    if precision <= 0.0 or recall <= 0.0:
        return 0.0

    beta = 1.2
    beta_sq = beta * beta
    denominator = recall + beta_sq * precision
    if denominator <= 0.0:
        return 0.0
    return float(((1.0 + beta_sq) * precision * recall) / denominator)


def compute_spice_score(reference_text: str, candidate_text: str) -> float:
    if Spice is None:
        raise ImportError("pycocoevalcap is required for SPICE.")

    gts, res = _build_coco_caption_payload(reference_text, candidate_text)
    spice_scorer = Spice()
    mean_score, _ = spice_scorer.compute_score(gts, res)
    return mean_score


def compute_all_semantic_metrics(
    reference_text: str,
    candidate_text: str,
    bf1_lang: str = "en",
    include_cider: bool = True,
    include_spice: bool = True,
) -> Dict[str, float]:
    bleu_scores = calculate_bleu(reference_text, candidate_text)
    try:
        bf1_scores = calculate_BF1_score(reference_text, candidate_text, lang=bf1_lang)
    except Exception as exc:
        LOGGER.warning("BF1 computation failed: %s", exc)
        bf1_scores = {
            "BF1 (Semantic Similarity)": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
        }

    results = {
        "bleu_total": float(bleu_scores.get("BLEU-Total", 0.0)),
        "bleu_1": float(bleu_scores.get("BLEU-1 (Word-level)", 0.0)),
        "bleu_2": float(bleu_scores.get("BLEU-2 (Phrase-level)", 0.0)),
        "bleu_3": float(bleu_scores.get("BLEU-3 (Tri-gram)", 0.0)),
        "bleu_4": float(bleu_scores.get("BLEU-4 (Sentence-level)", 0.0)),
        "meteor": _safe_metric_call("METEOR", compute_meteor_score, reference_text, candidate_text),
        "rouge_l": _safe_metric_call("ROUGE-L", compute_rouge_l_score, reference_text, candidate_text),
        "bf1": float(bf1_scores.get("BF1 (Semantic Similarity)", 0.0)),
        "bf1_precision": float(bf1_scores.get("Precision", 0.0)),
        "bf1_recall": float(bf1_scores.get("Recall", 0.0)),
    }
    if include_cider:
        results["cider"] = _safe_metric_call("CIDEr", compute_cider_score, reference_text, candidate_text)
    if include_spice:
        results["spice"] = _safe_metric_call("SPICE", compute_spice_score, reference_text, candidate_text)
    return results


def batch_translate_txt(input_dir, output_dir, auth_key):
    """
    Batch translate txt files to English using DeepL

    :param input_dir: Path to folder containing original txt files
    :param output_dir: Path to folder where translated files will be saved
    :param auth_key: Your DeepL API Authentication Key
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    if not files:
        print("No text files found in the input directory.")
        return

    print(f"Starting translation for {len(files)} files...")

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                merged_content = content.replace('\n', '').replace('\r', '')
                translated_text = translate_text_to_english(merged_content, api_key=auth_key)
                translated_text = translated_text.replace('\n', '').replace('\r', '')
            if not content:
                print(f"Skipping empty file: {file_name}")
                continue

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)

            print(f"Successfully translated: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("Batch translation completed.")


def get_matching_text(video_path, txt_dir, csv_dir):
    """
    Find matching txt file based on video path and extract content
    :param video_path: Full path to video file
    :param txt_dir: Directory containing text files
    :param csv_dir: Directory containing CSV files
    :return: Processed text content (newlines removed), returns None if file doesn't exist
    """
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    target_txt_path = os.path.join(txt_dir, f"{file_name}.txt")
    csv_path = os.path.join(csv_dir, f"{file_name}.csv")
    load_single_csv_with_multipart_labels = _load_single_csv_with_multipart_labels_runtime()
    _, label, label_total = load_single_csv_with_multipart_labels(csv_path, max_frames=124)

    if os.path.exists(target_txt_path):
        try:
            with open(target_txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                clean_content = content.replace('\n', '').replace('\r', '')
                return clean_content, label, label_total
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    else:
        print(f"Text file not found: {target_txt_path}")
        return None

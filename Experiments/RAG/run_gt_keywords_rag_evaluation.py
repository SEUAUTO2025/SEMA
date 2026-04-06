import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT_DIR)

DEFAULT_DATASET_ROOT = "dataset_ex"
DEFAULT_OUTPUT_ROOT = os.path.join("Experiments", "RAG", "evaluation", "results")
DEFAULT_LANGUAGE = "zh"
DEFAULT_RESPONSE_MODEL_NAME = "qwen-plus"
DEFAULT_KEYWORD_STAGE_PIPELINE = 4
DEFAULT_SEARCH_EMBEDDING_MODEL_NAME = "ali-text-embedding-v3"
DEFAULT_TOP_K = 34
DEFAULT_TRANSLATION_TARGET_LANG = "EN-US"
DEFAULT_SEMANTIC_METRIC_LANGUAGE = "en"
DEFAULT_SHOW_PROGRESS = True

def _load_rag_runtime() -> Dict[str, Any]:
    from RAG.Knowledge_Database.AI_dbmanager import KnowledgeDB
    from RAG.Knowledge_Database.AIdbconfig import session, session_en
    from RAG.Knowledge_Database.RAGFunc import (
        _get_llm_client_by_model,
        _get_keyword_polarity_library,
        _classify_comment_keywords,
        get_embedding,
        get_response,
    )

    return {
        "KnowledgeDB": KnowledgeDB,
        "session": session,
        "session_en": session_en,
        "get_llm_client_by_model": _get_llm_client_by_model,
        "get_keyword_polarity_library": _get_keyword_polarity_library,
        "classify_comment_keywords": _classify_comment_keywords,
        "get_embedding": get_embedding,
        "get_response": get_response,
    }


def _load_eval_runtime() -> Dict[str, Any]:
    from Tools.LLMTools.performance_test_tools import (
        compute_all_semantic_metrics,
        translate_texts_to_english,
    )

    return {
        "compute_all_semantic_metrics": compute_all_semantic_metrics,
        "translate_texts_to_english": translate_texts_to_english,
    }


def _resolve_project_path(path_str: Union[str, Path]) -> Path:
    path_obj = Path(path_str) if path_str else Path()
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(PROJECT_ROOT_DIR) / path_obj).resolve()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _mean_of_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _format_metric_value(value: Any) -> str:
    numeric_value = _safe_float(value)
    if np.isfinite(numeric_value):
        return "{0:.6f}".format(numeric_value)
    return "NaN"


def _print_sample_metric_scores(sample_name: str, row: Dict[str, Any], show_progress: bool) -> None:
    metric_items = [
        ("BLEU-T", row.get("bleu_total")),
        ("BLEU-1", row.get("bleu_1")),
        ("BLEU-2", row.get("bleu_2")),
        ("BLEU-3", row.get("bleu_3")),
        ("BLEU-4", row.get("bleu_4")),
        ("METEOR", row.get("meteor")),
        ("ROUGE-L", row.get("rouge_l")),
        ("BF1", row.get("bf1")),
        ("BF1-P", row.get("bf1_precision")),
        ("BF1-R", row.get("bf1_recall")),
    ]
    metric_text = " | ".join(
        "{0}={1}".format(metric_name, _format_metric_value(metric_value))
        for metric_name, metric_value in metric_items
    )
    messages = [
        "[Metrics] {0} | {1}".format(str(sample_name), metric_text),
        "[GT-EN] {0} | {1}".format(str(sample_name), str(row.get("gt_text_en", ""))),
        "[EVAL-EN] {0} | {1}".format(str(sample_name), str(row.get("evaluation_text_en", ""))),
    ]
    for message in messages:
        if show_progress:
            tqdm.write(message)
        else:
            print(message)


def _count_non_empty_keywords(keywords: List[str]) -> int:
    return int(sum(1 for item in keywords if str(item).strip()))


def _clean_text_for_eval(text: str) -> str:
    return str(text or "").replace("\n", "").replace("\r", "").strip()


def _split_keywords_same_as_main(keyword_sequence: Any) -> List[str]:
    # Keep the same split rule as main.py / batch_run_main_pipeline.py.
    if isinstance(keyword_sequence, list):
        return [str(item) for item in keyword_sequence]
    return str(keyword_sequence).split("&")


def _clean_keyword_list(raw_keywords: List[str]) -> List[str]:
    punctuation_chars = " \t\r\n。；;，,、.!?！？"
    cleaned = []
    for item in raw_keywords:
        normalized = str(item).strip().strip(punctuation_chars)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _normalize_keyword_for_match(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[\s,，。；;、:：!?！？\"'`·\(\)（）\[\]\-]+", "", normalized)
    return normalized


def _classify_keyword_polarity_with_fallback(
    runtime: Dict[str, Any],
    cleaned_keywords: List[str],
    language: str,
    keyword_stage_pipeline: int,
) -> Dict[str, List[str]]:
    exact_buckets = runtime["classify_comment_keywords"](
        keywords=cleaned_keywords,
        language=language,
        pipeline=int(keyword_stage_pipeline),
    )
    exact_positive = set(exact_buckets.get("positive", []))
    exact_negative = set(exact_buckets.get("negative", []))

    library = runtime["get_keyword_polarity_library"](
        language=language,
        pipeline=int(keyword_stage_pipeline),
    )
    positive_norms = [
        _normalize_keyword_for_match(item) for item in list(library.get("positive", [])) if str(item).strip()
    ]
    negative_norms = [
        _normalize_keyword_for_match(item) for item in list(library.get("negative", [])) if str(item).strip()
    ]

    buckets = {"positive": [], "negative": [], "unknown": []}
    seen = {key: set() for key in buckets}
    for keyword in cleaned_keywords:
        if keyword in exact_positive:
            bucket = "positive"
        elif keyword in exact_negative:
            bucket = "negative"
        else:
            keyword_norm = _normalize_keyword_for_match(keyword)
            bucket = "unknown"
            for lib_norm in positive_norms:
                if keyword_norm and lib_norm and (keyword_norm in lib_norm or lib_norm in keyword_norm):
                    bucket = "positive"
                    break
            if bucket == "unknown":
                for lib_norm in negative_norms:
                    if keyword_norm and lib_norm and (keyword_norm in lib_norm or lib_norm in keyword_norm):
                        bucket = "negative"
                        break

        if keyword not in seen[bucket]:
            buckets[bucket].append(keyword)
            seen[bucket].add(keyword)
    return buckets


def _read_utf8_text(path_obj: Path) -> str:
    with open(path_obj, "r", encoding="utf-8") as f:
        return _clean_text_for_eval(f.read())


def _load_label_scores_from_csv(csv_path: Path) -> Optional[Dict[str, float]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("CSV file is empty: {0}".format(csv_path))

    header = lines[0].strip().split(",")
    label_map = {
        "label_hand": "hand",
        "label_head": "head",
        "label_feet": "feet",
        "label_arm": "arm",
        "label_body": "body",
        "label_total": "total",
    }
    label_indices = {}
    for csv_label_name in label_map:
        if csv_label_name in header:
            label_indices[csv_label_name] = header.index(csv_label_name)

    if not label_indices:
        return None

    first_value_row = None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            first_value_row = stripped.split(",")
            break

    if first_value_row is None:
        return None

    labels = {
        "hand": 0.0,
        "head": 0.0,
        "feet": 0.0,
        "arm": 0.0,
        "body": 0.0,
        "total": 0.0,
    }
    for csv_label_name, target_name in label_map.items():
        idx = label_indices.get(csv_label_name)
        if idx is None or idx >= len(first_value_row):
            continue
        labels[target_name] = _safe_float(first_value_row[idx])
    return labels


def _build_empty_score_dict() -> Dict[str, Optional[float]]:
    return {
        "total_score": None,
        "head_score": None,
        "hand_score": None,
        "torso_score": None,
        "foot_score": None,
        "arm_score": None,
    }


def _build_score_dict_from_csv(csv_path: Path) -> Dict[str, Any]:
    labels = _load_label_scores_from_csv(csv_path)
    if labels is None:
        return {
            "score_dict": _build_empty_score_dict(),
            "score_source": "none_no_csv_labels",
        }
    return {
        "score_dict": {
            "total_score": _safe_float(labels.get("total")),
            "head_score": _safe_float(labels.get("head")),
            "hand_score": _safe_float(labels.get("hand")),
            "torso_score": _safe_float(labels.get("body")),
            "foot_score": _safe_float(labels.get("feet")),
            "arm_score": _safe_float(labels.get("arm")),
        },
        "score_source": "gt_csv_labels",
    }


def _collect_stems(directory: Path) -> Dict[str, Path]:
    if not directory.exists():
        return {}
    file_map = {}
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue
        file_map[file_path.stem] = file_path.resolve()
    return file_map


def _build_sample_manifest(dataset_root: Path) -> pd.DataFrame:
    keyword_dir = dataset_root / "keywords"
    txt_dir = dataset_root / "txt"
    csv_dir = dataset_root / "csv"
    video_dir = dataset_root / "video"

    keyword_map = _collect_stems(keyword_dir)
    txt_map = _collect_stems(txt_dir)
    csv_map = _collect_stems(csv_dir)
    video_map = _collect_stems(video_dir)

    all_stems = sorted(set(keyword_map) | set(txt_map) | set(csv_map) | set(video_map))
    rows = []
    for sample_name in all_stems:
        has_keyword = sample_name in keyword_map
        has_txt = sample_name in txt_map
        has_csv = sample_name in csv_map
        has_video = sample_name in video_map
        rows.append(
            {
                "sample_name": sample_name,
                "has_keyword": bool(has_keyword),
                "has_txt": bool(has_txt),
                "has_csv": bool(has_csv),
                "has_video": bool(has_video),
                "eligible_for_run": bool(has_keyword and has_txt and has_csv),
                "keyword_path": str(keyword_map.get(sample_name, "")),
                "txt_path": str(txt_map.get(sample_name, "")),
                "csv_path": str(csv_map.get(sample_name, "")),
                "video_path": str(video_map.get(sample_name, "")),
            }
        )
    return pd.DataFrame(rows)


def _manifest_missing_rows(manifest_df: pd.DataFrame) -> List[Dict[str, Any]]:
    failure_rows = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for row in manifest_df.to_dict(orient="records"):
        if bool(row.get("eligible_for_run")):
            continue
        missing_types = []
        if not bool(row.get("has_keyword")):
            missing_types.append("keywords")
        if not bool(row.get("has_txt")):
            missing_types.append("txt")
        if not bool(row.get("has_csv")):
            missing_types.append("csv")
        failure_rows.append(
            {
                "sample_name": str(row.get("sample_name", "")),
                "stage": "dataset_alignment",
                "error_type": "missing_required_files",
                "error_message": "Missing required files: {0}".format(", ".join(missing_types)),
                "missing_file_types": json.dumps(missing_types, ensure_ascii=False),
                "keyword_path": str(row.get("keyword_path", "")),
                "txt_path": str(row.get("txt_path", "")),
                "csv_path": str(row.get("csv_path", "")),
                "video_path": str(row.get("video_path", "")),
                "timestamp": timestamp,
            }
        )
    return failure_rows


def _build_run_dir(output_root: Union[str, Path], run_name: Optional[str] = None) -> Path:
    output_root_path = _resolve_project_path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    if run_name:
        safe_name = str(run_name).strip()
    else:
        safe_name = "gt_keywords_rag_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = output_root_path / safe_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _retrieve_dynamic_snippets(
    keywords: List[str],
    language: str,
    top_k: int,
    search_embedding_model_name: str,
    fallback_text: str,
) -> List[str]:
    runtime = _load_rag_runtime()
    query_texts = [str(item).strip() for item in keywords if str(item).strip()]
    if not query_texts:
        query_texts = [str(fallback_text).strip()]
    query_embeddings = runtime["get_embedding"](query_texts)
    db = runtime["KnowledgeDB"](
        runtime["session_en"] if str(language).lower().startswith("en") else runtime["session"]
    )
    return db.from_video_search(
        query_vec=query_embeddings,
        model_name=search_embedding_model_name,
        top_k=int(top_k),
    )


def _extract_chat_text_from_completion(completion: Any) -> str:
    if completion is None or not getattr(completion, "choices", None):
        raise ValueError("Translation completion returned empty response.")

    message = getattr(completion.choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return _clean_text_for_eval(content)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return _clean_text_for_eval("".join(parts))

    raise ValueError("Translation completion contains no text content.")


def _translate_texts_to_english_with_fallback(
    eval_runtime: Dict[str, Any],
    texts: List[str],
    target_lang: str,
) -> Dict[str, Any]:
    try:
        translated = eval_runtime["translate_texts_to_english"](
            texts=texts,
            target_lang=target_lang,
        )
        return {
            "texts": translated,
            "provider": "deepl_api",
        }
    except Exception:
        pass

    runtime = _load_rag_runtime()
    client = runtime["get_llm_client_by_model"]("qwen-plus")
    translated_texts = []
    for text in list(texts or []):
        cleaned_text = _clean_text_for_eval(text)
        if not cleaned_text:
            translated_texts.append("")
            continue
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the user text into natural English. "
                        "Return only the English translation."
                    ),
                },
                {
                    "role": "user",
                    "content": cleaned_text,
                },
            ],
            stream=False,
            temperature=0.0,
        )
        translated_texts.append(_extract_chat_text_from_completion(completion))

    return {
        "texts": translated_texts,
        "provider": "qwen-plus-fallback",
    }


def _run_one_sample(
    sample_row: Dict[str, Any],
    language: str,
    response_model_name: str,
    keyword_stage_pipeline: int,
    top_k: int,
    search_embedding_model_name: str,
    translation_target_lang: str,
    semantic_metric_language: str,
) -> Dict[str, Any]:
    runtime = _load_rag_runtime()
    eval_runtime = _load_eval_runtime()
    sample_name = str(sample_row["sample_name"])
    keyword_path = Path(str(sample_row["keyword_path"]))
    txt_path = Path(str(sample_row["txt_path"]))
    csv_path = Path(str(sample_row["csv_path"]))
    video_path = str(sample_row.get("video_path", "") or "")

    raw_keyword_text = _read_utf8_text(keyword_path)
    raw_keywords = _split_keywords_same_as_main(raw_keyword_text)
    cleaned_keywords = _clean_keyword_list(raw_keywords)
    keyword_polarity = _classify_keyword_polarity_with_fallback(
        runtime=runtime,
        cleaned_keywords=cleaned_keywords,
        language=language,
        keyword_stage_pipeline=int(keyword_stage_pipeline),
    )

    score_bundle = _build_score_dict_from_csv(csv_path)
    score_dict = dict(score_bundle["score_dict"])
    score_source = str(score_bundle["score_source"])
    retrieved_result = _retrieve_dynamic_snippets(
        keywords=cleaned_keywords,
        language=language,
        top_k=int(top_k),
        search_embedding_model_name=search_embedding_model_name,
        fallback_text=json.dumps(score_dict, ensure_ascii=False),
    )
    answer = runtime["get_response"](
        keywords=cleaned_keywords,
        score_dict=score_dict,
        retrieved_snippets=retrieved_result,
        keyword_polarity=keyword_polarity,
        language=language,
        model_name=response_model_name,
    )

    gt_text = _read_utf8_text(txt_path)
    translation_result = _translate_texts_to_english_with_fallback(
        eval_runtime=eval_runtime,
        texts=[gt_text, answer],
        target_lang=translation_target_lang,
    )
    gt_text_en, evaluation_text_en = translation_result["texts"]
    translation_provider = str(translation_result["provider"])
    semantic_metrics = eval_runtime["compute_all_semantic_metrics"](
        reference_text=gt_text_en,
        candidate_text=evaluation_text_en,
        bf1_lang=semantic_metric_language,
        include_cider=False,
        include_spice=False,
    )

    return {
        "sample_name": sample_name,
        "keyword_path": str(keyword_path),
        "txt_path": str(txt_path),
        "csv_path": str(csv_path),
        "video_path": video_path,
        "language": language,
        "response_model_name": response_model_name,
        "keyword_stage_pipeline": int(keyword_stage_pipeline),
        "search_embedding_model_name": search_embedding_model_name,
        "score_source": score_source,
        "retrieval_top_k": int(top_k),
        "raw_keyword_sequence": raw_keyword_text,
        "keyword_list_json": json.dumps(raw_keywords, ensure_ascii=False),
        "cleaned_keyword_list_json": json.dumps(cleaned_keywords, ensure_ascii=False),
        "keywords_count_raw_split": int(len(raw_keywords)),
        "keywords_count_non_empty": _count_non_empty_keywords(raw_keywords),
        "keyword_positive_count": int(len(keyword_polarity.get("positive", []))),
        "keyword_negative_count": int(len(keyword_polarity.get("negative", []))),
        "keyword_unknown_count": int(len(keyword_polarity.get("unknown", []))),
        "keyword_polarity_json": json.dumps(keyword_polarity, ensure_ascii=False),
        "retrieved_snippets_count": int(len(retrieved_result)),
        "retrieved_snippets_json": json.dumps(retrieved_result, ensure_ascii=False),
        "input_total_score": _safe_float(score_dict.get("total_score")),
        "input_head_score": _safe_float(score_dict.get("head_score")),
        "input_hand_score": _safe_float(score_dict.get("hand_score")),
        "input_torso_score": _safe_float(score_dict.get("torso_score")),
        "input_foot_score": _safe_float(score_dict.get("foot_score")),
        "input_arm_score": _safe_float(score_dict.get("arm_score")),
        "evaluation_text": str(answer),
        "gt_text": str(gt_text),
        "evaluation_text_en": str(evaluation_text_en),
        "gt_text_en": str(gt_text_en),
        "translation_provider": translation_provider,
        "translation_target_lang": translation_target_lang,
        "semantic_metric_language": semantic_metric_language,
        "bleu_total": _safe_float(semantic_metrics.get("bleu_total")),
        "bleu_1": _safe_float(semantic_metrics.get("bleu_1")),
        "bleu_2": _safe_float(semantic_metrics.get("bleu_2")),
        "bleu_3": _safe_float(semantic_metrics.get("bleu_3")),
        "bleu_4": _safe_float(semantic_metrics.get("bleu_4")),
        "meteor": _safe_float(semantic_metrics.get("meteor")),
        "rouge_l": _safe_float(semantic_metrics.get("rouge_l")),
        "bf1": _safe_float(semantic_metrics.get("bf1")),
        "bf1_precision": _safe_float(semantic_metrics.get("bf1_precision")),
        "bf1_recall": _safe_float(semantic_metrics.get("bf1_recall")),
    }


def _build_metric_summary(records_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "bleu_total",
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "bf1",
        "bf1_precision",
        "bf1_recall",
    ]
    rows = []
    for metric in metric_columns:
        if metric not in records_df.columns:
            continue
        series = pd.to_numeric(records_df[metric], errors="coerce")
        values = series[np.isfinite(series.to_numpy(dtype=float))]
        if values.empty:
            rows.append(
                {
                    "metric": metric,
                    "count": 0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "metric": metric,
                "count": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def run_batch(
    dataset_root: Union[str, Path],
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    run_name: Optional[str] = None,
    language: str = DEFAULT_LANGUAGE,
    response_model_name: str = DEFAULT_RESPONSE_MODEL_NAME,
    keyword_stage_pipeline: int = DEFAULT_KEYWORD_STAGE_PIPELINE,
    top_k: int = DEFAULT_TOP_K,
    search_embedding_model_name: str = DEFAULT_SEARCH_EMBEDDING_MODEL_NAME,
    translation_target_lang: str = DEFAULT_TRANSLATION_TARGET_LANG,
    semantic_metric_language: str = DEFAULT_SEMANTIC_METRIC_LANGUAGE,
    max_samples: Optional[int] = None,
    show_progress: bool = DEFAULT_SHOW_PROGRESS,
) -> Dict[str, Any]:
    dataset_root_path = _resolve_project_path(dataset_root)
    required_dirs = {
        "keywords": dataset_root_path / "keywords",
        "txt": dataset_root_path / "txt",
        "csv": dataset_root_path / "csv",
    }
    for dir_name, dir_path in required_dirs.items():
        if not dir_path.exists():
            raise FileNotFoundError("{0} directory not found: {1}".format(dir_name, dir_path))

    run_dir = _build_run_dir(output_root=output_root, run_name=run_name)
    sample_manifest_df = _build_sample_manifest(dataset_root_path)
    eligible_df = sample_manifest_df[sample_manifest_df["eligible_for_run"] == True].copy()
    eligible_df = eligible_df.sort_values("sample_name").reset_index(drop=True)
    if max_samples is not None:
        eligible_df = eligible_df.iloc[: int(max_samples)].copy()

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = _manifest_missing_rows(sample_manifest_df)
    iterator = tqdm(
        eligible_df.to_dict(orient="records"),
        total=int(eligible_df.shape[0]),
        desc="GT Keyword RAG Eval",
        disable=not show_progress,
    )

    for sample_row in iterator:
        try:
            row = _run_one_sample(
                sample_row=sample_row,
                language=language,
                response_model_name=response_model_name,
                keyword_stage_pipeline=int(keyword_stage_pipeline),
                top_k=int(top_k),
                search_embedding_model_name=search_embedding_model_name,
                translation_target_lang=translation_target_lang,
                semantic_metric_language=semantic_metric_language,
            )
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            records.append(row)
            _print_sample_metric_scores(
                sample_name=str(row.get("sample_name", "")),
                row=row,
                show_progress=show_progress,
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_name": str(sample_row.get("sample_name", "")),
                    "stage": "run_one_sample",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "missing_file_types": json.dumps([], ensure_ascii=False),
                    "keyword_path": str(sample_row.get("keyword_path", "")),
                    "txt_path": str(sample_row.get("txt_path", "")),
                    "csv_path": str(sample_row.get("csv_path", "")),
                    "video_path": str(sample_row.get("video_path", "")),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )

    records_df = pd.DataFrame(records)
    failures_df = pd.DataFrame(failures)
    metric_summary_df = _build_metric_summary(records_df)

    sample_manifest_csv = run_dir / "sample_manifest.csv"
    records_csv = run_dir / "records.csv"
    failed_samples_csv = run_dir / "failed_samples.csv"
    metric_summary_csv = run_dir / "metric_summary.csv"
    summary_json = run_dir / "summary.json"
    manifest_json = run_dir / "manifest.json"

    sample_manifest_df.to_csv(sample_manifest_csv, index=False, encoding="utf-8-sig")
    records_df.to_csv(records_csv, index=False, encoding="utf-8-sig")
    failures_df.to_csv(failed_samples_csv, index=False, encoding="utf-8-sig")
    metric_summary_df.to_csv(metric_summary_csv, index=False, encoding="utf-8-sig")

    summary = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root_path),
        "language": language,
        "response_model_name": response_model_name,
        "keyword_stage_pipeline": int(keyword_stage_pipeline),
        "search_embedding_model_name": search_embedding_model_name,
        "score_input_strategy": "csv_labels_if_available_else_none",
        "score_sources_in_records": sorted(
            list({str(item) for item in records_df.get("score_source", pd.Series(dtype=str)).dropna().tolist()})
        ),
        "retrieval_top_k": int(top_k),
        "translation_provider": "deepl_api_or_qwen-plus-fallback",
        "translation_providers_in_records": sorted(
            list({str(item) for item in records_df.get("translation_provider", pd.Series(dtype=str)).dropna().tolist()})
        ),
        "translation_target_lang": translation_target_lang,
        "semantic_metric_language": semantic_metric_language,
        "num_manifest_samples": int(sample_manifest_df.shape[0]),
        "num_eligible_samples": int(
            sample_manifest_df[sample_manifest_df["eligible_for_run"] == True].shape[0]
        ),
        "num_run_attempted_samples": int(eligible_df.shape[0]),
        "num_records": int(records_df.shape[0]),
        "num_failed_samples": int(failures_df.shape[0]),
        "dir_file_counts": {
            "keywords": int(sample_manifest_df["has_keyword"].sum()) if "has_keyword" in sample_manifest_df.columns else 0,
            "txt": int(sample_manifest_df["has_txt"].sum()) if "has_txt" in sample_manifest_df.columns else 0,
            "csv": int(sample_manifest_df["has_csv"].sum()) if "has_csv" in sample_manifest_df.columns else 0,
            "video": int(sample_manifest_df["has_video"].sum()) if "has_video" in sample_manifest_df.columns else 0,
        },
        "mean_semantic_metrics": {
            "bleu_total": _mean_of_column(records_df, "bleu_total"),
            "bleu_1": _mean_of_column(records_df, "bleu_1"),
            "bleu_2": _mean_of_column(records_df, "bleu_2"),
            "bleu_3": _mean_of_column(records_df, "bleu_3"),
            "bleu_4": _mean_of_column(records_df, "bleu_4"),
            "meteor": _mean_of_column(records_df, "meteor"),
            "rouge_l": _mean_of_column(records_df, "rouge_l"),
            "bf1": _mean_of_column(records_df, "bf1"),
            "bf1_precision": _mean_of_column(records_df, "bf1_precision"),
            "bf1_recall": _mean_of_column(records_df, "bf1_recall"),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    manifest = {
        "run_dir": str(run_dir),
        "sample_manifest_csv": str(sample_manifest_csv),
        "records_csv": str(records_csv),
        "failed_samples_csv": str(failed_samples_csv),
        "metric_summary_csv": str(metric_summary_csv),
        "summary_json": str(summary_json),
        "dataset_root": str(dataset_root_path),
        "language": language,
        "response_model_name": response_model_name,
        "keyword_stage_pipeline": int(keyword_stage_pipeline),
        "search_embedding_model_name": search_embedding_model_name,
        "score_input_strategy": "csv_labels_if_available_else_none",
        "retrieval_top_k": int(top_k),
        "translation_provider": "deepl_api_or_qwen-plus-fallback",
        "translation_target_lang": translation_target_lang,
        "semantic_metric_language": semantic_metric_language,
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG generation evaluation using human-annotated keyword sequences."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root that contains keywords/txt/csv/video subfolders.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for run artifacts.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional fixed run folder name under the output root.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language for retrieval/generation, usually zh or en.",
    )
    parser.add_argument(
        "--response-model-name",
        default=DEFAULT_RESPONSE_MODEL_NAME,
        help="Generation model name passed to get_response(...).",
    )
    parser.add_argument(
        "--keyword-stage-pipeline",
        type=int,
        default=DEFAULT_KEYWORD_STAGE_PIPELINE,
        help="Pipeline id used for keyword polarity classification. main.py default path uses 4.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k retrieved dynamic KB snippets.",
    )
    parser.add_argument(
        "--search-embedding-model-name",
        default=DEFAULT_SEARCH_EMBEDDING_MODEL_NAME,
        help="Embedding model name tag stored in the KB for from_video_search(...).",
    )
    parser.add_argument(
        "--translation-target-lang",
        default=DEFAULT_TRANSLATION_TARGET_LANG,
        help="Target language used before text metrics are computed.",
    )
    parser.add_argument(
        "--semantic-metric-language",
        default=DEFAULT_SEMANTIC_METRIC_LANGUAGE,
        help="Language flag used by semantic metrics such as BF1.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on eligible samples after sorting by sample name.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_batch(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        run_name=args.run_name,
        language=args.language,
        response_model_name=args.response_model_name,
        keyword_stage_pipeline=args.keyword_stage_pipeline,
        top_k=args.top_k,
        search_embedding_model_name=args.search_embedding_model_name,
        translation_target_lang=args.translation_target_lang,
        semantic_metric_language=args.semantic_metric_language,
        max_samples=args.max_samples,
        show_progress=not bool(args.disable_progress),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

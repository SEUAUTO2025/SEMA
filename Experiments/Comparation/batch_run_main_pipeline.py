import sys
import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RTMPOSE_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, "RTMPose")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT_DIR)
sys.path.insert(0, RTMPOSE_ROOT_DIR)

from RAG.Knowledge_Database.RAGFunc import get_response
from RAG.tokenize_search import Tokenize_SearchKeyword
from batch_eval_utils import (
    DEFAULT_HISTORY_SEARCH_ROOTS,
    SEMANTIC_METRIC_COLUMNS,
    build_historical_prediction_index,
    build_prediction_row,
    build_run_dir,
    export_batch_run_artifacts,
    load_gt_text_and_scores,
    merge_rows_by_keys,
    lookup_historical_prediction,
    normalize_eval_text,
    prepare_resume_failed_run,
    write_progress_records_and_failures,
)
from evaluation.eval_db.eval_db_manager import add_or_update_evaluation, get_evaluation, init_db

# Fixed run configuration (edit here before running).
DATASET_ROOT = "dataset"
MODEL_NAME = "SEMA_FULL"
LANGUAGE = "zh"
PIPELINE = 1
SUBPIPELINE = 4
ASSESSMENT_MODEL_NAME = "qwen3-vl-plus"
RESPONSE_MODEL_NAME = "qwen-plus"
VIDEO_EXT = ".mp4"
MAX_SAMPLES = None
SHOW_PROGRESS = True
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "evaluation", "results")
RESUME_FAILED_RUN_NAME = ""
SEMANTIC_METRIC_LANGUAGE = "zh"
BERT_SCORE_LANGUAGE = "en"
METRIC_PROTOCOL = "zh_raw_text_bleu_plus_bertscore_only"
PRINT_SKIP_ERRORS = True
PRINT_SKIP_TRACEBACK = False


class SampleStageError(Exception):
    def __init__(self, stage: str, original_exc: Exception):
        self.stage = str(stage)
        self.original_exc = original_exc
        super().__init__(
            "stage={0} failed: {1}: {2}".format(
                self.stage,
                type(original_exc).__name__,
                str(original_exc),
            )
        )


def _print_skip_event(
    sample_name: str,
    stage: str,
    error_type: str,
    error_message: str,
    traceback_text: str = "",
) -> None:
    if not PRINT_SKIP_ERRORS:
        return
    tqdm.write(
        "[MainBatch][SKIP] sample={0} stage={1} error_type={2} error={3}".format(
            sample_name,
            stage,
            error_type,
            error_message,
        )
    )
    if PRINT_SKIP_TRACEBACK and traceback_text:
        for line in str(traceback_text).rstrip().splitlines():
            tqdm.write(line)

def _resolve_project_path(path_str: Union[str, Path]) -> Path:
    path_obj = Path(path_str) if path_str else Path()
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(PROJECT_ROOT_DIR) / path_obj).resolve()


def _build_run_dir(output_root: Union[str, Path]) -> Path:
    output_root_path = _resolve_project_path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    run_dir = output_root_path / "main_batch_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _comment_to_keywords(comment: Any) -> List[str]:
    # Keep the same split rule as main.py.
    if isinstance(comment, list):
        return [str(item) for item in comment]
    return str(comment).split("&")


def _mean_of_column(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _ordered_unique_strings(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _build_resume_targets(
    resume_failed_rows: List[Dict[str, Any]],
    dataset_root_path: Path,
    video_ext: str,
) -> List[Dict[str, Any]]:
    ext = video_ext if str(video_ext).startswith(".") else ".{0}".format(video_ext)
    ext = ext.lower()
    video_dir = dataset_root_path / "video"
    targets: List[Dict[str, Any]] = []
    seen = set()

    for row in resume_failed_rows:
        sample_name = str(row.get("sample_name") or "").strip()
        target_model_name = str(row.get("model_name") or "").strip()
        if not sample_name or not target_model_name:
            continue
        key = (sample_name, target_model_name)
        if key in seen:
            continue
        seen.add(key)

        raw_video_path = str(row.get("video_path") or "").strip()
        if raw_video_path:
            video_path = _resolve_project_path(raw_video_path)
        else:
            video_path = (video_dir / "{0}{1}".format(sample_name, ext)).resolve()

        targets.append(
            {
                "sample_name": sample_name,
                "model_name": target_model_name,
                "video_path": video_path,
            }
        )
    return targets


def _run_one_sample(
    video_path: Path,
    txt_dir: Path,
    csv_dir: Path,
    model_name: str,
    language: str,
    pipeline: int,
    subpipeline: int,
    assessment_model_name: str,
    response_model_name: str,
) -> Dict[str, Any]:
    pose_csv_path = csv_dir / "{0}.csv".format(video_path.stem)

    # 1) main.py: Tokenize_SearchKeyword(...)
    try:
        scores, comment, retrieved_result, keyword_polarity = Tokenize_SearchKeyword(
            video_path=str(video_path),
            pose_csv_path=str(pose_csv_path),
            pipeline=pipeline,
            subpipeline=subpipeline,
            language=language,
            show=False,
            return_keyword_polarity=True,
            assessment_model_name=assessment_model_name,
        )
    except Exception as exc:
        raise SampleStageError("tokenize_search_keyword", exc)

    # 2) main.py: get_response(...)
    keywords = _comment_to_keywords(comment)
    try:
        answer = get_response(
            keywords=keywords,
            score_dict=scores,
            retrieved_snippets=retrieved_result,
            keyword_polarity=keyword_polarity,
            language=language,
            model_name=response_model_name,
        )
    except Exception as exc:
        raise SampleStageError("get_response", exc)

    try:
        add_or_update_evaluation(
            sample_name=video_path.stem,
            model_name=str(model_name),
            eval_text=str(answer),
        )
    except Exception as exc:
        raise SampleStageError("db_write_evaluation_text", exc)

    try:
        gt_payload = load_gt_text_and_scores(video_path=video_path, txt_dir=txt_dir, csv_dir=csv_dir)
    except Exception as exc:
        raise SampleStageError("load_gt_text_and_scores", exc)

    try:
        return build_prediction_row(
            base_row={
                "sample_name": video_path.stem,
                "video_path": str(video_path),
                "model_name": str(model_name),
                "db_model_name": str(model_name),
                "language": language,
                "pipeline": int(pipeline),
                "subpipeline": int(subpipeline),
                "keywords_count": int(len(keywords)),
                "retrieved_snippets_count": int(len(retrieved_result) if isinstance(retrieved_result, list) else 0),
                "semantic_metric_language": SEMANTIC_METRIC_LANGUAGE,
                "bert_score_language": BERT_SCORE_LANGUAGE,
            },
            evaluation_text=str(answer),
            gt_payload=gt_payload,
            pred_scores=scores,
            raw_model_output="",
            cache_source="fresh_inference",
            db_written=False,
            include_english_only_text_metrics=False,
            extra_fields={
                "keyword_polarity": json.dumps(keyword_polarity, ensure_ascii=False)
                if keyword_polarity is not None
                else "",
            },
        )
    except Exception as exc:
        raise SampleStageError("build_prediction_row", exc)


def run_batch(
    dataset_root: Union[str, Path],
    model_name: str = MODEL_NAME,
    language: str = LANGUAGE,
    pipeline: int = PIPELINE,
    subpipeline: int = SUBPIPELINE,
    assessment_model_name: str = ASSESSMENT_MODEL_NAME,
    response_model_name: str = RESPONSE_MODEL_NAME,
    output_root: Union[str, Path] = OUTPUT_ROOT,
    resume_failed_run_name: Optional[str] = RESUME_FAILED_RUN_NAME,
    max_samples: Optional[int] = MAX_SAMPLES,
    video_ext: str = VIDEO_EXT,
) -> Dict[str, Any]:
    dataset_root_path = _resolve_project_path(dataset_root)
    video_dir = dataset_root_path / "video"
    txt_dir = dataset_root_path / "txt"
    csv_dir = dataset_root_path / "csv"

    if not video_dir.exists():
        raise FileNotFoundError("Video directory not found: {0}".format(video_dir))
    if not txt_dir.exists():
        raise FileNotFoundError("TXT directory not found: {0}".format(txt_dir))
    if not csv_dir.exists():
        raise FileNotFoundError("CSV directory not found: {0}".format(csv_dir))

    ext = video_ext if str(video_ext).startswith(".") else ".{0}".format(video_ext)
    ext = ext.lower()
    video_files = sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() == ext])
    if max_samples is not None:
        video_files = video_files[: int(max_samples)]

    resume_ctx = prepare_resume_failed_run(
        output_root=output_root,
        resume_run_name=resume_failed_run_name,
        base_dir=SCRIPT_DIR,
    )
    if resume_ctx["enabled"]:
        run_dir = Path(resume_ctx["run_dir"])
        target_items = _build_resume_targets(
            resume_failed_rows=resume_ctx["resume_failed_rows"],
            dataset_root_path=dataset_root_path,
            video_ext=ext,
        )
        if max_samples is not None:
            target_items = target_items[: int(max_samples)]
        if len(target_items) == 0:
            raise ValueError("No resumable failed targets found in run: {0}".format(run_dir))
    else:
        run_dir = build_run_dir(output_root=output_root, run_prefix="main_batch", base_dir=SCRIPT_DIR)
        target_items = [
            {
                "sample_name": video_path.stem,
                "model_name": str(model_name),
                "video_path": video_path,
            }
            for video_path in video_files
        ]

    new_records: List[Dict[str, Any]] = []
    new_failures: List[Dict[str, Any]] = []
    if resume_ctx["enabled"]:
        records_to_export = list(resume_ctx["existing_records"])
        failures_to_export = list(resume_ctx["existing_failures"])
    else:
        records_to_export = []
        failures_to_export = []

    init_db()
    historical_cache = build_historical_prediction_index()

    iterator = tqdm(
        target_items,
        desc="Batch Main Pipeline Resume" if resume_ctx["enabled"] else "Batch Main Pipeline",
        disable=not SHOW_PROGRESS,
    )
    for target in iterator:
        sample_name = str(target["sample_name"])
        current_model_name = str(target["model_name"])
        video_path = Path(target["video_path"])
        sample_ts = datetime.now().isoformat(timespec="seconds")
        existing_eval = get_evaluation(sample_name=sample_name, model_name=current_model_name)
        cached_eval_text = normalize_eval_text(existing_eval.get("eval_text")) if existing_eval else ""
        if cached_eval_text:
            try:
                gt_payload = load_gt_text_and_scores(video_path=video_path, txt_dir=txt_dir, csv_dir=csv_dir)
                historical_row = lookup_historical_prediction(
                    cache_index=historical_cache,
                    sample_name=sample_name,
                    model_key=current_model_name,
                    eval_text=cached_eval_text,
                )
                extra_fields = {
                    "cached_from_db": True,
                    "source_records_csv": str(historical_row.get("source_records_csv") or "")
                    if historical_row
                    else "",
                }
                if historical_row is not None:
                    if "keywords_count" in historical_row:
                        extra_fields["keywords_count"] = historical_row.get("keywords_count")
                    if "retrieved_snippets_count" in historical_row:
                        extra_fields["retrieved_snippets_count"] = historical_row.get("retrieved_snippets_count")
                    if "keyword_polarity" in historical_row:
                        extra_fields["keyword_polarity"] = historical_row.get("keyword_polarity")
                else:
                    extra_fields["cache_note"] = (
                        "DB evaluation text exists, but no historical records.csv row with per-sample "
                        "prediction scores was found under evaluation/results."
                    )

                row = build_prediction_row(
                    base_row={
                        "sample_name": sample_name,
                        "video_path": str(video_path),
                        "model_name": current_model_name,
                        "db_model_name": current_model_name,
                        "language": language,
                        "pipeline": int(pipeline),
                        "subpipeline": int(subpipeline),
                        "semantic_metric_language": SEMANTIC_METRIC_LANGUAGE,
                        "bert_score_language": BERT_SCORE_LANGUAGE,
                    },
                    evaluation_text=cached_eval_text,
                    gt_payload=gt_payload,
                    pred_scores=historical_row,
                    raw_model_output=str(historical_row.get("raw_model_output") or "")
                    if historical_row
                    else "",
                    cache_source="db_records_cache" if historical_row else "db_text_only",
                    db_written=False,
                    include_english_only_text_metrics=False,
                    extra_fields=extra_fields,
                )
                row["timestamp"] = sample_ts
                new_records.append(row)
                processed_keys = [(sample_name, current_model_name)]
                records_to_export = merge_rows_by_keys(
                    existing_rows=records_to_export,
                    incoming_rows=[row],
                    key_fields=["sample_name", "model_name"],
                    remove_keys=processed_keys,
                )
                failures_to_export = merge_rows_by_keys(
                    existing_rows=failures_to_export,
                    incoming_rows=[],
                    key_fields=["sample_name", "model_name"],
                    remove_keys=processed_keys,
                )
                write_progress_records_and_failures(
                    run_dir=run_dir,
                    records=records_to_export,
                    failures=failures_to_export,
                )
                continue
            except Exception as exc:
                failure_stage = "cache_rebuild_record"
                error_type = type(exc).__name__
                tb = traceback.format_exc()
                fail = {
                    "sample_name": sample_name,
                    "video_path": str(video_path),
                    "model_name": current_model_name,
                    "language": language,
                    "pipeline": int(pipeline),
                    "subpipeline": int(subpipeline),
                    "stage": failure_stage,
                    "error_type": error_type,
                    "error_message": str(exc),
                    "traceback": tb,
                    "timestamp": sample_ts,
                }
                new_failures.append(fail)
                processed_keys = [(sample_name, current_model_name)]
                records_to_export = merge_rows_by_keys(
                    existing_rows=records_to_export,
                    incoming_rows=[],
                    key_fields=["sample_name", "model_name"],
                    remove_keys=processed_keys,
                )
                failures_to_export = merge_rows_by_keys(
                    existing_rows=failures_to_export,
                    incoming_rows=[fail],
                    key_fields=["sample_name", "model_name"],
                    remove_keys=processed_keys,
                )
                write_progress_records_and_failures(
                    run_dir=run_dir,
                    records=records_to_export,
                    failures=failures_to_export,
                )
                _print_skip_event(
                    sample_name=sample_name,
                    stage=failure_stage,
                    error_type=error_type,
                    error_message=str(exc),
                    traceback_text=tb,
                )
                continue

        try:
            row = _run_one_sample(
                video_path=video_path,
                txt_dir=txt_dir,
                csv_dir=csv_dir,
                model_name=current_model_name,
                language=language,
                pipeline=int(pipeline),
                subpipeline=int(subpipeline),
                assessment_model_name=assessment_model_name,
                response_model_name=response_model_name,
            )
            row["timestamp"] = sample_ts
            new_records.append(row)
            processed_keys = [(sample_name, current_model_name)]
            records_to_export = merge_rows_by_keys(
                existing_rows=records_to_export,
                incoming_rows=[row],
                key_fields=["sample_name", "model_name"],
                remove_keys=processed_keys,
            )
            failures_to_export = merge_rows_by_keys(
                existing_rows=failures_to_export,
                incoming_rows=[],
                key_fields=["sample_name", "model_name"],
                remove_keys=processed_keys,
            )
            write_progress_records_and_failures(
                run_dir=run_dir,
                records=records_to_export,
                failures=failures_to_export,
            )
        except Exception as exc:
            failure_stage = exc.stage if isinstance(exc, SampleStageError) else "unknown"
            error_type = (
                type(exc.original_exc).__name__ if isinstance(exc, SampleStageError) else type(exc).__name__
            )
            tb = traceback.format_exc()
            fail = {
                "sample_name": video_path.stem,
                "video_path": str(video_path),
                "model_name": current_model_name,
                "language": language,
                "pipeline": int(pipeline),
                "subpipeline": int(subpipeline),
                "stage": failure_stage,
                "error_type": error_type,
                "error_message": str(exc),
                "traceback": tb,
                "timestamp": sample_ts,
            }
            new_failures.append(fail)
            processed_keys = [(sample_name, current_model_name)]
            records_to_export = merge_rows_by_keys(
                existing_rows=records_to_export,
                incoming_rows=[],
                key_fields=["sample_name", "model_name"],
                remove_keys=processed_keys,
            )
            failures_to_export = merge_rows_by_keys(
                existing_rows=failures_to_export,
                incoming_rows=[fail],
                key_fields=["sample_name", "model_name"],
                remove_keys=processed_keys,
            )
            write_progress_records_and_failures(
                run_dir=run_dir,
                records=records_to_export,
                failures=failures_to_export,
            )
            _print_skip_event(
                sample_name=sample_name,
                stage=failure_stage,
                error_type=error_type,
                error_message=str(exc),
                traceback_text=tb,
            )

    records_df = pd.DataFrame(records_to_export)
    target_model_names = _ordered_unique_strings([item["model_name"] for item in target_items])

    summary = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root_path),
        "model_name": model_name,
        "target_model_names": target_model_names,
        "assessment_model_name": assessment_model_name,
        "response_model_name": response_model_name,
        "language": language,
        "semantic_metric_language": SEMANTIC_METRIC_LANGUAGE,
        "bert_score_language": BERT_SCORE_LANGUAGE,
        "metric_protocol": METRIC_PROTOCOL,
        "pipeline": int(pipeline),
        "subpipeline": int(subpipeline),
        "video_ext": ext,
        "num_video_files": int(len(target_items)),
        "resume_mode": bool(resume_ctx["enabled"]),
        "resume_failed_run_name": str(resume_ctx.get("run_name") or ""),
        "resume_target_count": int(len(target_items)) if resume_ctx["enabled"] else 0,
        "resume_existing_record_count": int(len(resume_ctx["existing_records"])) if resume_ctx["enabled"] else 0,
        "resume_existing_failure_count": int(len(resume_ctx["existing_failures"])) if resume_ctx["enabled"] else 0,
        "resume_new_record_count": int(len(new_records)),
        "resume_new_failure_count": int(len(new_failures)),
        "cache_source_counts": records_df["cache_source"].value_counts(dropna=False).to_dict()
        if "cache_source" in records_df.columns
        else {},
        "mean_mae": {
            "total": _mean_of_column(records_df, "mae_total"),
            "head": _mean_of_column(records_df, "mae_head"),
            "hand": _mean_of_column(records_df, "mae_hand"),
            "torso": _mean_of_column(records_df, "mae_torso"),
            "foot": _mean_of_column(records_df, "mae_foot"),
            "arm": _mean_of_column(records_df, "mae_arm"),
        },
        "mean_semantic_metrics": {
            "bleu_total": _mean_of_column(records_df, "bleu_total"),
            "bleu_1": _mean_of_column(records_df, "bleu_1"),
            "bleu_2": _mean_of_column(records_df, "bleu_2"),
            "bleu_3": _mean_of_column(records_df, "bleu_3"),
            "bleu_4": _mean_of_column(records_df, "bleu_4"),
            "bert_f1": _mean_of_column(records_df, "bert_f1"),
            "bert_precision": _mean_of_column(records_df, "bert_precision"),
            "bert_recall": _mean_of_column(records_df, "bert_recall"),
        },
    }

    manifest = {
        "run_dir": str(run_dir),
        "metric_protocol": METRIC_PROTOCOL,
        "semantic_metric_language": SEMANTIC_METRIC_LANGUAGE,
        "bert_score_language": BERT_SCORE_LANGUAGE,
        "history_search_roots": [str(path) for path in DEFAULT_HISTORY_SEARCH_ROOTS],
        "resume_mode": bool(resume_ctx["enabled"]),
        "resume_failed_run_name": str(resume_ctx.get("run_name") or ""),
    }
    exported = export_batch_run_artifacts(
        run_dir=run_dir,
        records=records_to_export,
        failures=failures_to_export,
        summary=summary,
        manifest=manifest,
        group_column=None,
        semantic_metric_columns=SEMANTIC_METRIC_COLUMNS,
    )
    return exported["summary"]


def main() -> None:
    summary = run_batch(
        dataset_root=DATASET_ROOT,
        model_name=MODEL_NAME,
        language=LANGUAGE,
        pipeline=PIPELINE,
        subpipeline=SUBPIPELINE,
        assessment_model_name=ASSESSMENT_MODEL_NAME,
        response_model_name=RESPONSE_MODEL_NAME,
        output_root=OUTPUT_ROOT,
        resume_failed_run_name=RESUME_FAILED_RUN_NAME,
        max_samples=MAX_SAMPLES,
        video_ext=VIDEO_EXT,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

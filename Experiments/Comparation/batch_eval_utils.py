import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from Tools.LLMTools.performance_test_tools import (
    calculate_BF1_score,
    calculate_bleu,
    get_matching_text,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_HISTORY_SEARCH_ROOTS = [
    Path(PROJECT_ROOT_DIR) / "evaluation" / "results",
    Path(SCRIPT_DIR) / "evaluation" / "results",
]

MAE_PART_COLUMNS = [
    ("total", "mae_total"),
    ("head", "mae_head"),
    ("hand", "mae_hand"),
    ("torso", "mae_torso"),
    ("foot", "mae_foot"),
    ("arm", "mae_arm"),
]

SEMANTIC_METRIC_COLUMNS = [
    ("bleu_total", "bleu_total"),
    ("bleu_1", "bleu_1"),
    ("bleu_2", "bleu_2"),
    ("bleu_3", "bleu_3"),
    ("bleu_4", "bleu_4"),
    ("bert_f1", "bert_f1"),
    ("bert_precision", "bert_precision"),
    ("bert_recall", "bert_recall"),
]

SEMANTIC_METRIC_COLUMNS_WITH_ENGLISH_ONLY = SEMANTIC_METRIC_COLUMNS + [
    ("cider", "cider"),
    ("meteor", "meteor"),
]


def resolve_path(
    path_str: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    path_obj = Path(path_str) if path_str else Path()
    if path_obj.is_absolute():
        return path_obj.resolve()
    anchor = Path(base_dir) if base_dir else Path(PROJECT_ROOT_DIR)
    return (anchor / path_obj).resolve()


def load_csv_rows(
    csv_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    path = resolve_path(csv_path, base_dir=base_dir)
    if not path.exists() or not path.is_file():
        return []
    if path.stat().st_size <= 0:
        return []
    try:
        frame = pd.read_csv(str(path), encoding="utf-8-sig", keep_default_na=False)
    except pd.errors.EmptyDataError:
        return []
    return frame.to_dict(orient="records")


def load_json_dict(
    json_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    path = resolve_path(json_path, base_dir=base_dir)
    if not path.exists() or not path.is_file():
        return {}
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def prepare_resume_failed_run(
    output_root: Union[str, Path],
    resume_run_name: Optional[str],
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    run_name = str(resume_run_name or "").strip()
    if not run_name:
        return {
            "enabled": False,
            "run_dir": None,
            "run_name": "",
            "resume_failed_rows": [],
            "existing_records": [],
            "existing_failures": [],
            "existing_summary": {},
            "existing_manifest": {},
        }

    output_root_path = resolve_path(output_root, base_dir=base_dir)
    run_dir = (output_root_path / run_name).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError("Resume run folder not found: {0}".format(run_dir))

    failed_csv = run_dir / "failed_samples.csv"
    if not failed_csv.exists():
        raise FileNotFoundError("Resume failed_samples.csv not found: {0}".format(failed_csv))

    resume_failed_rows = load_csv_rows(failed_csv)
    if len(resume_failed_rows) == 0:
        raise ValueError("Resume failed_samples.csv is empty: {0}".format(failed_csv))

    return {
        "enabled": True,
        "run_dir": run_dir,
        "run_name": run_name,
        "resume_failed_rows": resume_failed_rows,
        "existing_records": load_csv_rows(run_dir / "records.csv"),
        "existing_failures": load_csv_rows(failed_csv),
        "existing_summary": load_json_dict(run_dir / "summary.json"),
        "existing_manifest": load_json_dict(run_dir / "manifest.json"),
    }


def _merge_row_key(row: Dict[str, Any], key_fields: List[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(field) or "").strip() for field in key_fields)


def merge_rows_by_keys(
    existing_rows: List[Dict[str, Any]],
    incoming_rows: List[Dict[str, Any]],
    key_fields: List[str],
    remove_keys: Optional[List[Tuple[str, ...]]] = None,
) -> List[Dict[str, Any]]:
    keys_to_remove = set(remove_keys or [])
    if not keys_to_remove:
        keys_to_remove = {_merge_row_key(row, key_fields) for row in incoming_rows}

    merged = [
        row for row in list(existing_rows or [])
        if _merge_row_key(row, key_fields) not in keys_to_remove
    ]
    merged.extend(list(incoming_rows or []))
    return merged


def write_progress_records_and_failures(
    run_dir: Union[str, Path],
    records: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Path]:
    run_dir_path = resolve_path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    records_df = pd.DataFrame([standardize_record_aliases(row) for row in records])
    failures_df = pd.DataFrame(failures)

    records_csv = run_dir_path / "records.csv"
    failures_csv = run_dir_path / "failed_samples.csv"
    records_df.to_csv(str(records_csv), index=False, encoding="utf-8-sig")
    failures_df.to_csv(str(failures_csv), index=False, encoding="utf-8-sig")
    return {
        "records_csv": records_csv,
        "failed_samples_csv": failures_csv,
    }


def build_run_dir(
    output_root: Union[str, Path],
    run_prefix: str,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    output_root_path = resolve_path(output_root, base_dir=base_dir)
    output_root_path.mkdir(parents=True, exist_ok=True)
    run_dir = output_root_path / "{0}_{1}".format(
        str(run_prefix).strip(),
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def normalize_eval_text(text: Any) -> str:
    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass
    return str(text or "").replace("\r", " ").replace("\n", " ").strip()


def load_gt_text_and_scores(
    video_path: Union[str, Path],
    txt_dir: Union[str, Path],
    csv_dir: Union[str, Path],
) -> Dict[str, Any]:
    gt_result = get_matching_text(
        str(video_path),
        txt_dir=str(txt_dir),
        csv_dir=str(csv_dir),
    )
    if gt_result is None:
        raise RuntimeError("Ground truth text/labels not found.")

    gt_text, label, label_total = gt_result
    return {
        "gt_text": normalize_eval_text(gt_text),
        "gt_total": safe_float(label_total),
        "gt_head": safe_float(label.get("head")),
        "gt_hand": safe_float(label.get("hand")),
        "gt_torso": safe_float(label.get("body", label.get("torso"))),
        "gt_foot": safe_float(label.get("feet", label.get("foot"))),
        "gt_arm": safe_float(label.get("arm")),
    }


def compute_main_py_text_metrics(
    evaluation_text: str,
    gt_text: str,
    include_english_only_metrics: bool = True,
) -> Dict[str, float]:
    eval_text = normalize_eval_text(evaluation_text)
    gt_text_clean = normalize_eval_text(gt_text)

    bleu_scores = calculate_bleu(eval_text, gt_text_clean)
    try:
        bert_scores = calculate_BF1_score(eval_text, gt_text_clean, lang="en")
    except Exception:
        bert_scores = {
            "BF1 (Semantic Similarity)": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
        }

    results = {
        "bleu_total": safe_float(bleu_scores.get("BLEU-Total")),
        "bleu_1": safe_float(bleu_scores.get("BLEU-1 (Word-level)")),
        "bleu_2": safe_float(bleu_scores.get("BLEU-2 (Phrase-level)")),
        "bleu_3": safe_float(bleu_scores.get("BLEU-3 (Tri-gram)")),
        "bleu_4": safe_float(bleu_scores.get("BLEU-4 (Sentence-level)")),
        "bert_f1": safe_float(bert_scores.get("BF1 (Semantic Similarity)")),
        "bert_precision": safe_float(bert_scores.get("Precision")),
        "bert_recall": safe_float(bert_scores.get("Recall")),
    }
    if include_english_only_metrics:
        results["cider"] = float("nan")
        results["meteor"] = float("nan")
    return results


def standardize_record_aliases(row: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(row)
    eval_text = record.get("evaluation_text", record.get("eval_text"))
    if eval_text is not None:
        record["evaluation_text"] = normalize_eval_text(eval_text)
        record["eval_text"] = normalize_eval_text(eval_text)

    pred_aliases = [
        ("pred_total", "total_score"),
        ("pred_head", "head_score"),
        ("pred_hand", "hand_score"),
        ("pred_torso", "torso_score"),
        ("pred_foot", "foot_score"),
        ("pred_arm", "arm_score"),
    ]
    for pred_key, alt_key in pred_aliases:
        if pred_key not in record and alt_key in record:
            record[pred_key] = record.get(alt_key)
        if alt_key not in record and pred_key in record:
            record[alt_key] = record.get(pred_key)

    return record


def _normalize_model_key(row: Dict[str, Any]) -> str:
    db_model_name = str(row.get("db_model_name") or "").strip()
    if db_model_name:
        return db_model_name
    return str(row.get("model_name") or "").strip()


def _normalize_historical_row(
    row: Dict[str, Any],
    source_path: Path,
) -> Optional[Dict[str, Any]]:
    sample_name = str(row.get("sample_name") or "").strip()
    model_key = _normalize_model_key(row)
    if not sample_name or not model_key:
        return None

    normalized = standardize_record_aliases(dict(row))
    normalized["sample_name"] = sample_name
    normalized["model_key"] = model_key
    normalized["source_records_csv"] = str(source_path)
    normalized["source_mtime"] = float(source_path.stat().st_mtime)
    return normalized


def build_historical_prediction_index(
    search_roots: Optional[List[Union[str, Path]]] = None,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    roots = search_roots or DEFAULT_HISTORY_SEARCH_ROOTS
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for root in roots:
        root_path = resolve_path(root)
        if not root_path.exists():
            continue
        for records_csv in sorted(root_path.rglob("records.csv")):
            try:
                frame = pd.read_csv(str(records_csv), encoding="utf-8-sig")
            except Exception:
                continue

            for _, row in frame.iterrows():
                normalized = _normalize_historical_row(row.to_dict(), source_path=records_csv)
                if normalized is None:
                    continue
                key = (normalized["sample_name"], normalized["model_key"])
                index.setdefault(key, []).append(normalized)

    for records in index.values():
        records.sort(
            key=lambda item: (
                str(item.get("timestamp") or ""),
                safe_float(item.get("source_mtime")),
            ),
            reverse=True,
        )

    return index


def lookup_historical_prediction(
    cache_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    sample_name: str,
    model_key: str,
    eval_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    candidates = list(cache_index.get((str(sample_name).strip(), str(model_key).strip()), []))
    if not candidates:
        return None

    expected_text = normalize_eval_text(eval_text)
    if expected_text:
        for candidate in candidates:
            candidate_text = normalize_eval_text(
                candidate.get("evaluation_text", candidate.get("eval_text"))
            )
            if candidate_text == expected_text:
                return candidate

    for candidate in candidates:
        pred_total = safe_float(candidate.get("pred_total", candidate.get("total_score")))
        if np.isfinite(pred_total):
            return candidate
    return candidates[0]


def build_prediction_row(
    base_row: Dict[str, Any],
    evaluation_text: str,
    gt_payload: Dict[str, Any],
    pred_scores: Optional[Dict[str, Any]] = None,
    raw_model_output: str = "",
    cache_source: str = "fresh_inference",
    db_written: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    include_english_only_text_metrics: bool = True,
) -> Dict[str, Any]:
    row = dict(base_row)
    eval_text = normalize_eval_text(evaluation_text)
    pred_scores = dict(pred_scores or {})

    pred_total = safe_float(pred_scores.get("total_score", pred_scores.get("pred_total")))
    pred_head = safe_float(pred_scores.get("head_score", pred_scores.get("pred_head")))
    pred_hand = safe_float(pred_scores.get("hand_score", pred_scores.get("pred_hand")))
    pred_torso = safe_float(pred_scores.get("torso_score", pred_scores.get("pred_torso")))
    pred_foot = safe_float(pred_scores.get("foot_score", pred_scores.get("pred_foot")))
    pred_arm = safe_float(pred_scores.get("arm_score", pred_scores.get("pred_arm")))

    gt_total = safe_float(gt_payload.get("gt_total"))
    gt_head = safe_float(gt_payload.get("gt_head"))
    gt_hand = safe_float(gt_payload.get("gt_hand"))
    gt_torso = safe_float(gt_payload.get("gt_torso"))
    gt_foot = safe_float(gt_payload.get("gt_foot"))
    gt_arm = safe_float(gt_payload.get("gt_arm"))

    metric_payload = compute_main_py_text_metrics(
        eval_text,
        str(gt_payload.get("gt_text") or ""),
        include_english_only_metrics=include_english_only_text_metrics,
    )

    row.update(
        {
            "db_written": bool(db_written),
            "cache_source": str(cache_source),
            "prediction_scores_available": bool(np.isfinite(pred_total)),
            "pred_total": pred_total,
            "pred_head": pred_head,
            "pred_hand": pred_hand,
            "pred_torso": pred_torso,
            "pred_foot": pred_foot,
            "pred_arm": pred_arm,
            "total_score": pred_total,
            "head_score": pred_head,
            "hand_score": pred_hand,
            "torso_score": pred_torso,
            "foot_score": pred_foot,
            "arm_score": pred_arm,
            "gt_total": gt_total,
            "gt_head": gt_head,
            "gt_hand": gt_hand,
            "gt_torso": gt_torso,
            "gt_foot": gt_foot,
            "gt_arm": gt_arm,
            "mae_total": abs(pred_total - gt_total)
            if np.isfinite(pred_total) and np.isfinite(gt_total)
            else float("nan"),
            "mae_head": abs(pred_head - gt_head)
            if np.isfinite(pred_head) and np.isfinite(gt_head)
            else float("nan"),
            "mae_hand": abs(pred_hand - gt_hand)
            if np.isfinite(pred_hand) and np.isfinite(gt_hand)
            else float("nan"),
            "mae_torso": abs(pred_torso - gt_torso)
            if np.isfinite(pred_torso) and np.isfinite(gt_torso)
            else float("nan"),
            "mae_foot": abs(pred_foot - gt_foot)
            if np.isfinite(pred_foot) and np.isfinite(gt_foot)
            else float("nan"),
            "mae_arm": abs(pred_arm - gt_arm)
            if np.isfinite(pred_arm) and np.isfinite(gt_arm)
            else float("nan"),
            "evaluation_text": eval_text,
            "eval_text": eval_text,
            "gt_text": normalize_eval_text(gt_payload.get("gt_text")),
            "raw_model_output": str(raw_model_output or ""),
        }
    )
    row.update(metric_payload)
    if extra_fields:
        row.update(extra_fields)
    return standardize_record_aliases(row)


def _metric_stats(values: np.ndarray) -> Dict[str, Any]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _metric_summary_rows(
    records_df: pd.DataFrame,
    metric_columns: List[Tuple[str, str]],
    summary_kind: str,
    group_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if records_df.empty:
        return rows

    if group_column and group_column in records_df.columns:
        groups = records_df.groupby(group_column, dropna=False)
    else:
        groups = [(None, records_df)]

    for group_value, group_df in groups:
        for metric_name, column_name in metric_columns:
            if column_name not in group_df.columns:
                stats = _metric_stats(np.array([], dtype=float))
            else:
                series = pd.to_numeric(group_df[column_name], errors="coerce").to_numpy(dtype=float)
                stats = _metric_stats(series)

            row = {"metric": metric_name}
            if summary_kind == "mae":
                row["part"] = metric_name
                row.pop("metric", None)
                row.update(
                    {
                        "count": stats["count"],
                        "mean_mae": stats["mean"],
                        "std_mae": stats["std"],
                        "min_mae": stats["min"],
                        "max_mae": stats["max"],
                    }
                )
            else:
                row.update(stats)

            if group_value is not None:
                row[str(group_column)] = str(group_value)
            rows.append(row)
    return rows


def _rows_to_summary_dict(
    rows: List[Dict[str, Any]],
    group_column: Optional[str],
    key_field: str,
) -> Dict[str, Any]:
    if not rows:
        return {}

    if group_column and group_column in rows[0]:
        summary: Dict[str, Any] = {}
        for row in rows:
            group_value = str(row.get(group_column))
            key_value = str(row.get(key_field))
            payload = {k: row[k] for k in row.keys() if k not in {group_column, key_field}}
            summary.setdefault(group_value, {})[key_value] = payload
        return summary

    out: Dict[str, Any] = {}
    for row in rows:
        key_value = str(row.get(key_field))
        out[key_value] = {k: row[k] for k in row.keys() if k != key_field}
    return out


def _maybe_save_mae_plot(
    mae_rows_df: pd.DataFrame,
    output_path: Path,
    group_column: Optional[str] = None,
) -> Optional[str]:
    if plt is None or mae_rows_df.empty:
        return None

    plot_df = mae_rows_df.copy()
    if group_column and group_column in plot_df.columns:
        pivot_df = plot_df.pivot(index="part", columns=group_column, values="mean_mae")
        ax = pivot_df.plot(kind="bar", figsize=(10, 5))
        ax.legend(title=group_column)
    else:
        ax = plot_df.set_index("part")["mean_mae"].plot(kind="bar", figsize=(8, 4), color="#377eb8")

    ax.set_ylabel("MAE")
    ax.set_xlabel("")
    ax.set_title("MAE by Part")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    return str(output_path)


def _maybe_save_semantic_plot(
    semantic_rows_df: pd.DataFrame,
    output_path: Path,
    group_column: Optional[str] = None,
) -> Optional[str]:
    if plt is None or semantic_rows_df.empty:
        return None

    metric_order = [
        "bleu_total",
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "bert_f1",
        "bert_precision",
        "bert_recall",
    ]
    plot_df = semantic_rows_df[semantic_rows_df["metric"].isin(metric_order)].copy()
    if plot_df.empty:
        return None

    plot_df["metric"] = pd.Categorical(plot_df["metric"], categories=metric_order, ordered=True)
    plot_df = plot_df.sort_values("metric")

    if group_column and group_column in plot_df.columns:
        pivot_df = plot_df.pivot(index="metric", columns=group_column, values="mean")
        ax = pivot_df.plot(kind="bar", figsize=(11, 5))
        ax.legend(title=group_column)
    else:
        ax = plot_df.set_index("metric")["mean"].plot(kind="bar", figsize=(9, 4), color="#4daf4a")

    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Semantic Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    return str(output_path)


def _maybe_save_total_score_scatter(
    records_df: pd.DataFrame,
    output_path: Path,
    group_column: Optional[str] = None,
) -> Optional[str]:
    if plt is None or records_df.empty:
        return None

    plot_df = records_df.copy()
    if "pred_total" not in plot_df.columns or "gt_total" not in plot_df.columns:
        return None
    plot_df["pred_total"] = pd.to_numeric(plot_df["pred_total"], errors="coerce")
    plot_df["gt_total"] = pd.to_numeric(plot_df["gt_total"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["pred_total"]) & np.isfinite(plot_df["gt_total"])]
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    if group_column and group_column in plot_df.columns:
        for group_value, group_df in plot_df.groupby(group_column, dropna=False):
            ax.scatter(
                group_df["gt_total"],
                group_df["pred_total"],
                label=str(group_value),
                alpha=0.75,
            )
        ax.legend(title=group_column)
    else:
        ax.scatter(plot_df["gt_total"], plot_df["pred_total"], alpha=0.75, color="#e41a1c")

    all_vals = np.concatenate(
        [
            plot_df["gt_total"].to_numpy(dtype=float),
            plot_df["pred_total"].to_numpy(dtype=float),
        ]
    )
    lower = float(np.min(all_vals))
    upper = float(np.max(all_vals))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("GT Total Score")
    ax.set_ylabel("Pred Total Score")
    ax.set_title("Pred vs GT Total Score")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    return str(output_path)


def export_batch_run_artifacts(
    run_dir: Union[str, Path],
    records: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    summary: Dict[str, Any],
    manifest: Optional[Dict[str, Any]] = None,
    group_column: Optional[str] = None,
    semantic_metric_columns: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    run_dir_path = resolve_path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    records_df = pd.DataFrame([standardize_record_aliases(row) for row in records])
    failures_df = pd.DataFrame(failures)

    records_csv = run_dir_path / "records.csv"
    failures_csv = run_dir_path / "failed_samples.csv"
    mae_csv = run_dir_path / "mae_by_part.csv"
    semantic_csv = run_dir_path / "semantic_metrics_summary.csv"
    summary_json = run_dir_path / "summary.json"
    manifest_json = run_dir_path / "manifest.json"
    fig_mae_path = run_dir_path / "fig_mae_by_part.png"
    fig_semantic_path = run_dir_path / "fig_semantic_metrics.png"
    fig_scatter_path = run_dir_path / "fig_total_score_scatter.png"

    records_df.to_csv(str(records_csv), index=False, encoding="utf-8-sig")
    failures_df.to_csv(str(failures_csv), index=False, encoding="utf-8-sig")

    mae_rows = _metric_summary_rows(records_df, MAE_PART_COLUMNS, summary_kind="mae", group_column=group_column)
    semantic_rows = _metric_summary_rows(
        records_df,
        semantic_metric_columns or SEMANTIC_METRIC_COLUMNS_WITH_ENGLISH_ONLY,
        summary_kind="semantic",
        group_column=group_column,
    )

    mae_rows_df = pd.DataFrame(mae_rows)
    semantic_rows_df = pd.DataFrame(semantic_rows)
    mae_rows_df.to_csv(str(mae_csv), index=False, encoding="utf-8-sig")
    semantic_rows_df.to_csv(str(semantic_csv), index=False, encoding="utf-8-sig")

    fig_mae = _maybe_save_mae_plot(mae_rows_df, fig_mae_path, group_column=group_column)
    fig_semantic = _maybe_save_semantic_plot(semantic_rows_df, fig_semantic_path, group_column=group_column)
    fig_scatter = _maybe_save_total_score_scatter(records_df, fig_scatter_path, group_column=group_column)

    summary_payload = dict(summary)
    summary_payload.update(
        {
            "num_records": int(records_df.shape[0]),
            "num_failed_samples": int(failures_df.shape[0]),
            "mae_summary": _rows_to_summary_dict(mae_rows, group_column=group_column, key_field="part"),
            "semantic_summary": _rows_to_summary_dict(
                semantic_rows,
                group_column=group_column,
                key_field="metric",
            ),
        }
    )
    with open(str(summary_json), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    manifest_payload = dict(manifest or {})
    manifest_payload.update(
        {
            "run_dir": str(run_dir_path),
            "records_csv": str(records_csv),
            "failed_samples_csv": str(failures_csv),
            "mae_by_part_csv": str(mae_csv),
            "semantic_metrics_summary_csv": str(semantic_csv),
            "summary_json": str(summary_json),
        }
    )
    if fig_mae:
        manifest_payload["fig_mae_by_part"] = fig_mae
    if fig_semantic:
        manifest_payload["fig_semantic_metrics"] = fig_semantic
    if fig_scatter:
        manifest_payload["fig_total_score_scatter"] = fig_scatter

    with open(str(manifest_json), "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, ensure_ascii=False, indent=2)

    return {
        "records_df": records_df,
        "failures_df": failures_df,
        "summary": summary_payload,
        "manifest": manifest_payload,
    }

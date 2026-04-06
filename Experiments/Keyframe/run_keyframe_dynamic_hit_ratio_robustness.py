import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import run_keyframe_interval_iou_robustness as keyframe_interval_iou_module  # noqa: E402
from run_keyframe_interval_iou_robustness import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_DATASET_ROOT,
    DEFAULT_FASTDTW_FEATURE_MODE,
    DEFAULT_FASTDTW_RADIUS,
    DEFAULT_FASTDTW_REPO,
    DEFAULT_FASTDTW_TEMPLATE_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_COMPARISON,
    DEFAULT_TARGET_K,
    ROOT,
    _get_current_angle_cut_threshold_deg,
    _json_dumps,
    _parse_bool,
    _print_failure_reason,
    _resolve_path_from_root,
    _safe_float,
    apply_pose_noise,
    iter_dataset_samples,
    load_pose_sequence_from_csv,
    run_current_keyframe_extractor,
)

DEFAULT_GT_ANNOTATION_PATH = "dataset/keyframes/keyframes.txt"
# Follow the previously logged robustness settings:
# gaussian uses the paper-facing level for this script, while frame-mask
# reuses the IoU experiment's s1 severity.
DEFAULT_GAUSSIAN_SEVERITY = 0.001
DEFAULT_FRAME_MASK_SEVERITY = 0.1
DEFAULT_NOISE_REPEAT = 1

_GT_DYNAMIC_INTERVAL_CACHE: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run keyframe dynamic hit-ratio evaluation with clean / RTMPose-noisy "
            "conditions and an optional FastDTW baseline."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root with csv/video subfolders (relative paths are resolved against project ROOT).",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for evaluation outputs (relative paths are resolved against project ROOT).",
    )
    parser.add_argument(
        "--gt-annotation-path",
        default=DEFAULT_GT_ANNOTATION_PATH,
        help="CSV-style annotation file containing one dynamic interval per sample.",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=DEFAULT_TARGET_K,
        help="Final keyframe count after refinement / subsampling.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum number of samples to process.")
    parser.add_argument(
        "--noise-repeat",
        type=int,
        default=DEFAULT_NOISE_REPEAT,
        help="Repeat count for each RTMPose noise condition.",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="Base seed for deterministic noise.")
    parser.add_argument("--run-clean", type=_parse_bool, default=True, help="Whether to run the clean condition.")
    parser.add_argument("--run-noisy", type=_parse_bool, default=True, help="Whether to run the noisy condition.")
    parser.add_argument(
        "--run-comparison",
        type=_parse_bool,
        default=DEFAULT_RUN_COMPARISON,
        help="Whether to run the FastDTW comparison baseline on clean inputs.",
    )
    parser.add_argument(
        "--fastdtw-repo",
        default=DEFAULT_FASTDTW_REPO,
        help="Path to the local FastDTW source repo used as the comparison baseline implementation.",
    )
    parser.add_argument(
        "--fastdtw-template-manifest",
        default=DEFAULT_FASTDTW_TEMPLATE_MANIFEST,
        help="Manifest JSON for the reference/template keyframe video used by the FastDTW baseline.",
    )
    parser.add_argument(
        "--fastdtw-radius",
        type=int,
        default=DEFAULT_FASTDTW_RADIUS,
        help="FastDTW radius. Larger values are slower but closer to exact DTW.",
    )
    parser.add_argument(
        "--fastdtw-feature-mode",
        default=DEFAULT_FASTDTW_FEATURE_MODE,
        choices=["biomech_3"],
        help="Feature sequence definition used before FastDTW alignment.",
    )
    parser.add_argument(
        "--gaussian-severity",
        type=float,
        default=DEFAULT_GAUSSIAN_SEVERITY,
        help="Gaussian jitter sigma applied to normalized x/y pose coordinates.",
    )
    parser.add_argument(
        "--frame-mask-severity",
        type=float,
        default=DEFAULT_FRAME_MASK_SEVERITY,
        help="Frame-mask ratio applied to normalized x/y pose coordinates.",
    )
    return parser.parse_args()


def _read_text_with_fallback(path: Path) -> str:
    last_exc = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    raise RuntimeError("Failed to read dynamic keyframe annotation file: {0}".format(path)) from last_exc


def _parse_int_like(value: Any) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _infer_sample_key_from_path(path_text: str) -> str:
    return Path(str(path_text or "").strip()).stem.strip().lower()


def _build_dynamic_interval_index(annotation_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    if not annotation_path.exists():
        raise FileNotFoundError("Dynamic annotation file not found: {0}".format(annotation_path))

    text = _read_text_with_fallback(annotation_path)
    reader = csv.DictReader(StringIO(text))
    fieldnames = reader.fieldnames or []
    field_map = {
        str(name).strip().lower(): str(name)
        for name in fieldnames
        if str(name).strip()
    }
    url_key = field_map.get("url")
    start_key = field_map.get("start")
    end_key = field_map.get("end")
    if not url_key or not start_key or not end_key:
        raise ValueError(
            "Dynamic annotation file must include URL/Start/End columns: {0}".format(annotation_path)
        )

    intervals: Dict[str, List[Tuple[int, int]]] = {}
    for row in reader:
        sample_key = _infer_sample_key_from_path(str(row.get(url_key, "")))
        if not sample_key:
            continue
        start_frame = _parse_int_like(row.get(start_key))
        end_frame = _parse_int_like(row.get(end_key))
        if start_frame is None or end_frame is None:
            continue
        if start_frame < 0 or end_frame < start_frame:
            continue
        intervals.setdefault(sample_key, []).append((int(start_frame), int(end_frame)))
    return intervals


def _get_dynamic_interval_index(annotation_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    cache_key = str(annotation_path.resolve())
    cached = _GT_DYNAMIC_INTERVAL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parsed = _build_dynamic_interval_index(annotation_path.resolve())
    _GT_DYNAMIC_INTERVAL_CACHE[cache_key] = parsed
    return parsed


def load_gt_dynamic_interval_from_annotation(
    sample_path: str,
    annotation_path: str,
) -> Tuple[str, Optional[Tuple[int, int]]]:
    sample_key = _infer_sample_key_from_path(sample_path)
    if not sample_key:
        return "gt_pending", None
    interval_index = _get_dynamic_interval_index(_resolve_path_from_root(annotation_path))
    interval_list = interval_index.get(sample_key, [])
    if len(interval_list) == 0:
        return "gt_pending", None
    if len(interval_list) > 1:
        return "gt_invalid", None
    interval = interval_list[0]
    return "ok", (int(interval[0]), int(interval[1]))


def _load_unique_dynamic_interval_for_comparison(
    sample_path: str,
    annotation_path: str,
) -> Optional[Tuple[int, int]]:
    status, interval = load_gt_dynamic_interval_from_annotation(
        sample_path=sample_path,
        annotation_path=annotation_path,
    )
    if status != "ok":
        return None
    return interval


def extract_keyframes_with_dynamic_comparison_method(
    pose_seq: np.ndarray,
    video_path: str,
    target_k: int,
    annotation_path: str,
    fastdtw_repo: str = DEFAULT_FASTDTW_REPO,
    template_manifest_path: str = DEFAULT_FASTDTW_TEMPLATE_MANIFEST,
    fastdtw_radius: int = DEFAULT_FASTDTW_RADIUS,
    feature_mode: str = DEFAULT_FASTDTW_FEATURE_MODE,
) -> Dict[str, Any]:
    original_loader = keyframe_interval_iou_module.load_gt_interval_from_csv

    def _patched_loader(csv_path: str) -> Optional[Tuple[int, int]]:
        return _load_unique_dynamic_interval_for_comparison(
            sample_path=csv_path,
            annotation_path=annotation_path,
        )

    keyframe_interval_iou_module.load_gt_interval_from_csv = _patched_loader
    try:
        result = keyframe_interval_iou_module.extract_keyframes_with_comparison_method(
            pose_seq=pose_seq,
            video_path=video_path,
            target_k=target_k,
            fastdtw_repo=fastdtw_repo,
            template_manifest_path=template_manifest_path,
            fastdtw_radius=fastdtw_radius,
            feature_mode=feature_mode,
        )
    finally:
        keyframe_interval_iou_module.load_gt_interval_from_csv = original_loader

    selection_debug = dict(result.get("selection_debug", {}) or {})
    template_interval = selection_debug.pop("template_release_interval", None)
    aligned_count = selection_debug.pop("aligned_release_frames_count", None)
    aligned_bounds = selection_debug.pop("aligned_release_frame_bounds", None)

    if selection_debug.get("pred_interval_source") == "fastdtw_template_release_interval":
        selection_debug["pred_interval_source"] = "fastdtw_template_dynamic_interval"
    if template_interval is not None:
        selection_debug["template_dynamic_interval"] = template_interval
    if aligned_count is not None:
        selection_debug["aligned_dynamic_frames_count"] = aligned_count
    if aligned_bounds is not None:
        selection_debug["aligned_dynamic_frame_bounds"] = aligned_bounds
    selection_debug["interval_annotation_mode"] = "dynamic_only"
    selection_debug["interval_annotation_path"] = str(_resolve_path_from_root(annotation_path))

    result["selection_debug"] = selection_debug
    return result


def compute_keyframe_hit_ratio(
    frame_idx_list: List[int],
    gt_interval: Optional[Tuple[int, int]],
) -> Optional[Dict[str, Any]]:
    if gt_interval is None:
        return None
    keyframes = [int(v) for v in (frame_idx_list or [])]
    if len(keyframes) == 0:
        return None
    gt_start = int(gt_interval[0])
    gt_end = int(gt_interval[1])
    hit_keyframes = [frame for frame in keyframes if gt_start <= int(frame) <= gt_end]
    miss_keyframes = [frame for frame in keyframes if not (gt_start <= int(frame) <= gt_end)]
    keyframe_count = len(keyframes)
    hit_count = len(hit_keyframes)
    return {
        "keyframe_hit_ratio": float(hit_count / float(keyframe_count)),
        "hit_count": int(hit_count),
        "keyframe_count": int(keyframe_count),
        "hit_keyframes": [int(v) for v in hit_keyframes],
        "miss_keyframes": [int(v) for v in miss_keyframes],
    }


def _make_failure_row(
    sample_name: str,
    regime: str,
    condition_id: str,
    noise_type: str,
    severity_value: Optional[float],
    repeat_id: int,
    status: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "sample_name": sample_name,
        "regime": regime,
        "condition_id": condition_id,
        "noise_type": noise_type,
        "severity_value": severity_value,
        "repeat_id": repeat_id,
        "status": status,
        "error": error,
    }


def _make_eval_record(
    sample_name: str,
    regime: str,
    condition_id: str,
    noise_type: str,
    severity_value: float,
    repeat_id: int,
    gt_start_frame: Any,
    gt_end_frame: Any,
    status: str,
    error: str,
    pred_result: Optional[Dict[str, Any]] = None,
    ratio_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pred_result = dict(pred_result or {})
    ratio_result = dict(ratio_result or {})
    return {
        "sample_name": sample_name,
        "regime": regime,
        "condition_id": condition_id,
        "noise_type": noise_type,
        "severity_value": severity_value,
        "repeat_id": repeat_id,
        "pred_start_frame": (
            np.nan if pred_result.get("pred_start_frame") is None else int(pred_result["pred_start_frame"])
        ),
        "pred_end_frame": (
            np.nan if pred_result.get("pred_end_frame") is None else int(pred_result["pred_end_frame"])
        ),
        "pred_span_len": (
            np.nan if pred_result.get("pred_span_len") is None else int(pred_result["pred_span_len"])
        ),
        "gt_start_frame": gt_start_frame,
        "gt_end_frame": gt_end_frame,
        "keyframe_hit_ratio": (
            np.nan
            if ratio_result.get("keyframe_hit_ratio") is None
            else float(ratio_result["keyframe_hit_ratio"])
        ),
        "hit_count": np.nan if ratio_result.get("hit_count") is None else int(ratio_result["hit_count"]),
        "keyframe_count": (
            np.nan if ratio_result.get("keyframe_count") is None else int(ratio_result["keyframe_count"])
        ),
        "status": str(status),
        "error": str(error),
        "keyframes_json": _json_dumps(pred_result.get("frame_idx_list", []) or []),
        "hit_keyframes_json": _json_dumps(ratio_result.get("hit_keyframes", []) or []),
        "miss_keyframes_json": _json_dumps(ratio_result.get("miss_keyframes", []) or []),
        "selection_debug_json": _json_dumps(pred_result.get("selection_debug", {}) or {}),
    }


def _run_internal_hit_ratio_asserts() -> None:
    all_hit = compute_keyframe_hit_ratio([10, 20, 30], (10, 30))
    assert all_hit is not None
    assert np.isclose(all_hit["keyframe_hit_ratio"], 1.0)
    assert all_hit["hit_count"] == 3

    none_hit = compute_keyframe_hit_ratio([10, 20, 30], (31, 40))
    assert none_hit is not None
    assert np.isclose(none_hit["keyframe_hit_ratio"], 0.0)
    assert none_hit["hit_count"] == 0

    partial_hit = compute_keyframe_hit_ratio([10, 20, 30, 40], (15, 35))
    assert partial_hit is not None
    assert np.isclose(partial_hit["keyframe_hit_ratio"], 0.5)
    assert partial_hit["hit_keyframes"] == [20, 30]
    assert compute_keyframe_hit_ratio([], (0, 10)) is None
    assert compute_keyframe_hit_ratio([1, 2], None) is None


def _build_run_dir(output_root: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = Path(ROOT) / root
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / "rag_keyframe_hitratio_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_summary_rows(
    condition_rows: List[Dict[str, Any]],
    records_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    if records_df.empty or "condition_id" not in records_df.columns:
        for row in condition_rows:
            summary_rows.append(
                {
                    "condition_id": row["condition_id"],
                    "regime": row["regime"],
                    "noise_type": row["noise_type"],
                    "severity_value": row["severity_value"],
                    "valid_n": 0,
                    "mean_hit_ratio": float("nan"),
                    "var_hit_ratio": float("nan"),
                    "median_hit_ratio": float("nan"),
                    "min_hit_ratio": float("nan"),
                    "max_hit_ratio": float("nan"),
                }
            )
        return summary_rows
    for row in condition_rows:
        sub = records_df[records_df["condition_id"] == row["condition_id"]]
        valid = sub[(sub["status"] == "ok") & sub["keyframe_hit_ratio"].notna()]
        ratios = (
            valid["keyframe_hit_ratio"].to_numpy(dtype=float)
            if not valid.empty
            else np.array([], dtype=np.float64)
        )
        if ratios.size == 0:
            summary_rows.append(
                {
                    "condition_id": row["condition_id"],
                    "regime": row["regime"],
                    "noise_type": row["noise_type"],
                    "severity_value": row["severity_value"],
                    "valid_n": 0,
                    "mean_hit_ratio": float("nan"),
                    "var_hit_ratio": float("nan"),
                    "median_hit_ratio": float("nan"),
                    "min_hit_ratio": float("nan"),
                    "max_hit_ratio": float("nan"),
                }
            )
            continue
        summary_rows.append(
            {
                "condition_id": row["condition_id"],
                "regime": row["regime"],
                "noise_type": row["noise_type"],
                "severity_value": row["severity_value"],
                "valid_n": int(ratios.size),
                "mean_hit_ratio": float(np.mean(ratios)),
                "var_hit_ratio": float(np.var(ratios)),
                "median_hit_ratio": float(np.median(ratios)),
                "min_hit_ratio": float(np.min(ratios)),
                "max_hit_ratio": float(np.max(ratios)),
            }
        )
    return summary_rows


def _build_noisy_condition_rows(
    gaussian_severity: float,
    frame_mask_severity: float,
) -> List[Dict[str, Any]]:
    return [
        {
            "regime": "noisy_rtmpose",
            "condition_id": "gaussian_jitter_s1",
            "noise_type": "gaussian_jitter",
            "severity_value": float(gaussian_severity),
        },
        {
            "regime": "noisy_rtmpose",
            "condition_id": "frame_mask_s1",
            "noise_type": "frame_mask",
            "severity_value": float(frame_mask_severity),
        },
    ]


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _valid_ratio_series(df: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    sub = df.loc[mask].copy()
    if sub.empty:
        return np.array([], dtype=np.float64)
    sub = sub[(sub["status"] == "ok") & sub["keyframe_hit_ratio"].notna()]
    if sub.empty:
        return np.array([], dtype=np.float64)
    return sub["keyframe_hit_ratio"].to_numpy(dtype=np.float64)


def _build_plot_groups(
    records_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> List[Tuple[str, np.ndarray, str]]:
    groups: List[Tuple[str, np.ndarray, str]] = []
    groups.append(
        ("Our Method", _valid_ratio_series(records_df, records_df["regime"] == "clean"), "#1b9e77")
    )
    groups.append(
        (
            "Our Method\n(Gaussian Noise)",
            _valid_ratio_series(
                records_df,
                (records_df["regime"] == "noisy_rtmpose")
                & (records_df["noise_type"].astype(str) == "gaussian_jitter"),
            ),
            "#d95f02",
        )
    )
    groups.append(
        (
            "Our Method\n(Frame Mask)",
            _valid_ratio_series(
                records_df,
                (records_df["regime"] == "noisy_rtmpose")
                & (records_df["noise_type"].astype(str) == "frame_mask"),
            ),
            "#7570b3",
        )
    )
    groups.append(
        (
            "FastDTW",
            _valid_ratio_series(comparison_df, comparison_df["regime"] == "clean"),
            "#e7298a",
        )
    )
    return groups


def _summarize_plot_groups(groups: List[Tuple[str, np.ndarray, str]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for label, values, _ in groups:
        stats_key = str(label).replace("\n", " ")
        if values.size == 0:
            stats[stats_key] = {
                "n": 0.0,
                "mean": float("nan"),
                "median": float("nan"),
                "q1": float("nan"),
                "q3": float("nan"),
                "std": float("nan"),
            }
            continue
        stats[stats_key] = {
            "n": float(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "std": float(np.std(values)),
        }
    return stats


def _build_plot_stats_rows(groups: List[Tuple[str, np.ndarray, str]]) -> List[Dict[str, Any]]:
    stats_map = _summarize_plot_groups(groups)
    rows: List[Dict[str, Any]] = []
    for label, _, _ in groups:
        stats_key = str(label).replace("\n", " ")
        stats = stats_map.get(stats_key, {})
        rows.append(
            {
                "group_label": stats_key,
                "n": float(stats.get("n", float("nan"))),
                "mean": float(stats.get("mean", float("nan"))),
                "median": float(stats.get("median", float("nan"))),
                "q1": float(stats.get("q1", float("nan"))),
                "q3": float(stats.get("q3", float("nan"))),
                "std": float(stats.get("std", float("nan"))),
            }
        )
    return rows


def _draw_boxplot(
    groups: List[Tuple[str, np.ndarray, str]],
    out_png: Path,
    out_pdf: Path,
) -> None:
    labels = [label for label, _, _ in groups]
    values = [vals for _, vals, _ in groups]
    colors = [color for _, _, color in groups]
    positions = np.arange(1, len(groups) + 1, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.8},
        whiskerprops={"color": "#404040", "linewidth": 1.2},
        capprops={"color": "#404040", "linewidth": 1.2},
        boxprops={"linewidth": 1.2, "edgecolor": "#404040"},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

    for pos, vals, color in zip(positions, values, colors):
        mean_value = float(np.mean(vals))
        jitter = np.linspace(-0.11, 0.11, num=min(vals.size, 40))
        if vals.size > jitter.size:
            jitter = np.resize(jitter, vals.size)
        ax.scatter(
            np.full(vals.shape, pos, dtype=np.float64) + jitter,
            vals,
            s=11,
            color=color,
            alpha=0.16,
            linewidths=0.0,
            zorder=2,
        )
        ax.scatter(
            [pos],
            [mean_value],
            marker="D",
            s=32,
            color="#111111",
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )

    ax.set_ylabel("Keyframe Proportion")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", colors="#333333")
    ax.set_title("")

    fig.tight_layout(pad=0.35)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _evaluate_prediction(
    pred_result: Dict[str, Any],
    gt_status: str,
    gt_interval: Optional[Tuple[int, int]],
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    if gt_status != "ok":
        return gt_status, "" if gt_status == "gt_pending" else gt_status, None
    ratio_result = compute_keyframe_hit_ratio(
        [int(v) for v in (pred_result.get("frame_idx_list", []) or [])],
        gt_interval,
    )
    if ratio_result is None:
        return "extraction_failed", "empty_keyframe_list", None
    return "ok", "", ratio_result


def _stable_sample_seed(sample_name: str) -> int:
    seed = 0
    for idx, ch in enumerate(str(sample_name or "")):
        seed += (idx + 1) * ord(ch)
    return int(seed)


def _valid_ratio_records(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return records_df.copy()
    required_columns = {"status", "keyframe_hit_ratio"}
    if not required_columns.issubset(set(records_df.columns.tolist())):
        return records_df.iloc[0:0].copy()
    return records_df[
        (records_df["status"] == "ok") & records_df["keyframe_hit_ratio"].notna()
    ].copy()


def main() -> None:
    _run_internal_hit_ratio_asserts()
    _configure_matplotlib()
    args = _parse_args()

    dataset_root_path = Path(args.dataset_root)
    if not dataset_root_path.is_absolute():
        dataset_root_path = Path(ROOT) / dataset_root_path
    dataset_root = str(dataset_root_path.resolve())
    annotation_path = str(_resolve_path_from_root(str(args.gt_annotation_path)))
    samples = iter_dataset_samples(dataset_root)
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise RuntimeError("No valid samples found under: {0}".format(dataset_root))

    run_dir = _build_run_dir(str(args.output_root))
    angle_cut_threshold_deg = _get_current_angle_cut_threshold_deg()
    gaussian_severity = float(args.gaussian_severity)
    frame_mask_severity = float(args.frame_mask_severity)
    noisy_condition_rows = _build_noisy_condition_rows(
        gaussian_severity=gaussian_severity,
        frame_mask_severity=frame_mask_severity,
    )

    condition_rows: List[Dict[str, Any]] = []
    if args.run_clean:
        condition_rows.append(
            {
                "regime": "clean",
                "condition_id": "clean",
                "noise_type": "",
                "severity_value": 0.0,
            }
        )
    if args.run_noisy:
        condition_rows.extend(noisy_condition_rows)

    records: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []
    comparison_failed_rows: List[Dict[str, Any]] = []

    for sample in tqdm(samples, desc="Samples"):
        gt_status, gt_interval = load_gt_dynamic_interval_from_annotation(
            sample.csv_path,
            annotation_path=annotation_path,
        )
        gt_start_frame = np.nan if gt_interval is None else int(gt_interval[0])
        gt_end_frame = np.nan if gt_interval is None else int(gt_interval[1])

        try:
            pose_seq = load_pose_sequence_from_csv(sample.csv_path)
        except Exception as exc:
            error_text = "{0}: {1}".format(type(exc).__name__, exc)
            _print_failure_reason(
                sample_name=sample.sample_name,
                regime="load_pose",
                condition_id="load_pose",
                repeat_id=0,
                status="load_pose_failed",
                error=error_text,
            )
            failed_rows.append(
                _make_failure_row(
                    sample_name=sample.sample_name,
                    regime="load_pose",
                    condition_id="load_pose",
                    noise_type="",
                    severity_value=None,
                    repeat_id=0,
                    status="load_pose_failed",
                    error=error_text,
                )
            )
            records.append(
                _make_eval_record(
                    sample_name=sample.sample_name,
                    regime="load_pose",
                    condition_id="load_pose",
                    noise_type="",
                    severity_value=0.0,
                    repeat_id=0,
                    gt_start_frame=gt_start_frame,
                    gt_end_frame=gt_end_frame,
                    status="load_pose_failed",
                    error=error_text,
                )
            )
            if bool(args.run_comparison):
                comparison_failed_rows.append(
                    _make_failure_row(
                        sample_name=sample.sample_name,
                        regime="load_pose",
                        condition_id="load_pose",
                        noise_type="",
                        severity_value=None,
                        repeat_id=0,
                        status="load_pose_failed",
                        error=error_text,
                    )
                )
                comparison_records.append(
                    _make_eval_record(
                        sample_name=sample.sample_name,
                        regime="load_pose",
                        condition_id="load_pose",
                        noise_type="",
                        severity_value=0.0,
                        repeat_id=0,
                        gt_start_frame=gt_start_frame,
                        gt_end_frame=gt_end_frame,
                        status="load_pose_failed",
                        error=error_text,
                    )
                )
            continue

        if args.run_clean:
            try:
                clean_result = run_current_keyframe_extractor(
                    pose_seq=pose_seq,
                    video_path=sample.video_path,
                    target_k=int(args.target_k),
                    pred_start_keyframe_rank=1,
                )
                status, error_text, ratio_result = _evaluate_prediction(clean_result, gt_status, gt_interval)
                if status != "ok":
                    failed_rows.append(
                        _make_failure_row(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            repeat_id=0,
                            status=status,
                            error=error_text,
                        )
                    )
                records.append(
                    _make_eval_record(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean",
                        noise_type="",
                        severity_value=0.0,
                        repeat_id=0,
                        gt_start_frame=gt_start_frame,
                        gt_end_frame=gt_end_frame,
                        status=status,
                        error=error_text,
                        pred_result=clean_result,
                        ratio_result=ratio_result,
                    )
                )
            except Exception as exc:
                error_text = "{0}: {1}".format(type(exc).__name__, exc)
                _print_failure_reason(
                    sample_name=sample.sample_name,
                    regime="clean",
                    condition_id="clean",
                    repeat_id=0,
                    status="extraction_failed",
                    error=error_text,
                )
                failed_rows.append(
                    _make_failure_row(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean",
                        noise_type="",
                        severity_value=0.0,
                        repeat_id=0,
                        status="extraction_failed",
                        error=error_text,
                    )
                )
                records.append(
                    _make_eval_record(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean",
                        noise_type="",
                        severity_value=0.0,
                        repeat_id=0,
                        gt_start_frame=gt_start_frame,
                        gt_end_frame=gt_end_frame,
                        status="extraction_failed",
                        error=error_text,
                    )
                )

            if bool(args.run_comparison):
                try:
                    comparison_clean_result = extract_keyframes_with_dynamic_comparison_method(
                        pose_seq=pose_seq,
                        video_path=sample.video_path,
                        target_k=int(args.target_k),
                        annotation_path=annotation_path,
                        fastdtw_repo=str(args.fastdtw_repo),
                        template_manifest_path=str(args.fastdtw_template_manifest),
                        fastdtw_radius=int(args.fastdtw_radius),
                        feature_mode=str(args.fastdtw_feature_mode),
                    )
                    status, error_text, ratio_result = _evaluate_prediction(
                        comparison_clean_result,
                        gt_status,
                        gt_interval,
                    )
                    if status != "ok":
                        comparison_failed_rows.append(
                            _make_failure_row(
                                sample_name=sample.sample_name,
                                regime="clean",
                                condition_id="clean",
                                noise_type="",
                                severity_value=0.0,
                                repeat_id=0,
                                status=status,
                                error=error_text,
                            )
                        )
                    comparison_records.append(
                        _make_eval_record(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            repeat_id=0,
                            gt_start_frame=gt_start_frame,
                            gt_end_frame=gt_end_frame,
                            status=status,
                            error=error_text,
                            pred_result=comparison_clean_result,
                            ratio_result=ratio_result,
                        )
                    )
                except Exception as exc:
                    error_text = "{0}: {1}".format(type(exc).__name__, exc)
                    _print_failure_reason(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean_fastdtw",
                        repeat_id=0,
                        status="comparison_extract_failed",
                        error=error_text,
                    )
                    comparison_failed_rows.append(
                        _make_failure_row(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            repeat_id=0,
                            status="comparison_extract_failed",
                            error=error_text,
                        )
                    )
                    comparison_records.append(
                        _make_eval_record(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            repeat_id=0,
                            gt_start_frame=gt_start_frame,
                            gt_end_frame=gt_end_frame,
                            status="comparison_extract_failed",
                            error=error_text,
                        )
                    )

        if args.run_noisy:
            for repeat_id in range(int(args.noise_repeat)):
                for noise_rank, noise_row in enumerate(noisy_condition_rows):
                    rng_seed = (
                        int(args.base_seed)
                        + int(repeat_id)
                        + (100000 * _stable_sample_seed(sample.sample_name))
                        + (1000 * (noise_rank + 1))
                    )
                    rng = np.random.default_rng(rng_seed)
                    noisy_pose_seq = apply_pose_noise(
                        pose_seq=pose_seq,
                        noise_type=str(noise_row["noise_type"]),
                        severity_value=float(noise_row["severity_value"]),
                        rng=rng,
                    )
                    try:
                        noisy_result = run_current_keyframe_extractor(
                            pose_seq=noisy_pose_seq,
                            video_path=sample.video_path,
                            target_k=int(args.target_k),
                            pred_start_keyframe_rank=1,
                        )
                        status, error_text, ratio_result = _evaluate_prediction(
                            noisy_result,
                            gt_status,
                            gt_interval,
                        )
                        if status != "ok":
                            failed_rows.append(
                                _make_failure_row(
                                    sample_name=sample.sample_name,
                                    regime=str(noise_row["regime"]),
                                    condition_id=str(noise_row["condition_id"]),
                                    noise_type=str(noise_row["noise_type"]),
                                    severity_value=float(noise_row["severity_value"]),
                                    repeat_id=repeat_id,
                                    status=status,
                                    error=error_text,
                                )
                            )
                        records.append(
                            _make_eval_record(
                                sample_name=sample.sample_name,
                                regime=str(noise_row["regime"]),
                                condition_id=str(noise_row["condition_id"]),
                                noise_type=str(noise_row["noise_type"]),
                                severity_value=float(noise_row["severity_value"]),
                                repeat_id=repeat_id,
                                gt_start_frame=gt_start_frame,
                                gt_end_frame=gt_end_frame,
                                status=status,
                                error=error_text,
                                pred_result=noisy_result,
                                ratio_result=ratio_result,
                            )
                        )
                    except Exception as exc:
                        error_text = "{0}: {1}".format(type(exc).__name__, exc)
                        _print_failure_reason(
                            sample_name=sample.sample_name,
                            regime=str(noise_row["regime"]),
                            condition_id=str(noise_row["condition_id"]),
                            repeat_id=repeat_id,
                            status="noisy_extract_failed",
                            error=error_text,
                        )
                        failed_rows.append(
                            _make_failure_row(
                                sample_name=sample.sample_name,
                                regime=str(noise_row["regime"]),
                                condition_id=str(noise_row["condition_id"]),
                                noise_type=str(noise_row["noise_type"]),
                                severity_value=float(noise_row["severity_value"]),
                                repeat_id=repeat_id,
                                status="noisy_extract_failed",
                                error=error_text,
                            )
                        )
                        records.append(
                            _make_eval_record(
                                sample_name=sample.sample_name,
                                regime=str(noise_row["regime"]),
                                condition_id=str(noise_row["condition_id"]),
                                noise_type=str(noise_row["noise_type"]),
                                severity_value=float(noise_row["severity_value"]),
                                repeat_id=repeat_id,
                                gt_start_frame=gt_start_frame,
                                gt_end_frame=gt_end_frame,
                                status="noisy_extract_failed",
                                error=error_text,
                            )
                        )

    records_df = pd.DataFrame(records)
    failed_df = pd.DataFrame(failed_rows)
    summary_rows = _make_summary_rows(condition_rows, records_df)
    summary_df = pd.DataFrame(summary_rows)
    comparison_records_df = pd.DataFrame(comparison_records)
    comparison_failed_df = pd.DataFrame(comparison_failed_rows)
    comparison_condition_rows = [row for row in condition_rows if row["condition_id"] == "clean"]
    comparison_summary_rows = _make_summary_rows(comparison_condition_rows, comparison_records_df)
    comparison_summary_df = pd.DataFrame(comparison_summary_rows)

    valid_records = _valid_ratio_records(records_df)
    comparison_valid_records = _valid_ratio_records(comparison_records_df)

    plot_png = run_dir / "paper_hitratio_boxplot_ours_vs_fastdtw.png"
    plot_pdf = run_dir / "paper_hitratio_boxplot_ours_vs_fastdtw.pdf"
    plot_stats_csv = run_dir / "paper_hitratio_boxplot_stats.csv"
    plots_generated = False
    plot_skip_reason = "No valid hit-ratio records available."
    plot_group_stats: Dict[str, Dict[str, float]] = {}
    plot_stats_rows: List[Dict[str, Any]] = []

    if not records_df.empty or not comparison_records_df.empty:
        groups = _build_plot_groups(records_df, comparison_records_df)
        plot_group_stats = _summarize_plot_groups(groups)
        plot_stats_rows = _build_plot_stats_rows(groups)
        plottable_groups = [group for group in groups if group[1].size > 0]
        missing_labels = [label for label, values, _ in groups if values.size == 0]
        if plottable_groups:
            _draw_boxplot(plottable_groups, plot_png, plot_pdf)
            plots_generated = True
            if missing_labels:
                plot_skip_reason = "Excluded empty plot groups: {0}".format(", ".join(missing_labels))
        else:
            plot_skip_reason = "All plot groups are empty."

    summary = {
        "dataset_root": dataset_root,
        "gt_annotation_path": annotation_path,
        "run_dir": str(run_dir),
        "total_samples": int(len(samples)),
        "records_count": int(len(records_df)),
        "valid_hit_ratio_count": int(len(valid_records)),
        "clean_mean_hit_ratio": (
            _safe_float(summary_df.loc[summary_df["condition_id"] == "clean", "mean_hit_ratio"].iloc[0])
            if not summary_df[summary_df["condition_id"] == "clean"].empty
            else float("nan")
        ),
        "clean_var_hit_ratio": (
            _safe_float(summary_df.loc[summary_df["condition_id"] == "clean", "var_hit_ratio"].iloc[0])
            if not summary_df[summary_df["condition_id"] == "clean"].empty
            else float("nan")
        ),
        "angle_cut_threshold_deg": angle_cut_threshold_deg,
        "gaussian_severity": gaussian_severity,
        "plots_generated": plots_generated,
        "plot_skip_reason": "" if plots_generated else plot_skip_reason,
        "plot_group_stats": plot_group_stats,
        "comparison_fastdtw": {
            "enabled": bool(args.run_comparison),
            "records_count": int(len(comparison_records_df)),
            "valid_hit_ratio_count": int(len(comparison_valid_records)),
            "clean_mean_hit_ratio": (
                _safe_float(
                    comparison_summary_df.loc[
                        comparison_summary_df["condition_id"] == "clean",
                        "mean_hit_ratio",
                    ].iloc[0]
                )
                if not comparison_summary_df[comparison_summary_df["condition_id"] == "clean"].empty
                else float("nan")
            ),
            "clean_var_hit_ratio": (
                _safe_float(
                    comparison_summary_df.loc[
                        comparison_summary_df["condition_id"] == "clean",
                        "var_hit_ratio",
                    ].iloc[0]
                )
                if not comparison_summary_df[comparison_summary_df["condition_id"] == "clean"].empty
                else float("nan")
            ),
            "config": {
                "fastdtw_repo": str(_resolve_path_from_root(str(args.fastdtw_repo))),
                "template_manifest_path": str(_resolve_path_from_root(str(args.fastdtw_template_manifest))),
                "fastdtw_radius": int(args.fastdtw_radius),
                "feature_mode": str(args.fastdtw_feature_mode),
                "template_interval_policy": "load the unique dynamic interval for the template sample from --gt-annotation-path",
            },
        },
    }

    manifest = {
        "script_path": str((CURRENT_DIR / "run_keyframe_dynamic_hit_ratio_robustness.py").resolve()),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": dataset_root,
        "output_root": str(
            Path(args.output_root).resolve()
            if Path(args.output_root).is_absolute()
            else (Path(ROOT) / str(args.output_root)).resolve()
        ),
        "run_dir": str(run_dir),
        "gt_annotation_path": annotation_path,
        "target_k": int(args.target_k),
        "noise_repeat": int(args.noise_repeat),
        "base_seed": int(args.base_seed),
        "run_clean": bool(args.run_clean),
        "run_noisy": bool(args.run_noisy),
        "run_comparison": bool(args.run_comparison),
        "gaussian_severity": gaussian_severity,
        "frame_mask_severity": frame_mask_severity,
        "metric_definition": {
            "name": "keyframe_hit_ratio",
            "formula": "hit_count / keyframe_count",
            "hit_rule": "gt_start_frame <= keyframe_frame <= gt_end_frame",
            "gt_interval_policy": "use the unique Start/End interval in --gt-annotation-path; 0 records=gt_pending; >1 records=gt_invalid",
        },
        "plot_groups": [row["group_label"] for row in plot_stats_rows],
        "comparison_method": {
            "implemented": True,
            "function_name": "extract_keyframes_with_dynamic_comparison_method",
            "name": "fastdtw",
            "fastdtw_repo": str(_resolve_path_from_root(str(args.fastdtw_repo))),
            "template_manifest_path": str(_resolve_path_from_root(str(args.fastdtw_template_manifest))),
            "fastdtw_radius": int(args.fastdtw_radius),
            "feature_mode": str(args.fastdtw_feature_mode),
            "template_interval_policy": "load the unique dynamic interval for the template sample from --gt-annotation-path",
        },
    }

    records_csv = run_dir / "records.csv"
    summary_csv = run_dir / "summary_by_condition.csv"
    failed_csv = run_dir / "failed_samples.csv"
    comparison_records_csv = run_dir / "comparison_records_fastdtw.csv"
    comparison_summary_csv = run_dir / "comparison_summary_by_condition_fastdtw.csv"
    comparison_failed_csv = run_dir / "comparison_failed_samples_fastdtw.csv"
    summary_json = run_dir / "summary.json"
    manifest_json = run_dir / "manifest.json"

    records_df.to_csv(records_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(plot_stats_rows).to_csv(plot_stats_csv, index=False, encoding="utf-8-sig")
    if bool(args.run_comparison):
        comparison_records_df.to_csv(comparison_records_csv, index=False, encoding="utf-8-sig")
        comparison_summary_df.to_csv(comparison_summary_csv, index=False, encoding="utf-8-sig")
        comparison_failed_df.to_csv(comparison_failed_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] Output directory: {0}".format(run_dir))
    print("[Done] records.csv: {0}".format(records_csv))
    print("[Done] summary_by_condition.csv: {0}".format(summary_csv))
    print("[Done] failed_samples.csv: {0}".format(failed_csv))
    print("[Done] paper_hitratio_boxplot_stats.csv: {0}".format(plot_stats_csv))
    if bool(args.run_comparison):
        print("[Done] comparison_records_fastdtw.csv: {0}".format(comparison_records_csv))
        print("[Done] comparison_summary_by_condition_fastdtw.csv: {0}".format(comparison_summary_csv))
        print("[Done] comparison_failed_samples_fastdtw.csv: {0}".format(comparison_failed_csv))
    print("[Done] summary.json: {0}".format(summary_json))
    print("[Done] manifest.json: {0}".format(manifest_json))
    if plots_generated:
        print("[Done] boxplot_png: {0}".format(plot_png))
        print("[Done] boxplot_pdf: {0}".format(plot_pdf))
        for label, stats in plot_group_stats.items():
            print(
                "[Stats] {0}: n={1}, mean={2:.4f}, median={3:.4f}, q1={4:.4f}, q3={5:.4f}, std={6:.4f}".format(
                    label,
                    int(stats["n"]),
                    stats["mean"],
                    stats["median"],
                    stats["q1"],
                    stats["q3"],
                    stats["std"],
                )
            )
    else:
        print("[Info] Plot generation skipped: {0}".format(plot_skip_reason))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[Error] {0}: {1}".format(type(exc).__name__, exc))
        print(traceback.format_exc())
        raise

import argparse
import inspect
import json
import math
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from RTMPose.Bone_Feature_Extract import cal_math_features, extract_action_features
except ModuleNotFoundError as exc:
    if exc.name != "ruptures":
        raise

    ruptures_stub = types.ModuleType("ruptures")

    class _MissingRupturesBinseg(object):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ruptures is required for keyframe/change-point functions, "
                "but BIO_FEATURE_CAL only needs cal_math_features/extract_action_features."
            )

    ruptures_stub.Binseg = _MissingRupturesBinseg
    sys.modules["ruptures"] = ruptures_stub
    from RTMPose.Bone_Feature_Extract import cal_math_features, extract_action_features
from Tools.Exe_dataset.dataset_test_tools import (
    DEFAULT_FEATURE_GT_PART_MAP,
    DEFAULT_FEATURE_SPECS,
    DEFAULT_GT_FIXED_BINS,
    DEFAULT_THRESHOLD_SHAPE_MAP,
    build_grade_intervals,
    build_gt_quality_reference,
    build_gt_supervised_grade_standards,
    grade_value,
    load_single_csv_with_multipart_labels,
)

DEFAULT_TRAIN_DATASET_ROOT = PROJECT_ROOT / "dataset_choose"
DEFAULT_VAL_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "Experiments" / "BIO_FEATURE_CAL" / "evaluation" / "results"
)
GRADE_ORDER = ["Excellent", "Good", "Average", "Poor"]
GRADE_TO_SCORE = {
    "Excellent": 5.0,
    "Good": 4.0,
    "Average": 3.0,
    "Poor": 2.0,
}
SCORE_TO_GRADE = {
    5.0: "Excellent",
    4.0: "Good",
    3.0: "Average",
    2.0: "Poor",
}
HIGH_QUALITY_GRADES = {"Excellent", "Good"}
LOW_QUALITY_GRADES = {"Average", "Poor"}
THRESHOLD_METHOD_NAME = "gt_supervised_threshold_search"
BIO_FEATURE_SPECS = dict(DEFAULT_FEATURE_SPECS)
BIO_FEATURE_GT_PART_MAP = dict(DEFAULT_FEATURE_GT_PART_MAP)
BIO_THRESHOLD_SHAPE_MAP = dict(DEFAULT_THRESHOLD_SHAPE_MAP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether current biomechanics features guide body-part scoring "
            "with train/validation dataset roots under current repository settings."
        )
    )
    parser.add_argument(
        "--train-dataset-root",
        default=str(DEFAULT_TRAIN_DATASET_ROOT),
    )
    parser.add_argument(
        "--val-dataset-root",
        "--dataset-root",
        dest="val_dataset_root",
        default=str(DEFAULT_VAL_DATASET_ROOT),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-frames", type=int, default=124)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument(
        "--val-max-samples",
        "--max-samples",
        dest="val_max_samples",
        type=int,
        default=0,
    )
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--sort-files", action="store_true", default=True)
    return parser.parse_args()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _safe_corr(x: Sequence[Any], y: Sequence[Any], method: str) -> float:
    sx = pd.Series(list(x), dtype="float64")
    sy = pd.Series(list(y), dtype="float64")
    mask = sx.notna() & sy.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    return float(sx[mask].corr(sy[mask], method=method))


def _safe_mean(values: Sequence[Any]) -> float:
    arr = pd.Series(list(values), dtype="float64").dropna()
    if arr.empty:
        return float("nan")
    return float(arr.mean())


def _safe_std(values: Sequence[Any]) -> float:
    arr = pd.Series(list(values), dtype="float64").dropna()
    if arr.empty:
        return float("nan")
    return float(arr.std(ddof=0))


def _safe_quantile(values: Sequence[Any], q: float) -> float:
    arr = pd.Series(list(values), dtype="float64").dropna()
    if arr.empty:
        return float("nan")
    return float(arr.quantile(q))


def _safe_accuracy(lhs: Sequence[Any], rhs: Sequence[Any]) -> float:
    left = pd.Series(list(lhs), dtype="object")
    right = pd.Series(list(rhs), dtype="object")
    mask = left.notna() & right.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    return float((left[mask] == right[mask]).mean())


def _safe_score_accuracy(pred_scores: Sequence[Any], gt_scores: Sequence[Any]) -> float:
    pred = pd.Series(list(pred_scores), dtype="float64")
    gt = pd.Series(list(gt_scores), dtype="float64")
    mask = pred.notna() & gt.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    return float((pred[mask] == gt[mask]).mean())


def _quality_group(grade: Any) -> Optional[str]:
    text = str(grade)
    if text in HIGH_QUALITY_GRADES:
        return "high"
    if text in LOW_QUALITY_GRADES:
        return "low"
    return None


def _grouped_accuracy(pred_grades: Sequence[Any], gt_grades: Sequence[Any]) -> float:
    pred_groups = [_quality_group(v) for v in pred_grades]
    gt_groups = [_quality_group(v) for v in gt_grades]
    pred = pd.Series(pred_groups, dtype="object")
    gt = pd.Series(gt_groups, dtype="object")
    mask = pred.notna() & gt.notna()
    if int(mask.sum()) == 0:
        return float("nan")
    return float((pred[mask] == gt[mask]).mean())


def _current_extract_action_feature_settings() -> Dict[str, Any]:
    source = inspect.getsource(extract_action_features)
    out = {
        "window_size_angle": None,
        "window_size_chin": None,
        "window_size_head": None,
        "window_size_diff": None,
    }
    for key in list(out.keys()):
        pattern = r"%s\s*=\s*(\d+)" % re.escape(key)
        match = re.search(pattern, source)
        if match:
            out[key] = int(match.group(1))
    return out


def _build_total_score_buckets(
    sample_rows: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, float]]:
    totals = pd.Series(
        [row.get("label_total", np.nan) for row in sample_rows],
        dtype="float64",
    ).dropna()
    if totals.empty:
        return {}, {"q25": float("nan"), "q50": float("nan"), "q75": float("nan")}

    q25 = float(totals.quantile(0.25))
    q50 = float(totals.quantile(0.50))
    q75 = float(totals.quantile(0.75))
    sample_to_bucket = {}
    for row in sample_rows:
        value = float(row.get("label_total", np.nan))
        bucket = "Unknown"
        if not np.isnan(value):
            if value <= q25:
                bucket = "P0_25"
            elif value <= q50:
                bucket = "P25_50"
            elif value <= q75:
                bucket = "P50_75"
            else:
                bucket = "P75_100"
        sample_to_bucket[str(row.get("sample_name", ""))] = bucket
    return sample_to_bucket, {"q25": q25, "q50": q50, "q75": q75}


def _restore_csv_pose_to_current_scale(
    data: np.ndarray,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    all_data = torch.from_numpy(data).float()
    all_data = all_data[:, :, :, 0]
    all_data = all_data.permute(1, 2, 0).contiguous()
    all_data[:, :, 0] = all_data[:, :, 0] * float(image_width)
    all_data[:, :, 1] = all_data[:, :, 1] * float(image_height)
    return all_data


def _extract_sample_feature_row(
    csv_path: Path,
    max_frames: int,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    data, labels, label_total = load_single_csv_with_multipart_labels(
        str(csv_path),
        max_frames=max_frames,
    )
    pose_tensor = _restore_csv_pose_to_current_scale(
        data=data,
        image_width=image_width,
        image_height=image_height,
    )
    math_features = cal_math_features(pose_tensor, plot_angle_curve=False)
    action_features = extract_action_features(math_features)
    non_zero_frames = int(
        np.sum(np.any(np.abs(data[:, :, :, 0]) > 0, axis=(0, 2)))
    )
    return {
        "sample_name": csv_path.stem,
        "csv_path": str(csv_path),
        "frame_count": non_zero_frames,
        "label_total": float(label_total),
        "label_hand": float(labels.get("hand", np.nan)),
        "label_head": float(labels.get("head", np.nan)),
        "label_feet": float(labels.get("feet", np.nan)),
        "label_arm": float(labels.get("arm", np.nan)),
        "label_body": float(labels.get("body", np.nan)),
        "max_angle_avg": float(action_features.get("max_angle_avg", np.nan)),
        "min_dist_avg": float(action_features.get("min_dist_avg", np.nan)),
        "max_head_turn_ratio_avg": float(action_features.get("max_head_turn_ratio_avg", np.nan)),
        "min_torso_tilt_avg": float(action_features.get("min_torso_tilt_avg", np.nan)),
        "ankle_shoulder_width_ratio_absdiff_avg": float(
            action_features.get("ankle_shoulder_width_ratio_absdiff_avg", np.nan)
        ),
    }


def collect_sample_feature_rows(
    dataset_root: Path,
    max_frames: int,
    image_width: int,
    image_height: int,
    max_samples: int,
    sort_files: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    csv_dir = dataset_root / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError("CSV folder not found: %s" % csv_dir)

    csv_files = list(csv_dir.glob("*.csv"))
    if sort_files:
        csv_files = sorted(csv_files)
    if max_samples > 0:
        csv_files = csv_files[:max_samples]

    rows = []
    failed_rows = []
    for csv_path in csv_files:
        try:
            rows.append(
                _extract_sample_feature_row(
                    csv_path=csv_path,
                    max_frames=max_frames,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        except Exception as exc:
            failed_rows.append(
                {
                    "sample_name": csv_path.stem,
                    "csv_path": str(csv_path),
                    "error": str(exc),
                }
            )
    return rows, failed_rows


def _alignment_value(value: Any, standard: Dict[str, Any]) -> float:
    val = float(value)
    if np.isnan(val):
        return float("nan")
    mode = str(standard.get("mode", "one_sided")).strip().lower()
    if mode == "two_sided":
        center = float(standard.get("center", np.nan))
        if np.isnan(center):
            return float("nan")
        return float(-abs(val - center))

    direction = str(standard.get("direction", "")).strip().lower()
    if direction == "higher_is_better":
        return val
    if direction == "lower_is_better":
        return float(-val)
    return val


def _is_same_score(lhs: Any, rhs: Any) -> bool:
    left = float(lhs)
    right = float(rhs)
    if np.isnan(left) or np.isnan(right):
        return False
    return bool(abs(left - right) < 1e-8)


def _score_to_grade(score: Any) -> str:
    value = float(score)
    if np.isnan(value):
        return "Unknown"
    for candidate, grade in SCORE_TO_GRADE.items():
        if abs(value - candidate) < 1e-8:
            return grade
    return "Unknown"


def _relaxed_pred_score_for_eval(pred_score: Any, gt_part_score_bucket: Any) -> float:
    pred = float(pred_score)
    gt_bucket = float(gt_part_score_bucket)
    if np.isnan(pred) or np.isnan(gt_bucket):
        return float("nan")
    if abs(gt_bucket - RELAXED_TOP_SCORE) < 1e-8:
        for accepted in RELAXED_ACCEPTABLE_TOP_PRED_SCORES:
            if abs(pred - accepted) < 1e-8:
                return float(RELAXED_TOP_SCORE)
    return pred


def _relaxed_score_correct(pred_score: Any, gt_part_score_bucket: Any) -> float:
    relaxed_pred = _relaxed_pred_score_for_eval(pred_score, gt_part_score_bucket)
    gt_bucket = float(gt_part_score_bucket)
    if np.isnan(relaxed_pred) or np.isnan(gt_bucket):
        return float("nan")
    return 1.0 if abs(relaxed_pred - gt_bucket) < 1e-8 else 0.0


def _nanmedian(values: Sequence[Any]) -> float:
    arr = pd.Series(list(values), dtype="float64").dropna()
    if arr.empty:
        return float("nan")
    return float(arr.median())


def _midpoint(lhs: Any, rhs: Any) -> float:
    left = float(lhs)
    right = float(rhs)
    if np.isnan(left) and np.isnan(right):
        return float("nan")
    if np.isnan(left):
        return right
    if np.isnan(right):
        return left
    return float((left + right) / 2.0)


def _fill_missing_grade_representatives(
    values: Sequence[Any],
    fallback_value: float,
) -> List[float]:
    raw_values = [float(v) for v in values]
    if all(np.isnan(v) for v in raw_values):
        return [float(fallback_value)] * len(raw_values)

    filled = list(raw_values)
    for idx, value in enumerate(filled):
        if not np.isnan(value):
            continue
        prev_value = float("nan")
        next_value = float("nan")

        for prev_idx in range(idx - 1, -1, -1):
            candidate = raw_values[prev_idx]
            if not np.isnan(candidate):
                prev_value = float(candidate)
                break
        for next_idx in range(idx + 1, len(raw_values)):
            candidate = raw_values[next_idx]
            if not np.isnan(candidate):
                next_value = float(candidate)
                break

        if not np.isnan(prev_value) and not np.isnan(next_value):
            filled[idx] = float((prev_value + next_value) / 2.0)
        elif not np.isnan(prev_value):
            filled[idx] = prev_value
        elif not np.isnan(next_value):
            filled[idx] = next_value
        else:
            filled[idx] = float(fallback_value)

    return [float(v) for v in filled]


def _enforce_non_decreasing(values: Sequence[Any]) -> List[float]:
    out = []
    current = float("-inf")
    for value in values:
        item = float(value)
        if np.isnan(item):
            item = current if current != float("-inf") else 0.0
        current = max(current, item)
        out.append(float(current))
    return out


def _build_grade_count_map(
    sample_rows_with_gt: Sequence[Dict[str, Any]],
    feature_key: str,
) -> Dict[str, int]:
    gt_col = "%s_gt_quality_grade" % feature_key
    return {
        grade: int(
            sum(1 for row in sample_rows_with_gt if str(row.get(gt_col, "")) == grade)
        )
        for grade in GRADE_ORDER
    }


def _collect_feature_values_by_grade(
    sample_rows_with_gt: Sequence[Dict[str, Any]],
    feature_key: str,
) -> Dict[str, List[float]]:
    gt_col = "%s_gt_quality_grade" % feature_key
    values_by_grade = {grade: [] for grade in GRADE_ORDER}
    for row in sample_rows_with_gt:
        grade = str(row.get(gt_col, "Unknown"))
        value = float(row.get(feature_key, np.nan))
        if grade in values_by_grade and not np.isnan(value):
            values_by_grade[grade].append(value)
    return values_by_grade


def _fit_simple_one_sided_standard(
    sample_rows_with_gt: Sequence[Dict[str, Any]],
    feature_key: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    direction = str(DEFAULT_FEATURE_SPECS[feature_key]["direction"])
    values_by_grade = _collect_feature_values_by_grade(sample_rows_with_gt, feature_key)
    all_values = []
    for value_list in values_by_grade.values():
        all_values.extend(value_list)
    overall_median = _nanmedian(all_values)
    if np.isnan(overall_median):
        overall_median = 0.0

    if direction == "higher_is_better":
        representative_order = ["Poor", "Average", "Good", "Excellent"]
        threshold_pairs = [
            ("average", "Poor", "Average"),
            ("good", "Average", "Good"),
            ("excellent", "Good", "Excellent"),
        ]
    else:
        representative_order = ["Excellent", "Good", "Average", "Poor"]
        threshold_pairs = [
            ("excellent", "Excellent", "Good"),
            ("good", "Good", "Average"),
            ("average", "Average", "Poor"),
        ]

    raw_representatives = {
        grade: _nanmedian(values_by_grade[grade]) for grade in GRADE_ORDER
    }
    ordered_raw_values = [raw_representatives[grade] for grade in representative_order]
    ordered_filled_values = _fill_missing_grade_representatives(
        ordered_raw_values,
        fallback_value=float(overall_median),
    )
    ordered_monotonic_values = sorted(float(v) for v in ordered_filled_values)
    monotonic_representatives = {
        representative_order[idx]: float(ordered_monotonic_values[idx])
        for idx in range(len(representative_order))
    }

    standard = {
        "mode": "one_sided",
        "direction": direction,
    }
    threshold_construction = {}
    for threshold_key, lower_grade, upper_grade in threshold_pairs:
        threshold_value = _midpoint(
            monotonic_representatives[lower_grade],
            monotonic_representatives[upper_grade],
        )
        standard[threshold_key] = float(threshold_value)
        threshold_construction[threshold_key] = {
            "from_grade_pair": [lower_grade, upper_grade],
            "from_values": [
                float(monotonic_representatives[lower_grade]),
                float(monotonic_representatives[upper_grade]),
            ],
            "midpoint": float(threshold_value),
        }

    metadata = {
        "feature_key": feature_key,
        "display_name": DEFAULT_FEATURE_SPECS[feature_key]["display_name"],
        "threshold_method": SIMPLE_THRESHOLD_METHOD,
        "mode": "one_sided",
        "direction": direction,
        "grade_counts": _build_grade_count_map(sample_rows_with_gt, feature_key),
        "overall_feature_median": float(overall_median),
        "representative_order": representative_order,
        "raw_feature_representatives_by_grade": {
            grade: float(raw_representatives[grade]) for grade in GRADE_ORDER
        },
        "filled_feature_representatives_by_grade": {
            representative_order[idx]: float(ordered_filled_values[idx])
            for idx in range(len(representative_order))
        },
        "monotonic_feature_representatives_by_grade": monotonic_representatives,
        "threshold_construction": threshold_construction,
    }
    return standard, metadata


def _fit_simple_two_sided_standard(
    sample_rows_with_gt: Sequence[Dict[str, Any]],
    feature_key: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    direction = str(DEFAULT_FEATURE_SPECS[feature_key]["direction"])
    values_by_grade = _collect_feature_values_by_grade(sample_rows_with_gt, feature_key)
    all_values = []
    for value_list in values_by_grade.values():
        all_values.extend(value_list)
    overall_feature_median = _nanmedian(all_values)
    if np.isnan(overall_feature_median):
        overall_feature_median = 0.0

    excellent_values = list(values_by_grade["Excellent"])
    high_quality_values = list(values_by_grade["Excellent"]) + list(values_by_grade["Good"])
    if excellent_values:
        center_source = "excellent_feature_median"
        center_candidates = excellent_values
    elif high_quality_values:
        center_source = "excellent_good_feature_median"
        center_candidates = high_quality_values
    else:
        center_source = "overall_feature_median"
        center_candidates = list(all_values)

    center = _nanmedian(center_candidates)
    if np.isnan(center):
        center = float(overall_feature_median)

    deviation_values_by_grade = {}
    for grade in GRADE_ORDER:
        deviation_values_by_grade[grade] = [
            float(abs(value - center)) for value in values_by_grade[grade]
        ]

    all_deviations = []
    for value_list in deviation_values_by_grade.values():
        all_deviations.extend(value_list)
    overall_deviation_median = _nanmedian(all_deviations)
    if np.isnan(overall_deviation_median):
        overall_deviation_median = 0.0

    representative_order = ["Excellent", "Good", "Average", "Poor"]
    raw_deviation_representatives = {
        grade: _nanmedian(deviation_values_by_grade[grade]) for grade in GRADE_ORDER
    }
    ordered_raw_values = [
        raw_deviation_representatives[grade] for grade in representative_order
    ]
    ordered_filled_values = _fill_missing_grade_representatives(
        ordered_raw_values,
        fallback_value=float(overall_deviation_median),
    )
    ordered_monotonic_values = sorted(float(v) for v in ordered_filled_values)
    monotonic_deviation_representatives = {
        representative_order[idx]: float(ordered_monotonic_values[idx])
        for idx in range(len(representative_order))
    }

    excellent_dev = _midpoint(
        monotonic_deviation_representatives["Excellent"],
        monotonic_deviation_representatives["Good"],
    )
    good_dev = _midpoint(
        monotonic_deviation_representatives["Good"],
        monotonic_deviation_representatives["Average"],
    )
    average_dev = _midpoint(
        monotonic_deviation_representatives["Average"],
        monotonic_deviation_representatives["Poor"],
    )

    standard = {
        "mode": "two_sided",
        "direction": direction,
        "center": float(center),
        "excellent_dev": float(excellent_dev),
        "good_dev": float(good_dev),
        "average_dev": float(average_dev),
    }
    metadata = {
        "feature_key": feature_key,
        "display_name": DEFAULT_FEATURE_SPECS[feature_key]["display_name"],
        "threshold_method": SIMPLE_THRESHOLD_METHOD,
        "mode": "two_sided",
        "direction": direction,
        "grade_counts": _build_grade_count_map(sample_rows_with_gt, feature_key),
        "center": float(center),
        "center_source": center_source,
        "center_candidate_count": int(len(center_candidates)),
        "overall_feature_median": float(overall_feature_median),
        "overall_deviation_median": float(overall_deviation_median),
        "feature_medians_by_grade": {
            grade: _nanmedian(values_by_grade[grade]) for grade in GRADE_ORDER
        },
        "raw_deviation_representatives_by_grade": {
            grade: float(raw_deviation_representatives[grade]) for grade in GRADE_ORDER
        },
        "filled_deviation_representatives_by_grade": {
            representative_order[idx]: float(ordered_filled_values[idx])
            for idx in range(len(representative_order))
        },
        "monotonic_deviation_representatives_by_grade": monotonic_deviation_representatives,
        "threshold_construction": {
            "excellent_dev": {
                "from_grade_pair": ["Excellent", "Good"],
                "midpoint": float(excellent_dev),
            },
            "good_dev": {
                "from_grade_pair": ["Good", "Average"],
                "midpoint": float(good_dev),
            },
            "average_dev": {
                "from_grade_pair": ["Average", "Poor"],
                "midpoint": float(average_dev),
            },
        },
    }
    return standard, metadata


def apply_thresholds_to_rows(
    sample_rows_with_gt: Sequence[Dict[str, Any]],
    grade_standards: Dict[str, Dict[str, Any]],
    feature_gt_part_map: Dict[str, str],
    total_score_bucket_map: Dict[str, str],
    eval_scope: str,
) -> List[Dict[str, Any]]:
    estimate_rows = []
    for row in sample_rows_with_gt:
        sample_name = str(row.get("sample_name", ""))
        for feature_key, part_key in feature_gt_part_map.items():
            standard = grade_standards[feature_key]
            feature_value = float(row.get(feature_key, np.nan))
            gt_grade = str(row.get("%s_gt_quality_grade" % feature_key, "Unknown"))
            gt_part_score_raw = float(row.get("label_%s" % part_key, np.nan))
            pred_grade = grade_value(feature_value, standard)
            pred_score = float(GRADE_TO_SCORE.get(pred_grade, np.nan))
            gt_quality_group = _quality_group(gt_grade)
            pred_quality_group = _quality_group(pred_grade)
            pred_score_error_raw = float("nan")
            if not np.isnan(pred_score) and not np.isnan(gt_part_score_raw):
                pred_score_error_raw = float(abs(pred_score - gt_part_score_raw))
            estimate_rows.append(
                {
                    "sample_name": sample_name,
                    "csv_path": row.get("csv_path", ""),
                    "frame_count": int(row.get("frame_count", 0)),
                    "eval_scope": eval_scope,
                    "feature_key": feature_key,
                    "feature_display_name": BIO_FEATURE_SPECS[feature_key]["display_name"],
                    "part_key": part_key,
                    "feature_value": feature_value,
                    "gt_grade": gt_grade,
                    "gt_part_score_raw": gt_part_score_raw,
                    "gt_quality_group": gt_quality_group,
                    "pred_grade": pred_grade,
                    "pred_score": pred_score,
                    "pred_quality_group": pred_quality_group,
                    "pred_score_error_raw": pred_score_error_raw,
                    "grouped_correct": (
                        1.0
                        if gt_quality_group is not None and gt_quality_group == pred_quality_group
                        else 0.0
                    )
                    if gt_quality_group is not None and pred_quality_group is not None
                    else float("nan"),
                    "alignment_value": _alignment_value(feature_value, standard),
                    "label_total": float(row.get("label_total", np.nan)),
                    "total_score_quartile": total_score_bucket_map.get(sample_name, "Unknown"),
                }
            )
    return estimate_rows


def build_simple_feature_diagnostics(
    sample_rows_with_gt: Sequence[Dict[str, Any]],
    grade_standards: Dict[str, Dict[str, Any]],
    feature_gt_part_map: Dict[str, str],
    threshold_fit_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    df = pd.DataFrame(sample_rows_with_gt)
    features = {}
    ranking = []

    for feature_key, part_key in feature_gt_part_map.items():
        feature_df = df.copy()
        gt_col = "%s_gt_quality_grade" % feature_key
        feature_value_medians = {}
        grade_counts = {}
        for grade in GRADE_ORDER:
            grade_df = feature_df[feature_df[gt_col] == grade]
            feature_value_medians[grade] = _nanmedian(grade_df[feature_key])
            grade_counts[grade] = int(len(grade_df))

        alignment_values = [
            _alignment_value(value, grade_standards[feature_key])
            for value in feature_df[feature_key].tolist()
        ]
        alignment_spearman = _safe_corr(
            alignment_values,
            feature_df["label_%s" % part_key],
            method="spearman",
        )
        features[feature_key] = {
            "feature_key": feature_key,
            "display_name": DEFAULT_FEATURE_SPECS[feature_key]["display_name"],
            "part_key": part_key,
            "threshold_shape": DEFAULT_THRESHOLD_SHAPE_MAP.get(feature_key, "one_sided"),
            "threshold_method": SIMPLE_THRESHOLD_METHOD,
            "grade_counts": grade_counts,
            "feature_medians_by_gt_grade": feature_value_medians,
            "feature_vs_gt_score_pearson": _safe_corr(
                feature_df[feature_key],
                feature_df["label_%s" % part_key],
                method="pearson",
            ),
            "feature_vs_gt_score_spearman": _safe_corr(
                feature_df[feature_key],
                feature_df["label_%s" % part_key],
                method="spearman",
            ),
            "alignment_vs_gt_score_pearson": _safe_corr(
                alignment_values,
                feature_df["label_%s" % part_key],
                method="pearson",
            ),
            "alignment_vs_gt_score_spearman": alignment_spearman,
            "threshold_standard": grade_standards[feature_key],
            "threshold_fit_meta": threshold_fit_meta[feature_key],
        }
        ranking.append(
            {
                "feature_key": feature_key,
                "part_key": part_key,
                "alignment_vs_gt_score_spearman": alignment_spearman,
                "abs_alignment_vs_gt_score_spearman": (
                    float(abs(alignment_spearman))
                    if not np.isnan(alignment_spearman)
                    else float("nan")
                ),
            }
        )

    ranking = sorted(
        ranking,
        key=lambda item: (
            -1.0
            if np.isnan(float(item["abs_alignment_vs_gt_score_spearman"]))
            else float(item["abs_alignment_vs_gt_score_spearman"])
        ),
        reverse=True,
    )
    return {
        "threshold_method": SIMPLE_THRESHOLD_METHOD,
        "relaxed_evaluation_rule": {
            "name": RELAXED_EVAL_RULE_NAME,
            "description": "If GT bucket score is 5, predicted bucket score 4 or 5 is accepted.",
        },
        "features": features,
        "ranking": ranking,
    }


def fit_thresholds_on_rows(
    sample_rows: Sequence[Dict[str, Any]],
    feature_gt_part_map: Dict[str, str],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
]:
    rows_with_gt, gt_quality_meta = build_gt_quality_reference(
        sample_feature_rows=list(sample_rows),
        feature_gt_part_map=feature_gt_part_map,
        gt_label_strategy="fixed_bins",
        gt_fixed_bins=DEFAULT_GT_FIXED_BINS,
        show_progress=False,
    )
    grade_standards, fit_metrics = build_gt_supervised_grade_standards(
        sample_feature_rows=rows_with_gt,
        feature_specs=BIO_FEATURE_SPECS,
        feature_gt_part_map=feature_gt_part_map,
        threshold_shape_map=BIO_THRESHOLD_SHAPE_MAP,
        show_progress=False,
        max_search_candidates=101,
    )
    grade_intervals = build_grade_intervals(
        grade_standards=grade_standards,
        feature_specs=BIO_FEATURE_SPECS,
    )
    return rows_with_gt, gt_quality_meta, grade_standards, fit_metrics, {
        "grade_intervals": grade_intervals,
    }


def attach_gt_quality_reference(
    sample_rows: Sequence[Dict[str, Any]],
    feature_gt_part_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows_with_gt, gt_quality_meta = build_gt_quality_reference(
        sample_feature_rows=list(sample_rows),
        feature_gt_part_map=feature_gt_part_map,
        gt_label_strategy="fixed_bins",
        gt_fixed_bins=DEFAULT_GT_FIXED_BINS,
        show_progress=False,
    )
    return rows_with_gt, gt_quality_meta


def build_sample_overlap_summary(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    train_names = sorted(
        {
            str(row.get("sample_name", "")).strip()
            for row in train_rows
            if str(row.get("sample_name", "")).strip()
        }
    )
    val_names = sorted(
        {
            str(row.get("sample_name", "")).strip()
            for row in val_rows
            if str(row.get("sample_name", "")).strip()
        }
    )
    overlap = sorted(set(train_names) & set(val_names))
    return {
        "train_sample_count": int(len(train_names)),
        "val_sample_count": int(len(val_names)),
        "overlap_sample_count": int(len(overlap)),
        "overlap_ratio_vs_train": (
            float(len(overlap)) / float(len(train_names))
            if train_names
            else float("nan")
        ),
        "overlap_ratio_vs_val": (
            float(len(overlap)) / float(len(val_names))
            if val_names
            else float("nan")
        ),
        "overlap_sample_names": overlap,
    }


def build_part_score_distribution(
    sample_rows_with_gt: Sequence[Dict[str, Any]],
    feature_gt_part_map: Dict[str, str],
) -> pd.DataFrame:
    records = []
    for feature_key, part_key in feature_gt_part_map.items():
        part_col = "label_%s" % part_key
        gt_col = "%s_gt_quality_grade" % feature_key
        feature_values = [
            float(row.get(feature_key, np.nan))
            for row in sample_rows_with_gt
            if not np.isnan(float(row.get(part_col, np.nan)))
        ]
        total = len(feature_values)
        grouped = {}
        for row in sample_rows_with_gt:
            score_value = float(row.get(part_col, np.nan))
            if np.isnan(score_value):
                continue
            grouped.setdefault(score_value, []).append(row)
        for score_value in sorted(grouped.keys()):
            rows = grouped[score_value]
            feature_series = [float(row.get(feature_key, np.nan)) for row in rows]
            gt_modes = pd.Series(
                [str(row.get(gt_col, "Unknown")) for row in rows],
                dtype="object",
            ).mode(dropna=True)
            records.append(
                {
                    "feature_key": feature_key,
                    "feature_display_name": DEFAULT_FEATURE_SPECS[feature_key]["display_name"],
                    "part_key": part_key,
                    "part_score": score_value,
                    "sample_count": len(rows),
                    "sample_ratio": (
                        float(len(rows)) / float(total) * 100.0 if total > 0 else float("nan")
                    ),
                    "feature_mean": _safe_mean(feature_series),
                    "feature_std": _safe_std(feature_series),
                    "feature_median": _safe_quantile(feature_series, 0.5),
                    "feature_q25": _safe_quantile(feature_series, 0.25),
                    "feature_q75": _safe_quantile(feature_series, 0.75),
                    "gt_grade_mode": gt_modes.iloc[0] if not gt_modes.empty else "Unknown",
                }
            )
    return pd.DataFrame(records)


def build_evaluation_summary(estimate_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(estimate_rows)
    records = []
    if df.empty:
        return pd.DataFrame(records)

    for eval_scope in sorted(df["eval_scope"].dropna().unique().tolist()):
        scope_df = df[df["eval_scope"] == eval_scope]
        for feature_key in sorted(scope_df["feature_key"].dropna().unique().tolist()):
            feature_df = scope_df[scope_df["feature_key"] == feature_key].copy()
            records.append(
                {
                    "eval_scope": eval_scope,
                    "feature_key": feature_key,
                    "feature_display_name": BIO_FEATURE_SPECS[feature_key]["display_name"],
                    "part_key": str(feature_df["part_key"].iloc[0]) if not feature_df.empty else "",
                    "threshold_mode": str(BIO_THRESHOLD_SHAPE_MAP.get(feature_key, "one_sided")),
                    "sample_count": int(len(feature_df)),
                    "accuracy": _grouped_accuracy(
                        feature_df["pred_grade"],
                        feature_df["gt_grade"],
                    ),
                    "pred_score_mae_raw": _safe_mean(feature_df["pred_score_error_raw"]),
                }
            )
    return pd.DataFrame(records)


def build_group_summary(
    estimate_rows: Sequence[Dict[str, Any]],
    group_column: str,
) -> pd.DataFrame:
    df = pd.DataFrame(estimate_rows)
    records = []
    if df.empty:
        return pd.DataFrame(records)

    for eval_scope in sorted(df["eval_scope"].dropna().unique().tolist()):
        scope_df = df[df["eval_scope"] == eval_scope]
        for feature_key in sorted(scope_df["feature_key"].dropna().unique().tolist()):
            feature_df = scope_df[scope_df["feature_key"] == feature_key]
            for group_value, group_df in feature_df.groupby(group_column, dropna=False):
                records.append(
                    {
                        "eval_scope": eval_scope,
                        "feature_key": feature_key,
                        "feature_display_name": DEFAULT_FEATURE_SPECS[feature_key]["display_name"],
                        "part_key": str(group_df["part_key"].iloc[0]) if not group_df.empty else "",
                        "group_type": group_column,
                        "group_value": str(group_value),
                        "sample_count": int(len(group_df)),
                        "feature_mean": _safe_mean(group_df["feature_value"]),
                        "feature_std": _safe_std(group_df["feature_value"]),
                        "gt_score_mean": _safe_mean(group_df["gt_part_score_raw"]),
                        "pred_score_mean": _safe_mean(group_df["pred_score"]),
                        "strict_bucket_accuracy": _safe_score_accuracy(
                            group_df["pred_score"],
                            group_df["gt_part_score_bucket"],
                        ),
                        "relaxed_bucket_accuracy_5to45": _safe_mean(
                            group_df["relaxed_score_correct_5to45"]
                        ),
                        "strict_grade_accuracy": _safe_accuracy(
                            group_df["pred_grade"],
                            group_df["gt_grade"],
                        ),
                        "relaxed_grade_accuracy_5to45": _safe_accuracy(
                            group_df["pred_grade_relaxed_for_eval"],
                            group_df["gt_grade"],
                        ),
                        "grouped_grade_accuracy": _grouped_accuracy(
                            group_df["pred_grade"],
                            group_df["gt_grade"],
                        ),
                        "pred_score_mae_raw": _safe_mean(group_df["pred_score_error_raw"]),
                    }
                )
    return pd.DataFrame(records)


def plot_feature_boxplots(
    estimate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    eval_scope: str,
) -> Optional[str]:
    df = pd.DataFrame(estimate_rows)
    if df.empty:
        return None

    scope_df = df[df["eval_scope"] == eval_scope]
    if scope_df.empty:
        return None

    feature_order = list(DEFAULT_FEATURE_GT_PART_MAP.keys())
    fig, axes = plt.subplots(1, len(feature_order), figsize=(5.0 * len(feature_order), 4.5))
    if len(feature_order) == 1:
        axes = [axes]
    for idx, feature_key in enumerate(feature_order):
        ax = axes[idx]
        feature_df = scope_df[scope_df["feature_key"] == feature_key]
        grouped_values = []
        labels = []
        for grade in GRADE_ORDER:
            grade_df = feature_df[feature_df["gt_grade"] == grade]
            series = pd.Series(grade_df["feature_value"], dtype="float64").dropna()
            if series.empty:
                continue
            grouped_values.append(series.to_numpy())
            labels.append(grade)
        if grouped_values:
            ax.boxplot(grouped_values, tick_labels=labels, patch_artist=True)
        ax.set_title(DEFAULT_FEATURE_SPECS[feature_key]["display_name"])
        ax.set_xlabel("GT Grade")
        ax.set_ylabel("Feature Value")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def plot_prediction_scatter(
    estimate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    eval_scope: str,
) -> Optional[str]:
    df = pd.DataFrame(estimate_rows)
    if df.empty:
        return None

    scope_df = df[df["eval_scope"] == eval_scope]
    if scope_df.empty:
        return None

    feature_order = list(DEFAULT_FEATURE_GT_PART_MAP.keys())
    fig, axes = plt.subplots(1, len(feature_order), figsize=(5.0 * len(feature_order), 4.5))
    if len(feature_order) == 1:
        axes = [axes]
    for idx, feature_key in enumerate(feature_order):
        ax = axes[idx]
        feature_df = scope_df[scope_df["feature_key"] == feature_key]
        ax.scatter(
            feature_df["gt_part_score_raw"],
            feature_df["pred_score"],
            s=18,
            alpha=0.7,
        )
        ax.set_title(DEFAULT_FEATURE_SPECS[feature_key]["display_name"])
        ax.set_xlabel("GT Part Score")
        ax.set_ylabel("Estimated Score")
        ax.set_xlim(1.8, 5.2)
        ax.set_ylim(1.8, 5.2)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    train_dataset_root = Path(args.train_dataset_root)
    val_dataset_root = Path(args.val_dataset_root)
    output_root = Path(args.output_root)
    run_name = args.run_name.strip()
    if not run_name:
        run_name = "bio_feature_guidance_%s" % datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_gt_part_map = dict(BIO_FEATURE_GT_PART_MAP)
    current_extract_settings = _current_extract_action_feature_settings()

    train_sample_rows, train_failed_rows = collect_sample_feature_rows(
        dataset_root=train_dataset_root,
        max_frames=int(args.max_frames),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        max_samples=int(args.train_max_samples),
        sort_files=bool(args.sort_files),
    )
    val_sample_rows, val_failed_rows = collect_sample_feature_rows(
        dataset_root=val_dataset_root,
        max_frames=int(args.max_frames),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        max_samples=int(args.val_max_samples),
        sort_files=bool(args.sort_files),
    )
    overlap_summary = build_sample_overlap_summary(
        train_rows=train_sample_rows,
        val_rows=val_sample_rows,
    )
    val_total_score_bucket_map, val_total_score_quartiles = _build_total_score_buckets(
        val_sample_rows
    )

    train_rows_with_gt, train_gt_quality_meta, grade_standards, fit_metrics, extra = fit_thresholds_on_rows(
        sample_rows=train_sample_rows,
        feature_gt_part_map=feature_gt_part_map,
    )
    grade_intervals = extra["grade_intervals"]
    val_rows_with_gt, val_gt_quality_meta = attach_gt_quality_reference(
        sample_rows=val_sample_rows,
        feature_gt_part_map=feature_gt_part_map,
    )

    validation_estimates = apply_thresholds_to_rows(
        sample_rows_with_gt=val_rows_with_gt,
        grade_standards=grade_standards,
        feature_gt_part_map=feature_gt_part_map,
        total_score_bucket_map=val_total_score_bucket_map,
        eval_scope="validation",
    )

    evaluation_summary = build_evaluation_summary(validation_estimates)

    score_thresholds = {
        "gt_fixed_bins": {
            "average": float(DEFAULT_GT_FIXED_BINS[0]),
            "good": float(DEFAULT_GT_FIXED_BINS[1]),
            "excellent": float(DEFAULT_GT_FIXED_BINS[2]),
        },
        "quality_groups": {
            "high_quality": sorted(HIGH_QUALITY_GRADES),
            "low_quality": sorted(LOW_QUALITY_GRADES),
        },
        "feature_gt_part_map": feature_gt_part_map,
        "threshold_shape_map": dict(BIO_THRESHOLD_SHAPE_MAP),
        "threshold_method": THRESHOLD_METHOD_NAME,
        "reported_metrics": ["accuracy", "pred_score_mae_raw"],
        "threshold_fit_source": "train_dataset_only",
        "evaluation_source": "validation_dataset_only",
        "train_dataset_root": str(train_dataset_root),
        "val_dataset_root": str(val_dataset_root),
        "train_val_overlap": overlap_summary,
        "coordinate_scale": {
            "source": "dataset/csv normalized xy restored to current pixel scale",
            "image_width": int(args.image_width),
            "image_height": int(args.image_height),
        },
        "current_extract_action_features": current_extract_settings,
    }

    train_sample_features_path = run_dir / "train_sample_features.csv"
    train_failed_samples_path = run_dir / "train_failed_samples.csv"
    val_sample_features_path = run_dir / "val_sample_features.csv"
    val_failed_samples_path = run_dir / "val_failed_samples.csv"
    sample_estimates_validation_path = run_dir / "sample_estimates_validation.csv"
    evaluation_summary_path = run_dir / "correlation_summary.csv"
    score_thresholds_path = run_dir / "score_thresholds.json"
    thresholds_train_path = run_dir / "thresholds_train.json"
    threshold_intervals_path = run_dir / "threshold_intervals_train.json"
    threshold_fit_metrics_train_path = run_dir / "threshold_fit_metrics_train.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    save_dataframe(train_sample_features_path, pd.DataFrame(train_sample_rows))
    save_dataframe(val_sample_features_path, pd.DataFrame(val_sample_rows))
    save_dataframe(
        train_failed_samples_path,
        pd.DataFrame(train_failed_rows, columns=["sample_name", "csv_path", "error"]),
    )
    save_dataframe(
        val_failed_samples_path,
        pd.DataFrame(val_failed_rows, columns=["sample_name", "csv_path", "error"]),
    )
    save_dataframe(sample_estimates_validation_path, pd.DataFrame(validation_estimates))
    save_dataframe(evaluation_summary_path, evaluation_summary)
    save_json(score_thresholds_path, score_thresholds)
    save_json(thresholds_train_path, grade_standards)
    save_json(threshold_intervals_path, grade_intervals)
    save_json(threshold_fit_metrics_train_path, fit_metrics)

    summary = {
        "train_dataset_root": str(train_dataset_root),
        "train_csv_root": str(train_dataset_root / "csv"),
        "val_dataset_root": str(val_dataset_root),
        "val_csv_root": str(val_dataset_root / "csv"),
        "train_processed_sample_count": len(train_sample_rows),
        "train_failed_sample_count": len(train_failed_rows),
        "val_processed_sample_count": len(val_sample_rows),
        "val_failed_sample_count": len(val_failed_rows),
        "train_val_overlap": overlap_summary,
        "feature_gt_part_map": feature_gt_part_map,
        "threshold_method": THRESHOLD_METHOD_NAME,
        "threshold_fit_scope": "train_dataset_only",
        "evaluation_scope": "validation_dataset_only",
        "cross_validation_used": False,
        "reported_metrics": ["accuracy", "pred_score_mae_raw"],
        "accuracy_definition": "binary_grouped_accuracy (Excellent/Good -> high, Average/Poor -> low)",
        "binary_group_definition": {
            "high_quality": sorted(HIGH_QUALITY_GRADES),
            "low_quality": sorted(LOW_QUALITY_GRADES),
        },
        "train_gt_quality_meta": train_gt_quality_meta,
        "val_gt_quality_meta": val_gt_quality_meta,
        "current_extract_action_features": current_extract_settings,
        "val_total_score_quartiles": val_total_score_quartiles,
        "threshold_fit_metrics_train": fit_metrics,
        "threshold_fit_sample_count": len(train_rows_with_gt),
        "evaluation_summary": evaluation_summary.to_dict(orient="records"),
    }
    save_json(summary_path, summary)

    manifest = {
        "experiment_name": "bio_feature_guidance_validation",
        "run_dir": str(run_dir),
        "train_dataset_root": str(train_dataset_root),
        "val_dataset_root": str(val_dataset_root),
        "settings_policy": "current_repository_settings_only",
        "coordinate_scale": {
            "source": "dataset/csv normalized xy",
            "restored_image_width": int(args.image_width),
            "restored_image_height": int(args.image_height),
        },
        "current_extract_action_features": current_extract_settings,
        "feature_gt_part_map": feature_gt_part_map,
        "gt_fixed_bins": DEFAULT_GT_FIXED_BINS,
        "threshold_shape_map": dict(BIO_THRESHOLD_SHAPE_MAP),
        "threshold_method": THRESHOLD_METHOD_NAME,
        "threshold_fit_scope": "train_dataset_only",
        "evaluation_scope": "validation_dataset_only",
        "cross_validation_used": False,
        "reported_metrics": ["accuracy", "pred_score_mae_raw"],
        "accuracy_definition": "binary_grouped_accuracy (Excellent/Good -> high, Average/Poor -> low)",
        "binary_group_definition": {
            "high_quality": sorted(HIGH_QUALITY_GRADES),
            "low_quality": sorted(LOW_QUALITY_GRADES),
        },
        "train_max_samples": int(args.train_max_samples),
        "val_max_samples": int(args.val_max_samples),
        "train_val_overlap": overlap_summary,
        "outputs": {
            "train_sample_features_csv": str(train_sample_features_path),
            "train_failed_samples_csv": str(train_failed_samples_path),
            "val_sample_features_csv": str(val_sample_features_path),
            "val_failed_samples_csv": str(val_failed_samples_path),
            "sample_estimates_validation_csv": str(sample_estimates_validation_path),
            "correlation_summary_csv": str(evaluation_summary_path),
            "score_thresholds_json": str(score_thresholds_path),
            "thresholds_train_json": str(thresholds_train_path),
            "threshold_intervals_train_json": str(threshold_intervals_path),
            "threshold_fit_metrics_train_json": str(threshold_fit_metrics_train_path),
            "summary_json": str(summary_path),
        },
    }
    save_json(manifest_path, manifest)

    print("=" * 80)
    print("BIO feature guidance validation finished")
    print("=" * 80)
    print("train_dataset_root:", train_dataset_root)
    print("val_dataset_root:", val_dataset_root)
    print("train_processed_sample_count:", len(train_sample_rows))
    print("train_failed_sample_count:", len(train_failed_rows))
    print("val_processed_sample_count:", len(val_sample_rows))
    print("val_failed_sample_count:", len(val_failed_rows))
    print("train_val_overlap_sample_count:", int(overlap_summary["overlap_sample_count"]))
    print("run_dir:", run_dir)
    print("threshold_method:", THRESHOLD_METHOD_NAME)
    print("threshold_fit_scope: train_dataset_only")
    print("evaluation_scope: validation_dataset_only")
    print("reported_metrics: accuracy + pred_score_mae_raw")
    print("current_extract_action_features:", current_extract_settings)
    print("correlation_summary_csv:", evaluation_summary_path)
    print("summary_json:", summary_path)
    print("manifest_json:", manifest_path)


if __name__ == "__main__":
    main()

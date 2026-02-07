import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Tools.Exe_dataset.dataset_exe import MAX_FRAMES, load_single_csv_with_multipart_labels
from RTMPose.Bone_Feature_Extract import cal_math_features, extract_action_features

GRADE_ORDER = ["Excellent", "Good", "Average", "Poor"]

DEFAULT_FEATURE_SPECS: dict[str, dict[str, str]] = {
    "max_angle_avg": {
        "direction": "higher_is_better",
        "display_name": "Hand-Shoulder-Hand Angle",
    },
    "min_dist_avg": {
        "direction": "lower_is_better",
        "display_name": "Left Hand to Chin Distance",
    },
    "min_x_diff_avg": {
        "direction": "lower_is_better",
        "display_name": "Shoulder-Foot X Difference",
    },
}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _feature_keys_or_default(feature_keys: Optional[list[str]]) -> list[str]:
    if feature_keys is None:
        return list(DEFAULT_FEATURE_SPECS.keys())
    return list(feature_keys)


def collect_action_features_from_csv_folder(
    csv_folder_path: str,
    skip_first_n: int = 0,
    max_frames: int = MAX_FRAMES,
    sort_files: bool = True,
    feature_keys: Optional[list[str]] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Collect per-sample action features from a CSV folder.
    Returns:
        sample_feature_rows: each row contains sample_name, label_total and feature values.
        failed_rows: rows that failed to parse/process.
    """
    csv_folder = Path(csv_folder_path)
    if not csv_folder.exists():
        return [], [{"sample_name": "", "csv_path": str(csv_folder), "error": "csv_folder_not_found"}]

    csv_files = list(csv_folder.glob("*.csv"))
    if sort_files:
        csv_files = sorted(csv_files)
    if skip_first_n > 0:
        csv_files = csv_files[skip_first_n:]

    selected_keys = _feature_keys_or_default(feature_keys)
    sample_feature_rows: list[dict] = []
    failed_rows: list[dict] = []

    for csv_file in csv_files:
        try:
            data, _, label_total = load_single_csv_with_multipart_labels(str(csv_file), max_frames=max_frames)

            all_data = torch.from_numpy(data).float()
            all_data = all_data[:, :, :, 0]
            all_data = all_data.permute(1, 2, 0)

            math_feature = cal_math_features(all_data)
            action_features = extract_action_features(math_feature)

            row: dict[str, Any] = {
                "sample_name": csv_file.stem,
                "csv_path": str(csv_file),
                "label_total": _safe_float(label_total),
            }
            for key in selected_keys:
                row[key] = _safe_float(action_features.get(key, np.nan))
            sample_feature_rows.append(row)
        except Exception as e:
            failed_rows.append(
                {
                    "sample_name": csv_file.stem,
                    "csv_path": str(csv_file),
                    "error": str(e),
                }
            )

    return sample_feature_rows, failed_rows


def compute_feature_statistics(
    sample_feature_rows: list[dict],
    feature_keys: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Compute statistics for each feature:
    mean/std/min/max/median/q10/q25/q75/q90/values/count
    """
    selected_keys = _feature_keys_or_default(feature_keys)
    stats: dict[str, dict] = {}

    for key in selected_keys:
        vals = [_safe_float(row.get(key, np.nan)) for row in sample_feature_rows]
        arr = np.array(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]

        if arr.size == 0:
            stats[key] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "q10": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
                "q90": float("nan"),
                "values": [],
                "count": 0,
            }
            continue

        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q10": float(np.percentile(arr, 10)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "q90": float(np.percentile(arr, 90)),
            "values": [float(x) for x in arr.tolist()],
            "count": int(arr.size),
        }

    return stats


def build_quantile_grade_standards(
    stats: dict[str, dict],
    feature_specs: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    """
    Build quantile-based grade standards:
      higher_is_better: excellent=q75, good=median, average=q25
      lower_is_better: excellent=q25, good=median, average=q75
    """
    specs = feature_specs or DEFAULT_FEATURE_SPECS
    grade_standards: dict[str, dict] = {}

    for feature_key, spec in specs.items():
        if feature_key not in stats:
            continue
        s = stats[feature_key]
        direction = spec["direction"]

        if direction == "higher_is_better":
            excellent = _safe_float(s["q75"])
            good = _safe_float(s["median"])
            average = _safe_float(s["q25"])
        else:
            excellent = _safe_float(s["q25"])
            good = _safe_float(s["median"])
            average = _safe_float(s["q75"])

        grade_standards[feature_key] = {
            "direction": direction,
            "excellent": excellent,
            "good": good,
            "average": average,
        }

    return grade_standards


def grade_value(value: Any, standard: dict) -> str:
    """Map a numeric value to one of: Excellent/Good/Average/Poor."""
    val = _safe_float(value)
    if np.isnan(val):
        return "Unknown"

    direction = standard.get("direction")
    excellent = _safe_float(standard.get("excellent"))
    good = _safe_float(standard.get("good"))
    average = _safe_float(standard.get("average"))

    if any(np.isnan(x) for x in [excellent, good, average]):
        return "Unknown"

    if direction == "higher_is_better":
        if val >= excellent:
            return "Excellent"
        if val >= good:
            return "Good"
        if val >= average:
            return "Average"
        return "Poor"

    if val <= excellent:
        return "Excellent"
    if val <= good:
        return "Good"
    if val <= average:
        return "Average"
    return "Poor"


def grade_samples(
    sample_feature_rows: list[dict],
    grade_standards: dict[str, dict],
) -> list[dict]:
    """Add per-feature grade labels for each sample row."""
    graded_rows: list[dict] = []
    for row in sample_feature_rows:
        out = dict(row)
        for feature_key, standard in grade_standards.items():
            out[f"{feature_key}_grade"] = grade_value(row.get(feature_key), standard)
        graded_rows.append(out)
    return graded_rows


def summarize_grade_distribution(
    sample_grade_rows: list[dict],
    grade_standards: dict[str, dict],
) -> dict[str, dict]:
    """
    Count and percentage per grade for each feature.
    Returns:
        {
          feature_key: {
            "counts": {...},
            "percentages": {...},
            "total": int
          }
        }
    """
    summary: dict[str, dict] = {}
    for feature_key in grade_standards.keys():
        grade_col = f"{feature_key}_grade"
        counts = {g: 0 for g in GRADE_ORDER}
        total = 0
        for row in sample_grade_rows:
            grade = row.get(grade_col)
            if grade in counts:
                counts[grade] += 1
                total += 1

        percentages = {}
        for g in GRADE_ORDER:
            if total > 0:
                percentages[g] = float(counts[g] / total * 100.0)
            else:
                percentages[g] = 0.0

        summary[feature_key] = {
            "counts": counts,
            "percentages": percentages,
            "total": total,
        }
    return summary


def plot_grade_distribution_pies(
    distribution_summary: dict[str, dict],
    output_folder: str = "output_charts",
    feature_specs: Optional[dict[str, dict]] = None,
) -> list[str]:
    """Plot pie charts for each feature grade distribution."""
    specs = feature_specs or DEFAULT_FEATURE_SPECS
    os.makedirs(output_folder, exist_ok=True)
    output_paths: list[str] = []
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    for feature_key, item in distribution_summary.items():
        counts_map = item.get("counts", {})
        total = int(item.get("total", 0))
        if total <= 0:
            continue

        counts = [int(counts_map.get(g, 0)) for g in GRADE_ORDER]
        if sum(counts) == 0:
            continue

        display_name = specs.get(feature_key, {}).get("display_name", feature_key)
        fig, ax = plt.subplots(figsize=(9, 7))
        wedges, _, autotexts = ax.pie(
            counts,
            labels=GRADE_ORDER,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11, "weight": "bold"},
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(12)

        legend_labels = []
        for grade in GRADE_ORDER:
            c = int(counts_map.get(grade, 0))
            p = float(item.get("percentages", {}).get(grade, 0.0))
            legend_labels.append(f"{grade}: {c} ({p:.1f}%)")

        ax.set_title(f"{display_name} - Grade Distribution (Total: {total})", fontsize=13, weight="bold", pad=16)
        ax.legend(wedges, legend_labels, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10)
        fig.tight_layout()

        out_path = os.path.join(output_folder, f"{feature_key}_distribution.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(out_path)

    return output_paths


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_distribution_report(
    stats: dict[str, dict],
    grade_standards: dict[str, dict],
    sample_feature_rows: list[dict],
    sample_grade_rows: list[dict],
    failed_rows: list[dict],
    distribution_summary: dict[str, dict],
    output_folder: str = "report_output",
) -> dict[str, str]:
    """Save report artifacts and return output path mapping."""
    os.makedirs(output_folder, exist_ok=True)

    feature_stats_path = os.path.join(output_folder, "feature_stats.json")
    grade_standards_path = os.path.join(output_folder, "grade_standards.json")
    sample_features_path = os.path.join(output_folder, "sample_features.csv")
    sample_grades_path = os.path.join(output_folder, "sample_grades.csv")
    failed_samples_path = os.path.join(output_folder, "failed_samples.csv")
    grade_distribution_path = os.path.join(output_folder, "grade_distribution.json")
    standards_txt_path = os.path.join(output_folder, "biomechanics_grade_standards.txt")

    with open(feature_stats_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(stats), f, ensure_ascii=False, indent=2)
    with open(grade_standards_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(grade_standards), f, ensure_ascii=False, indent=2)
    with open(grade_distribution_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(distribution_summary), f, ensure_ascii=False, indent=2)

    pd.DataFrame(sample_feature_rows).to_csv(sample_features_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_grade_rows).to_csv(sample_grades_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failed_rows).to_csv(failed_samples_path, index=False, encoding="utf-8-sig")

    with open(standards_txt_path, "w", encoding="utf-8") as f:
        f.write("Biomechanics Grade Standards\n")
        f.write("=" * 60 + "\n")
        for feature_key, standard in grade_standards.items():
            display_name = DEFAULT_FEATURE_SPECS.get(feature_key, {}).get("display_name", feature_key)
            f.write(f"\n{feature_key} ({display_name}):\n")
            f.write(f"  Direction: {standard['direction']}\n")
            f.write(f"  Excellent threshold: {standard['excellent']:.6f}\n")
            f.write(f"  Good threshold: {standard['good']:.6f}\n")
            f.write(f"  Average threshold: {standard['average']:.6f}\n")
            feature_stats = stats.get(feature_key, {})
            f.write(f"  Count: {feature_stats.get('count', 0)}\n")
            f.write(f"  Mean: {_safe_float(feature_stats.get('mean')):.6f}\n")
            f.write(f"  Std: {_safe_float(feature_stats.get('std')):.6f}\n")
            f.write(f"  Min: {_safe_float(feature_stats.get('min')):.6f}\n")
            f.write(f"  Median: {_safe_float(feature_stats.get('median')):.6f}\n")
            f.write(f"  Max: {_safe_float(feature_stats.get('max')):.6f}\n")

    return {
        "feature_stats_json": feature_stats_path,
        "grade_standards_json": grade_standards_path,
        "sample_features_csv": sample_features_path,
        "sample_grades_csv": sample_grades_path,
        "failed_samples_csv": failed_samples_path,
        "grade_distribution_json": grade_distribution_path,
        "biomechanics_grade_standards_txt": standards_txt_path,
    }


def run_dataset_level_distribution_analysis(
    csv_folder_path: str,
    skip_first_n: int = 0,
    max_frames: int = MAX_FRAMES,
    sort_files: bool = True,
    feature_specs: Optional[dict[str, dict]] = None,
    output_folder: str = "report_output",
    chart_output_folder: str = "output_charts",
) -> dict:
    """
    One-stop orchestration:
    collect -> stats -> standards -> grading -> summary -> plot -> save
    """
    specs = feature_specs or DEFAULT_FEATURE_SPECS
    feature_keys = list(specs.keys())

    sample_feature_rows, failed_rows = collect_action_features_from_csv_folder(
        csv_folder_path=csv_folder_path,
        skip_first_n=skip_first_n,
        max_frames=max_frames,
        sort_files=sort_files,
        feature_keys=feature_keys,
    )
    stats = compute_feature_statistics(sample_feature_rows, feature_keys=feature_keys)
    grade_standards = build_quantile_grade_standards(stats, feature_specs=specs)
    sample_grade_rows = grade_samples(sample_feature_rows, grade_standards)
    distribution_summary = summarize_grade_distribution(sample_grade_rows, grade_standards)
    chart_paths = plot_grade_distribution_pies(
        distribution_summary=distribution_summary,
        output_folder=chart_output_folder,
        feature_specs=specs,
    )
    artifacts = save_distribution_report(
        stats=stats,
        grade_standards=grade_standards,
        sample_feature_rows=sample_feature_rows,
        sample_grade_rows=sample_grade_rows,
        failed_rows=failed_rows,
        distribution_summary=distribution_summary,
        output_folder=output_folder,
    )
    artifacts["chart_paths"] = chart_paths

    return {
        "stats": stats,
        "grade_standards": grade_standards,
        "distribution_summary": distribution_summary,
        "sample_feature_rows": sample_feature_rows,
        "sample_grade_rows": sample_grade_rows,
        "failed_rows": failed_rows,
        "artifacts": artifacts,
    }
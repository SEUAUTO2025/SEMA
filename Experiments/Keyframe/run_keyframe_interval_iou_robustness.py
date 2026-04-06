import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import traceback
import types
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RTMPose"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import ruptures  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    if "ruptures" not in sys.modules:
        sys.modules["ruptures"] = types.ModuleType("ruptures")

from RTMPose.Bone_Feature_Extract import (
    cal_math_features,
    extract_keyframes_with_ruptures_poseparts_2d,
    refine_keyframes_with_absdiff,
)

DEFAULT_DATASET_ROOT = "dataset" #这里统一按项目ROOT解析
DEFAULT_OUTPUT_ROOT = "Experiments/Keyframe/evaluation" #这里会计算到根目录下
DEFAULT_TARGET_K = 6
DEFAULT_PRED_START_KEYFRAME_RANK = 1
DEFAULT_PRED_END_CLOSE_GAP_MAX = 10
DEFAULT_CLEAN_REPEAT = 1
DEFAULT_NOISE_REPEAT = 1
DEFAULT_BASE_SEED = 20260313
DEFAULT_RUN_COMPARISON = True
DEFAULT_FASTDTW_REPO = "Experiments/baselines/fastdtw"
DEFAULT_FASTDTW_TEMPLATE_MANIFEST = "output_keyframes/manifest.json"
DEFAULT_FASTDTW_RADIUS = 5
DEFAULT_FASTDTW_FEATURE_MODE = "biomech_3"

NOISE_SPECS = {
    "gaussian_jitter": [0.002, 0.005, 0.01],
    "joint_mask": [0.15, 0.30, 0.45],
    "frame_mask": [0.15, 0.30, 0.50],
}

_GT_RELEASE_INTERVAL_CACHE: dict[str, dict[str, tuple[int, int]]] = {}
_FASTDTW_CALLABLE_CACHE: dict[str, Any] = {}
_FASTDTW_TEMPLATE_CACHE: dict[str, dict[str, Any]] = {}

@dataclass(frozen=True)
class SampleRef:
    sample_name: str
    video_path: str
    csv_path: str


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_enabled_noise_types(raw: str) -> list[str]:
    items = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not items:
        return list(NOISE_SPECS.keys())
    invalid = [item for item in items if item not in NOISE_SPECS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported noise types: {invalid}")
    return items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run keyframe interval IoU evaluation and RTMPose-noise robustness analysis."
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
    parser.add_argument("--target-k", type=int, default=DEFAULT_TARGET_K, help="Final keyframe count after refinement.")
    parser.add_argument(
        "--pred-start-keyframe-rank",
        type=int,
        default=DEFAULT_PRED_START_KEYFRAME_RANK,
        help="1-based keyframe rank used as interval start (default: 3 means third keyframe).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum number of samples to process.")
    parser.add_argument("--noise-repeat", type=int, default=DEFAULT_NOISE_REPEAT, help="Repeat count for each noise condition.")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED, help="Base seed for deterministic noise.")
    parser.add_argument("--run-clean", type=_parse_bool, default=True, help="Whether to run the clean condition.")
    parser.add_argument("--run-noisy", type=_parse_bool, default=True, help="Whether to run noisy RTMPose conditions.")
    parser.add_argument(
        "--run-comparison",
        type=_parse_bool,
        default=DEFAULT_RUN_COMPARISON,
        help="Whether to run the FastDTW comparison baseline on the same clean/noisy inputs.",
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
        "--enabled-noise-types",
        type=_parse_enabled_noise_types,
        default=list(NOISE_SPECS.keys()),
        help="Comma-separated noise types from: gaussian_jitter,joint_mask,frame_mask",
    )
    return parser.parse_args()


def iter_dataset_samples(dataset_root: str) -> list[SampleRef]:
    root = Path(dataset_root)
    video_dir = root / "video"
    csv_dir = root / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    samples: list[SampleRef] = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        video_path = video_dir / f"{csv_path.stem}.mp4"
        samples.append(
            SampleRef(
                sample_name=csv_path.stem,
                video_path=str(video_path.resolve()) if video_path.exists() else "",
                csv_path=str(csv_path.resolve()),
            )
        )
    return samples


def load_pose_sequence_from_csv(csv_path: str) -> np.ndarray:
    csv_file = Path(str(csv_path or "").strip())
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"CSV has no rows: {csv_file}")
    if "frame" not in df.columns:
        raise ValueError(f"CSV missing required column 'frame': {csv_file}")

    num_joints = 133
    coord_cols: list[str] = []
    for joint_idx in range(num_joints):
        coord_cols.extend([f"x{joint_idx}", f"y{joint_idx}", f"z{joint_idx}"])

    missing_cols = [name for name in coord_cols if name not in df.columns]
    if missing_cols:
        head = ",".join(missing_cols[:6])
        raise ValueError(f"CSV missing pose columns ({head} ...): {csv_file}")

    frame_values = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(frame_values) & (frame_values >= 0.0)
    if not np.any(valid_mask):
        raise ValueError(f"CSV has no valid non-negative frame indices: {csv_file}")

    frame_idx = frame_values[valid_mask].astype(np.int64)
    coord_matrix = (
        df.loc[valid_mask, coord_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    coord_matrix = np.nan_to_num(coord_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    t = int(frame_idx.max()) + 1
    pose_seq = np.zeros((t, num_joints, 3), dtype=np.float64)
    for row_idx, frame_id in enumerate(frame_idx.tolist()):
        pose_seq[int(frame_id), :, :] = coord_matrix[row_idx].reshape(num_joints, 3)

    return pose_seq


def run_current_keyframe_extractor(
    pose_seq: np.ndarray,
    video_path: str,
    target_k: int,
    pred_start_keyframe_rank: int,
    pred_end_close_gap_max: int = DEFAULT_PRED_END_CLOSE_GAP_MAX,
) -> dict[str, Any]:
    base_result = extract_keyframes_with_ruptures_poseparts_2d(
        pose_seq,
        k=max(1, int(target_k)) + 3,
        print_all_frame_scores=False,
        print_selection_debug=False,
        candidate_multiplier=2
    )
    refined_result = refine_keyframes_with_absdiff(
        video_path=video_path,
        keyframe_result=base_result,
        k=max(1, int(target_k)),
    )
    frame_idx_list = [int(v) for v in (refined_result.get("frame_idx", []) or [])]
    frame_idx_list = sorted(set(frame_idx_list))
    if len(frame_idx_list) == 0:
        raise RuntimeError(
            f"No refined keyframes available after extraction/refinement (target_k={int(target_k)})."
        )
    start_rank = max(1, int(pred_start_keyframe_rank))
    if start_rank > len(frame_idx_list):
        raise RuntimeError(
            f"Requested pred_start_keyframe_rank={start_rank} exceeds available refined keyframes={len(frame_idx_list)}: {frame_idx_list}"
        )
    start_pos = start_rank - 1
    pred_start_frame = int(frame_idx_list[start_pos])
    selection_debug = refined_result.get("selection_debug")
    if not selection_debug:
        selection_debug = base_result.get("selection_debug", {})
    selection_debug = dict(selection_debug)
    focus_end_value = _parse_int_like(selection_debug.get("focus_end"))
    last_keyframe = int(frame_idx_list[-1])
    close_gap_max = max(0, int(pred_end_close_gap_max))
    if focus_end_value is None:
        pred_end_frame = int(last_keyframe)
        pred_end_source = "last_keyframe_fallback"
    else:
        focus_end_value = int(focus_end_value)
        #FIXME 斟酌一下这里的逻辑
        if abs(int(last_keyframe) - int(focus_end_value)) <= int(close_gap_max):
            pred_end_frame = int(last_keyframe)
            pred_end_source = "last_keyframe_close_to_focus_end"
        else:
            pred_end_frame = int(focus_end_value)
            pred_end_source = "focus_end"
    selection_debug["pred_start_keyframe_rank_requested"] = int(pred_start_keyframe_rank)
    selection_debug["pred_start_keyframe_rank_used"] = int(start_pos + 1)
    selection_debug["pred_end_close_gap_max"] = int(close_gap_max)
    selection_debug["last_keyframe"] = int(last_keyframe)
    selection_debug["focus_end_raw"] = None if focus_end_value is None else int(focus_end_value)
    selection_debug["pred_end_source"] = str(pred_end_source)
    selection_debug["pred_end_frame_used"] = int(pred_end_frame)
    return {
        "frame_idx_list": frame_idx_list,
        "pred_start_frame": pred_start_frame,
        "pred_end_frame": pred_end_frame,
        "pred_span_len": int(pred_end_frame - pred_start_frame + 1),
        "selection_debug": selection_debug,
    }


def _parse_int_like(value: Any) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _read_text_with_fallback(path: Path) -> str:
    last_exc: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to read keyframe annotation file: {path}") from last_exc


def _build_release_interval_index(keyframe_dir: Path) -> dict[str, tuple[int, int]]:
    if not keyframe_dir.exists():
        return {}

    raw_intervals: dict[str, list[tuple[int, int]]] = {}
    for txt_path in sorted(keyframe_dir.glob("*.txt")):
        text = _read_text_with_fallback(txt_path)
        reader = csv.DictReader(StringIO(text))
        fieldnames = reader.fieldnames or []
        field_map = {
            str(name).strip().lower(): str(name)
            for name in fieldnames
            if str(name).strip()
        }
        url_key = field_map.get("url")
        tag_key = field_map.get("tag")
        start_key = field_map.get("start")
        end_key = field_map.get("end")
        if not url_key or not tag_key or not start_key or not end_key:
            continue

        for row in reader:
            tag_value = str(row.get(tag_key, "")).strip().lower()
            if tag_value != "release":
                continue
            video_name = str(row.get(url_key, "")).strip()
            if not video_name:
                continue
            start_frame = _parse_int_like(row.get(start_key))
            end_frame = _parse_int_like(row.get(end_key))
            if start_frame is None or end_frame is None:
                continue
            if start_frame < 0 or end_frame < start_frame:
                continue

            sample_key = Path(video_name).stem.strip().lower()
            if not sample_key:
                continue
            raw_intervals.setdefault(sample_key, []).append((int(start_frame), int(end_frame)))

    merged: dict[str, tuple[int, int]] = {}
    for sample_key, interval_list in raw_intervals.items():
        starts = [interval[0] for interval in interval_list]
        ends = [interval[1] for interval in interval_list]
        merged[sample_key] = (int(min(starts)), int(max(ends)))
    return merged


def _get_release_interval_index(keyframe_dir: Path) -> dict[str, tuple[int, int]]:
    cache_key = str(keyframe_dir.resolve())
    cached = _GT_RELEASE_INTERVAL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parsed = _build_release_interval_index(keyframe_dir)
    _GT_RELEASE_INTERVAL_CACHE[cache_key] = parsed
    return parsed


def load_gt_interval_from_csv(csv_path: str) -> Optional[tuple[int, int]]:
    path_text = str(csv_path or "").strip()
    if not path_text:
        return None
    sample_file = Path(path_text).resolve()
    sample_key = sample_file.stem.strip().lower()
    if not sample_key:
        return None

    # dataset/csv/<sample>.csv or dataset/video/<sample>.mp4 -> dataset/keyframes/*.txt
    try:
        parent_name = sample_file.parent.name.strip().lower()
        if parent_name in {"csv", "video"}:
            keyframe_dir = sample_file.parent.parent / "keyframes"
        else:
            keyframe_dir = sample_file.parent / "keyframes"
    except Exception:
        return None

    release_index = _get_release_interval_index(keyframe_dir)
    return release_index.get(sample_key)


def interval_to_set(start_frame: Optional[int], end_frame: Optional[int]) -> set[int]:
    if start_frame is None or end_frame is None:
        return set()
    try:
        start = int(start_frame)
        end = int(end_frame)
    except Exception:
        return set()
    if start < 0 or end < 0 or end < start:
        return set()
    return set(range(start, end + 1))


def compute_interval_iou(
    pred_interval: Optional[tuple[int, int]],
    gt_interval: Optional[tuple[int, int]],
) -> Optional[float]:
    if pred_interval is None or gt_interval is None:
        return None
    pred_set = interval_to_set(*pred_interval)
    gt_set = interval_to_set(*gt_interval)
    if not gt_set:
        return None
    if not pred_set:
        return 0.0
    union = pred_set | gt_set
    if not union:
        return None
    inter = pred_set & gt_set
    return float(len(inter) / len(union))


def apply_pose_noise(
    pose_seq: np.ndarray,
    noise_type: str,
    severity_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pose = np.asarray(pose_seq, dtype=np.float64)
    noisy = np.array(pose, copy=True)
    if noisy.ndim != 3 or noisy.shape[2] != 3:
        raise ValueError(f"Unexpected pose sequence shape for noise injection: {noisy.shape}")

    t, v, _ = noisy.shape
    if t == 0:
        return noisy

    if noise_type == "gaussian_jitter":
        noisy[:, :, :2] += rng.normal(loc=0.0, scale=float(severity_value), size=(t, v, 2))
        noisy[:, :, :2] = np.clip(noisy[:, :, :2], 0.0, 1.0)
        return noisy

    temporal_median_xy = np.median(pose[:, :, :2], axis=0)

    if noise_type == "joint_mask":
        mask = rng.random((t, v)) < float(severity_value)
        for coord in range(2):
            coord_values = noisy[:, :, coord]
            fill_values = np.broadcast_to(temporal_median_xy[:, coord], (t, v))
            coord_values[mask] = fill_values[mask]
        noisy[:, :, :2] = np.clip(noisy[:, :, :2], 0.0, 1.0)
        return noisy

    if noise_type == "frame_mask":
        if t <= 1:
            return noisy
        mask_count = int(round(float(severity_value) * t))
        mask_count = min(max(mask_count, 0), t - 1)
        if mask_count == 0:
            return noisy
        masked_frames = np.sort(rng.choice(np.arange(t), size=mask_count, replace=False))
        valid_mask = np.ones(t, dtype=bool)
        valid_mask[masked_frames] = False
        valid_idx = np.flatnonzero(valid_mask)
        for joint_idx in range(v):
            for coord in range(2):
                series = noisy[:, joint_idx, coord]
                interp = np.interp(np.arange(t), valid_idx, series[valid_idx])
                series[~valid_mask] = interp[~valid_mask]
        noisy[:, :, :2] = np.clip(noisy[:, :, :2], 0.0, 1.0)
        return noisy

    raise ValueError(f"Unsupported noise_type: {noise_type}")


def _resolve_path_from_root(path_text: str) -> Path:
    path = Path(str(path_text or "").strip())
    if path.is_absolute():
        return path.resolve()
    return (Path(ROOT) / path).resolve()


def _load_fastdtw_callable(fastdtw_repo: str):
    repo_path = _resolve_path_from_root(fastdtw_repo)
    cache_key = str(repo_path)
    cached = _FASTDTW_CALLABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    module_path = repo_path / "fastdtw" / "fastdtw.py"
    if not module_path.exists():
        raise FileNotFoundError(f"FastDTW source file not found: {module_path}")
    module_name = "_external_fastdtw_" + re.sub(r"[^0-9a-zA-Z_]+", "_", cache_key)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load FastDTW module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fastdtw_fn = getattr(module, "fastdtw", None)
    if fastdtw_fn is None or not callable(fastdtw_fn):
        raise ImportError(f"No callable fastdtw(...) found in: {module_path}")
    _FASTDTW_CALLABLE_CACHE[cache_key] = fastdtw_fn
    return fastdtw_fn


def _normalize_feature_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.size == 0:
        return matrix
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (matrix - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _build_fastdtw_feature_sequence(pose_seq: np.ndarray, feature_mode: str) -> np.ndarray:
    feature_mode = str(feature_mode or DEFAULT_FASTDTW_FEATURE_MODE).strip().lower()
    if feature_mode != "biomech_3":
        raise ValueError(f"Unsupported fastdtw feature mode: {feature_mode}")
    math_features = cal_math_features(pose_seq, plot_angle_curve=False)
    feature_columns = [
        np.asarray(math_features["shoulder_wrist_angle"].detach().cpu().numpy(), dtype=np.float64),
        np.asarray(math_features["left_hand_to_chin_dist"].detach().cpu().numpy(), dtype=np.float64),
        np.asarray(math_features["torso_tilt_deg"].detach().cpu().numpy(), dtype=np.float64),
    ]
    feature_matrix = np.stack(feature_columns, axis=1)
    return _normalize_feature_matrix(feature_matrix)


def _infer_dataset_root_from_video_path(video_path: str) -> Path:
    video_file = Path(str(video_path or "").strip()).resolve()
    if video_file.exists() and video_file.parent.name.strip().lower() == "video":
        return video_file.parent.parent.resolve()
    return (Path(ROOT) / DEFAULT_DATASET_ROOT).resolve()


def _load_fastdtw_template_reference(
    video_path: str,
    template_manifest_path: str,
    feature_mode: str,
) -> dict[str, Any]:
    dataset_root = _infer_dataset_root_from_video_path(video_path)
    manifest_path = _resolve_path_from_root(template_manifest_path)
    cache_key = f"{str(dataset_root)}::{str(manifest_path)}::{str(feature_mode)}"
    cached = _FASTDTW_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not manifest_path.exists():
        raise FileNotFoundError(f"FastDTW template manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_video_path = str(manifest.get("video_path", "")).strip()
    template_sample_name = Path(template_video_path).stem.strip()
    if not template_sample_name:
        raise ValueError(f"Template manifest missing video_path stem: {manifest_path}")
    template_csv_path = (dataset_root / "csv" / f"{template_sample_name}.csv").resolve()
    if not template_csv_path.exists():
        raise FileNotFoundError(f"Template CSV not found: {template_csv_path}")
    template_pose_seq = load_pose_sequence_from_csv(str(template_csv_path))
    template_feature_seq = _build_fastdtw_feature_sequence(template_pose_seq, feature_mode=feature_mode)
    template_interval = load_gt_interval_from_csv(str(template_csv_path))
    if template_interval is None:
        raise RuntimeError(f"Template release interval not found for sample: {template_sample_name}")
    template_keyframes = [
        int(row.get("frame_idx"))
        for row in (manifest.get("keyframes", []) or [])
        if _parse_int_like(row.get("frame_idx")) is not None
    ]
    reference = {
        "template_sample_name": template_sample_name,
        "template_csv_path": str(template_csv_path),
        "template_manifest_path": str(manifest_path),
        "template_interval": (int(template_interval[0]), int(template_interval[1])),
        "template_keyframes": sorted(set(int(v) for v in template_keyframes if int(v) >= 0)),
        "template_feature_seq": template_feature_seq,
    }
    _FASTDTW_TEMPLATE_CACHE[cache_key] = reference
    return reference


def _subsample_sorted_frames(frame_idx_list: list[int], target_k: int) -> list[int]:
    unique_frames = sorted(set(int(v) for v in frame_idx_list if int(v) >= 0))
    if target_k <= 0 or len(unique_frames) <= target_k:
        return unique_frames
    positions = np.linspace(0, len(unique_frames) - 1, num=int(target_k))
    selected = [unique_frames[int(round(pos))] for pos in positions.tolist()]
    return sorted(set(int(v) for v in selected))


def _map_template_frames_via_path(path: list[tuple[int, int]], template_frames: list[int]) -> list[int]:
    if not path or not template_frames:
        return []
    aligned_frames_by_template: dict[int, list[int]] = {}
    for sample_idx, template_idx in path:
        aligned_frames_by_template.setdefault(int(template_idx), []).append(int(sample_idx))
    mapped_frames: list[int] = []
    for template_frame in template_frames:
        aligned_sample_frames = aligned_frames_by_template.get(int(template_frame), [])
        if not aligned_sample_frames:
            continue
        mapped_frames.append(int(round(float(np.median(aligned_sample_frames)))))
    return sorted(set(mapped_frames))


def _map_template_interval_via_path(
    path: list[tuple[int, int]],
    template_interval: tuple[int, int],
) -> tuple[Optional[tuple[int, int]], list[int]]:
    if not path:
        return None, []
    start_frame = int(template_interval[0])
    end_frame = int(template_interval[1])
    aligned_sample_frames = sorted(
        set(
            int(sample_idx)
            for sample_idx, template_idx in path
            if start_frame <= int(template_idx) <= end_frame
        )
    )
    if not aligned_sample_frames:
        return None, []
    return (
        (int(aligned_sample_frames[0]), int(aligned_sample_frames[-1])),
        aligned_sample_frames,
    )


def extract_keyframes_with_comparison_method(
    pose_seq: np.ndarray,
    video_path: str,
    target_k: int,
    fastdtw_repo: str = DEFAULT_FASTDTW_REPO,
    template_manifest_path: str = DEFAULT_FASTDTW_TEMPLATE_MANIFEST,
    fastdtw_radius: int = DEFAULT_FASTDTW_RADIUS,
    feature_mode: str = DEFAULT_FASTDTW_FEATURE_MODE,
):
    if np.asarray(pose_seq).ndim != 3 or np.asarray(pose_seq).shape[0] <= 0:
        raise ValueError(f"Invalid pose sequence for FastDTW baseline: shape={np.asarray(pose_seq).shape}")
    fastdtw_fn = _load_fastdtw_callable(fastdtw_repo)
    template_ref = _load_fastdtw_template_reference(
        video_path=video_path,
        template_manifest_path=template_manifest_path,
        feature_mode=feature_mode,
    )
    sample_feature_seq = _build_fastdtw_feature_sequence(pose_seq, feature_mode=feature_mode)
    radius = max(1, int(fastdtw_radius))
    distance, path = fastdtw_fn(
        sample_feature_seq,
        template_ref["template_feature_seq"],
        radius=radius,
        dist=2,
    )
    alignment_path = [(int(i), int(j)) for i, j in (path or [])]
    pred_interval, aligned_release_frames = _map_template_interval_via_path(
        alignment_path,
        template_ref["template_interval"],
    )
    mapped_keyframes = _map_template_frames_via_path(
        alignment_path,
        list(template_ref["template_keyframes"]),
    )
    frame_idx_list = _subsample_sorted_frames(mapped_keyframes, target_k=max(1, int(target_k)))
    pred_interval_source = "fastdtw_template_release_interval"
    if pred_interval is None:
        if frame_idx_list:
            pred_interval = (int(frame_idx_list[0]), int(frame_idx_list[-1]))
            pred_interval_source = "fastdtw_template_keyframes_fallback"
        else:
            raise RuntimeError("FastDTW alignment produced neither mapped release frames nor mapped keyframes.")
    pred_start_frame = int(pred_interval[0])
    pred_end_frame = int(pred_interval[1])
    if pred_end_frame < pred_start_frame:
        raise RuntimeError(
            f"FastDTW mapped invalid interval: start={pred_start_frame}, end={pred_end_frame}"
        )
    selection_debug = {
        "comparison_method": "fastdtw",
        "feature_mode": str(feature_mode),
        "fastdtw_repo": str(_resolve_path_from_root(fastdtw_repo)),
        "fastdtw_radius": int(radius),
        "distance_pnorm": 2,
        "alignment_distance": float(distance),
        "alignment_path_len": int(len(alignment_path)),
        "template_sample_name": str(template_ref["template_sample_name"]),
        "template_csv_path": str(template_ref["template_csv_path"]),
        "template_manifest_path": str(template_ref["template_manifest_path"]),
        "template_release_interval": [
            int(template_ref["template_interval"][0]),
            int(template_ref["template_interval"][1]),
        ],
        "template_keyframes": [int(v) for v in template_ref["template_keyframes"]],
        "mapped_template_keyframes_raw": [int(v) for v in mapped_keyframes],
        "aligned_release_frames_count": int(len(aligned_release_frames)),
        "aligned_release_frame_bounds": (
            []
            if not aligned_release_frames
            else [int(aligned_release_frames[0]), int(aligned_release_frames[-1])]
        ),
        "pred_interval_source": str(pred_interval_source),
    }
    return {
        "implemented": True,
        "frame_idx_list": frame_idx_list,
        "pred_start_frame": pred_start_frame,
        "pred_end_frame": pred_end_frame,
        "pred_span_len": int(pred_end_frame - pred_start_frame + 1),
        "selection_debug": selection_debug,
    }


def _build_run_dir(output_root: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = Path(ROOT) / root
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"rag_keyframe_iou_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _get_current_angle_cut_threshold_deg() -> Optional[float]:
    source_path = Path(ROOT) / "RTMPose" / "Bone_Feature_Extract.py"
    try:
        text = source_path.read_text(encoding="utf-8")
    except Exception:
        return None
    match = re.search(r"angle_cut_threshold_deg\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _build_noise_conditions(enabled_noise_types: list[str]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for noise_type in enabled_noise_types:
        severity_values = NOISE_SPECS.get(noise_type, [])
        for severity_rank, severity_value in enumerate(severity_values, start=1):
            conditions.append(
                {
                    "regime": "noisy_rtmpose",
                    "condition_id": f"{noise_type}_s{severity_rank}",
                    "noise_type": noise_type,
                    "severity_value": float(severity_value),
                    "severity_rank": int(severity_rank),
                }
            )
    return conditions


def _make_failure_row(
    sample_name: str,
    regime: str,
    condition_id: str,
    noise_type: str,
    severity_value: Optional[float],
    severity_rank: Optional[int],
    repeat_id: int,
    status: str,
    error: str,
) -> dict[str, Any]:
    return {
        "sample_name": sample_name,
        "regime": regime,
        "condition_id": condition_id,
        "noise_type": noise_type,
        "severity_value": severity_value,
        "severity_rank": severity_rank,
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
    severity_rank: int,
    repeat_id: int,
    gt_start_frame: Any,
    gt_end_frame: Any,
    gt_span_len: Any,
    status: str,
    error: str,
    pred_result: Optional[dict[str, Any]] = None,
    iou_value: Optional[float] = None,
) -> dict[str, Any]:
    pred_result = dict(pred_result or {})
    return {
        "sample_name": sample_name,
        "regime": regime,
        "condition_id": condition_id,
        "noise_type": noise_type,
        "severity_value": severity_value,
        "severity_rank": severity_rank,
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
        "gt_span_len": gt_span_len,
        "iou": iou_value if iou_value is not None else np.nan,
        "status": str(status),
        "error": str(error),
        "keyframes_json": _json_dumps(pred_result.get("frame_idx_list", []) or []),
        "selection_debug_json": _json_dumps(pred_result.get("selection_debug", {}) or {}),
    }


def _print_pred_interval(
    *,
    sample_name: str,
    regime: str,
    condition_id: str,
    repeat_id: int,
    pred_start_frame: int,
    pred_end_frame: int,
) -> None:
    print(
        f"[Interval] sample={sample_name} regime={regime} "
        f"condition={condition_id} repeat={repeat_id} "
        f"pred_start={int(pred_start_frame)} pred_end={int(pred_end_frame)}"
    )


def _print_failure_reason(
    *,
    sample_name: str,
    regime: str,
    condition_id: str,
    repeat_id: int,
    status: str,
    error: str,
) -> None:
    tqdm.write(
        f"[Error] sample={sample_name} regime={regime} "
        f"condition={condition_id} repeat={repeat_id} "
        f"status={status} error={error}"
    )


def _run_internal_iou_asserts() -> None:
    assert math.isclose(compute_interval_iou((10, 20), (10, 20)), 1.0)
    assert math.isclose(compute_interval_iou((10, 20), (21, 30)), 0.0)
    assert math.isclose(compute_interval_iou((10, 20), (20, 30)), 1.0 / 21.0)
    assert math.isclose(compute_interval_iou((10, 20), (12, 18)), 7.0 / 11.0)
    assert compute_interval_iou((10, 20), None) is None


def _plot_iou_vs_severity(summary_df: pd.DataFrame, out_path: str, clean_mean_iou: float) -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for noise_type, group in summary_df.groupby("noise_type"):
        if not noise_type:
            continue
        group = group.sort_values("severity_rank")
        ax.plot(
            group["severity_value"],
            group["mean_iou"],
            marker="o",
            linewidth=2.0,
            label=str(noise_type),
        )
    ax.axhline(clean_mean_iou, linestyle="--", linewidth=1.6, color="#222222", label="clean")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Keyframe Interval IoU vs Noise Severity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iou_drop_bar(summary_df: pd.DataFrame, out_path: str, clean_mean_iou: float) -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return
    noise_types = [noise_type for noise_type in summary_df["noise_type"].dropna().unique().tolist() if noise_type]
    if not noise_types:
        return

    x = np.arange(len(noise_types), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    width = 0.22

    for offset_idx, severity_rank in enumerate(sorted(summary_df["severity_rank"].dropna().unique().tolist())):
        group = summary_df[summary_df["severity_rank"] == severity_rank]
        drops = []
        for noise_type in noise_types:
            row = group[group["noise_type"] == noise_type]
            if row.empty:
                drops.append(np.nan)
            else:
                mean_iou = float(row.iloc[0]["mean_iou"])
                drops.append(clean_mean_iou - mean_iou if np.isfinite(mean_iou) else np.nan)
        ax.bar(x + (offset_idx - 1) * width, drops, width=width, label=f"s{int(severity_rank)}")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, rotation=15)
    ax.set_ylabel("Mean IoU Drop From Clean")
    ax.set_title("Noise-Induced IoU Drop")
    ax.legend(title="Severity")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iou_boxplot(records_df: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    if records_df.empty:
        return

    plot_frames = []
    clean_df = records_df[(records_df["regime"] == "clean") & records_df["iou"].notna()]
    if not clean_df.empty:
        plot_frames.append(("clean", clean_df["iou"].to_numpy(dtype=float)))

    noisy_df = records_df[(records_df["regime"] == "noisy_rtmpose") & records_df["iou"].notna()]
    if not noisy_df.empty:
        max_rank = noisy_df.groupby("noise_type")["severity_rank"].max().to_dict()
        for noise_type, severity_rank in sorted(max_rank.items()):
            sub = noisy_df[
                (noisy_df["noise_type"] == noise_type)
                & (noisy_df["severity_rank"] == severity_rank)
            ]
            if sub.empty:
                continue
            plot_frames.append((f"{noise_type}_s{int(severity_rank)}", sub["iou"].to_numpy(dtype=float)))

    if not plot_frames:
        return

    labels = [item[0] for item in plot_frames]
    values = [item[1] for item in plot_frames]
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Per-sample IoU")
    ax.set_title("IoU Distribution Under Clean and Highest-Severity Noise")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_summary_rows(
    condition_rows: list[dict[str, Any]],
    records_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for row in condition_rows:
        sub = records_df[records_df["condition_id"] == row["condition_id"]]
        valid = sub[(sub["status"] == "ok") & sub["iou"].notna()]
        ious = valid["iou"].to_numpy(dtype=float) if not valid.empty else np.array([], dtype=np.float64)
        if ious.size == 0:
            summary_rows.append(
                {
                    "condition_id": row["condition_id"],
                    "regime": row["regime"],
                    "noise_type": row["noise_type"],
                    "severity_value": row["severity_value"],
                    "severity_rank": row["severity_rank"],
                    "valid_n": 0,
                    "mean_iou": float("nan"),
                    "median_iou": float("nan"),
                    "std_iou": float("nan"),
                    "min_iou": float("nan"),
                    "max_iou": float("nan"),
                }
            )
            continue
        summary_rows.append(
            {
                "condition_id": row["condition_id"],
                "regime": row["regime"],
                "noise_type": row["noise_type"],
                "severity_value": row["severity_value"],
                "severity_rank": row["severity_rank"],
                "valid_n": int(ious.size),
                "mean_iou": float(np.mean(ious)),
                "median_iou": float(np.median(ious)),
                "std_iou": float(np.std(ious)),
                "min_iou": float(np.min(ious)),
                "max_iou": float(np.max(ious)),
            }
        )
    return summary_rows


def main() -> None:
    _run_internal_iou_asserts()
    args = _parse_args()
    if int(args.pred_start_keyframe_rank) < 1:
        raise ValueError("--pred-start-keyframe-rank must be >= 1")

    dataset_root_path = Path(args.dataset_root)
    if not dataset_root_path.is_absolute():
        dataset_root_path = Path(ROOT) / dataset_root_path
    dataset_root = str(dataset_root_path.resolve())
    samples = iter_dataset_samples(dataset_root)
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise RuntimeError(f"No valid samples found under: {dataset_root}")

    run_dir = _build_run_dir(args.output_root)
    angle_cut_threshold_deg = _get_current_angle_cut_threshold_deg()
    enabled_noise_types = list(args.enabled_noise_types)

    condition_rows: list[dict[str, Any]] = []
    if args.run_clean:
        condition_rows.append(
            {
                "regime": "clean",
                "condition_id": "clean",
                "noise_type": "",
                "severity_value": 0.0,
                "severity_rank": 0,
            }
        )
    if args.run_noisy:
        condition_rows.extend(_build_noise_conditions(enabled_noise_types))

    records: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    comparison_failed_rows: list[dict[str, Any]] = []
    noise_conditions = _build_noise_conditions(enabled_noise_types) if args.run_noisy else []
    comparison_preflight_error = ""
    if bool(args.run_comparison):
        try:
            _load_fastdtw_callable(str(args.fastdtw_repo))
            manifest_path = _resolve_path_from_root(str(args.fastdtw_template_manifest))
            if not manifest_path.exists():
                raise FileNotFoundError(f"FastDTW template manifest not found: {manifest_path}")
        except Exception as exc:
            comparison_preflight_error = f"{type(exc).__name__}: {exc}"

    for sample_idx, sample in enumerate(tqdm(samples, desc="Keyframe IoU Robustness")):
        try:
            pose_seq = load_pose_sequence_from_csv(sample.csv_path)
        except Exception as exc:
            error_text = f"load_pose_sequence_failed: {exc}"
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
                    severity_rank=0,
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
                    severity_rank=0,
                    repeat_id=0,
                    gt_start_frame=np.nan,
                    gt_end_frame=np.nan,
                    gt_span_len=np.nan,
                    status="extraction_failed",
                    error=error_text,
                )
            )
            if bool(args.run_comparison):
                comparison_failed_rows.append(
                    _make_failure_row(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean",
                        noise_type="",
                        severity_value=0.0,
                        severity_rank=0,
                        repeat_id=0,
                        status="extraction_failed",
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
                        severity_rank=0,
                        repeat_id=0,
                        gt_start_frame=np.nan,
                        gt_end_frame=np.nan,
                        gt_span_len=np.nan,
                        status="extraction_failed",
                        error=error_text,
                    )
                )
            continue

        gt_interval = load_gt_interval_from_csv(sample.csv_path)
        gt_status = "gt_pending"
        gt_start_frame = np.nan
        gt_end_frame = np.nan
        gt_span_len = np.nan
        if gt_interval is not None:
            gt_set = interval_to_set(*gt_interval)
            if gt_set:
                gt_status = "gt_ready"
                gt_start_frame = int(gt_interval[0])
                gt_end_frame = int(gt_interval[1])
                gt_span_len = int(gt_interval[1] - gt_interval[0] + 1)
            else:
                gt_status = "gt_invalid"

        if args.run_clean:
            try:
                clean_result = run_current_keyframe_extractor(
                    pose_seq=pose_seq,
                    video_path=sample.video_path,
                    target_k=int(args.target_k),
                    pred_start_keyframe_rank=int(args.pred_start_keyframe_rank),
                    pred_end_close_gap_max=int(DEFAULT_PRED_END_CLOSE_GAP_MAX),
                )
                _print_pred_interval(
                    sample_name=sample.sample_name,
                    regime="clean",
                    condition_id="clean",
                    repeat_id=0,
                    pred_start_frame=int(clean_result["pred_start_frame"]),
                    pred_end_frame=int(clean_result["pred_end_frame"]),
                )
                pred_interval = (
                    int(clean_result["pred_start_frame"]),
                    int(clean_result["pred_end_frame"]),
                )
                iou_value = compute_interval_iou(pred_interval, gt_interval)
                status = "ok" if iou_value is not None else gt_status
                if status != "ok":
                    failed_rows.append(
                        _make_failure_row(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            severity_rank=0,
                            repeat_id=0,
                            status=status,
                            error="" if status == "gt_pending" else status,
                        )
                    )
                records.append(
                    _make_eval_record(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean",
                        noise_type="",
                        severity_value=0.0,
                        severity_rank=0,
                        repeat_id=0,
                        gt_start_frame=gt_start_frame,
                        gt_end_frame=gt_end_frame,
                        gt_span_len=gt_span_len,
                        status=status,
                        error="",
                        pred_result=clean_result,
                        iou_value=iou_value,
                    )
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
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
                        severity_rank=0,
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
                        severity_rank=0,
                        repeat_id=0,
                        gt_start_frame=gt_start_frame,
                        gt_end_frame=gt_end_frame,
                        gt_span_len=gt_span_len,
                        status="extraction_failed",
                        error=error_text,
                    )
                )

        if bool(args.run_comparison) and args.run_clean:
            if comparison_preflight_error:
                comparison_failed_rows.append(
                    _make_failure_row(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean",
                        noise_type="",
                        severity_value=0.0,
                        severity_rank=0,
                        repeat_id=0,
                        status="comparison_extract_failed",
                        error=comparison_preflight_error,
                    )
                )
                comparison_records.append(
                    _make_eval_record(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean",
                        noise_type="",
                        severity_value=0.0,
                        severity_rank=0,
                        repeat_id=0,
                        gt_start_frame=gt_start_frame,
                        gt_end_frame=gt_end_frame,
                        gt_span_len=gt_span_len,
                        status="comparison_extract_failed",
                        error=comparison_preflight_error,
                    )
                )
            else:
                try:
                    comparison_clean_result = extract_keyframes_with_comparison_method(
                        pose_seq=pose_seq,
                        video_path=sample.video_path,
                        target_k=int(args.target_k),
                        fastdtw_repo=str(args.fastdtw_repo),
                        template_manifest_path=str(args.fastdtw_template_manifest),
                        fastdtw_radius=int(args.fastdtw_radius),
                        feature_mode=str(args.fastdtw_feature_mode),
                    )
                    _print_pred_interval(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean_fastdtw",
                        repeat_id=0,
                        pred_start_frame=int(comparison_clean_result["pred_start_frame"]),
                        pred_end_frame=int(comparison_clean_result["pred_end_frame"]),
                    )
                    comparison_pred_interval = (
                        int(comparison_clean_result["pred_start_frame"]),
                        int(comparison_clean_result["pred_end_frame"]),
                    )
                    comparison_iou = compute_interval_iou(comparison_pred_interval, gt_interval)
                    comparison_status = "ok" if comparison_iou is not None else gt_status
                    if comparison_status != "ok":
                        comparison_failed_rows.append(
                            _make_failure_row(
                                sample_name=sample.sample_name,
                                regime="clean",
                                condition_id="clean",
                                noise_type="",
                                severity_value=0.0,
                                severity_rank=0,
                                repeat_id=0,
                                status=comparison_status,
                                error="" if comparison_status == "gt_pending" else comparison_status,
                            )
                        )
                    comparison_records.append(
                        _make_eval_record(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            severity_rank=0,
                            repeat_id=0,
                            gt_start_frame=gt_start_frame,
                            gt_end_frame=gt_end_frame,
                            gt_span_len=gt_span_len,
                            status=comparison_status,
                            error="",
                            pred_result=comparison_clean_result,
                            iou_value=comparison_iou,
                        )
                    )
                except Exception as exc:
                    comparison_error_text = f"{type(exc).__name__}: {exc}"
                    _print_failure_reason(
                        sample_name=sample.sample_name,
                        regime="clean",
                        condition_id="clean_fastdtw",
                        repeat_id=0,
                        status="comparison_extract_failed",
                        error=comparison_error_text,
                    )
                    comparison_failed_rows.append(
                        _make_failure_row(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            severity_rank=0,
                            repeat_id=0,
                            status="comparison_extract_failed",
                            error=comparison_error_text,
                        )
                    )
                    comparison_records.append(
                        _make_eval_record(
                            sample_name=sample.sample_name,
                            regime="clean",
                            condition_id="clean",
                            noise_type="",
                            severity_value=0.0,
                            severity_rank=0,
                            repeat_id=0,
                            gt_start_frame=gt_start_frame,
                            gt_end_frame=gt_end_frame,
                            gt_span_len=gt_span_len,
                            status="comparison_extract_failed",
                            error=comparison_error_text,
                        )
                    )

        for condition_idx, condition in enumerate(noise_conditions):
            for repeat_id in range(int(args.noise_repeat)):
                seed = int(args.base_seed) + sample_idx * 10000 + condition_idx * 100 + repeat_id
                rng = np.random.default_rng(seed)
                noisy_pose_seq = None
                try:
                    noisy_pose_seq = apply_pose_noise(
                        pose_seq=pose_seq,
                        noise_type=str(condition["noise_type"]),
                        severity_value=float(condition["severity_value"]),
                        rng=rng,
                    )
                    noisy_result = run_current_keyframe_extractor(
                        pose_seq=noisy_pose_seq,
                        video_path=sample.video_path,
                        target_k=int(args.target_k),
                        pred_start_keyframe_rank=int(args.pred_start_keyframe_rank),
                        pred_end_close_gap_max=int(DEFAULT_PRED_END_CLOSE_GAP_MAX),
                    )
                    _print_pred_interval(
                        sample_name=sample.sample_name,
                        regime="noisy_rtmpose",
                        condition_id=str(condition["condition_id"]),
                        repeat_id=int(repeat_id),
                        pred_start_frame=int(noisy_result["pred_start_frame"]),
                        pred_end_frame=int(noisy_result["pred_end_frame"]),
                    )
                    pred_interval = (
                        int(noisy_result["pred_start_frame"]),
                        int(noisy_result["pred_end_frame"]),
                    )
                    iou_value = compute_interval_iou(pred_interval, gt_interval)
                    status = "ok" if iou_value is not None else gt_status
                    if status != "ok":
                        failed_rows.append(
                            _make_failure_row(
                                sample_name=sample.sample_name,
                                regime="noisy_rtmpose",
                                condition_id=str(condition["condition_id"]),
                                noise_type=str(condition["noise_type"]),
                                severity_value=float(condition["severity_value"]),
                                severity_rank=int(condition["severity_rank"]),
                                repeat_id=repeat_id,
                                status=status,
                                error="" if status == "gt_pending" else status,
                            )
                        )
                    records.append(
                        _make_eval_record(
                            sample_name=sample.sample_name,
                            regime="noisy_rtmpose",
                            condition_id=str(condition["condition_id"]),
                            noise_type=str(condition["noise_type"]),
                            severity_value=float(condition["severity_value"]),
                            severity_rank=int(condition["severity_rank"]),
                            repeat_id=int(repeat_id),
                            gt_start_frame=gt_start_frame,
                            gt_end_frame=gt_end_frame,
                            gt_span_len=gt_span_len,
                            status=status,
                            error="",
                            pred_result=noisy_result,
                            iou_value=iou_value,
                        )
                    )
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    _print_failure_reason(
                        sample_name=sample.sample_name,
                        regime="noisy_rtmpose",
                        condition_id=str(condition["condition_id"]),
                        repeat_id=int(repeat_id),
                        status="noisy_extract_failed",
                        error=error_text,
                    )
                    failed_rows.append(
                        _make_failure_row(
                            sample_name=sample.sample_name,
                            regime="noisy_rtmpose",
                            condition_id=str(condition["condition_id"]),
                            noise_type=str(condition["noise_type"]),
                            severity_value=float(condition["severity_value"]),
                            severity_rank=int(condition["severity_rank"]),
                            repeat_id=repeat_id,
                            status="noisy_extract_failed",
                            error=error_text,
                        )
                    )
                    records.append(
                        _make_eval_record(
                            sample_name=sample.sample_name,
                            regime="noisy_rtmpose",
                            condition_id=str(condition["condition_id"]),
                            noise_type=str(condition["noise_type"]),
                            severity_value=float(condition["severity_value"]),
                            severity_rank=int(condition["severity_rank"]),
                            repeat_id=int(repeat_id),
                            gt_start_frame=gt_start_frame,
                            gt_end_frame=gt_end_frame,
                            gt_span_len=gt_span_len,
                            status="noisy_extract_failed",
                            error=error_text,
                        )
                    )
                if bool(args.run_comparison):
                    if noisy_pose_seq is None:
                        comparison_error_text = "noisy_pose_sequence_unavailable"
                        comparison_failed_rows.append(
                            _make_failure_row(
                                sample_name=sample.sample_name,
                                regime="noisy_rtmpose",
                                condition_id=str(condition["condition_id"]),
                                noise_type=str(condition["noise_type"]),
                                severity_value=float(condition["severity_value"]),
                                severity_rank=int(condition["severity_rank"]),
                                repeat_id=int(repeat_id),
                                status="comparison_extract_failed",
                                error=comparison_error_text,
                            )
                        )
                        comparison_records.append(
                            _make_eval_record(
                                sample_name=sample.sample_name,
                                regime="noisy_rtmpose",
                                condition_id=str(condition["condition_id"]),
                                noise_type=str(condition["noise_type"]),
                                severity_value=float(condition["severity_value"]),
                                severity_rank=int(condition["severity_rank"]),
                                repeat_id=int(repeat_id),
                                gt_start_frame=gt_start_frame,
                                gt_end_frame=gt_end_frame,
                                gt_span_len=gt_span_len,
                                status="comparison_extract_failed",
                                error=comparison_error_text,
                            )
                        )
                    elif comparison_preflight_error:
                        comparison_failed_rows.append(
                            _make_failure_row(
                                sample_name=sample.sample_name,
                                regime="noisy_rtmpose",
                                condition_id=str(condition["condition_id"]),
                                noise_type=str(condition["noise_type"]),
                                severity_value=float(condition["severity_value"]),
                                severity_rank=int(condition["severity_rank"]),
                                repeat_id=int(repeat_id),
                                status="comparison_extract_failed",
                                error=comparison_preflight_error,
                            )
                        )
                        comparison_records.append(
                            _make_eval_record(
                                sample_name=sample.sample_name,
                                regime="noisy_rtmpose",
                                condition_id=str(condition["condition_id"]),
                                noise_type=str(condition["noise_type"]),
                                severity_value=float(condition["severity_value"]),
                                severity_rank=int(condition["severity_rank"]),
                                repeat_id=int(repeat_id),
                                gt_start_frame=gt_start_frame,
                                gt_end_frame=gt_end_frame,
                                gt_span_len=gt_span_len,
                                status="comparison_extract_failed",
                                error=comparison_preflight_error,
                            )
                        )
                    else:
                        try:
                            comparison_noisy_result = extract_keyframes_with_comparison_method(
                                pose_seq=noisy_pose_seq,
                                video_path=sample.video_path,
                                target_k=int(args.target_k),
                                fastdtw_repo=str(args.fastdtw_repo),
                                template_manifest_path=str(args.fastdtw_template_manifest),
                                fastdtw_radius=int(args.fastdtw_radius),
                                feature_mode=str(args.fastdtw_feature_mode),
                            )
                            _print_pred_interval(
                                sample_name=sample.sample_name,
                                regime="noisy_rtmpose",
                                condition_id=f"{str(condition['condition_id'])}_fastdtw",
                                repeat_id=int(repeat_id),
                                pred_start_frame=int(comparison_noisy_result["pred_start_frame"]),
                                pred_end_frame=int(comparison_noisy_result["pred_end_frame"]),
                            )
                            comparison_pred_interval = (
                                int(comparison_noisy_result["pred_start_frame"]),
                                int(comparison_noisy_result["pred_end_frame"]),
                            )
                            comparison_iou = compute_interval_iou(comparison_pred_interval, gt_interval)
                            comparison_status = "ok" if comparison_iou is not None else gt_status
                            if comparison_status != "ok":
                                comparison_failed_rows.append(
                                    _make_failure_row(
                                        sample_name=sample.sample_name,
                                        regime="noisy_rtmpose",
                                        condition_id=str(condition["condition_id"]),
                                        noise_type=str(condition["noise_type"]),
                                        severity_value=float(condition["severity_value"]),
                                        severity_rank=int(condition["severity_rank"]),
                                        repeat_id=int(repeat_id),
                                        status=comparison_status,
                                        error="" if comparison_status == "gt_pending" else comparison_status,
                                    )
                                )
                            comparison_records.append(
                                _make_eval_record(
                                    sample_name=sample.sample_name,
                                    regime="noisy_rtmpose",
                                    condition_id=str(condition["condition_id"]),
                                    noise_type=str(condition["noise_type"]),
                                    severity_value=float(condition["severity_value"]),
                                    severity_rank=int(condition["severity_rank"]),
                                    repeat_id=int(repeat_id),
                                    gt_start_frame=gt_start_frame,
                                    gt_end_frame=gt_end_frame,
                                    gt_span_len=gt_span_len,
                                    status=comparison_status,
                                    error="",
                                    pred_result=comparison_noisy_result,
                                    iou_value=comparison_iou,
                                )
                            )
                        except Exception as exc:
                            comparison_error_text = f"{type(exc).__name__}: {exc}"
                            _print_failure_reason(
                                sample_name=sample.sample_name,
                                regime="noisy_rtmpose",
                                condition_id=f"{str(condition['condition_id'])}_fastdtw",
                                repeat_id=int(repeat_id),
                                status="comparison_extract_failed",
                                error=comparison_error_text,
                            )
                            comparison_failed_rows.append(
                                _make_failure_row(
                                    sample_name=sample.sample_name,
                                    regime="noisy_rtmpose",
                                    condition_id=str(condition["condition_id"]),
                                    noise_type=str(condition["noise_type"]),
                                    severity_value=float(condition["severity_value"]),
                                    severity_rank=int(condition["severity_rank"]),
                                    repeat_id=int(repeat_id),
                                    status="comparison_extract_failed",
                                    error=comparison_error_text,
                                )
                            )
                            comparison_records.append(
                                _make_eval_record(
                                    sample_name=sample.sample_name,
                                    regime="noisy_rtmpose",
                                    condition_id=str(condition["condition_id"]),
                                    noise_type=str(condition["noise_type"]),
                                    severity_value=float(condition["severity_value"]),
                                    severity_rank=int(condition["severity_rank"]),
                                    repeat_id=int(repeat_id),
                                    gt_start_frame=gt_start_frame,
                                    gt_end_frame=gt_end_frame,
                                    gt_span_len=gt_span_len,
                                    status="comparison_extract_failed",
                                    error=comparison_error_text,
                                )
                            )

    records_df = pd.DataFrame(records)
    failed_df = pd.DataFrame(failed_rows)
    summary_rows = _make_summary_rows(condition_rows, records_df)
    summary_df = pd.DataFrame(summary_rows)
    comparison_records_df = pd.DataFrame(comparison_records)
    comparison_failed_df = pd.DataFrame(comparison_failed_rows)
    comparison_summary_rows = _make_summary_rows(condition_rows, comparison_records_df)
    comparison_summary_df = pd.DataFrame(comparison_summary_rows)

    clean_row = summary_df[summary_df["condition_id"] == "clean"]
    clean_mean_iou = float("nan")
    if not clean_row.empty:
        clean_mean_iou = _safe_float(clean_row.iloc[0]["mean_iou"])
    comparison_clean_row = comparison_summary_df[comparison_summary_df["condition_id"] == "clean"]
    comparison_clean_mean_iou = float("nan")
    if not comparison_clean_row.empty:
        comparison_clean_mean_iou = _safe_float(comparison_clean_row.iloc[0]["mean_iou"])

    valid_records = records_df[(records_df["status"] == "ok") & records_df["iou"].notna()]
    comparison_valid_records = comparison_records_df[
        (comparison_records_df["status"] == "ok") & comparison_records_df["iou"].notna()
    ]
    gt_pending_count = int((records_df["status"] == "gt_pending").sum()) if not records_df.empty else 0
    gt_invalid_count = int((records_df["status"] == "gt_invalid").sum()) if not records_df.empty else 0
    extraction_failed_count = int((records_df["status"] == "extraction_failed").sum()) if not records_df.empty else 0
    noisy_extract_failed_count = int((records_df["status"] == "noisy_extract_failed").sum()) if not records_df.empty else 0
    comparison_gt_pending_count = (
        int((comparison_records_df["status"] == "gt_pending").sum()) if not comparison_records_df.empty else 0
    )
    comparison_gt_invalid_count = (
        int((comparison_records_df["status"] == "gt_invalid").sum()) if not comparison_records_df.empty else 0
    )
    comparison_extract_failed_count = (
        int((comparison_records_df["status"] == "comparison_extract_failed").sum())
        if not comparison_records_df.empty
        else 0
    )

    fig_iou_vs_severity = run_dir / "fig_iou_vs_severity.png"
    fig_iou_drop_bar = run_dir / "fig_iou_drop_bar.png"
    fig_iou_boxplot = run_dir / "fig_iou_boxplot.png"
    plots_generated = False
    plot_skip_reason = "No valid GT-backed IoU records available."

    if not summary_df.empty and np.isfinite(clean_mean_iou) and not valid_records.empty:
        noisy_summary_df = summary_df[summary_df["regime"] == "noisy_rtmpose"].copy()
        if not noisy_summary_df.empty and noisy_summary_df["mean_iou"].notna().any():
            _plot_iou_vs_severity(noisy_summary_df, str(fig_iou_vs_severity), clean_mean_iou)
            _plot_iou_drop_bar(noisy_summary_df, str(fig_iou_drop_bar), clean_mean_iou)
            _plot_iou_boxplot(valid_records, str(fig_iou_boxplot))
            plots_generated = True
        else:
            plot_skip_reason = "No noisy GT-backed IoU records available."

    summary = {
        "dataset_root": dataset_root,
        "pose_source": "dataset_csv_pose_sequence",
        "run_dir": str(run_dir),
        "total_samples": int(len(samples)),
        "pred_start_keyframe_rank": int(args.pred_start_keyframe_rank),
        "pred_interval_definition": "start=keyframe_rank, end=last_keyframe_if_close_else_focus_end",
        "records_count": int(len(records_df)),
        "valid_iou_count": int(len(valid_records)),
        "gt_pending_count": gt_pending_count,
        "gt_invalid_count": gt_invalid_count,
        "extraction_failed_count": extraction_failed_count,
        "noisy_extract_failed_count": noisy_extract_failed_count,
        "clean_mean_iou": clean_mean_iou,
        "angle_cut_threshold_deg": angle_cut_threshold_deg,
        "plots_generated": plots_generated,
        "plot_skip_reason": "" if plots_generated else plot_skip_reason,
        "noise_condition_means": {
            str(row["condition_id"]): _safe_float(row["mean_iou"])
            for row in summary_rows
            if row["regime"] == "noisy_rtmpose"
        },
        "comparison_fastdtw": {
            "enabled": bool(args.run_comparison),
            "preflight_error": str(comparison_preflight_error),
            "records_count": int(len(comparison_records_df)),
            "valid_iou_count": int(len(comparison_valid_records)),
            "gt_pending_count": int(comparison_gt_pending_count),
            "gt_invalid_count": int(comparison_gt_invalid_count),
            "extract_failed_count": int(comparison_extract_failed_count),
            "clean_mean_iou": comparison_clean_mean_iou,
            "noise_condition_means": {
                str(row["condition_id"]): _safe_float(row["mean_iou"])
                for row in comparison_summary_rows
                if row["regime"] == "noisy_rtmpose"
            },
            "config": {
                "fastdtw_repo": str(_resolve_path_from_root(str(args.fastdtw_repo))),
                "template_manifest_path": str(_resolve_path_from_root(str(args.fastdtw_template_manifest))),
                "fastdtw_radius": int(args.fastdtw_radius),
                "feature_mode": str(args.fastdtw_feature_mode),
            },
        },
    }

    manifest = {
        "script_path": str((Path(ROOT) / "Experiments" / "Keyframe" / "run_keyframe_interval_iou_robustness.py").resolve()),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": dataset_root,
        "output_root": str(
            Path(args.output_root).resolve()
            if Path(args.output_root).is_absolute()
            else (Path(ROOT) / args.output_root).resolve()
        ),
        "run_dir": str(run_dir),
        "target_k": int(args.target_k),
        "pred_start_keyframe_rank": int(args.pred_start_keyframe_rank),
        "pred_end_close_gap_max": int(DEFAULT_PRED_END_CLOSE_GAP_MAX),
        "clean_repeat": int(DEFAULT_CLEAN_REPEAT),
        "noise_repeat": int(args.noise_repeat),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
        "base_seed": int(args.base_seed),
        "run_clean": bool(args.run_clean),
        "run_noisy": bool(args.run_noisy),
        "enabled_noise_types": enabled_noise_types,
        "noise_registry": {
            noise_type: [float(v) for v in NOISE_SPECS[noise_type]]
            for noise_type in enabled_noise_types
        },
        "angle_cut_threshold_deg": angle_cut_threshold_deg,
        "pose_extractor": {
            "function_name": "load_pose_sequence_from_csv",
            "current_behavior": "read pose sequence directly from dataset/csv/*.csv and use it as keyframe extractor input",
        },
        "gt_interval_loader": {
            "implemented": True,
            "function_name": "load_gt_interval_from_csv",
            "current_behavior": "loads Tag=Release start/end from dataset/keyframes/*.txt by sample stem",
        },
        "pred_interval_policy": {
            "start": "keyframe rank (1-based, configurable by --pred-start-keyframe-rank; must be <= available refined keyframe count)",
            "end": "if |last_keyframe - focus_end| <= pred_end_close_gap_max then use last_keyframe, else use focus_end",
        },
        "comparison_method": {
            "implemented": True,
            "function_name": "extract_keyframes_with_comparison_method",
            "default_enabled": bool(DEFAULT_RUN_COMPARISON),
            "name": "fastdtw",
            "fastdtw_repo": str(_resolve_path_from_root(str(args.fastdtw_repo))),
            "template_manifest_path": str(_resolve_path_from_root(str(args.fastdtw_template_manifest))),
            "fastdtw_radius": int(args.fastdtw_radius),
            "feature_mode": str(args.fastdtw_feature_mode),
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
    if bool(args.run_comparison):
        comparison_records_df.to_csv(comparison_records_csv, index=False, encoding="utf-8-sig")
        comparison_summary_df.to_csv(comparison_summary_csv, index=False, encoding="utf-8-sig")
        comparison_failed_df.to_csv(comparison_failed_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] Output directory: {run_dir}")
    print(f"[Done] records.csv: {records_csv}")
    print(f"[Done] summary_by_condition.csv: {summary_csv}")
    print(f"[Done] failed_samples.csv: {failed_csv}")
    if bool(args.run_comparison):
        print(f"[Done] comparison_records_fastdtw.csv: {comparison_records_csv}")
        print(f"[Done] comparison_summary_by_condition_fastdtw.csv: {comparison_summary_csv}")
        print(f"[Done] comparison_failed_samples_fastdtw.csv: {comparison_failed_csv}")
    print(f"[Done] summary.json: {summary_json}")
    print(f"[Done] manifest.json: {manifest_json}")
    if not plots_generated:
        print(f"[Info] Plot generation skipped: {plot_skip_reason}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Error] {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise

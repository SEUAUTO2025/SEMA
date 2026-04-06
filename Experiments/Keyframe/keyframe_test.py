import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "RTMPose") not in sys.path:
    sys.path.insert(0, str(ROOT / "RTMPose"))

try:
    import ruptures  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    if "ruptures" not in sys.modules:
        # Bone_Feature_Extract has an internal diff-event fallback when Binseg is unavailable.
        sys.modules["ruptures"] = types.ModuleType("ruptures")

from RTMPose.Bone_Feature_Extract import extract_keyframes_with_ruptures_poseparts_2d


DEFAULT_DATASET_ROOT = "dataset"
DEFAULT_TARGET_K = 6
DEFAULT_CANDIDATE_MULTIPLIER = 2

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a sample CSV and return extracted keyframe sequence."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root with csv/ subfolder (relative paths are resolved against project ROOT).",
    )
    parser.add_argument(
        "--sample-name",
        default="89",
        help="Sample stem in dataset/csv, e.g. 103. Ignored if --csv-path is provided.",
    )
    parser.add_argument(
        "--csv-path",
        default="",
        help="Direct path to sample CSV. Overrides --sample-name.",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=DEFAULT_TARGET_K,
        help="Target number of keyframes to extract (default: 8).",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=DEFAULT_CANDIDATE_MULTIPLIER,
        help="Candidate multiplier for keyframe extraction.",
    )
    parser.add_argument(
        "--print-all-frame-scores",
        action="store_true",
        help="Print per-frame score table from extractor.",
    )
    parser.add_argument(
        "--print-selection-debug",
        action="store_true",
        help="Print keyframe selection debug logs from extractor.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save the output JSON.",
    )
    return parser.parse_args()


def _resolve_from_root(path_text: str) -> Path:
    path = Path(str(path_text).strip())
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _resolve_csv_path(dataset_root: str, sample_name: str, csv_path: str) -> Path:
    direct = str(csv_path or "").strip()
    if direct:
        return _resolve_from_root(direct)
    sample = str(sample_name or "").strip()
    if not sample:
        raise ValueError("Provide either --csv-path or --sample-name.")
    return (_resolve_from_root(dataset_root) / "csv" / f"{sample}.csv").resolve()


def _load_pose_sequence_from_csv(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV has no rows: {csv_path}")
    if "frame" not in df.columns:
        raise ValueError(f"CSV missing required column 'frame': {csv_path}")

    num_joints = 133
    coord_cols: list[str] = []
    for joint_idx in range(num_joints):
        coord_cols.extend([f"x{joint_idx}", f"y{joint_idx}", f"z{joint_idx}"])

    missing_cols = [name for name in coord_cols if name not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing pose columns, e.g. {missing_cols[:6]}")

    frame_values = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(frame_values) & (frame_values >= 0.0)
    if not np.any(valid_mask):
        raise ValueError(f"CSV has no valid non-negative frame indices: {csv_path}")

    frame_idx = frame_values[valid_mask].astype(np.int64)
    coord_matrix = (
        df.loc[valid_mask, coord_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    coord_matrix = np.nan_to_num(coord_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    total_frames = int(frame_idx.max()) + 1
    pose_seq = np.zeros((total_frames, num_joints, 3), dtype=np.float64)
    for row_idx, frame_id in enumerate(frame_idx.tolist()):
        pose_seq[int(frame_id), :, :] = coord_matrix[row_idx].reshape(num_joints, 3)

    return pose_seq


def _serialize_result(result: dict[str, Any], csv_path: Path, target_k: int) -> dict[str, Any]:
    frame_idx = [int(v) for v in (result.get("frame_idx", []) or [])]
    score = [float(v) for v in (result.get("score", []) or [])]
    dominant_part = [str(v) for v in (result.get("dominant_part", []) or [])]
    dominant_point = [str(v) for v in (result.get("dominant_point", []) or [])]
    selection_debug = result.get("selection_debug", {})

    return {
        "sample_name": csv_path.stem,
        "csv_path": str(csv_path),
        "target_k": int(target_k),
        "extracted_k": int(len(frame_idx)),
        "frame_idx": frame_idx,
        "score": score,
        "dominant_part": dominant_part,
        "dominant_point": dominant_point,
        "selection_debug": selection_debug,
    }


def main() -> None:
    args = _parse_args()
    csv_path = _resolve_csv_path(
        dataset_root=str(args.dataset_root),
        sample_name=str(args.sample_name),
        csv_path=str(args.csv_path),
    )
    pose_seq = _load_pose_sequence_from_csv(csv_path)

    target_k = max(1, int(args.target_k))
    candidate_multiplier = max(1, int(args.candidate_multiplier))
    result = extract_keyframes_with_ruptures_poseparts_2d(
        pose_seq,
        k=target_k,
        candidate_multiplier=candidate_multiplier,
        print_all_frame_scores=bool(args.print_all_frame_scores),
        print_selection_debug=bool(args.print_selection_debug),
    )
    payload = _serialize_result(result=result, csv_path=csv_path, target_k=target_k)

    if int(payload["extracted_k"]) != int(target_k):
        print(
            f"[Warn] target_k={target_k}, extracted_k={payload['extracted_k']}",
            file=sys.stderr,
        )

    save_json = str(args.save_json or "").strip()
    if save_json:
        output_path = _resolve_from_root(save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Done] json_saved={output_path}", file=sys.stderr)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

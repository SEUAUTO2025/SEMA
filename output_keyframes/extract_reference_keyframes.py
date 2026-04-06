import argparse
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RTMPose"))  # Allow `import rtmlib` inside RTMPose modules.

from RTMPose.Bone_Feature_Extract import (
    Keypoint_Extract,
    extract_keyframes_with_ruptures_poseparts_2d,
    refine_keyframes_with_absdiff,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract keyframes for a specific video and save them as numeric filenames "
            "(1.jpg, 2.jpg, ...) in temporal order."
        )
    )
    parser.add_argument(
        "--video-path",
        default="dataset/video/414.mp4",
        help="Path to input video.",
    )
    parser.add_argument(
        "--output-dir",
        default="output_keyframes",
        help="Directory to save numeric keyframe images.",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=8,
        help="Target number of keyframes after refinement.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
        help="Width used in keypoint normalization.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
        help="Height used in keypoint normalization.",
    )
    return parser.parse_args()


def _cleanup_numeric_jpgs(output_dir: Path) -> None:
    for file in output_dir.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() != ".jpg":
            continue
        if file.stem.isdigit():
            file.unlink()


def _extract_temporal_keyframes(video_path: str, target_k: int, image_width: int, image_height: int) -> list[int]:
    _, normalized_data = Keypoint_Extract(
        video_path,
        image_width=image_width,
        image_height=image_height,
        show_draw_selection=False,
        draw_math_feature_points=False,
    )
    base = extract_keyframes_with_ruptures_poseparts_2d(
        normalized_data,
        k=max(1, int(target_k)) + 3,
        print_all_frame_scores=False,
        print_selection_debug=False,
    )
    refined = refine_keyframes_with_absdiff(
        video_path=video_path,
        keyframe_result=base,
        k=max(1, int(target_k)),
    )
    frame_idx = [int(v) for v in (refined.get("frame_idx", []) or [])]
    # Enforce "appearance order": strictly sorted by source frame index.
    return sorted(set(frame_idx))


def _save_keyframes(video_path: str, output_dir: Path, frame_indices: list[int]) -> list[dict]:
    """
    Save keyframes by one-pass sequential decoding to avoid random-seek drift.

    Note:
    `cap.set(CAP_PROP_POS_FRAMES, idx)` can be inaccurate on compressed videos.
    We therefore read frames in temporal order and capture exact target indices.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    targets = sorted(int(v) for v in frame_indices)
    if not targets:
        cap.release()
        return []

    records = []
    target_ptr = 0
    read_idx = 0
    next_target = targets[target_ptr]
    try:
        while cap.isOpened() and target_ptr < len(targets):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if read_idx == next_target:
                rank = target_ptr + 1
                file_name = f"{rank}.jpg"
                save_path = output_dir / file_name
                written = cv2.imwrite(str(save_path), frame)
                if not written:
                    raise RuntimeError(f"Failed to save keyframe: {save_path}")
                records.append(
                    {
                        "rank": int(rank),
                        "frame_idx": int(next_target),
                        "file_name": file_name,
                    }
                )
                target_ptr += 1
                if target_ptr < len(targets):
                    next_target = targets[target_ptr]
            read_idx += 1
    finally:
        cap.release()

    if len(records) != len(targets):
        missing = [idx for idx in targets if idx not in {int(r["frame_idx"]) for r in records}]
        raise RuntimeError(f"Failed to read all requested keyframes. Missing frame indices: {missing}")

    return records


def main() -> None:
    args = _parse_args()
    video_path = os.path.abspath(args.video_path)
    output_dir = Path(os.path.abspath(args.output_dir))
    target_k = max(1, int(args.target_k))

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_numeric_jpgs(output_dir)

    frame_indices = _extract_temporal_keyframes(
        video_path=video_path,
        target_k=target_k,
        image_width=int(args.image_width),
        image_height=int(args.image_height),
    )
    if not frame_indices:
        raise RuntimeError("No keyframes extracted from the video.")

    records = _save_keyframes(video_path=video_path, output_dir=output_dir, frame_indices=frame_indices)
    if not records:
        raise RuntimeError("No keyframe images were saved.")

    manifest = {
        "video_path": video_path,
        "output_dir": str(output_dir),
        "target_k": target_k,
        "saved_count": len(records),
        "keyframes": records,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[Done] Saved {len(records)} keyframes to: {output_dir}")
    for row in records:
        print(f"- rank={row['rank']} frame_idx={row['frame_idx']} file={row['file_name']}")


if __name__ == "__main__":
    main()

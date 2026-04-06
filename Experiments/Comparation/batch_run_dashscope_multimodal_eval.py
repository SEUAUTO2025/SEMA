import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(ROOT, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, PROJECT_ROOT_DIR)

import base64
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from batch_eval_utils import (
    DEFAULT_HISTORY_SEARCH_ROOTS,
    build_historical_prediction_index,
    build_prediction_row,
    build_run_dir,
    export_batch_run_artifacts,
    load_gt_text_and_scores,
    merge_rows_by_keys,
    lookup_historical_prediction,
    normalize_eval_text,
    prepare_resume_failed_run,
    standardize_record_aliases,
    write_progress_records_and_failures,
)
from evaluation.eval_db.eval_db_manager import add_or_update_evaluation, get_evaluation, init_db


# Fixed run configuration (edit here before running).
DATASET_ROOT = os.path.join(PROJECT_ROOT_DIR, "dataset")
VIDEO_EXT = ".mp4"
MAX_SAMPLES = None
TARGET_K = 8
LANGUAGE = "zh"  # "zh" or "en"

# Fixed model list (DashScope / Alibaba).
MODEL_LIST = [
    "glm-4.6v",
    "kimi-k2.5",
    "qwen3-vl-plus"
]
MODEL_NAME_PREFIX = "dashscope::"
GLM_MODEL_NAME_PREFIX = "zhipu::"

OUTPUT_ROOT = os.path.join(ROOT, "evaluation", "results")
print(OUTPUT_ROOT)
RESUME_FAILED_RUN_NAME = ""
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TEMPERATURE = 0.2
MAX_TOKENS = 900
PRINT_INTERMEDIATE_OUTPUT = True
RAW_OUTPUT_MAX_CHARS: Optional[int] = None  # None means print full output.
PRINT_PARSED_PAYLOAD = True
PRINT_SKIP_ERRORS = True
PRINT_SKIP_TRACEBACK = False
UPLOAD_IMAGE_MAX_BYTES = 220 * 1024
UPLOAD_TOTAL_IMAGE_MAX_BYTES = 1400 * 1024
UPLOAD_MIN_IMAGE_MAX_BYTES = 64 * 1024
UPLOAD_DEFAULT_LONG_SIDES = [1280, 1024, 896, 768, 640]
UPLOAD_AGGRESSIVE_LONG_SIDES = [896, 768, 640, 512, 448, 384]
UPLOAD_DEFAULT_JPEG_QUALITIES = [80, 70, 60, 50, 40, 32]
UPLOAD_AGGRESSIVE_JPEG_QUALITIES = [65, 55, 45, 35, 28, 22]

# DashScope thinking strategy.
QWEN_ENABLE_THINKING = False
QWEN_THINKING_BUDGET = 4096  # Medium mapping.
KIMI_ENABLE_THINKING = False  # User-confirmed default.

# Zhipu GLM config.
ZHIPU_API_KEY_ENV = "ZHIPU_API_KEY"
GLM_THINKING_ENABLED = False

REFERENCE_ACTION_LIBRARY = {
    "zh": {
        "positive": [
            "身体纵轴垂直于地面",
            "身体侧向对着射箭方向",
            "头部水平转动到位",
            "前手、前肩、后肩在一条线上，后手小臂与箭在一条线上",
            "后手靠在脸颊下颚位置，弓弦靠在下巴和鼻子上",
            "瞄准时后手没有松滑弦",
            "双脚开步与肩同宽",
            "撒放时后手顺势向后延展",
            "撒放时前手放松推弓",
        ],
        "negative": [
            "身体重心倾斜",
            "身体没有转到侧对",
            "头部没有转动到位",
            "前手、前肩、后肩不共线或后手小臂与箭不共线",
            "后手悬空没有贴实，弓弦没有贴上下巴和鼻子",
            "瞄准时后手松滑弦",
            "双脚随意站立",
            "撒放时后手定在原地或向外、向前",
            "撒放时前手抓弓",
        ],
    },
    "en": {
        "positive": [
            "Body vertical axis is perpendicular to the ground",
            "Body is side-on to the shooting direction",
            "Head rotates horizontally to the target",
            "Front hand, front shoulder, and rear shoulder are aligned, rear forearm aligned with arrow",
            "Rear hand is anchored at cheek/jaw and string touches chin and nose",
            "During aiming, rear hand does not slip or loosen the string",
            "Feet stance is shoulder-width",
            "At release, rear hand extends naturally backward",
            "At release, bow/front hand relaxes and pushes the bow",
        ],
        "negative": [
            "Body center of mass is tilted or leaning",
            "Body is not rotated to a side-on stance",
            "Head does not rotate enough",
            "Shoulder-arm line is broken or rear forearm not aligned with arrow",
            "Rear hand is floating and string does not touch chin and nose",
            "During aiming, rear hand slips or loosens the string",
            "Feet are placed arbitrarily",
            "At release, rear hand stays in place or moves outward or forward",
            "At release, bow/front hand grabs the bow",
        ],
    },
}


def _ensure_results_subdir(path_str: str) -> str:
    root = Path("evaluation") / "results"
    p = Path(path_str)
    if p.parts[:2] == ("evaluation", "results"):
        return str(p)
    if p.is_absolute():
        return str(root / p.name)
    return str(root / p)


def _build_run_dir(output_root: str) -> str:
    root = Path(_ensure_results_subdir(output_root))
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"dashscope_mm_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _extract_json_object_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Model output is empty.")

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return text[idx: idx + end]
    raise ValueError("No valid JSON object found in model output.")


def _response_to_text(resp: Any) -> str:
    if resp is None or not getattr(resp, "choices", None):
        raise ValueError("Empty response from model.")

    message = getattr(resp.choices[0], "message", None)
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
            else:
                text_attr = getattr(item, "text", None)
                if text_attr:
                    parts.append(str(text_attr))
        return "".join(parts).strip()

    raise ValueError("Unsupported response content format.")


def _truncate_for_print(text: str) -> str:
    if RAW_OUTPUT_MAX_CHARS is None:
        return text
    if RAW_OUTPUT_MAX_CHARS <= 0:
        return text
    s = str(text)
    if len(s) <= RAW_OUTPUT_MAX_CHARS:
        return s
    return f"{s[:RAW_OUTPUT_MAX_CHARS]} ...[truncated {len(s) - RAW_OUTPUT_MAX_CHARS} chars]"


def _print_intermediate_output(
    sample_name: str,
    model_name: str,
    raw_text: str,
    payload: Optional[Dict[str, Any]] = None,
    request_model: Optional[str] = None,
    print_raw: bool = True,
) -> None:
    if not PRINT_INTERMEDIATE_OUTPUT:
        return
    model_tag = request_model or model_name
    if print_raw:
        print(f"\n[DashScope][{sample_name}][{model_name}] REQUEST_MODEL={model_tag} RAW OUTPUT BEGIN")
        print(_truncate_for_print(raw_text))
        print(f"[DashScope][{sample_name}][{model_name}] RAW OUTPUT END")
    if PRINT_PARSED_PAYLOAD and payload is not None:
        print(f"[DashScope][{sample_name}][{model_name}] PARSED PAYLOAD:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_skip_event(
    sample_name: str,
    stage: str,
    error_type: str,
    error_message: str,
    model_name: str = "",
    traceback_text: str = "",
) -> None:
    if not PRINT_SKIP_ERRORS:
        return
    model_tag = str(model_name or "ALL_MODELS")
    tqdm.write(
        "[DashScope][SKIP] sample={0} model={1} stage={2} error_type={3} error={4}".format(
            sample_name,
            model_tag,
            stage,
            error_type,
            error_message,
        )
    )
    if PRINT_SKIP_TRACEBACK and traceback_text:
        for line in str(traceback_text).rstrip().splitlines():
            tqdm.write(line)


def _print_payload_retry_event(sample_name: str, model_name: str, error_message: str) -> None:
    tqdm.write(
        "[DashScope][RETRY] sample={0} model={1} action=retry_with_aggressive_image_compression error={2}".format(
            sample_name,
            model_name,
            error_message,
        )
    )


def _to_int_score(val: Any, key: str, lo: int, hi: int) -> int:
    try:
        num = float(val)
    except Exception as exc:
        raise ValueError(f"Score '{key}' is not numeric: {val}") from exc
    if int(num) != num:
        raise ValueError(f"Score '{key}' must be an integer, got: {val}")
    out = int(num)
    if out < lo or out > hi:
        raise ValueError(f"Score '{key}' out of range [{lo}, {hi}]: {out}")
    return out


def _parse_eval_payload(raw_text: str) -> Dict[str, Any]:
    required = [
        "evaluation_text",
        "total_score",
        "head_score",
        "hand_score",
        "torso_score",
        "foot_score",
        "arm_score",
    ]
    json_text = _extract_json_object_text(raw_text)
    try:
        payload = json.loads(json_text)
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON output: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON output must be an object.")

    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    eval_text = str(payload.get("evaluation_text") or "").strip()
    if not eval_text:
        raise ValueError("evaluation_text is empty.")

    head = _to_int_score(payload.get("head_score"), "head_score", 0, 5)
    hand = _to_int_score(payload.get("hand_score"), "hand_score", 0, 5)
    torso = _to_int_score(payload.get("torso_score"), "torso_score", 0, 5)
    foot = _to_int_score(payload.get("foot_score"), "foot_score", 0, 5)
    arm = _to_int_score(payload.get("arm_score"), "arm_score", 0, 5)
    total = _to_int_score(payload.get("total_score"), "total_score", 0, 25)

    if total != (head + hand + torso + foot + arm):
        raise ValueError(
            "total_score must equal head_score + hand_score + torso_score + foot_score + arm_score."
        )

    return {
        "evaluation_text": eval_text,
        "total_score": total,
        "head_score": head,
        "hand_score": hand,
        "torso_score": torso,
        "foot_score": foot,
        "arm_score": arm,
    }


def _compose_db_text(model_name: str, payload: Dict[str, Any], language: str) -> str:
    # DB eval_text should store only the model-generated assessment text.
    _ = model_name
    _ = language
    return str(payload.get("evaluation_text") or "").strip()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_gt_scores(csv_dir: Path, sample_name: str) -> Dict[str, float]:
    from Tools.Exe_dataset.dataset_test_tools import load_single_csv_with_multipart_labels

    csv_path = csv_dir / f"{sample_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")
    _, labels, label_total = load_single_csv_with_multipart_labels(str(csv_path), max_frames=124)
    return {
        "gt_total": _safe_float(label_total),
        "gt_head": _safe_float(labels.get("head")),
        "gt_hand": _safe_float(labels.get("hand")),
        "gt_torso": _safe_float(labels.get("body", labels.get("torso"))),
        "gt_foot": _safe_float(labels.get("feet", labels.get("foot"))),
        "gt_arm": _safe_float(labels.get("arm")),
    }


def _format_reference_action_text(language: str) -> str:
    lang_key = "en" if str(language).lower().startswith("en") else "zh"
    ref = REFERENCE_ACTION_LIBRARY.get(lang_key, REFERENCE_ACTION_LIBRARY["zh"])
    positives = ref.get("positive", [])
    negatives = ref.get("negative", [])

    if lang_key == "en":
        pos_lines = "\n".join([f"- {x}" for x in positives]) if positives else "- (empty)"
        neg_lines = "\n".join([f"- {x}" for x in negatives]) if negatives else "- (empty)"
        return (
            "Reference action standard lexicon (must be explicitly considered):\n"
            "Positive standards:\n"
            f"{pos_lines}\n"
            "Negative standards:\n"
            f"{neg_lines}"
        )

    pos_lines = "\n".join([f"- {x}" for x in positives]) if positives else "- (空)"
    neg_lines = "\n".join([f"- {x}" for x in negatives]) if negatives else "- (空)"
    return (
        "参考动作标准词库（必须显式参考）：\n"
        "正向标准：\n"
        f"{pos_lines}\n"
        "反向标准：\n"
        f"{neg_lines}"
    )


def _build_prompts(language: str, target_k: int) -> Tuple[str, str]:
    ref_text = _format_reference_action_text(language=language)
    frame_count = max(1, int(target_k))
    if str(language).lower().startswith("en"):
        system_prompt = (
            "You are a professional archery action-quality evaluator.\n"
            "Use only the provided uniformly sampled video frames.\n"
            "You must explicitly evaluate against the provided reference action standard lexicon.\n"
            "Return JSON only (no markdown, no explanation) with keys:\n"
            "evaluation_text,total_score,head_score,hand_score,torso_score,foot_score,arm_score.\n"
            "Scoring constraints: each part score must be an integer in [0,5]. "
            "total_score must be an integer in [0,25] and equal to the sum of five part scores.\n"
            "evaluation_text should briefly include both correct and incorrect actions.\n"
            f"{ref_text}"
        )
        user_note = (
            "{0} frames are uniformly sampled across the full video and ordered from early to late. ".format(
                frame_count
            )
            +
            "Please evaluate this archer."
        )
        return system_prompt, user_note

    system_prompt = (
        "你是专业射箭动作质量评估教练。\n"
        "只能使用用户提供的视频均匀采样帧对视频中用户的射箭动作质量进行评估。\n"
        # "必须显式参考提供的动作标准词库进行评估。\n"
        "输出必须仅包含JSON格式,字段必须是:\n"
        "evaluation_text,total_score,head_score,hand_score,torso_score,foot_score,arm_score。\n"
        "评分约束:各部位分数必须是0-5的整数,总分必须是0-25的整数,且等于五个部位分数之和。\n"
        "evaluation_text=的格式为：这位同学的正确动作如下：......这位同学的错误动作如下：......\n"
        # f"{ref_text}"
    )
    user_note = "已提供全视频均匀采样的{0}帧，按时间从早到晚排序，请对该运动员进行动作质量评估。".format(
        frame_count
    )
    return system_prompt, user_note


def _resize_frame_to_long_side(frame: np.ndarray, max_long_side: int) -> np.ndarray:
    if max_long_side <= 0:
        return frame
    height, width = frame.shape[:2]
    current_long_side = max(height, width)
    if current_long_side <= max_long_side:
        return frame
    scale = float(max_long_side) / float(current_long_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _encode_frame_as_jpeg_bytes(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, quality)))],
    )
    if not ok:
        raise ValueError("Failed to JPEG-encode sampled frame.")
    return buf.tobytes()


def _build_base64_image_item(jpeg_bytes: bytes) -> Dict[str, Any]:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,{0}".format(encoded)},
    }


def _compress_frame_for_upload(
    frame: np.ndarray,
    max_image_bytes: int,
    long_side_candidates: List[int],
    quality_candidates: List[int],
) -> Dict[str, Any]:
    last_result: Optional[Dict[str, Any]] = None
    for long_side in long_side_candidates:
        resized = _resize_frame_to_long_side(frame, max_long_side=int(long_side))
        resized_height, resized_width = resized.shape[:2]
        for quality in quality_candidates:
            jpeg_bytes = _encode_frame_as_jpeg_bytes(resized, quality=int(quality))
            result = {
                "item": _build_base64_image_item(jpeg_bytes),
                "jpeg_bytes": int(len(jpeg_bytes)),
                "width": int(resized_width),
                "height": int(resized_height),
                "long_side": int(max(resized_width, resized_height)),
                "jpeg_quality": int(quality),
            }
            last_result = result
            if result["jpeg_bytes"] <= int(max_image_bytes):
                return result

    if last_result is None:
        raise ValueError("Failed to build compressed image payload for sampled frame.")
    return last_result


def _compress_frames_for_upload(
    frames: List[np.ndarray],
    compression_mode: str = "default",
) -> Dict[str, Any]:
    normalized_mode = "aggressive" if str(compression_mode or "").strip().lower() == "aggressive" else "default"
    if not frames:
        raise ValueError("No frames available for upload compression.")

    if normalized_mode == "aggressive":
        long_sides = list(UPLOAD_AGGRESSIVE_LONG_SIDES)
        qualities = list(UPLOAD_AGGRESSIVE_JPEG_QUALITIES)
        per_image_limit = max(
            int(UPLOAD_MIN_IMAGE_MAX_BYTES),
            min(
                int(UPLOAD_IMAGE_MAX_BYTES),
                int(UPLOAD_TOTAL_IMAGE_MAX_BYTES / max(1, len(frames))),
            ),
        )
        compressed = [
            _compress_frame_for_upload(
                frame=frame,
                max_image_bytes=per_image_limit,
                long_side_candidates=long_sides,
                quality_candidates=qualities,
            )
            for frame in frames
        ]
    else:
        compressed = [
            _compress_frame_for_upload(
                frame=frame,
                max_image_bytes=int(UPLOAD_IMAGE_MAX_BYTES),
                long_side_candidates=list(UPLOAD_DEFAULT_LONG_SIDES),
                quality_candidates=list(UPLOAD_DEFAULT_JPEG_QUALITIES),
            )
            for frame in frames
        ]
        total_bytes = sum(int(item["jpeg_bytes"]) for item in compressed)
        if total_bytes > int(UPLOAD_TOTAL_IMAGE_MAX_BYTES):
            normalized_mode = "aggressive"
            tighter_limit = max(
                int(UPLOAD_MIN_IMAGE_MAX_BYTES),
                min(
                    int(UPLOAD_IMAGE_MAX_BYTES),
                    int(UPLOAD_TOTAL_IMAGE_MAX_BYTES / max(1, len(frames))),
                ),
            )
            compressed = [
                _compress_frame_for_upload(
                    frame=frame,
                    max_image_bytes=tighter_limit,
                    long_side_candidates=list(UPLOAD_AGGRESSIVE_LONG_SIDES),
                    quality_candidates=list(UPLOAD_AGGRESSIVE_JPEG_QUALITIES),
                )
                for frame in frames
            ]

    total_bytes = sum(int(item["jpeg_bytes"]) for item in compressed)
    return {
        "compression_mode": normalized_mode,
        "image_items": [item["item"] for item in compressed],
        "image_meta": [
            {
                "jpeg_bytes": int(item["jpeg_bytes"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "long_side": int(item["long_side"]),
                "jpeg_quality": int(item["jpeg_quality"]),
            }
            for item in compressed
        ],
        "total_image_bytes": int(total_bytes),
        "max_image_bytes": int(max(int(item["jpeg_bytes"]) for item in compressed)),
    }


def _extract_keyframe_images(
    video_path: str,
    target_k: int,
    show: bool = False,
    compression_mode: str = "default",
) -> Dict[str, Any]:
    del show  # Uniform frame sampling path does not use visualization switches.
    target_k = int(target_k)
    if target_k <= 0:
        raise ValueError(f"target_k must be > 0, got {target_k}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        probe_cap = cv2.VideoCapture(video_path)
        if not probe_cap.isOpened():
            raise ValueError(f"Failed to open video for frame probing: {video_path}")
        total_frames = 0
        while True:
            ok, _ = probe_cap.read()
            if not ok:
                break
            total_frames += 1
        probe_cap.release()

    if total_frames <= 0:
        raise ValueError(f"No decodable frames found in video: {video_path}")

    frame_idx = np.linspace(0, total_frames - 1, num=target_k)
    frame_idx = [int(round(x)) for x in frame_idx]

    read_cap = cv2.VideoCapture(video_path)
    if not read_cap.isOpened():
        raise ValueError(f"Failed to reopen video: {video_path}")

    raw_frames: List[np.ndarray] = []
    used_idx: List[int] = []
    for idx in frame_idx:
        read_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
        ok, frame = read_cap.read()
        if not ok or frame is None:
            continue
        raw_frames.append(frame)
        used_idx.append(int(idx))
    read_cap.release()

    if not raw_frames:
        raise ValueError("No sampled frames available for model input.")
    compressed_payload = _compress_frames_for_upload(frames=raw_frames, compression_mode=compression_mode)
    return {
        "image_items": compressed_payload["image_items"],
        "keyframe_count": len(raw_frames),
        "frame_idx": used_idx,
        "compression_mode": str(compressed_payload["compression_mode"]),
        "total_image_bytes": int(compressed_payload["total_image_bytes"]),
        "max_image_bytes": int(compressed_payload["max_image_bytes"]),
        "image_meta": compressed_payload["image_meta"],
    }


def _build_dashscope_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def _is_glm_model(model_name: str) -> bool:
    return str(model_name).strip().lower() in {"glm-4.6v", "zhipu/glm-4.6v"}


def _provider_for_model(model_name: str) -> str:
    return "zhipu" if _is_glm_model(model_name) else "dashscope"


def _db_model_name_for(model_name: str) -> str:
    if _is_glm_model(model_name):
        return f"{GLM_MODEL_NAME_PREFIX}{model_name}"
    return f"{MODEL_NAME_PREFIX}{model_name}"


def _build_zhipu_client(api_key: str):
    try:
        from zai import ZhipuAiClient  # type: ignore
    except Exception as exc:
        raise ImportError(
            "glm-4.6v requires Zhipu SDK `zai`. Install it before running this model."
        ) from exc
    return ZhipuAiClient(api_key=api_key)


def _build_dashscope_extra_body(model_name: str) -> Dict[str, Any]:
    if _is_glm_model(model_name):
        return {}
    if model_name == "qwen3-vl-plus":
        if bool(QWEN_ENABLE_THINKING):
            return {
                "enable_thinking": True,
                "thinking_budget": int(QWEN_THINKING_BUDGET),
            }
        return {"enable_thinking": False}
    if model_name in {"kimi/kimi-k2.5", "kimi-k2.5"}:
        return {
            "enable_thinking": bool(KIMI_ENABLE_THINKING),
        }
    return {}


def _is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("model" in text) and ("not found" in text or "invalid" in text)


def _is_unsupported_param_error(exc: Exception) -> bool:
    text = str(exc).lower()
    keywords = [
        "unsupported",
        "unknown parameter",
        "invalid parameter",
        "enable_thinking",
        "thinking_budget",
        "extra_body",
    ]
    return any(k in text for k in keywords)


def _is_request_payload_too_large_error(exc: Exception) -> bool:
    text = str(exc).lower()
    size_terms = [
        "too large",
        "too big",
        "payload too large",
        "request body too large",
        "request too large",
        "content too large",
        "file too large",
        "image too large",
        "size limit",
        "maximum size",
        "超范围",
        "超出",
        "过大",
    ]
    scope_terms = [
        "request",
        "payload",
        "body",
        "content",
        "image",
        "file",
        "input",
        "消息",
        "请求",
        "图片",
        "上传",
    ]
    return any(term in text for term in size_terms) and any(term in text for term in scope_terms)


def _candidate_request_models(model_name: str) -> List[str]:
    if _is_glm_model(model_name):
        return ["glm-4.6v"]
    if model_name == "kimi/kimi-k2.5":
        # Compatible fallback aliases for different provider rollouts.
        return ["kimi/kimi-k2.5", "kimi-k2.5", "kimi-k2-5-preview"]
    return [model_name]


def _evaluate_with_glm_model(
    zhipu_client: Any,
    sample_name: str,
    model_name: str,
    language: str,
    keyframe_images: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, str, Dict[str, Any]]:
    system_prompt, user_note = _build_prompts(language=language, target_k=len(keyframe_images))
    # Follow Zhipu multimodal format: single user message with image_url + text content.
    user_content = list(keyframe_images) + [
        {
            "type": "text",
            "text": f"{system_prompt}\n\n{user_note}",
        }
    ]
    request_model = "glm-4.6v"
    request_kwargs: Dict[str, Any] = {
        "model": request_model,
        "messages": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
    }
    thinking_cfg: Dict[str, Any] = {}
    if bool(GLM_THINKING_ENABLED):
        thinking_cfg = {"type": "enabled"}
        request_kwargs["thinking"] = thinking_cfg

    resp = zhipu_client.chat.completions.create(**request_kwargs)
    raw_text = _response_to_text(resp)
    _print_intermediate_output(
        sample_name=sample_name,
        model_name=model_name,
        request_model=request_model,
        raw_text=raw_text,
    )
    payload = _parse_eval_payload(raw_text)
    _print_intermediate_output(
        sample_name=sample_name,
        model_name=model_name,
        request_model=request_model,
        raw_text=raw_text,
        payload=payload,
        print_raw=False,
    )
    extra_used = {"thinking": thinking_cfg} if thinking_cfg else {}
    return payload, raw_text, request_model, extra_used


def _evaluate_with_model(
    client: OpenAI,
    sample_name: str,
    model_name: str,
    language: str,
    keyframe_images: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, str, Dict[str, Any]]:
    system_prompt, user_note = _build_prompts(language=language, target_k=len(keyframe_images))
    user_content = [{"type": "text", "text": user_note}] + keyframe_images
    extra_body = _build_dashscope_extra_body(model_name)
    model_candidates = _candidate_request_models(model_name)

    last_exc: Optional[Exception] = None
    for request_model in model_candidates:
        request_kwargs: Dict[str, Any] = {
            "model": request_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        try:
            resp = client.chat.completions.create(**request_kwargs)
            raw_text = _response_to_text(resp)
            _print_intermediate_output(
                sample_name=sample_name,
                model_name=model_name,
                request_model=request_model,
                raw_text=raw_text,
            )
            payload = _parse_eval_payload(raw_text)
            _print_intermediate_output(
                sample_name=sample_name,
                model_name=model_name,
                request_model=request_model,
                raw_text=raw_text,
                payload=payload,
                print_raw=False,
            )
            return payload, raw_text, request_model, extra_body
        except Exception as exc:
            last_exc = exc
            # Retry once without extra_body if provider rejects thinking fields.
            if extra_body and _is_unsupported_param_error(exc):
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("extra_body", None)
                try:
                    resp = client.chat.completions.create(**retry_kwargs)
                    raw_text = _response_to_text(resp)
                    _print_intermediate_output(
                        sample_name=sample_name,
                        model_name=model_name,
                        request_model=request_model,
                        raw_text=raw_text,
                    )
                    payload = _parse_eval_payload(raw_text)
                    _print_intermediate_output(
                        sample_name=sample_name,
                        model_name=model_name,
                        request_model=request_model,
                        raw_text=raw_text,
                        payload=payload,
                        print_raw=False,
                    )
                    return payload, raw_text, request_model, {}
                except Exception as exc2:
                    last_exc = exc2
            # Model alias fallback for not-found style errors.
            if _is_model_not_found_error(last_exc):
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected evaluation failure for model: {model_name}")


def _collect_video_files(
    dataset_root: str,
    video_ext: str,
    max_samples: Optional[int],
) -> Tuple[Path, str, List[Path]]:
    dataset_root_path = Path(dataset_root)
    video_dir = dataset_root_path / "video"
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    ext = video_ext if str(video_ext).startswith(".") else f".{video_ext}"
    ext = ext.lower()
    video_files = sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() == ext])
    if max_samples is not None:
        video_files = video_files[: int(max_samples)]
    return dataset_root_path, ext, video_files


def _build_model_score_summary(records_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if records_df.empty:
        return []
    grouped = records_df.groupby("db_model_name", dropna=False)
    summary = []
    for model_name, grp in grouped:
        total_col = "pred_total" if "pred_total" in grp else "total_score"
        head_col = "pred_head" if "pred_head" in grp else "head_score"
        hand_col = "pred_hand" if "pred_hand" in grp else "hand_score"
        torso_col = "pred_torso" if "pred_torso" in grp else "torso_score"
        foot_col = "pred_foot" if "pred_foot" in grp else "foot_score"
        arm_col = "pred_arm" if "pred_arm" in grp else "arm_score"
        summary.append(
            {
                "model_name": str(model_name),
                "count": int(grp.shape[0]),
                "mean_total": float(pd.to_numeric(grp[total_col], errors="coerce").mean()),
                "mean_head": float(pd.to_numeric(grp[head_col], errors="coerce").mean()),
                "mean_hand": float(pd.to_numeric(grp[hand_col], errors="coerce").mean()),
                "mean_torso": float(pd.to_numeric(grp[torso_col], errors="coerce").mean()),
                "mean_foot": float(pd.to_numeric(grp[foot_col], errors="coerce").mean()),
                "mean_arm": float(pd.to_numeric(grp[arm_col], errors="coerce").mean()),
            }
        )
    return summary


def _build_model_mae_summary(records_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if records_df.empty:
        return []
    grouped = records_df.groupby("db_model_name", dropna=False)
    summary = []
    for model_name, grp in grouped:
        summary.append(
            {
                "model_name": str(model_name),
                "count": int(grp.shape[0]),
                "mae_total": float(grp["mae_total"].mean()) if "mae_total" in grp else float("nan"),
                "mae_head": float(grp["mae_head"].mean()) if "mae_head" in grp else float("nan"),
                "mae_hand": float(grp["mae_hand"].mean()) if "mae_hand" in grp else float("nan"),
                "mae_torso": float(grp["mae_torso"].mean()) if "mae_torso" in grp else float("nan"),
                "mae_foot": float(grp["mae_foot"].mean()) if "mae_foot" in grp else float("nan"),
                "mae_arm": float(grp["mae_arm"].mean()) if "mae_arm" in grp else float("nan"),
            }
        )
    return summary


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


def _resolve_resume_video_path(
    row: Dict[str, Any],
    dataset_root_path: Path,
    video_ext: str,
) -> Path:
    sample_name = str(row.get("sample_name") or "").strip()
    raw_video_path = str(row.get("video_path") or "").strip()
    if raw_video_path:
        path_obj = Path(raw_video_path)
        if path_obj.is_absolute():
            return path_obj.resolve()
        return (Path(ROOT) / path_obj).resolve()
    ext = video_ext if str(video_ext).startswith(".") else ".{0}".format(video_ext)
    return (dataset_root_path / "video" / "{0}{1}".format(sample_name, ext.lower())).resolve()


def _build_resume_sample_groups(
    resume_failed_rows: List[Dict[str, Any]],
    dataset_root_path: Path,
    video_ext: str,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    group_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in resume_failed_rows:
        sample_name = str(row.get("sample_name") or "").strip()
        model_name = str(row.get("model_name") or "").strip()
        if not sample_name or not model_name:
            continue
        video_path = _resolve_resume_video_path(row=row, dataset_root_path=dataset_root_path, video_ext=video_ext)
        group_key = (sample_name, str(video_path))
        group = group_map.get(group_key)
        if group is None:
            group = {
                "sample_name": sample_name,
                "video_path": video_path,
                "models": [],
            }
            group_map[group_key] = group
            groups.append(group)
        if model_name not in group["models"]:
            group["models"].append(model_name)
    return groups


def run_batch_dashscope(
    dataset_root: str = DATASET_ROOT,
    models: Optional[List[str]] = None,
    language: str = LANGUAGE,
    target_k: int = TARGET_K,
    output_root: str = OUTPUT_ROOT,
    resume_failed_run_name: Optional[str] = RESUME_FAILED_RUN_NAME,
    max_samples: Optional[int] = MAX_SAMPLES,
    video_ext: str = VIDEO_EXT,
) -> Dict[str, Any]:
    models = list(models or MODEL_LIST)
    if not models:
        raise ValueError("models cannot be empty.")

    dataset_root_path, ext, video_files = _collect_video_files(
        dataset_root=dataset_root,
        video_ext=video_ext,
        max_samples=max_samples,
    )
    csv_dir = dataset_root_path / "csv"
    txt_dir = dataset_root_path / "txt"
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    if not txt_dir.exists():
        raise FileNotFoundError(f"TXT directory not found: {txt_dir}")

    resume_ctx = prepare_resume_failed_run(
        output_root=output_root,
        resume_run_name=resume_failed_run_name,
        base_dir=ROOT,
    )
    if resume_ctx["enabled"]:
        target_groups = _build_resume_sample_groups(
            resume_failed_rows=resume_ctx["resume_failed_rows"],
            dataset_root_path=dataset_root_path,
            video_ext=ext,
        )
        if max_samples is not None:
            target_groups = target_groups[: int(max_samples)]
        if len(target_groups) == 0:
            raise ValueError("No resumable failed targets found in run: {0}".format(resume_ctx["run_dir"]))
        models = _ordered_unique_strings(
            [model_name for group in target_groups for model_name in group["models"]]
        )
        run_dir = Path(resume_ctx["run_dir"])
    else:
        target_groups = [
            {
                "sample_name": video_path.stem,
                "video_path": video_path,
                "models": list(models),
            }
            for video_path in video_files
        ]
        run_dir = build_run_dir(output_root=output_root, run_prefix="dashscope_mm_batch", base_dir=ROOT)

    if len(models) > 15:
        raise ValueError("evaluation DB supports at most 15 model names per sample.")

    need_dashscope = any(not _is_glm_model(m) for m in models)
    need_zhipu = any(_is_glm_model(m) for m in models)

    ali_api_key = os.getenv("ALI_API_KEY", "").strip()
    zhipu_api_key = os.getenv(ZHIPU_API_KEY_ENV, "").strip()
    if need_dashscope and not ali_api_key:
        raise EnvironmentError("ALI_API_KEY is required for DashScope models.")
    if need_zhipu and not zhipu_api_key:
        raise EnvironmentError(f"{ZHIPU_API_KEY_ENV} is required for glm-4.6v.")

    new_records: List[Dict[str, Any]] = []
    new_failures: List[Dict[str, Any]] = []
    if resume_ctx["enabled"]:
        records_to_export = list(resume_ctx["existing_records"])
        failures_to_export = list(resume_ctx["existing_failures"])
    else:
        records_to_export = []
        failures_to_export = []

    init_db()
    client = _build_dashscope_client(api_key=ali_api_key) if need_dashscope else None
    zhipu_client = _build_zhipu_client(api_key=zhipu_api_key) if need_zhipu else None
    historical_cache = build_historical_prediction_index()

    iterator = tqdm(
        target_groups,
        desc="DashScope MM Resume" if resume_ctx["enabled"] else "DashScope MM Eval",
    )
    for target_group in iterator:
        video_path = Path(target_group["video_path"])
        sample_name = str(target_group["sample_name"])
        sample_models = list(target_group["models"])
        sample_ts = datetime.now().isoformat(timespec="seconds")
        sample_failures: Dict[str, Dict[str, Any]] = {}

        try:
            gt_payload = load_gt_text_and_scores(video_path=video_path, txt_dir=txt_dir, csv_dir=csv_dir)
        except Exception as exc:
            tb = traceback.format_exc()
            _print_skip_event(
                sample_name=sample_name,
                model_name="ALL_MODELS",
                stage="gt_loading",
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_text=tb,
            )
            processed_keys = [(sample_name, model_name) for model_name in sample_models]
            gt_fail_rows: List[Dict[str, Any]] = []
            for model_name in sample_models:
                fail = {
                    "sample_name": sample_name,
                    "video_path": str(video_path),
                    "provider": _provider_for_model(model_name),
                    "model_name": model_name,
                    "db_model_name": _db_model_name_for(model_name),
                    "stage": "gt_loading",
                    "timestamp": sample_ts,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": tb,
                }
                new_failures.append(fail)
                gt_fail_rows.append(fail)
            records_to_export = merge_rows_by_keys(
                existing_rows=records_to_export,
                incoming_rows=[],
                key_fields=["sample_name", "model_name"],
                remove_keys=processed_keys,
            )
            failures_to_export = merge_rows_by_keys(
                existing_rows=failures_to_export,
                incoming_rows=gt_fail_rows,
                key_fields=["sample_name", "model_name"],
                remove_keys=processed_keys,
            )
            write_progress_records_and_failures(
                run_dir=run_dir,
                records=records_to_export,
                failures=failures_to_export,
            )
            continue

        existing_eval_map: Dict[str, Optional[Dict[str, Any]]] = {}
        models_to_infer: List[str] = []
        for model_name in sample_models:
            db_model_name = _db_model_name_for(model_name)
            existing_eval = get_evaluation(sample_name=sample_name, model_name=db_model_name)
            existing_eval_map[model_name] = existing_eval
            existing_text = normalize_eval_text(existing_eval.get("eval_text")) if existing_eval else ""
            if not existing_text:
                models_to_infer.append(model_name)

        keyframe_data: Optional[Dict[str, Any]] = None
        if models_to_infer:
            try:
                keyframe_data = _extract_keyframe_images(
                    str(video_path),
                    target_k=int(target_k),
                    show=False,
                    compression_mode="default",
                )
            except Exception as exc:
                tb = traceback.format_exc()
                for model_name in models_to_infer:
                    db_model_name = _db_model_name_for(model_name)
                    sample_failures[model_name] = {
                        "sample_name": sample_name,
                        "video_path": str(video_path),
                        "provider": _provider_for_model(model_name),
                        "model_name": model_name,
                        "db_model_name": db_model_name,
                        "stage": "frame_sampling",
                        "timestamp": sample_ts,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": tb,
                    }

        for model_name in sample_models:
            db_model_name = _db_model_name_for(model_name)
            existing_eval = existing_eval_map.get(model_name)
            cached_eval_text = normalize_eval_text(existing_eval.get("eval_text")) if existing_eval else ""
            base_row = {
                "sample_name": sample_name,
                "video_path": str(video_path),
                "provider": _provider_for_model(model_name),
                "model_name": model_name,
                "db_model_name": db_model_name,
                "language": language,
                "target_k": int(target_k),
                "keyframe_count": int(keyframe_data["keyframe_count"]) if keyframe_data else 0,
                "frame_idx": ",".join(str(x) for x in keyframe_data["frame_idx"]) if keyframe_data else "",
                "image_compression_mode": str(keyframe_data.get("compression_mode") or "") if keyframe_data else "",
                "total_image_bytes": int(keyframe_data.get("total_image_bytes") or 0) if keyframe_data else 0,
                "max_image_bytes": int(keyframe_data.get("max_image_bytes") or 0) if keyframe_data else 0,
                "timestamp": sample_ts,
            }
            if cached_eval_text:
                historical_row = lookup_historical_prediction(
                    cache_index=historical_cache,
                    sample_name=sample_name,
                    model_key=db_model_name,
                    eval_text=cached_eval_text,
                )
                extra_fields = {
                    "request_model": str(historical_row.get("request_model") or "")
                    if historical_row
                    else "",
                    "extra_body_used": str(historical_row.get("extra_body_used") or "")
                    if historical_row
                    else "",
                    "source_records_csv": str(historical_row.get("source_records_csv") or "")
                    if historical_row
                    else "",
                    "cached_from_db": True,
                }
                if historical_row is None:
                    extra_fields["cache_note"] = (
                        "DB evaluation text exists, but no historical records.csv row with per-sample "
                        "prediction scores was found under evaluation/results."
                    )
                row = build_prediction_row(
                    base_row=base_row,
                    evaluation_text=cached_eval_text,
                    gt_payload=gt_payload,
                    pred_scores=historical_row,
                    raw_model_output=str(historical_row.get("raw_model_output") or "")
                    if historical_row
                    else "",
                    cache_source="db_records_cache" if historical_row else "db_text_only",
                    db_written=False,
                    extra_fields=extra_fields,
                )
                new_records.append(row)
                processed_keys = [(sample_name, model_name)]
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

            if model_name in sample_failures:
                fail = dict(base_row, **sample_failures[model_name])
                new_failures.append(fail)
                processed_keys = [(sample_name, model_name)]
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
                    model_name=model_name,
                    stage=str(fail.get("stage") or "unknown"),
                    error_type=str(fail.get("error_type") or "UnknownError"),
                    error_message=str(fail.get("error_message") or ""),
                    traceback_text=str(fail.get("traceback") or ""),
                )
                continue

            try:
                active_keyframe_data = keyframe_data
                while True:
                    if active_keyframe_data is None:
                        raise RuntimeError("Keyframe payload is not initialized.")
                    base_row["image_compression_mode"] = str(active_keyframe_data.get("compression_mode") or "")
                    base_row["total_image_bytes"] = int(active_keyframe_data.get("total_image_bytes") or 0)
                    base_row["max_image_bytes"] = int(active_keyframe_data.get("max_image_bytes") or 0)
                    try:
                        if _is_glm_model(model_name):
                            if zhipu_client is None:
                                raise RuntimeError("Zhipu client is not initialized.")
                            payload, raw_text, request_model, extra_body_used = _evaluate_with_glm_model(
                                zhipu_client=zhipu_client,
                                sample_name=sample_name,
                                model_name=model_name,
                                language=language,
                                keyframe_images=active_keyframe_data["image_items"],
                            )
                        else:
                            if client is None:
                                raise RuntimeError("DashScope client is not initialized.")
                            payload, raw_text, request_model, extra_body_used = _evaluate_with_model(
                                client=client,
                                sample_name=sample_name,
                                model_name=model_name,
                                language=language,
                                keyframe_images=active_keyframe_data["image_items"],
                            )
                        keyframe_data = active_keyframe_data
                        break
                    except Exception as exc:
                        if (
                            active_keyframe_data is not None
                            and str(active_keyframe_data.get("compression_mode") or "") != "aggressive"
                            and _is_request_payload_too_large_error(exc)
                        ):
                            _print_payload_retry_event(
                                sample_name=sample_name,
                                model_name=model_name,
                                error_message=str(exc),
                            )
                            active_keyframe_data = _extract_keyframe_images(
                                str(video_path),
                                target_k=int(target_k),
                                show=False,
                                compression_mode="aggressive",
                            )
                            continue
                        raise
                db_text = _compose_db_text(model_name=db_model_name, payload=payload, language=language)
                add_or_update_evaluation(
                    sample_name=sample_name,
                    model_name=db_model_name,
                    eval_text=db_text,
                )

                row = build_prediction_row(
                    base_row=base_row,
                    evaluation_text=payload["evaluation_text"],
                    gt_payload=gt_payload,
                    pred_scores=payload,
                    raw_model_output=raw_text,
                    cache_source="fresh_inference",
                    db_written=True,
                    extra_fields={
                        "request_model": request_model,
                        "extra_body_used": json.dumps(extra_body_used, ensure_ascii=False)
                        if extra_body_used
                        else "",
                        "cached_from_db": False,
                    },
                )
                new_records.append(row)
                processed_keys = [(sample_name, model_name)]
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
                historical_cache.setdefault((sample_name, db_model_name), []).insert(
                    0,
                    standardize_record_aliases(row),
                )
            except Exception as exc:
                fail = dict(base_row)
                fail.update(
                    {
                        "stage": "model_inference_or_parsing",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                new_failures.append(fail)
                processed_keys = [(sample_name, model_name)]
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
                    model_name=model_name,
                    stage=str(fail.get("stage") or "unknown"),
                    error_type=str(fail.get("error_type") or "UnknownError"),
                    error_message=str(fail.get("error_message") or ""),
                    traceback_text=str(fail.get("traceback") or ""),
                )
    records_df = pd.DataFrame([standardize_record_aliases(row) for row in records_to_export])

    model_runtime_config = []
    for model_name in models:
        model_runtime_config.append(
            {
                "model_name": model_name,
                "provider": _provider_for_model(model_name),
                "db_model_name": _db_model_name_for(model_name),
                "request_candidates": _candidate_request_models(model_name),
                "extra_body_default": _build_dashscope_extra_body(model_name),
                "glm_thinking_enabled": bool(GLM_THINKING_ENABLED) if _is_glm_model(model_name) else None,
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root_path),
        "video_ext": ext,
        "num_video_files": int(len(target_groups)),
        "language": language,
        "target_k": int(target_k),
        "models": models,
        "resume_mode": bool(resume_ctx["enabled"]),
        "resume_failed_run_name": str(resume_ctx.get("run_name") or ""),
        "resume_target_count": int(
            sum(len(group["models"]) for group in target_groups)
        ) if resume_ctx["enabled"] else 0,
        "resume_existing_record_count": int(len(resume_ctx["existing_records"])) if resume_ctx["enabled"] else 0,
        "resume_existing_failure_count": int(len(resume_ctx["existing_failures"])) if resume_ctx["enabled"] else 0,
        "resume_new_record_count": int(len(new_records)),
        "resume_new_failure_count": int(len(new_failures)),
        "model_score_summary": _build_model_score_summary(records_df),
        "model_mae_summary": _build_model_mae_summary(records_df),
        "cache_source_counts": records_df["cache_source"].value_counts(dropna=False).to_dict()
        if "cache_source" in records_df.columns
        else {},
        "model_runtime_config": model_runtime_config,
        "spec_source": {
            "dashscope_base_url": DASHSCOPE_BASE_URL,
            "zhipu_api_key_env": ZHIPU_API_KEY_ENV,
            "qwen_enable_thinking": QWEN_ENABLE_THINKING,
            "qwen_thinking_budget": QWEN_THINKING_BUDGET if QWEN_ENABLE_THINKING else None,
            "kimi_enable_thinking": KIMI_ENABLE_THINKING,
            "glm_thinking_enabled": GLM_THINKING_ENABLED,
            "upload_image_max_bytes": int(UPLOAD_IMAGE_MAX_BYTES),
            "upload_total_image_max_bytes": int(UPLOAD_TOTAL_IMAGE_MAX_BYTES),
            "upload_min_image_max_bytes": int(UPLOAD_MIN_IMAGE_MAX_BYTES),
            "upload_default_long_sides": list(UPLOAD_DEFAULT_LONG_SIDES),
            "upload_aggressive_long_sides": list(UPLOAD_AGGRESSIVE_LONG_SIDES),
            "upload_default_jpeg_qualities": list(UPLOAD_DEFAULT_JPEG_QUALITIES),
            "upload_aggressive_jpeg_qualities": list(UPLOAD_AGGRESSIVE_JPEG_QUALITIES),
        },
    }

    manifest = {
        "run_dir": str(run_dir),
        "group_column": "db_model_name",
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
        group_column="db_model_name",
    )
    return exported["summary"]


def main() -> None:
    summary = run_batch_dashscope(
        dataset_root=DATASET_ROOT,
        models=MODEL_LIST,
        language=LANGUAGE,
        target_k=TARGET_K,
        output_root=OUTPUT_ROOT,
        resume_failed_run_name=RESUME_FAILED_RUN_NAME,
        max_samples=MAX_SAMPLES,
        video_ext=VIDEO_EXT,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

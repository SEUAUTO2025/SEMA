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
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

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
SEMANTIC_METRIC_LANGUAGE = "zh"
BERT_SCORE_LANGUAGE = "en"
METRIC_PROTOCOL = "zh_raw_text_bleu_plus_bertscore_only"

# Fixed model list (OpenRouter).
MODEL_LIST = [
    "google/gemini-3-flash-preview",
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "meta-llama/llama-4-maverick",
    "openai/gpt-5.2"
]
MODEL_NAME_PREFIX = "openrouter::"

OUTPUT_ROOT = os.path.join(ROOT, "evaluation", "results")
RESUME_FAILED_RUN_NAME = ""
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
REQUEST_TIMEOUT_SEC = 45

TEMPERATURE = 0.2
MAX_TOKENS = 900
PRINT_INTERMEDIATE_OUTPUT = True
RAW_OUTPUT_MAX_CHARS: Optional[int] = None  # None means print full output.
PRINT_PARSED_PAYLOAD = True
PRINT_SKIP_ERRORS = True
PRINT_SKIP_TRACEBACK = False

# OpenRouter reasoning defaults (medium thinking).
REASONING_EFFORT = "medium"
REASONING_MAX_TOKENS = 1024
EXCLUDE_REASONING_FROM_RESPONSE = True

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
    run_dir = root / f"openrouter_mm_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print_raw: bool = True,
) -> None:
    if not PRINT_INTERMEDIATE_OUTPUT:
        return
    if print_raw:
        print(f"\n[OpenRouter][{sample_name}][{model_name}] RAW OUTPUT BEGIN")
        print(_truncate_for_print(raw_text))
        print(f"[OpenRouter][{sample_name}][{model_name}] RAW OUTPUT END")
    if PRINT_PARSED_PAYLOAD and payload is not None:
        print(f"[OpenRouter][{sample_name}][{model_name}] PARSED PAYLOAD:")
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
        "[OpenRouter][SKIP] sample={0} model={1} stage={2} error_type={3} error={4}".format(
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
        "evaluation_text,total_score,head_score,hand_score,torso_score,foot_score,arm_score.\n"
        "评分约束:各部位分数必须是0-5的整数,总分必须是0-25的整数,且等于五个部位分数之和.\n"
        "evaluation_text=的格式为：这位同学的正确动作如下：......这位同学的错误动作如下：......\n"
        # f"{ref_text}"
    )
    user_note = "已提供全视频均匀采样的{0}帧，按时间从早到晚排序，请对该运动员进行动作质量评估。".format(
        frame_count
    )
    return system_prompt, user_note


def _extract_keyframe_images(video_path: str, target_k: int, show: bool = False) -> Dict[str, Any]:
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

    image_items: List[Dict[str, Any]] = []
    used_idx: List[int] = []
    for idx in frame_idx:
        read_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
        ok, frame = read_cap.read()
        if not ok or frame is None:
            continue
        ok_enc, buf = cv2.imencode(".jpg", frame)
        if not ok_enc:
            continue
        encoded = base64.b64encode(buf.tobytes()).decode("ascii")
        image_items.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
        used_idx.append(int(idx))
    read_cap.release()

    if not image_items:
        raise ValueError("No sampled frames available for model input.")
    return {
        "image_items": image_items,
        "keyframe_count": len(image_items),
        "frame_idx": used_idx,
    }


def _build_openrouter_client(api_key: str) -> OpenAI:
    headers = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.getenv("OPENROUTER_APP_TITLE", "SEMA-OpenRouter-MM-Batch").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers if headers else None,
    )


def _fetch_openrouter_supported_parameters(
    api_key: str,
    target_models: List[str],
) -> Dict[str, Set[str]]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        OPENROUTER_MODELS_URL,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Unexpected /models payload: missing 'data' list.")

    all_params: Dict[str, Set[str]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        supported = item.get("supported_parameters") or []
        if isinstance(supported, list):
            all_params[model_id] = {str(x) for x in supported}
        else:
            all_params[model_id] = set()

    missing = [m for m in target_models if m not in all_params]
    if missing:
        raise ValueError(f"Target models not found in OpenRouter /models: {missing}")

    return {m: all_params[m] for m in target_models}


def _build_reasoning_extra_body(
    model_name: str,
    supported_params: Set[str],
) -> Optional[Dict[str, Any]]:
    if "reasoning" not in supported_params:
        return None

    # Provider-specific reasoning shape.
    if model_name == "google/gemini-3-flash-preview":
        return {"reasoning": {"max_tokens": REASONING_MAX_TOKENS}}
    if model_name == "anthropic/claude-sonnet-4":
        return {
            "reasoning": {
                "max_tokens": REASONING_MAX_TOKENS,
                "exclude": EXCLUDE_REASONING_FROM_RESPONSE,
            }
        }
    if model_name == "x-ai/grok-4-fast":
        return {
            "reasoning": {
                "effort": REASONING_EFFORT,
                "exclude": EXCLUDE_REASONING_FROM_RESPONSE,
            }
        }

    return {
        "reasoning": {
            "effort": REASONING_EFFORT,
            "exclude": EXCLUDE_REASONING_FROM_RESPONSE,
        }
    }


def _is_unsupported_param_error(exc: Exception) -> bool:
    text = str(exc).lower()
    keywords = [
        "unsupported",
        "unknown parameter",
        "invalid parameter",
        "extra_body",
        "reasoning",
    ]
    return any(k in text for k in keywords)


def _build_no_web_request_fields(supported_params: Set[str]) -> Dict[str, Any]:
    """
    Best-effort hard disable for external/tool/web retrieval.
    Only pass fields advertised by model supported_parameters.
    """
    out: Dict[str, Any] = {}
    if "plugins" in supported_params:
        out["plugins"] = []
    if "tools" in supported_params:
        out["tools"] = []
    if "tool_choice" in supported_params:
        out["tool_choice"] = "none"
    return out


def _evaluate_with_model(
    client: OpenAI,
    sample_name: str,
    model_name: str,
    language: str,
    keyframe_images: List[Dict[str, Any]],
    supported_params: Set[str],
) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]], bool]:
    system_prompt, user_note = _build_prompts(language=language, target_k=len(keyframe_images))
    user_content = [{"type": "text", "text": user_note}] + keyframe_images

    request_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if "temperature" in supported_params:
        request_kwargs["temperature"] = TEMPERATURE
    if "max_tokens" in supported_params:
        request_kwargs["max_tokens"] = MAX_TOKENS

    # Explicitly disable web/tool retrieval when the model accepts these fields.
    request_kwargs.update(_build_no_web_request_fields(supported_params=supported_params))

    extra_body = _build_reasoning_extra_body(model_name=model_name, supported_params=supported_params)
    reasoning_applied = False
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    try:
        resp = client.chat.completions.create(**request_kwargs)
        reasoning_applied = bool(extra_body)
    except Exception as exc:
        # Soft fallback: if reasoning body is rejected, retry without it.
        if not extra_body or not _is_unsupported_param_error(exc):
            raise
        fallback_kwargs = dict(request_kwargs)
        fallback_kwargs.pop("extra_body", None)
        resp = client.chat.completions.create(**fallback_kwargs)
        extra_body = None
        reasoning_applied = False

    raw_text = _response_to_text(resp)
    _print_intermediate_output(sample_name=sample_name, model_name=model_name, raw_text=raw_text)
    payload = _parse_eval_payload(raw_text)
    _print_intermediate_output(
        sample_name=sample_name,
        model_name=model_name,
        raw_text=raw_text,
        payload=payload,
        print_raw=False,
    )
    return payload, raw_text, extra_body, reasoning_applied


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


def run_batch_openrouter(
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
        run_dir = build_run_dir(output_root=output_root, run_prefix="openrouter_mm_batch", base_dir=ROOT)

    if len(models) > 15:
        raise ValueError("evaluation DB supports at most 15 model names per sample.")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required.")

    new_records: List[Dict[str, Any]] = []
    new_failures: List[Dict[str, Any]] = []
    if resume_ctx["enabled"]:
        records_to_export = list(resume_ctx["existing_records"])
        failures_to_export = list(resume_ctx["existing_failures"])
    else:
        records_to_export = []
        failures_to_export = []

    init_db()
    client = _build_openrouter_client(api_key=api_key)
    supported_map = _fetch_openrouter_supported_parameters(api_key=api_key, target_models=models)
    historical_cache = build_historical_prediction_index()

    iterator = tqdm(
        target_groups,
        desc="OpenRouter MM Resume" if resume_ctx["enabled"] else "OpenRouter MM Eval",
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
                    "provider": "openrouter",
                    "model_name": model_name,
                    "db_model_name": f"{MODEL_NAME_PREFIX}{model_name}",
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
            db_model_name = f"{MODEL_NAME_PREFIX}{model_name}"
            existing_eval = get_evaluation(sample_name=sample_name, model_name=db_model_name)
            existing_eval_map[model_name] = existing_eval
            existing_text = normalize_eval_text(existing_eval.get("eval_text")) if existing_eval else ""
            if not existing_text:
                models_to_infer.append(model_name)

        keyframe_data: Optional[Dict[str, Any]] = None
        if models_to_infer:
            try:
                keyframe_data = _extract_keyframe_images(str(video_path), target_k=int(target_k), show=False)
            except Exception as exc:
                tb = traceback.format_exc()
                for model_name in models_to_infer:
                    db_model_name = f"{MODEL_NAME_PREFIX}{model_name}"
                    sample_failures[model_name] = {
                        "sample_name": sample_name,
                        "video_path": str(video_path),
                        "provider": "openrouter",
                        "model_name": model_name,
                        "db_model_name": db_model_name,
                        "stage": "frame_sampling",
                        "timestamp": sample_ts,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": tb,
                    }

        for model_name in sample_models:
            db_model_name = f"{MODEL_NAME_PREFIX}{model_name}"
            supported_params = supported_map.get(model_name, set())
            existing_eval = existing_eval_map.get(model_name)
            cached_eval_text = normalize_eval_text(existing_eval.get("eval_text")) if existing_eval else ""
            base_row = {
                "sample_name": sample_name,
                "video_path": str(video_path),
                "provider": "openrouter",
                "model_name": model_name,
                "db_model_name": db_model_name,
                "supported_parameters": ",".join(sorted(supported_params)),
                "language": language,
                "target_k": int(target_k),
                "keyframe_count": int(keyframe_data["keyframe_count"]) if keyframe_data else 0,
                "frame_idx": ",".join(str(x) for x in keyframe_data["frame_idx"]) if keyframe_data else "",
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
                    "reasoning_applied": bool(historical_row.get("reasoning_applied"))
                    if historical_row
                    else False,
                    "reasoning_extra_body": str(historical_row.get("reasoning_extra_body") or "")
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
                    include_english_only_text_metrics=False,
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
                payload, raw_text, reasoning_extra_body, reasoning_applied = _evaluate_with_model(
                    client=client,
                    sample_name=sample_name,
                    model_name=model_name,
                    language=language,
                    keyframe_images=keyframe_data["image_items"],
                    supported_params=supported_params,
                )
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
                    include_english_only_text_metrics=False,
                    extra_fields={
                        "reasoning_applied": bool(reasoning_applied),
                        "reasoning_extra_body": json.dumps(reasoning_extra_body, ensure_ascii=False)
                        if reasoning_extra_body
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
        supported_params = supported_map.get(model_name, set())
        model_runtime_config.append(
            {
                "model_name": model_name,
                "db_model_name": f"{MODEL_NAME_PREFIX}{model_name}",
                "supported_parameters": sorted(supported_params),
                "no_web_fields_if_supported": _build_no_web_request_fields(
                    supported_params=supported_params,
                ),
                "reasoning_payload_if_used": _build_reasoning_extra_body(
                    model_name=model_name,
                    supported_params=supported_params,
                ),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root_path),
        "video_ext": ext,
        "num_video_files": int(len(target_groups)),
        "language": language,
        "semantic_metric_language": SEMANTIC_METRIC_LANGUAGE,
        "bert_score_language": BERT_SCORE_LANGUAGE,
        "metric_protocol": METRIC_PROTOCOL,
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
            "openrouter_models_api": OPENROUTER_MODELS_URL,
            "reasoning_default": REASONING_EFFORT,
            "web_search_policy": "disabled_if_supported_via_plugins_tools",
        },
    }

    manifest = {
        "run_dir": str(run_dir),
        "group_column": "db_model_name",
        "history_search_roots": [str(path) for path in DEFAULT_HISTORY_SEARCH_ROOTS],
        "semantic_metric_language": SEMANTIC_METRIC_LANGUAGE,
        "bert_score_language": BERT_SCORE_LANGUAGE,
        "metric_protocol": METRIC_PROTOCOL,
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
        semantic_metric_columns=SEMANTIC_METRIC_COLUMNS,
    )
    return exported["summary"]


def main() -> None:
    summary = run_batch_openrouter(
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

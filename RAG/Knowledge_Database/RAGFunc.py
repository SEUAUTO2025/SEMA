"""
@filename: RAG.py
@description: Implementation of all RAG functions
"""
import os
import sys
import re
from pathlib import Path
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    RecursiveCharacterTextSplitter = None
import numpy as np
import pandas as pd
from openai import OpenAI
import requests
import json
_LANGUAGEBIND_IMPORT_ERROR = None
try:
    from RAG.Knowledge_Database.languagebind_main.languagebind import LanguageBindImageTokenizer,LanguageBind, to_device, LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor, transform_dict
except ModuleNotFoundError as exc:
    _LANGUAGEBIND_IMPORT_ERROR = exc
    LanguageBindImageTokenizer = None
    LanguageBind = None
    to_device = None
    LanguageBindVideo = None
    LanguageBindVideoTokenizer = None
    LanguageBindVideoProcessor = None
    transform_dict = None
import torch
import json
from typing import List, Union, Dict, Any, Literal
import base64
_BONE_FEATURE_IMPORT_ERROR = None
try:
    from RTMPose.Bone_Feature_Extract import *
except ModuleNotFoundError as exc:
    _BONE_FEATURE_IMPORT_ERROR = exc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_MODEL_CLIENT_ROUTE = {
    "qwen3-vl-plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "ALI_API_KEY",
    },
    "google/gemini-3-flash-preview": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    # Backward-compat for existing QA defaults.
    "qwen-plus": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "ALI_API_KEY",
    },
}

def _get_llm_client_by_model(model_name: str) -> OpenAI:
    resolved_model_name = str(model_name or "").strip()
    if not resolved_model_name:
        raise ValueError("model_name must be a non-empty string.")

    route = _MODEL_CLIENT_ROUTE.get(resolved_model_name)
    if route is None:
        supported = ", ".join(sorted(_MODEL_CLIENT_ROUTE.keys()))
        raise ValueError(
            f"Unsupported model_name: '{resolved_model_name}'. "
            f"Supported models: {supported}"
        )

    api_key_env = str(route["api_key_env"]).strip()
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"{api_key_env} is required for model '{resolved_model_name}'."
        )

    return OpenAI(
        api_key=api_key,
        base_url=str(route["base_url"]).strip(),
    )


def split_and_merge(text, chunk_size=500):
    """
    Split text using LangChain, then merge into chunks close to chunk_size
    - Does not cross paragraphs
    - Does not break sentences
    - Punctuation at end goes to previous chunk
    - Remove leading punctuation from chunks
    """
    if RecursiveCharacterTextSplitter is None:
        normalized_text = str(text or "").replace("\r\n", "\n").strip()
        if not normalized_text:
            return []

        paragraphs = [p.strip() for p in normalized_text.split("\n\n") if p.strip()]
        chunks = []
        sentence_re = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")

        for paragraph in paragraphs:
            sentences = [s.strip() for s in sentence_re.findall(paragraph) if s.strip()]
            if not sentences:
                sentences = [paragraph]

            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", ".", "！", "!", "？", "?", ";", ";"]
    )
    pieces = splitter.split_text(text)

    chunks = []
    current_chunk = ""
    current_len = 0
    punctuations = {".", ".", "！", "!", "？", "?", ";", ";"}

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if "\n\n" in piece:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_len = 0
            piece = piece.replace("\n\n", "").strip()
            if not piece:
                continue

        if piece in punctuations:
            current_chunk += piece
            current_len += len(piece)
            continue

        piece_len = len(piece)
        if current_len + piece_len <= chunk_size:
            current_chunk += piece
            current_len += piece_len
        else:
            if current_chunk:
                while current_chunk and current_chunk[0] in punctuations:
                    current_chunk = current_chunk[1:].lstrip()
                if current_chunk:
                    chunks.append(current_chunk.strip())
            current_chunk = piece
            current_len = piece_len

    if current_chunk:
        while current_chunk and current_chunk[0] in punctuations:
            current_chunk = current_chunk[1:].lstrip()
        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

def get_embedding(texts):
    """Call OpenAI API to get text embedding vectors"""
    if isinstance(texts, str):
        input_texts = [texts]
    else:
        input_texts = list(texts or [])

    if not input_texts:
        return []

    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    embeddings = []
    batch_size = 10
    for start_idx in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[start_idx:start_idx + batch_size]
        response = client.embeddings.create(
            model="text-embedding-v4",
            input=batch_texts,
            dimensions=1024,
            encoding_format="float"
        )
        embeddings.extend(
            [np.array(item.embedding, dtype=np.float32) for item in response.data]
        )
    return embeddings


def video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        encoded = base64.b64encode(video_file.read()).decode("utf-8")
    return encoded

def _media_type_from_ext(path: str) -> str:
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


def _load_openai_images_from_folder(folder: str) -> list[dict]:
    folder_path = os.path.abspath(str(folder or "").strip())
    if not folder_path or not os.path.isdir(folder_path):
        raise ValueError(
            f"Template keyframes directory not found: {folder_path}. "
            "Please run extract_reference_keyframes.py first."
        )

    ordered_paths = []
    numeric_files = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        stem, ext = os.path.splitext(name)
        if not os.path.isfile(full_path):
            continue
        # Pipeline 4 template keyframes: only *.jpg named as integers (1.jpg, 2.jpg, ...).
        if ext.lower() != ".jpg":
            continue
        stem_stripped = str(stem).strip()
        if not stem_stripped.isdigit():
            continue
        numeric_files.append((int(stem_stripped), os.path.abspath(full_path)))

    numeric_files.sort(key=lambda x: x[0])
    ordered_paths = [p for _, p in numeric_files]

    if not ordered_paths:
        raise ValueError(
            f"No numeric .jpg template keyframes found in {folder_path}. "
            "Expected files like 1.jpg, 2.jpg, 3.jpg."
        )

    openai_images = []
    for img_path in ordered_paths:
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read template keyframe image: {img_path}. {e}") from e

        media_type = _media_type_from_ext(img_path)
        openai_images.append(
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{b64}",
            }
        )

    return openai_images


_KEYWORD_POLARITY_LIBRARY = {
    "zh": {
        "video": {
            "positive": [
                "身体纵轴垂直于地面",
                "身体侧向对着射箭方向",
                "头部水平转动",
                "前手,前肩,后肩在一条线上,后手小臂与箭在一条线上",
                "后手靠在脸颊下颌位置,弓弦靠在下巴和鼻子上",
                "瞄准时后手没有松滑弦",
                "双脚开步与肩同宽",
                "撒放时后手顺势向后延展",
                "撒放时前手放松推弓",
            ],
            "negative": [
                "身体重心歪斜",
                "身体没有转到侧对",
                "头部没有转动到位",
                "小臂与箭有角度",
                "后手悬空没有贴实,弦没有贴上下巴和鼻子",
                "瞄准时后手松滑弦",
                "双脚随意站立",
                "撒放时后手定在原地或向外,向前",
                "撒放时前手握弓",
            ],
        },
        "keyframes": {
            "positive": [
                "身体纵轴垂直于地面",
                "身体侧向对着射箭方向",
                "头部水平转动",
                "前手,前肩,后肩在一条线上,后手小臂与箭在一条线上",
                "后手靠在脸颊下颌位置,弓弦靠在下巴和鼻子上",
                "瞄准时后手未松滑弦",
                "双脚开步与肩同宽",
                "撒放时后手顺势向后延展",
                "撒放时前手放松推弓",
            ],
            "negative": [
                "身体重心歪斜",
                "身体没有转到侧对",
                "头部没有转动到位",
                "小臂与箭有角度",
                "后手悬空没有贴实,弦没有贴上下巴和鼻子",
                "瞄准时后手松滑弦",
                "双脚随意站立",
                "撒放时后手定在原地或向外,向前",
                "撒放时前手握弓",
            ],
        },
    },
    "en": {
        "video": {
            "positive": [
                "Body vertical axis is perpendicular to the ground",
                "body is side-on to the shooting direction",
                "Head rotates horizontally",
                "Front hand, front shoulder, and rear shoulder are aligned on one line",
                "front forearm is internally rotated",
                "rear forearm is aligned with the arrow",
                "Bow-hand fingers are relaxed",
                "rear hand is anchored at the cheek/jaw area",
                "the bowstring touches the chin and nose",
                "during aiming the rear hand does not slip/loosen the string",
                "Feet stance is shoulder-width",
                "at release the rear hand extends backward",
                "at release the front/bow hand relaxes and pushes the bow",
            ],
            "negative": [
                "Body center of mass is tilted/leaning",
                "body is not rotated to a side-on stance",
                "Head does not rotate enough",
                "Front shoulder has an angle and is not opened",
                "forearm is turned outward",
                "forearm is not aligned with the arrow",
                "Fingers are stiff or clenched",
                "rear hand is floating and not anchored",
                "string does not touch chin and nose",
                "before/after release the rear hand slowly slides backward",
                "at release the rear hand stays in place or moves outward/forward",
                "at release the front/bow hand grabs the bow",
                "Feet are placed arbitrarily",
            ],
        },
        "keyframes": {
            "positive": [
                "Body vertical axis is perpendicular to the ground",
                "body is side-on to the shooting direction",
                "Head rotates horizontally",
                "Front hand, front shoulder, and rear shoulder are aligned on one line",
                "front forearm is internally rotated",
                "rear forearm is aligned with the arrow",
                "Bow-hand fingers are relaxed",
                "rear hand is anchored at the cheek/jaw area",
                "the bowstring touches the chin and nose",
                "during aiming the rear hand does not slip/loosen the string",
                "Feet stance is shoulder-width",
                "at release the rear hand extends backward",
                "at release the front/bow hand relaxes and pushes the bow",
            ],
            "negative": [
                "Body center of mass is tilted/leaning",
                "body is not rotated to a side-on stance",
                "Head does not rotate enough",
                "Front shoulder has an angle and is not opened",
                "forearm is turned outward",
                "forearm is not aligned with the arrow",
                "Fingers are stiff or clenched",
                "rear hand is floating and not anchored",
                "string does not touch chin and nose",
                "before/after release the rear hand slowly slides backward",
                "at release the rear hand stays in place or moves outward/forward",
                "at release the front/bow hand grabs the bow",
                "Feet are placed arbitrarily",
            ],
        },
    },
}


def _get_keyword_polarity_library(language: str, pipeline: int) -> dict[str, list[str]]:
    lang = "en" if str(language or "zh").strip().lower().startswith("en") else "zh"
    stage = "video" if int(pipeline) == 1 else "keyframes"
    library = _KEYWORD_POLARITY_LIBRARY.get(lang, {}).get(stage, {})
    return {
        "positive": list(library.get("positive", [])),
        "negative": list(library.get("negative", [])),
    }


def _classify_comment_keywords(
    keywords: list[str],
    language: str,
    pipeline: int,
) -> dict[str, list[str]]:
    library = _get_keyword_polarity_library(language=language, pipeline=pipeline)
    positive_map = {
        _normalize_text_for_match(item): item
        for item in library.get("positive", [])
        if str(item).strip()
    }
    negative_map = {
        _normalize_text_for_match(item): item
        for item in library.get("negative", [])
        if str(item).strip()
    }

    buckets = {"positive": [], "negative": [], "unknown": []}
    seen = {key: set() for key in buckets}
    for keyword in keywords or []:
        keyword_text = str(keyword).strip()
        if not keyword_text:
            continue
        keyword_norm = _normalize_text_for_match(keyword_text)
        if keyword_norm in positive_map:
            bucket = "positive"
        elif keyword_norm in negative_map:
            bucket = "negative"
        else:
            bucket = "unknown"
        if keyword_text not in seen[bucket]:
            buckets[bucket].append(keyword_text)
            seen[bucket].add(keyword_text)
    return buckets


def _load_pose_sequence_from_csv(csv_path: Union[str, Path]) -> np.ndarray:
    csv_file = Path(str(csv_path or "").strip()).resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"Pose CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"Pose CSV has no rows: {csv_file}")
    if "frame" not in df.columns:
        raise ValueError(f"Pose CSV missing required column 'frame': {csv_file}")

    num_joints = 133
    coord_cols = []
    for joint_idx in range(num_joints):
        coord_cols.extend(["x{0}".format(joint_idx), "y{0}".format(joint_idx), "z{0}".format(joint_idx)])

    missing_cols = [name for name in coord_cols if name not in df.columns]
    if missing_cols:
        raise ValueError(f"Pose CSV missing pose columns, e.g. {missing_cols[:6]}")

    frame_values = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(frame_values) & (frame_values >= 0.0)
    if not np.any(valid_mask):
        raise ValueError(f"Pose CSV has no valid non-negative frame indices: {csv_file}")

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


def get_video_ori_keywords(
    video_path,
    pose_csv_path=None,
    pipeline=1,
    target_k=8,
    model_name='qwen3-vl-plus',
    temperature=0.3,
    language='zh',
    show=False,
    template_keyframes_dir = None,
    progress_callback=None,
) -> dict:
    """
    Multi-stage multimodal assessment for archery posture.

    Pipeline semantics:
        1) Video only: user content contains ONLY video_url (no extra text).
        2) Keyframes only: extracted keyframes are sent as input_image parts (Responses API).
        3) Keyframes + biomechanics metrics: keyframes + metrics text (Responses API).
        4) Keyframes + biomechanics metrics + template keyframes loaded from folder.

    Return:
        A normalized assessment dict with keys:
            total_score, head_score, hand_score, torso_score, foot_score, arm_score,
            comment, comment_polarity
        where `comment` is a keyword list split by comma, and `comment_polarity`
        groups those keywords into positive / negative / unknown.
        When `pose_csv_path` is provided, pipelines 2/3/4 read pose keypoints from
        that CSV and continue with the existing keyframe / biomechanics logic.
    """
    if int(pipeline) in (2, 3, 4) and _BONE_FEATURE_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "RTMPose.Bone_Feature_Extract dependencies are unavailable: "
            f"{_BONE_FEATURE_IMPORT_ERROR}"
        )

    def _normalize_language(lang_in: str) -> str:
        lang_norm = str(lang_in or "en").strip().lower()
        if lang_norm in ("zh", "zh-cn", "zh_hans", "cn", "chinese", "中文"):
            return "zh"
        if lang_norm in ("en", "en-us", "english"):
            return "en"
        raise ValueError("language must be 'en' or 'zh'")

    def _notify_progress(stage: str, message: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(stage, message)
            except Exception:
                pass

    def _chat_completion_output_text(resp) -> str:
        if resp is None or not getattr(resp, "choices", None):
            raise ValueError("Chat completion returned empty response.")

        message = getattr(resp.choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        parts.append(str(item["text"]))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
            if parts:
                return "".join(parts).strip()

        raise ValueError("Chat completion contains no textual content.")

    def _to_chat_content(input_content: list[dict]) -> list[dict]:
        chat_content = []
        for item in input_content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "input_text":
                chat_content.append({"type": "text", "text": str(item.get("text", ""))})
            elif item_type == "input_image":
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if image_url:
                    chat_content.append({"type": "image_url", "image_url": {"url": str(image_url)}})
        if not chat_content:
            raise ValueError("No valid chat content converted from keyframe inputs.")
        return chat_content

    def _extract_json_object_text(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            raise ValueError("Model response is empty; expected a JSON object string.")

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
        raise ValueError("No valid JSON object found in model response.")

    def _normalize_comment_keywords(raw_comment) -> list[str]:
        if isinstance(raw_comment, list):
            keywords = [str(item).strip() for item in raw_comment]
            return [k for k in keywords if k]

        text = str(raw_comment or "").replace(",", ",")
        parts = [p.strip() for p in text.split(",")]
        return [p for p in parts if p]

    def _parse_assessment_payload(raw_text: str) -> dict:
        required_keys = (
            "total_score",
            "head_score",
            "hand_score",
            "torso_score",
            "foot_score",
            "arm_score",
            "comment",
        )
        json_text = _extract_json_object_text(raw_text)
        try:
            payload = json.loads(json_text)
        except Exception as e:
            raise ValueError(f"Failed to parse model JSON response: {e}") from e

        if not isinstance(payload, dict):
            raise ValueError("Model response JSON is not an object.")

        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise ValueError(f"Model response JSON missing required keys: {missing}")

        out = {}
        for key in required_keys[:-1]:
            try:
                out[key] = int(payload[key])
            except Exception as e:
                raise ValueError(f"Invalid score field '{key}': {payload.get(key)}") from e
        out["comment"] = _normalize_comment_keywords(payload.get("comment"))
        return out

    def _finalize_assessment_payload(raw_text: str) -> dict:
        payload = _parse_assessment_payload(raw_text)
        payload["comment_polarity"] = _classify_comment_keywords(
            keywords=payload.get("comment", []),
            language=lang,
            pipeline=pipeline,
        )
        return payload

    lang = _normalize_language(language)
    pipeline = int(pipeline)
    if pipeline not in (1, 2, 3, 4):
        raise ValueError(
            "pipeline must be 1 (video), 2 (keyframes), 3 (keyframes+metrics), "
            "or 4 (keyframes+metrics+template keyframes)"
        )


    client = _get_llm_client_by_model(model_name)

    system_base_map = {
        "en": (
            "You are a professional archery coach.\n"
            "Use ONLY the user-provided inputs (video/keyframes and optional biomechanics metrics).\n"
            "You must NOT search for information or draw upon your own prior knowledge.\n"
            "Output MUST be ONLY a JSON object string with keys: "
            "'total_score', 'head_score', 'hand_score', 'torso_score', 'foot_score', 'arm_score', 'comment'.\n"
            "All scores MUST be integers: each part score in [0,5], total_score in [0,25].\n"
            "total_score MUST equal head_score + hand_score + torso_score + foot_score + arm_score.\n"
            "In 'comment', output a concise keyword sequence.\n"
            "Use short phrases separated by commas.\n"
            "Do not add extra keys, markdown, code fences, or any other text."
        ),
        "zh": (
            "你是一名专业的射箭教练.\n"
            "请参考下文完成指定的任务,下面先提出对输入及输出的要求.\n"
            "只能使用用户提供的输入(视频/关键帧以及可选的人体生物力学指标),不得搜索或引用已知知识,也不能输出不基于实时评判的结果\n"
            "输出必须且只能是一个 JSON 对象字符串,键必须为:"
            "'total_score', 'head_score', 'hand_score', 'torso_score', 'foot_score', 'arm_score', 'comment'.\n"
            "所有分数必须为整数:各部位分数范围[0,5],总分 total_score 范围[0,25].\n"
            "comment字段输出简洁的关键句序列\n"
            "用短语,使用&号分隔各序列中的关键句(关键句1&关键句2......这样的形式).\n"
            "不要添加额外键、Markdown、代码块或任何其他文本"
        ),
    }

    system_task_video_map = {
        "en": (
            "\nTask:\n"
            "You will receive a full archery video. Observe and judge whether the archer achieves the following.\n"
            "Static / aiming-phase observations (mainly from earlier parts):\n"
            "- Body upright/vertical axis stable\n"
            "- Head rotates sufficiently toward target\n"
            "- Arms form a straight line / alignment is correct\n"
            "- Rear hand firmly anchored against the chin\n"
            "- Nose touches the bowstring (string touches chin/nose)\n"
            "- Bow hand fingers are relaxed (not clenched)\n"
            "- Before release, rear hand does NOT slip/loosen the string during aiming\n"
            "Release / follow-through observations (around release and after):\n"
            "- At release, rear hand extends backward (follow-through)\n"
            "- At release, front/bow hand relaxes and pushes/lets the bow move naturally (not grabbing)\n"
            "Then output the required JSON with preliminary scores and a keyword-style comment.\n"
            "Keywords MUST be concise, and MUST be selected from the following keyword library (use whichever items apply).\n"
            "Keyword library: separate keywords with \",\".\n"
            "1. Torso: (1) Correct: Body vertical axis is perpendicular to the ground; body is side-on to the shooting direction; "
            "(2) Incorrect: Body center of mass is tilted/leaning; body is not rotated to a side-on stance.\n"
            "2. Arms: (1) Correct: Front hand, front shoulder, and rear shoulder are aligned on one line; front forearm is internally rotated; "
            "rear forearm is aligned with the arrow; "
            "(2) Incorrect: Front shoulder has an angle and is not opened; forearm is turned outward; forearm is not aligned with the arrow.\n"
            "3. Head: (1) Correct: Head rotates horizontally; (2) Incorrect: Head does not rotate enough.\n"
            "4. Feet: (1) Correct: Feet stance is shoulder-width; (2) Incorrect: Feet are placed arbitrarily.\n"
            "5. Hands: (1) Correct: Bow-hand fingers are relaxed; rear hand is anchored at the cheek/jaw area; "
            "the bowstring touches the chin and nose; during aiming the rear hand does not slip/loosen the string; "
            "at release the rear hand extends backward; at release the front/bow hand relaxes and pushes the bow; "
            "(2) Incorrect: Fingers are stiff or clenched; rear hand is floating and not anchored; string does not touch chin and nose; "
            "before/after release the rear hand slowly slides backward; at release the rear hand stays in place or moves outward/forward; "
            "at release the front/bow hand grabs the bow.\n"
        ),
        "zh": (
            "\n任务:\n"
            "你将看到一个射箭短视频\n"
            "从视频中观察并判断下面的动作,如果符合则comment中加入对应标准后第一个括号内关键句库中的正确动作(R),反之加上对应的错误动作(W)(要求一字不差),并按照评分规则(S)对分数进行初评,再按下文中的信息修正你的评价(如果出现)\n"
            "1.身体是否直立(R:身体纵轴垂直于地面/W:身体重心歪斜)(S:可认为基本垂直的情况躯干部位给4/5分,出现歪斜给2/3分)\n"
            "2.身体是否转到侧对(S:一般给出R:身体侧向对着射箭方向 的评价,除非极其离谱,给出W:身体没有转到侧对)\n"
            "3.头部是否转动到位(R:头部水平转动/W:头部没有转动到位)(S:转动到位给5分,不到位给3分,极其离谱情况给1/2分)\n"
            "4.手臂是否成一条直线(R:前手,前肩,后肩在一条线上,后手小臂与箭在一条线上/W:小臂与箭有角度)(S:后肩部有角度给2/3分,否则4/5分)\n"
            "5.后手是否牢牢靠在下巴上(R:后手靠在脸颊下颌位置,弓弦靠在下巴和鼻子上/W:后手悬空没有贴实,弦没有贴上下巴和鼻子)(S:手部评分规则参看[手部评分规则])\n"
            "6.后手在未放箭的瞄准阶段是否有移动现象(R:瞄准时后手没有松滑弦/W:瞄准时后手松滑弦)\n"
            "7.双脚开步与肩同宽(R:双脚开步与肩同宽/W:双脚随意站立)(S:一般给出满分,出现极其离谱的情况可以给0/1分)\n"
            "8.放箭时后手是否在箭射出时向后延展(随动)(R:撒放时后手顺势向后延展/W:撒放时后手定在原地或向外,向前)\n"
            "9.箭射出后弓是否自然转动(R:撒放时前手放松推弓/W:撒放时前手握弓)\n"
            "随后按要求输出JSON,给出初步评分与关键词化comment(关键词应精炼,必须使用以下的表达(符合哪项表达就使用哪项表达))\n"
            "评分规则:\n"
            "[手部评分规则]:三条标准:靠下巴(手指必须盖住下巴的一部分,牢牢贴合才算贴合,否则就是没有贴合),撒放向后移动,推弓(撒放后弓必须自然转动)这三条完全满足5分,少一条扣一分"
        )
    }

    system_task_keyframes_map = {
        "en": (
            "\nTask:\n"
            "You will receive a time-ordered sequence of keyframes extracted from the video (early -> late).\n"
            "Use earlier keyframes to judge static posture/aiming quality:\n"
            "- Body upright/vertical axis stable\n"
            "- Head rotates sufficiently toward target\n"
            "- Arms form a straight line / alignment is correct\n"
            "- Rear hand firmly anchored against the chin\n"
            "- Nose touches the bowstring (string touches chin/nose)\n"
            "- Bow hand fingers are relaxed\n"
            "- Before release, rear hand does NOT slip/loosen the string during aiming\n"
            "Use later keyframes AND their changes to judge release/follow-through quality:\n"
            "- At release, rear hand extends backward (follow-through)\n"
            "- At release, front/bow hand relaxes and pushes/lets the bow move naturally (not grabbing)\n"
            "Important: use the changes across later keyframes to infer release quality, not a single still.\n"
            "Then output the required JSON with preliminary scores and a keyword-style comment.\n"
            "Keywords MUST be concise, and MUST be selected from the following keyword library (use whichever items apply).\n"
            "Keyword library: separate keywords with \",\".\n"
            "1. Torso: (1) Correct: Body vertical axis is perpendicular to the ground; body is side-on to the shooting direction; "
            "(2) Incorrect: Body center of mass is tilted/leaning; body is not rotated to a side-on stance.\n"
            "2. Arms: (1) Correct: Front hand, front shoulder, and rear shoulder are aligned on one line; front forearm is internally rotated; "
            "rear forearm is aligned with the arrow; "
            "(2) Incorrect: Front shoulder has an angle and is not opened; forearm is turned outward; forearm is not aligned with the arrow.\n"
            "3. Head: (1) Correct: Head rotates horizontally; (2) Incorrect: Head does not rotate enough.\n"
            "4. Feet: (1) Correct: Feet stance is shoulder-width; (2) Incorrect: Feet are placed arbitrarily.\n"
            "5. Hands: (1) Correct: Bow-hand fingers are relaxed; rear hand is anchored at the cheek/jaw area; "
            "the bowstring touches the chin and nose; during aiming the rear hand does not slip/loosen the string; "
            "at release the rear hand extends backward; at release the front/bow hand relaxes and pushes the bow; "
            "(2) Incorrect: Fingers are stiff or clenched; rear hand is floating and not anchored; string does not touch chin and nose; "
            "before/after release the rear hand slowly slides backward; at release the rear hand stays in place or moves outward/forward; "
            "at release the front/bow hand grabs the bow.\n"
        ),
        "zh": (
            "\n任务:\n"
            "你将看到一组按时间顺序排列的关键帧(从早到晚).\n"
            "从关键帧中人物动作随时间的变化中观察并判断下面的动作,如果符合则关键词序列中直接加入对应标准后括号内词库中的正确动作字符串(R),反之加上对应的错误动作字符串(W)(一字不差),并按照评分规则(S)对分数进行初评,再按[生物力学特征信息],[模范]中的信息修正你的评价(如果出现)\n"
            "1.身体是否直立(R:身体纵轴垂直于地面/W:身体重心歪斜)(S:可认为基本垂直的情况躯干部位给4/5分,出现歪斜给2/3分,请在给出的第一帧中判断)\n"
            "2.身体是否转到侧对(R:身体侧向对着射箭方向/W:身体没有转到侧对)(S:一般为R)\n"
            "3.头部是否转动到位(R:头部水平转动/W:头部没有转动到位)(S:转动到位给5分,不到位给3分,极其离谱情况给1/2分,请在给出的第一帧中判断)\n"
            "4.瞄准阶段手臂是否成一条直线(R:前手,前肩,后肩在一条线上,后手小臂与箭在一条线上/W:小臂与箭有角度)(S:请遵循[生物力学特征]中的数值进行评判,max_angle_avg数值超过176.47,评分为5分,175.77-176.47为4分,167.32-175.77给3分,其余情况给2分.4分及以上评价为R,否则为W)\n"
            "5.瞄准阶段后手是否牢牢靠在下巴上(R:后手靠在脸颊下颌位置,弓弦靠在下巴和鼻子上/W:后手悬空没有贴实,弦没有贴上下巴和鼻子)(S:手部评分规则参看[手部评分规则])\n"
            "6.瞄准阶段后手是否有松弦现象(R:瞄准时后手未松滑弦/W:瞄准时后手松滑弦)(S:一般给出R)\n"
            "7.双脚开步与肩同宽(R:双脚开步与肩同宽/W:双脚随意站立)(S:一般为R)\n"
            "8.撒放阶段后手是否向后延展(R:撒放时后手顺势向后延展/W:撒放时后手定在原地或向外,向前)\n"
            "9.撒放时前手是否放松推弓(R:撒放时前手放松推弓/W:撒放时前手握弓)(S:撒放阶段弓体是否与第一帧的弓体相比发生了明显的旋转,应特别注意,并非平移,无旋转即错误)\n"
            "随后按要求输出JSON,给出初步评分与关键词化comment(关键词应精炼,必须使用以下的表达(符合哪项表达就使用哪项表达))\n"
            "评分规则:\n"
            "[手部评分规则]:三条标准:靠下巴(手指必须盖住下巴的一部分,牢牢贴合才算贴合,否则就是没有贴合),撒放向后移动,推弓(撒放后弓必须自然转动)这三条完全满足5分,少一条扣一分,第一条请在给出的第一帧中判断,其余两条请在后面撒放的关键帧序列中比对手部位置差异来判断"
            "注:“瞄准阶段”指箭尚未射出,动作处于稳定拉弓瞄准的阶段,“撒放阶段”指箭射出瞬间及之后的跟随动作."
        ),
    }
    
    system_task_keyframes_compare_map = {
        "en": (
            "\nTask:\n"
            "You will receive two time-ordered keyframe sequences: the student sequence and the template sequence (both early -> late).\n"
            "Do NOT first make an independent judgment and then revise it. Judge directly by comparing the student sequence against the template sequence.\n"
            "For actions 1, 2, 3, 4, 5, and 7, directly compare the FIRST frame of the student sequence with the FIRST frame of the template sequence.\n"
            "For action 6, judge from the student's aiming-stage keyframes whether the rear hand slips or loosens the string before release.\n"
            "For actions 8 and 9, compare the temporal changes in the student sequence against the temporal changes in the template sequence, especially the later release/follow-through frames.\n"
            "Use the template sequence as the full-score reference motion.\n"
            "Then output the required JSON with preliminary scores and a keyword-style comment.\n"
            "Keywords MUST be concise, and MUST be selected from the following keyword library (use whichever items apply).\n"
            "Keyword library: separate keywords with \",\".\n"
            "1. Torso: (1) Correct: Body vertical axis is perpendicular to the ground; body is side-on to the shooting direction; "
            "(2) Incorrect: Body center of mass is tilted/leaning; body is not rotated to a side-on stance.\n"
            "2. Arms: (1) Correct: Front hand, front shoulder, and rear shoulder are aligned on one line; front forearm is internally rotated; "
            "rear forearm is aligned with the arrow; "
            "(2) Incorrect: Front shoulder has an angle and is not opened; forearm is turned outward; forearm is not aligned with the arrow.\n"
            "3. Head: (1) Correct: Head rotates horizontally; (2) Incorrect: Head does not rotate enough.\n"
            "4. Feet: (1) Correct: Feet stance is shoulder-width; (2) Incorrect: Feet are placed arbitrarily.\n"
            "5. Hands: (1) Correct: Bow-hand fingers are relaxed; rear hand is anchored at the cheek/jaw area; "
            "the bowstring touches the chin and nose; during aiming the rear hand does not slip/loosen the string; "
            "at release the rear hand extends backward; at release the front/bow hand relaxes and pushes the bow; "
            "(2) Incorrect: Fingers are stiff or clenched; rear hand is floating and not anchored; string does not touch chin and nose; "
            "before/after release the rear hand slowly slides backward; at release the rear hand stays in place or moves outward/forward; "
            "at release the front/bow hand grabs the bow.\n"
        ),
        "zh": (
            "\n任务:\n"
            "你将看到两组按时间顺序排列的关键帧序列:学生动作关键帧序列和模范动作关键帧序列(均为从早到晚).\n"
            "对两组关键帧进行对比评估,并逐条判断学生动作样本与模范动作对应的动作标准差异(模范动作完全符合标准),如果符合则关键词序列中一字不差地加入对应标准后括号内词库中的正确动作字符串(用R表示),反之加上对应的错误动作字符串(用W表示),并按照评分规则(S)对分数进行评估.\n"
            # "对于标准1,2,3,4,5,7(静态动作)请直接对比两组关键帧序列中的第一帧并判断,"
            # "对于标准8,9,请参考学生序列与模范序列随时间的变化进行判断,尤其关注撒放阶段的关键帧.\n"
            "随后按要求输出JSON,给出初步评分与关键词化comment(关键词应精炼,必须使用以下的表达(符合哪项表达就使用哪项表达))\n"
            "评分规则:\n"
            "1.身体是否直立(R:身体纵轴垂直于地面/W:身体重心歪斜)(S:可认为基本垂直的情况躯干部位给4/5分,出现歪斜给2/3分,请在对比两组关键帧序列第一帧后判断)\n"
            "2.身体是否转到侧对(R:身体侧向对着射箭方向/W:身体没有转到侧对)(S:一般给出R,请在对比两组关键帧序列第一帧后判断)\n"
            "3.头部是否转动到位(R:头部水平转动/W:头部没有转动到位)(S:转动到位给5分,不到位给3分)\n"
            "4.瞄准阶段手臂是否成一条直线(R:前手,前肩,后肩在一条线上,后手小臂与箭在一条线上/W:小臂与箭有角度)(S:请遵循[生物力学特征]中的数值进行评判,max_angle_avg数值超过给4/5分,否则给2/3分)\n"
            "5.瞄准阶段后手是否牢牢靠在下巴上(R:后手靠在脸颊下颌位置,弓弦靠在下巴和鼻子上/W:后手悬空没有贴实,弦没有贴上下巴和鼻子)(S:手部评分规则参看[手部评分规则])\n"
            "6.瞄准阶段后手是否有松弦现象(R:瞄准时后手未松滑弦/W:瞄准时后手松滑弦)(S:一般为R)\n"
            "7.双脚开步与肩同宽(R:双脚开步与肩同宽/W:双脚随意站立)(S:一般为R,极少数情况为W,请在对比两组关键帧序列第一帧后判断)\n"
            "8.撒放阶段后手是否向后延展(R:撒放时后手顺势向后延展/W:撒放时后手定在原地或向外,向前)(S:请参考两组序列后半段的时序变化进行判断)\n"
            "9.撒放阶段前手是否放松推弓(R:撒放时前手放松推弓/W:撒放时前手握弓)(S:前手手指是否张开,让弓体随重力发生明显的旋转,应特别注意,无旋转即错误.请参考两组序列后半段的时序变化进行判断)\n"
            "[手部评分规则]:三条标准:靠下巴(手指必须盖住下巴的一部分,牢牢贴合才算贴合,否则就是没有贴合),撒放向后移动,推弓(撒放后弓必须自然转动)这三条完全满足5分,少一条扣一分,第一条请在对比两组关键帧序列第一帧后判断,其余两条请在两组序列后半段关键帧的时序变化中对比判断\n"
            "注:1.“瞄准阶段”指箭尚未射出,动作处于稳定拉弓瞄准的阶段,“撒放阶段”指箭射出瞬间及之后的跟随动作.2.前手指握弓手,后手指勾弦手.\n"
        ),
    }
    
    # system_metrics_rubric_map = {
    #     "en": (
    #         "\nMetrics rubric (use ONLY for judgment):\n"
    #         "Do NOT include any metric numeric values OR threshold numbers in 'comment'; 'comment' must contain no numbers.\n"
    #         "(1) Hand-Shoulder-Elbow Angle: >=175.67 Excellent (arm=5, MUST include a positive keyword about arm straightness); "
    #         "172.26-175.67 Good (arm=4, MUST include a positive keyword); "
    #         "128.64-172.26 Average (if >=150 then arm=3, otherwise arm=2, MUST include a negative keyword about arm straightness); "
    #         "<128.64 Poor (arm=1-2, MUST include a negative keyword).\n"
    #         "(2) Hand-to-Chin Distance (two-sided target range): 232.57-274.02 Excellent "
    #         "(if neither 'follow-through rear hand extends backward' nor 'bow hand relaxes and pushes' appears, hand=3; "
    #         "if exactly one appears, hand=4; if both appear, hand=5; MUST include a positive keyword about hand anchored at chin; "
    #         "include other hand keywords if applicable); "
    #         "228.58-232.57 OR 274.02-278.02 Average "
    #         "(if neither of the two release actions appears, hand=2; if one or both appear, hand=3; MUST include a negative keyword about hand anchored at chin; "
    #         "include other hand keywords if applicable); "
    #         "otherwise assign hand=0/1 as appropriate and use ONLY negative keywords.\n"
    #         "(3) X-difference of two midpoints (two-sided target range): 61.95-87.05 Excellent "
    #         "(if 68-80 then torso=5, else torso=4; MUST include ALL positive torso keywords); "
    #         "53.54-61.95 OR 87.05-95.46 Good (torso=3; may include negative keywords; MUST NOT include any positive torso keywords); "
    #         "45.99-53.54 OR 95.46-103.00 Average (torso=2; MUST include negative keywords); "
    #         "<45.99 OR >103.00 Poor (torso=0/1; MUST include negative keywords)."
    #     ),
    #     "zh": (
    #         "\n[生物力学特征信息](判断特定部位动作质量的关键依据,请严格根据指标数值和下面的规则修正你的评分和评估文本.下文中要给出的评价请均在词库中提取):\n"
    #         "不要在 comment 中出现任何指标数值或阈值数字;comment 中不得出现数字.\n"
    #         "(1)max_angle_avg:>=175.67 优秀(手臂部位5分,必须给出有关手臂伸直的正面评价(十分接近这个值也认定为优秀));172.26-175.67 良好(4分,同样必须给出正面评价);"
    #         "128.64-172.26 中等(大于150度3分,小于且在区间内给2分,必须给出有关手臂伸直的负面评价);<128.64 较差(1-2,同样必须给出负面评价).\n"
    #     ),
    # }
    
    

    if pipeline == 1:
        system_prompt = system_base_map[lang] + system_task_video_map[lang]

        video_base64 = video_to_base64(video_path)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            },
                        },
                    ],
                }
            ],
            stream=True,
            temperature=temperature,
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
                continue
            if getattr(delta, "content", None):
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "Complete Response" + "=" * 20 + "\n")
                    is_answering = True
                print(delta.content, end='', flush=True)
                answer_content += delta.content
        return _finalize_assessment_payload(answer_content)

    # pipeline 2/3/4: keyframes extraction
    if pose_csv_path:
        _notify_progress("loading_pose_csv", "正在读取CSV姿态序列")
        pose_seq = _load_pose_sequence_from_csv(pose_csv_path)
        data = pose_seq
        normalized_data = pose_seq
    else:
        data, normalized_data = Keypoint_Extract(
            video_path,
            show_draw_selection=show,
            display_wait=1,
            image_width=1920,
            image_height=1080,
            draw_math_feature_points=show,
            progress_callback=progress_callback,
        )
    _notify_progress("selecting_keyframes", "正在筛选关键帧")
    base_keyframes = extract_keyframes_with_ruptures_poseparts_2d(
        normalized_data,
        k=target_k + 3,
        print_all_frame_scores=show,
        print_selection_debug=show,
        candidate_multiplier=2
    ) #这个放大倍数不能设的过大，会导致rupture找不到，然后直接退化
    _notify_progress("refining_keyframes", "正在细化关键帧")
    key_frame_lists = refine_keyframes_with_absdiff(
        video_path=video_path,
        keyframe_result=base_keyframes,
        k=target_k,
    )
    _notify_progress("rendering_keyframes", "正在整理关键帧输入")
    keyframes_dict = extract_show_keyframes_by_index(video_path, key_frame_lists, show=show)
    keyframe_image_items = list(keyframes_dict.get("openai_input_images", []) or [])
    if not keyframe_image_items:
        raise ValueError("No keyframes extracted; pipeline 2/3/4 requires non-empty keyframes.")

    if pipeline == 2:
        system_prompt = system_base_map[lang] + system_task_keyframes_map[lang]
        ordering_note = (
            "Keyframes are ordered from early to late."
            if lang == "en"
            else "关键帧按时间从早到晚排序."
        )
        raw_user_content = [{"type": "input_text", "text": ordering_note}] + keyframe_image_items
        chat_user_content = _to_chat_content(raw_user_content)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_user_content},
            ],
            temperature=temperature,
        )
        return _finalize_assessment_payload(_chat_completion_output_text(resp))

    math_feature = cal_math_features(keypoints_data=data,plot_angle_curve=show)
    metrics = extract_action_features(math_feature)
    system_prompt = system_base_map[lang] + system_task_keyframes_map[lang]     
    # if show == True:
    #     print("max_angle_avg:", metrics.get("max_angle_avg"))
    #     print("min_dist_avg:", metrics.get("min_dist_avg"))
    #     print("min_x_diff_avg:", metrics.get("min_x_diff_avg"))
    # for k in ("max_angle_avg", "min_dist_avg", "min_x_diff_avg"):
    #     if k not in metrics:
    #         raise ValueError(f"Missing metric: {k}")

    ordering_note = (
        "Keyframes are ordered from early to late."
        if lang == "en"
        else "关键帧按时间从早到晚排序."
    )
    metrics_header = "\n\nMetrics:\n" if lang == "en" else "\n\n[生物力学特征]\n"
    metrics_text = (
        metrics_header
        + f"- max_angle_avg: {metrics['max_angle_avg']}\n"
        # + f"- min_dist_avg: {metrics['min_dist_avg']}\n"
        # + f"- min_x_diff_avg: {metrics['min_x_diff_avg']}"
    )
    if pipeline == 3:
        _notify_progress("generating_assessment", "正在生成动作评估")
        p3_system_prompt = system_prompt # + system_metrics_rubric_map[lang]
        user_text = ordering_note + metrics_text
        raw_user_content = [{"type": "input_text", "text": user_text}] + keyframe_image_items
        chat_user_content = _to_chat_content(raw_user_content)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": p3_system_prompt},
                {"role": "user", "content": chat_user_content},
            ],
            temperature=temperature,
        )
        print(resp)
        return _finalize_assessment_payload(_chat_completion_output_text(resp))

    if pipeline == 4:
        _notify_progress("loading_template_keyframes", "正在加载模板关键帧")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_template_dir = os.path.join(repo_root, "output_keyframes")
        template_dir = template_keyframes_dir if template_keyframes_dir else default_template_dir
        if not os.path.isabs(template_dir):
            template_dir = os.path.abspath(os.path.join(repo_root, template_dir))
        template_keyframe_items = _load_openai_images_from_folder(template_dir)

        pipeline4_system_prompt = system_base_map[lang] + system_task_keyframes_compare_map[lang] #+ system_metrics_rubric_map[lang]
        if lang == "en":
            student_note = "Student keyframes are ordered from early to late." #+ metrics_text
            template_note = "Template keyframes are ordered from early to late."
        else:
            student_note = "学生关键帧按时间从早到晚排序." #+ metrics_text
            template_note = "模范关键帧按时间从早到晚排序."

        raw_user_content = (
            [{"type": "input_text", "text": student_note}]
            + keyframe_image_items
            + [{"type": "input_text", "text": template_note}]
            + template_keyframe_items
            + [{"type": "input_text", "text": metrics_text}]
        )
        chat_user_content = _to_chat_content(raw_user_content)
        _notify_progress("generating_assessment", "正在生成动作评估")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": pipeline4_system_prompt},
                {"role": "user", "content": chat_user_content},
            ],
            temperature=temperature,
        )
        return _finalize_assessment_payload(_chat_completion_output_text(resp))

_PART_ALIASES = {
    "head": ["head", "neck", "头", "头部", "颈", "颈部"],
    "hand": ["hand", "hands", "palm", "fingers", "wrist", "手", "手部", "手指", "手腕", "后手", "前手", "握弓手", "推弓手"],
    "arm": ["arm", "arms", "elbow", "shoulder", "forearm", "upperarm", "手臂", "臂", "肩", "肩部", "肘", "肘部", "小臂", "大臂", "共线", "一条线", "直线"],
    "torso": ["torso", "body", "trunk", "core", "posture", "躯干", "身体", "中轴", "重心", "背部", "上身"],
    "foot": ["foot", "feet", "leg", "stance", "脚", "足", "足部", "双脚", "步态", "站姿", "开步", "与肩同宽"],
}

def _normalize_language_qa(language: str) -> str:
    language_norm = str(language or "en").strip().lower()
    if language_norm in ("zh", "zh-cn", "zh_hans", "cn", "chinese", "中文"):
        return "zh"
    if language_norm in ("en", "en-us", "english"):
        return "en"
    return "zh"


def _normalize_text_for_match(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(text or "").strip().lower())


def _normalize_part_name(part: str):
    part_norm = _normalize_text_for_match(part)
    for canon_part, aliases in _PART_ALIASES.items():
        for alias in aliases:
            if part_norm == _normalize_text_for_match(alias):
                return canon_part
    return None


def _sanitize_question_text(question: str) -> str:
    text = str(question or "")
    text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Question is empty after sanitization.")
    return text


def _extract_json_object_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("LLM output is empty.")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return text[i : i + end]
    raise ValueError("No valid JSON object found in LLM output.")


def _chat_completion_output_text(resp) -> str:
    if resp is None or not getattr(resp, "choices", None):
        raise ValueError("Chat completion returned empty response.")
    message = getattr(resp.choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(str(txt))
            else:
                txt = getattr(item, "text", None)
                if txt:
                    parts.append(str(txt))
        return "".join(parts).strip()
    return str(content or "").strip()


def _split_question_with_llm(question_clean: str, language: str, model: str) -> dict:
    lang = _normalize_language_qa(language)
    client = _get_llm_client_by_model(model)
    system_prompt_map = {
       "zh": (
            "你是一个专业的问答问题拆分器，面向“视频中人的射箭动作分析”场景."
            "你的任务是把用户问题字符串整理提炼为后续大模型可以正确理解的两类内容并识别下文提到的两个特征,并返回JSON对象字符串.\n"
            "JSON字段固定为technical_part(内容1), knowledge_part(内容2), technical_scope(特征1), target_parts(特征2)\n"
            "1)technical_part:与视频中人物射箭动作技术相关的提问(字符串)\n"
            "2)knowledge_part:与射箭相关的通用知识内容(如规则、器材、术语解释、训练方法原理、比赛常识等)(字符串).\n"
            "特征1的所有可能结果为:overall/parts/none.判断准则如下:\n"
            "- overall:技术问题主要针对整体动作或完整流程.\n"
            "- parts:技术问题主要针对具体身体部位.\n"
            "- none:问题不涉及视频动作技术.\n"
            "如特征1的结果为parts,则对特征2进行识别,找出技术问题主要针对下列哪个或哪些身体部位:head/hand/arm/torso/foot,结果应该为列表.若特征1为其他结果则返回空列表."
            "根据问题指向选择一个或多个部位；否则 target_parts 为空列表."
        ),
        "en": (
            "You are a QA question splitter. "
            "Output JSON object string only, with keys: technical_part, knowledge_part, technical_scope, target_parts. "
            "technical_scope must be one of overall/parts/none. "
            "target_parts can only include head/hand/arm/torso/foot. "
            "Use empty string/list if a field is not applicable."
        ),
    }
    user_prompt_map = {
        "zh": f"问题:{question_clean}\n",
        "en": f"Question: {question_clean}\nSplit the question and return JSON only.",
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt_map[lang]},
            {"role": "user", "content": user_prompt_map[lang]},
        ],
        temperature=0.2,
    )
    raw_text = _chat_completion_output_text(resp)
    payload = json.loads(_extract_json_object_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Question split output is not a JSON object.")

    technical_part = str(payload.get("technical_part", "") or "").strip()
    knowledge_part = str(payload.get("knowledge_part", "") or "").strip()
    technical_scope = str(payload.get("technical_scope", "none") or "none").strip().lower()
    if technical_scope not in {"overall", "parts", "none"}:
        technical_scope = "none"

    target_parts_raw = payload.get("target_parts", [])
    if not isinstance(target_parts_raw, list):
        target_parts_raw = []
    target_parts = []
    for part in target_parts_raw:
        canon_part = _normalize_part_name(str(part))
        if canon_part and canon_part not in target_parts:
            target_parts.append(canon_part)
    return {
        "technical_part": technical_part,
        "knowledge_part": knowledge_part,
        "technical_scope": technical_scope,
        "target_parts": target_parts,
    }

def _map_keyword_to_part(keyword: str, language: str):
    _ = _normalize_language_qa(language)
    keyword_norm = _normalize_text_for_match(keyword)
    if not keyword_norm:
        return None

    part_order = ["arm", "hand", "torso", "head", "foot"]
    best_part = None
    best_count = 0
    best_first_pos = None

    for part in part_order:
        part_count = 0
        part_first_pos = None
        for alias in _PART_ALIASES.get(part, []):
            alias_norm = _normalize_text_for_match(alias)
            if not alias_norm:
                continue
            hit_count = keyword_norm.count(alias_norm)
            if hit_count <= 0:
                continue
            part_count += hit_count
            hit_pos = keyword_norm.find(alias_norm)
            if hit_pos >= 0 and (part_first_pos is None or hit_pos < part_first_pos):
                part_first_pos = hit_pos

        if part_count <= 0:
            continue

        if best_part is None:
            best_part = part
            best_count = part_count
            best_first_pos = part_first_pos
            continue

        if part_count > best_count:
            best_part = part
            best_count = part_count
            best_first_pos = part_first_pos
            continue

        if part_count == best_count:
            current_pos = part_first_pos if part_first_pos is not None else 10**9
            best_pos = best_first_pos if best_first_pos is not None else 10**9
            if current_pos < best_pos:
                best_part = part
                best_first_pos = part_first_pos

    if best_part:
        return best_part
    return None

def _filter_keywords_for_parts(keywords: list[str], parts: list[str], language: str) -> list[str]:
    target_parts = []
    for part in parts:
        canon_part = _normalize_part_name(part)
        if canon_part and canon_part not in target_parts:
            target_parts.append(canon_part)
    if not target_parts:
        return []

    filtered = []
    seen = set()
    for kw in keywords:
        kw_str = str(kw).strip()
        if not kw_str:
            continue
        part = _map_keyword_to_part(kw_str, language)
        if part in target_parts and kw_str not in seen:
            filtered.append(kw_str)
            seen.add(kw_str)
    return filtered


# def _retrieve_by_keywords(
#     keywords: list[str],
#     session,
#     embedding_model: str,
#     top_k_per_keyword: int,
# ) -> list[str]:
#     from RAG.Knowledge_Database.AI_dbmanager import KnowledgeDB

#     db = KnowledgeDB(session=session)
#     snippets = []
#     seen = set()
#     for kw in keywords:
#         kw_text = str(kw).strip()
#         if not kw_text:
#             continue
#         try:
#             search_results = db.search(
#                 kw_text,
#                 embed_fn=get_embedding,
#                 model_name=embedding_model,
#                 top_k=top_k_per_keyword,
#             )
#         except Exception as e:
#             print(f"[QA] keyword retrieval failed for '{kw_text}': {e}")
#             continue

#         for result in search_results:
#             chunk = result[0] if isinstance(result, tuple) else result
#             chunk_text = getattr(chunk, "text", None)
#             if chunk_text is None and isinstance(chunk, str):
#                 chunk_text = chunk
#             chunk_text = str(chunk_text or "").strip()
#             if not chunk_text or chunk_text in seen:
#                 continue
#             snippets.append(chunk_text)
#             seen.add(chunk_text)
#     return snippets


def _retrieve_knowledge_by_question(
    question_text: str,
    session,
    embedding_model: str,
    title_top_k: int,
    chunk_top_k: int,
) -> list[str]:
    from RAG.Knowledge_Database.AI_dbmanager import KnowledgeDB

    db = KnowledgeDB(session=session)
    try:
        search_results = db.search_knowledge_two_stage(
            str(question_text or "").strip(),
            embed_fn=get_embedding,
            model_name=embedding_model,
            title_top_k=title_top_k,
            chunk_top_k=chunk_top_k,
        )
    except Exception as e:
        print(f"[QA] knowledge retrieval failed: {e}")
        return []

    snippets = []
    seen = set()
    for result in search_results:
        chunk = result[0] if isinstance(result, tuple) else result
        chunk_text = getattr(chunk, "text", None)
        if chunk_text is None and isinstance(chunk, str):
            chunk_text = chunk
        chunk_text = str(chunk_text or "").strip()
        if not chunk_text or chunk_text in seen:
            continue
        snippets.append(chunk_text)
        seen.add(chunk_text)
    return snippets

def _compress_eval_text(
    eval_text: str,
    scope: str,
    target_parts: list[str],
    language: str,
    model: str,
) -> str:
    text = str(eval_text or "").strip()
    if not text:
        return ""

    scope_norm = str(scope or "overall").strip().lower()
    target_part_list = []
    for part in target_parts or []:
        canon_part = _normalize_part_name(part)
        if canon_part and canon_part not in target_part_list:
            target_part_list.append(canon_part)
    target_part_text = ", ".join(target_part_list) if target_part_list else "(none)"

    lang = _normalize_language_qa(language)
    client = _get_llm_client_by_model(model)
    system_prompt_map = {
        "zh": (
            "你是射箭评估文本提取与语言修缮助手.\n"
            "若 scope=parts: 只提取与 target_parts 对应部位相关的评价信息,忽略其他部位.\n"
            "若 scope=overall: 提炼整体动作评价信息.\n"
            "将结果压缩成1-2句动作要点,不要添加原文之外的信息.\n"
            "若 scope=parts 且原文无对应部位信息,返回空字符串.\n"
            "输出仅为纯文本."
        ),
        "en": (
            "You are an archery-evaluation extraction and summarization assistant.\n"
            "If scope=parts: extract ONLY evaluation content related to target_parts and ignore other parts.\n"
            "If scope=overall: summarize the overall action evaluation.\n"
            "Compress into 1-2 short action-focused sentences without adding extra facts.\n"
            "If scope=parts and no relevant information exists, return an empty string.\n"
            "Output plain text only."
        ),
    }
    user_prompt_map = {
        "zh": (
            f"scope: {scope_norm}\n"
            f"target_parts: {target_part_text}\n"
            f"原文: {text}\n"
        ),
        "en": (
            f"scope: {scope_norm}\n"
            f"target_parts: {target_part_text}\n"
            f"source: {text}\n"
            "Extract first, then summarize."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_map[lang]},
                {"role": "user", "content": user_prompt_map[lang]},
            ],
            temperature=0.2,
        )
        summary = _chat_completion_output_text(resp).strip()
        if summary:
            return summary
    except Exception as e:
        print(f"[QA] eval text compression failed: {e}")

    if scope_norm == "parts":
        return ""
    return text[:220]


def _compose_qa_answer(
    question_tech: str,
    tech_summary: str,
    keywords: list[str],
    knowledge_snippets: list[str],
    language: str,
    tech_suggestions = None,
    knowledge_question: str = "",
    model: str = "qwen-plus",
) -> str:
    
    lang = _normalize_language_qa(language)
    tech_summary = str(tech_summary or "").strip()
    keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    tech_suggestions = [str(s).strip() for s in (tech_suggestions or []) if str(s).strip()]
    knowledge_question = str(knowledge_question or "").strip()
    knowledge_snippets = [str(s).strip() for s in (knowledge_snippets or []) if str(s).strip()]

    client = _get_llm_client_by_model(model)
    keyword_text = ", ".join(keywords[:12]) if keywords else "(none)"
    suggestion_seed = ";".join(tech_suggestions[:8]) if tech_suggestions else "(none)"
    snippet_text = "\n".join([f"- {s}" for s in knowledge_snippets[:8]]) if knowledge_snippets else "- (empty)"

    if question_tech != "":
        if knowledge_question != "":
            system_prompt_map = {
                "zh": (
                    "你是射箭问答整合助手.\n"
                    "你需要在一次回答中完成三个模块并最终拼接输出,保证语义一致性,逻辑严谨性和专业性:\n"
                    "1) 技术动作分析: 按照tech_summary给出当前针对部位的技术动作分析(尽可能使用原句进行分析)\n"
                    "2) 改进建议: 依据keywords对question_tech给出改进建议,可使用你的通用知识.\n"
                    "3) 知识问题回答: 仅依据knowledge_snippets中能够回答knowledge_question的片段进行回答"
                    ",不得引入 snippets 外信息,你的任务是将零碎的信息整合成段落.\n"
                    "若 snippets 不足,必须写“根据已检索片段无法确定”.\n"
                    "最后直接输出三行,格式固定为:\n"
                    "对您问题中涉及的技术动作进行分析:...\n针对不足之处的改进建议:...\n关于知识问题的回答:...\n"
                    "不要输出中间推理、不要分点编号、不要额外标题."
                ),
                "en": (
                    "You are an archery QA integrator.\n"
                    "In ONE response, complete three modules and merge them with semantic consistency:\n"
                    "1) Technical Action Analysis: based on tech_summary.\n"
                    "2) Improvement Advice: based on question + keywords; general knowledge is allowed.\n"
                    "3) Knowledge Answer: answer knowledge_question using ONLY knowledge_snippets.\n"
                    "If snippets are insufficient, you must output 'Cannot determine from retrieved snippets.'\n"
                    "Return exactly three lines:\n"
                    "Technical Action Analysis:...\nImprovement Advice:...\nKnowledge Answer:...\n"
                    "No extra headings or reasoning."
                ),
            }
        else:
            system_prompt_map = {
                "zh": (
                    "你是射箭问答整合助手.\n"
                    "你需要在一次回答中完成两个模块并最终拼接输出,保证语义一致性,逻辑严谨性和专业性:\n"
                    "1) 技术动作分析: 按照tech_summary给出当前针对部位的技术动作分析(尽可能使用原句进行分析)\n"
                    "2) 改进建议: 依据 question_tech + keywords 给出改进建议,可使用你的通用知识.\n"
                    ",不得引入snippets外信息,你的任务是将零碎的信息整合成段落.\n"
                    "若 snippets 不足,必须写“根据已检索片段无法确定”.\n"
                    "最后直接输出两行,格式固定为:\n"
                    "对您问题中涉及的技术动作进行分析:...\n针对不足之处的改进建议:...\n"
                    "不要输出中间推理、不要分点编号、不要额外标题."
                ),
                "en": (
                    "You are an archery QA integrator.\n"
                    "In ONE response, complete three modules and merge them with semantic consistency:\n"
                    "1) Technical Action Analysis: based on tech_summary.\n"
                    "2) Improvement Advice: based on question + keywords; general knowledge is allowed.\n"
                    "3) Knowledge Answer: answer knowledge_question using ONLY knowledge_snippets.\n"
                    "If snippets are insufficient, you must output 'Cannot determine from retrieved snippets.'\n"
                    "Return exactly two lines:\n"
                    "Technical Action Analysis:...\nImprovement Advice:...\n"
                    "No extra headings or reasoning."
                ),
            }
    else:
        system_prompt_map = {
                "zh": (
                    "你是射箭问答整合助手.\n"
                    "你需要在一次回答中完成一个模块并最终拼接输出,保证语义一致性,逻辑严谨性和专业性:\n"
                    "1) 知识问题回答: 仅依据knowledge_snippets中能够回答knowledge_question的片段进行回答"
                    ",不得引入snippets外信息,你的任务是将零碎的信息整合成段落.\n"
                    "若 snippets 不足,必须写“根据已检索片段无法确定”.\n"
                    "最后直接输出一行,格式固定为:\n"
                    "关于知识问题的回答:\n"
                    "不要输出中间推理、不要分点编号、不要额外标题."
                ),
                "en": (
                    "You are an archery QA integrator.\n"
                    "In ONE response, complete three modules and merge them with semantic consistency:\n"
                    "1) Technical Action Analysis: based on tech_summary.\n"
                    "2) Improvement Advice: based on question + keywords; general knowledge is allowed.\n"
                    "3) Knowledge Answer: answer knowledge_question using ONLY knowledge_snippets.\n"
                    "If snippets are insufficient, you must output 'Cannot determine from retrieved snippets.'\n"
                    "Return exactly two lines:\n"
                    "Technical Action Analysis:...\nImprovement Advice:...\n"
                    "No extra headings or reasoning."
                ),
            }
    user_prompt_map = {
        "zh": (
            f"question_tech: {question_tech}\n"
            f"tech_summary: {tech_summary if tech_summary else '(none)'}\n"
            f"keywords: {keyword_text}\n"
            # f"tech_suggestions_seed: {suggestion_seed}\n"
            f"knowledge_question: {knowledge_question if knowledge_question else '(none)'}\n"
            f"knowledge_snippets:\n{snippet_text}\n"
            "请按系统要求一次性完成并拼接."
        ),
        "en": (
            f"question_tech: {question_tech}\n"
            f"tech_summary: {tech_summary if tech_summary else '(none)'}\n"
            f"keywords: {keyword_text}\n"
            # f"tech_suggestions_seed: {suggestion_seed}\n"
            f"knowledge_question: {knowledge_question if knowledge_question else '(none)'}\n"
            f"knowledge_snippets:\n{snippet_text}\n"
            "Complete and merge in one pass following the required format."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_map[lang]},
                {"role": "user", "content": user_prompt_map[lang]},
            ],
            temperature=0.2,
        )
        merged = _chat_completion_output_text(resp).strip()
        if merged:
            return merged
    except Exception as e:
        print(f"[QA] one-shot compose failed: {e}")

    if lang == "zh":
        knowledge_fallback = "根据已检索片段无法确定." if knowledge_question else "无知识型问题."
        advice_fallback = (
            ";".join(tech_suggestions[:8])
            if tech_suggestions
            else "当前暂无建议库,可先围绕关键词逐项练习并复查动作稳定性."
        )
        return "\n".join(
            [
                f"技术动作分析:{tech_summary if tech_summary else '暂无可用技术动作摘要.'}",
                f"改进建议:{advice_fallback}",
                f"知识问题回答:{knowledge_fallback}",
            ]
        )

    knowledge_fallback = "Cannot determine from retrieved snippets." if knowledge_question else "No knowledge question."
    advice_fallback = (
        "; ".join(tech_suggestions[:8])
        if tech_suggestions
        else "No suggestion DB is available yet; focus on keyword-related drills and re-check movement stability."
    )
    return "\n".join(
        [
            f"Technical Action Analysis: {tech_summary if tech_summary else 'No technical summary available.'}",
            f"Improvement Advice: {advice_fallback}",
            f"Knowledge Answer: {knowledge_fallback}",
        ]
    )
    
def answer_archery_question(
    keywords: list[str],
    evaluation_text: str,
    question: str,
    tech_session,
    knowledge_session,
    language: str = "zh",
    embedding_model: str = "ali-text-embedding-v4",
    classifier_model: str = "qwen-plus",
    summarizer_model: str = "qwen-plus",
    top_k_knowledge_title: int = 5,
    top_k_knowledge_chunks: int  = 8,
    show : bool = False
) -> str:
    """
    基于“技术评估上下文 + 静态知识库”生成射箭问答回复。

    流程概览:
    1) 规范化语言与输入问题
    2) 用 LLM 拆分问题(技术细节 / 知识问答 / 作用范围)
    3) 按范围汇总评估文本并提取技术动作摘要
    4) 对知识子问题做向量检索补充
    5) 基于关键词生成改进建议 + 基于知识片段回答知识问题
    6) 统一拼接为最终自然语言答案
    """
    try:
        # 统一语言标识并清洗关键词，避免空白词污染检索。
        lang = _normalize_language_qa(language) #return zh/en
        clean_keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
        try:
            # 对用户问题做基础净化(去噪/长度与格式保护)。
            question_clean = _sanitize_question_text(question)#去所有特殊符号和标点，转换成空格
        except Exception:
            return "问题为空或无有效内容,请提供具体问题." if lang == "zh" else "The question is empty after sanitization. Please provide a concrete question."

        # 当问题拆分失败时，退化为“全部按技术问题处理”的保底策略。
        split_fallback = {
            "technical_part": question_clean,
            "knowledge_part": "",
            "technical_scope": "overall",
            "target_parts": [],
        }
        try:
            split_result = _split_question_with_llm(
                question_clean=question_clean,
                language=lang,
                model=classifier_model,
            )
            if show == True:
                print(split_result)
        except Exception as e:
            print(f"[QA] question split failed, use fallback: {e}")
            split_result = split_fallback

        # 解析拆分结果并约束 scope 枚举值，防止异常输出影响后续分支。
        technical_part = str(split_result.get("technical_part", "") or "").strip()
        knowledge_part = str(split_result.get("knowledge_part", "") or "").strip()
        technical_scope = str(split_result.get("technical_scope", "none") or "none").strip().lower()
        if technical_scope not in {"overall", "parts", "none"}:
            technical_scope = "none"

        # 部位名称做标准化并去重，用于“分部位技术检索”。
        target_parts_raw = split_result.get("target_parts", [])
        target_parts = []
        if isinstance(target_parts_raw, list):
            for part in target_parts_raw:
                canon_part = _normalize_part_name(str(part)) #match the correct name for each part
                if canon_part and canon_part not in target_parts:
                    target_parts.append(canon_part)

        tech_summary = ""
        tech_suggestions = None
        advice_keywords = clean_keywords

        if technical_part:
            if technical_scope == "parts" and target_parts:
                # 分部位问题: 优先使用部位过滤后的关键词，并抽取对应部位评估摘要。
                scoped_keywords = _filter_keywords_for_parts(clean_keywords, target_parts, lang)
                if not scoped_keywords:
                    # 过滤后为空时回退到原始关键词，避免检索结果为空。
                    scoped_keywords = clean_keywords
                advice_keywords = scoped_keywords
                tech_summary = _compress_eval_text(
                    eval_text=evaluation_text,
                    scope="parts",
                    target_parts=target_parts,
                    language=lang,
                    model=summarizer_model,
                )
            else:
                # 整体技术问题: 使用整体评估摘要 + 全关键词建议生成。
                tech_summary = _compress_eval_text(
                    eval_text=evaluation_text,
                    scope="overall",
                    target_parts=[],
                    language=lang,
                    model=summarizer_model,
                )
        knowledge_snippets = []
        if knowledge_part:
            # 知识型子问题单独检索，避免与技术建议混淆。
            knowledge_snippets = _retrieve_knowledge_by_question(
                question_text=knowledge_part,
                session=knowledge_session,
                embedding_model=embedding_model,
                title_top_k=top_k_knowledge_title,
                chunk_top_k=top_k_knowledge_chunks,
            )
            if show == True:
                print(knowledge_snippets)

        # 汇总技术摘要、关键词建议、知识片段问答，生成最终文本。
        return _compose_qa_answer(
            question_tech=technical_part,
            tech_summary=tech_summary,
            keywords=advice_keywords,
            tech_suggestions=tech_suggestions,
            knowledge_question=knowledge_part,
            knowledge_snippets=knowledge_snippets,
            language=lang,
            model=summarizer_model,
        )
        
    except Exception as e:
        print(f"[QA] answer_archery_question failed: {e}")
        lang = _normalize_language_qa(language)
        if lang == "zh":
            return "问答处理失败,请稍后重试."
        return "QA processing failed. Please try again later."

def get_response(
    keywords: list[str],
    score_dict: dict,
    retrieved_snippets: list[str],
    keyword_polarity = None,
    language: str = "en",
    model_name: str = "qwen-plus",
) -> str:
    """
    Generate final assessment text from:
    - keyword sequence
    - keyword polarity labels
    - preliminary score dictionary
    - retrieved database snippets
    - routed model name
    """
    client = _get_llm_client_by_model(model_name)

    language_norm = str(language or "en").strip().lower()
    if language_norm in ("zh", "zh-cn", "zh_hans", "cn", "chinese", "中文"):
        lang = "zh"
    elif language_norm in ("en", "en-us", "english"):
        lang = "en"
    else:
        raise ValueError("language must be 'en' or 'zh'")

    if not isinstance(keywords, (list, tuple)):
        raise ValueError("keywords must be a list of strings")
    if not isinstance(score_dict, dict):
        raise ValueError("score_dict must be a dict")
    if not isinstance(retrieved_snippets, (list, tuple)):
        raise ValueError("retrieved_snippets must be a list of strings")
    if keyword_polarity is not None and not isinstance(keyword_polarity, dict):
        raise ValueError("keyword_polarity must be a dict when provided")

    def _normalize_snippet_item(item) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("text", "content", "chunk", "document", "comment"):
                value = item.get(key)
                if value:
                    return str(value).strip()
            return json.dumps(item, ensure_ascii=False)
        return str(item).strip()

    def _normalize_keyword_polarity_payload(raw_payload, keyword_list: list[str]) -> dict[str, list[str]]:
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        normalized = {"positive": [], "negative": [], "unknown": []}
        seen = {key: set() for key in normalized}

        for bucket in normalized:
            raw_items = payload.get(bucket, [])
            if not isinstance(raw_items, (list, tuple)):
                continue
            for item in raw_items:
                text = str(item).strip()
                if text and text not in seen[bucket]:
                    normalized[bucket].append(text)
                    seen[bucket].add(text)

        covered = {
            _normalize_text_for_match(item)
            for bucket in ("positive", "negative", "unknown")
            for item in normalized[bucket]
        }
        for keyword in keyword_list:
            keyword_text = str(keyword).strip()
            keyword_norm = _normalize_text_for_match(keyword_text)
            if keyword_text and keyword_norm not in covered and keyword_text not in seen["unknown"]:
                normalized["unknown"].append(keyword_text)
                seen["unknown"].add(keyword_text)
        return normalized

    def _format_keyword_polarity_text(payload: dict[str, list[str]], lang_code: str) -> str:
        if lang_code == "zh":
            labels = {
                "positive": "正面关键词",
                "negative": "负面关键词",
                "unknown": "未识别极性关键词",
            }
            empty_text = "（空）"
        else:
            labels = {
                "positive": "Positive keywords",
                "negative": "Negative keywords",
                "unknown": "Unknown-polarity keywords",
            }
            empty_text = "(empty)"

        lines = []
        for bucket in ("positive", "negative", "unknown"):
            items = payload.get(bucket, [])
            rendered = ", ".join(items) if items else empty_text
            lines.append(f"- {labels[bucket]}: {rendered}")
        return "\n".join(lines)

    cleaned_keywords = [str(k).strip() for k in keywords if str(k).strip()]
    cleaned_snippets = [_normalize_snippet_item(s) for s in retrieved_snippets]
    cleaned_snippets = [s for s in cleaned_snippets if s]
    score_payload = {
        key: score_dict.get(key)
        for key in ("total_score", "head_score", "hand_score", "torso_score", "foot_score", "arm_score")
    }
    resolved_keyword_polarity = _normalize_keyword_polarity_payload(
        keyword_polarity if keyword_polarity is not None else score_dict.get("keyword_polarity"),
        cleaned_keywords,
    )

    system_instruction_map = {
        "en": (
            "You are a professional archery evaluation coach.\n"
            "Use only user-provided inputs: keyword sequence, keyword polarity labels, score dictionary, and retrieved DB snippets.\n"
            "Do not search or use external knowledge.\n"
            "Output must be plain assessment text only (not JSON, no markdown, no code block).\n"
            "Write a coherent professional summary and improvement suggestions.\n"
            "Use this format: This student's correct actions are: ... This student's incorrect actions are: ...\n"
            "Treat positive keywords as correct actions and negative keywords as incorrect actions. Do not invert their polarity.\n"
            "Prefer retrieved DB snippets when semantically consistent with keywords and scores.\n"
            "Avoid contradictions and keep length around 550 characters."
        ),
        "zh": (
            "你是一名专业的射箭评估教练.\n"
            "只能使用用户提供的输入:关键词序列、关键词极性标注、评估分数字典、数据库检索片段.\n"
            "不得搜索或引用外部知识,你的任务是根据输入内容识别表述的一致性并进行筛选、对输入信息进行拼接、生成流畅的评估文本.\n"
            "输出必须是纯文本评估结果(不要 含有JSON、Markdown 或代码块,也不要含有特殊字符).\n"
            "写出连贯、专业的动作质量评估与改进建议.\n"
            "确保最终评估文本中不存在语义完全相反(自相矛盾)的语句后再进行输出.\n"
            "只做评估工作,不进行额外的解释或说明(根据数据库中语句......可得到等类似的措辞不要出现,也不要出现数据)\n"
            "行文格式必须为:这位同学的正确动作如下: ... 这位同学的错误动作如下: ...\n"
            "务必严格参考提供的关键词极性标注:正面关键词写入正确动作,负面关键词写入错误动作,不要把极性写反.\n"
            "当数据库片段与关键词及分数字典语义一致时优先使用数据库片段表达,如不一致则使用关键词序列中的表达.\n"
            "文本整体长度控制在约 550 字符."
            "数据库检索得到的文本片段并非完全正确,请仔细甄别,只使用与关键词序列语义相同的关键词."
            "注意,射箭人的动作并非一定有缺点,如果关键词序列中无负面评价,“这位同学的错误动作如下：”冒号后不填写任何东西即可.\n"
        ),
    }

    system_instruction = system_instruction_map[lang]
    keyword_text = "\n".join([f"- {k}" for k in cleaned_keywords]) if cleaned_keywords else "- (empty)"
    keyword_polarity_text = _format_keyword_polarity_text(resolved_keyword_polarity, lang)
    snippet_text = "\n".join([f"- {s}" for s in cleaned_snippets]) if cleaned_snippets else "- (empty)"
    score_json = json.dumps(score_payload, ensure_ascii=False, indent=2)
    user_prompt_map = {
        "en": (
            "Keywords:\n"
            f"{keyword_text}\n\n"
            "Keyword polarity labels:\n"
            f"{keyword_polarity_text}\n\n"
            "Score dictionary(Head, hands, feet, arms, torso and overall movement quality assessment):\n"
            f"{score_json}\n\n"
            "Retrieved database snippets:\n"
            f"{snippet_text}\n\n"
            "Return only the final assessment text string."
        ),
        "zh": (
            "关键词序列:\n"
            f"{keyword_text}\n\n"
            "关键词极性标注:\n"
            f"{keyword_polarity_text}\n\n"
            "评估分数字典(头部,手部,足部,手臂,躯干及整体动作质量评分):\n"
            f"{score_json}\n\n"
            "数据库检索片段:\n"
            f"{snippet_text}\n\n"
            "只返回最终评估文本字符串."
        ),
    }

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt_map[lang]},
    ]

    answer_content = ""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=0.3,
        )

        print("\n" + "=" * 20 + " Generating Assessment Text " + "=" * 20)
        is_answering = False

        for chunk in completion:
            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                if not is_answering:
                    print(f"\n[Thinking]: {delta.reasoning_content}", end="", flush=True)

            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "-" * 45)
                    is_answering = True
                print(delta.content, end="", flush=True)
                answer_content += delta.content

        print("\n" + "=" * 45)

    except Exception as e:
        print(f"API call error: {e}")
        return f"Error: {e}"

    return answer_content.strip()

def get_embedding_languagebind_text(texts):
    """Get text embeddings using LanguageBind model"""
    if _LANGUAGEBIND_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "LanguageBind dependencies are unavailable: "
            f"{_LANGUAGEBIND_IMPORT_ERROR}"
        )
    texts = texts if isinstance(texts, list) else [texts]
    device = torch.device("cuda:0")
    
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        'lb203/LanguageBind_Image',
        cache_dir='./cache_dir/tokenizer_cache_dir'
    )
    
    model = LanguageBind(clip_type={'image': 'LanguageBind_Image'},
                         cache_dir='./cache_dir').to(device)
    model.eval()
    
    inputs = {
        'language': to_device(tokenizer(texts, max_length=77, 
                                        padding='max_length', 
                                        truncation=True, 
                                        return_tensors='pt'), 
                                        device)
    }
    
    with torch.no_grad():
        embeddings = model(inputs)['language']
    return embeddings.cpu().numpy()

def get_embedding_languagebind_video(video_path):
    """Get video embeddings using LanguageBind model"""
    if _LANGUAGEBIND_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "LanguageBind dependencies are unavailable: "
            f"{_LANGUAGEBIND_IMPORT_ERROR}"
        )
    video_path = video_path if isinstance(video_path, list) else [video_path]
    clip_type = {
        'video': 'LanguageBind_Video_FT',
    }
    model = LanguageBind(clip_type=clip_type)
    model = model.to(device)
    model.eval()
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    inputs = {
        'video': to_device(modality_transform['video'](video_path), device)
    }
    
    with torch.no_grad():
        embeddings = model(inputs)['video']
    return embeddings.cpu().numpy()

def construct_complex_prompt(scores: Dict, prompt: str, comment_text: List[str]):
    """
    Construct a complex prompt containing structured scores, retrieved comments, and user instructions
    """
    scores_str = json.dumps(scores, indent=2, ensure_ascii=False)

    formatted_comments = "\n".join([f"{i+1}. {text}" for i, text in enumerate(comment_text)])
    print(formatted_comments)

    final_content = f"""Please evaluate the archer's posture based on the following information:
    1. Movement Scores for Each Body Part of Athletes (json):
    {scores_str}
    2. Evaluation keywords and phrases provided by professional coaches retrieved from the database:
    {formatted_comments}
    """
    return final_content


import base64
import os
from typing import Any

import cv2
from openai import OpenAI
import numpy as np

def extract_frames_by_indices(video_path: str, frame_indices):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frames.append(frame)
    finally:
        cap.release()

    return frames

def cv_png_to_base64(img_bgr: np.ndarray, png_compression: int = 3) -> str:
    png_compression = int(max(0, min(9, png_compression)))
    ok, buf = cv2.imencode(".png", img_bgr,
                           [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    if not ok:
        raise RuntimeError("Decode failed")

    return base64.b64encode(buf.tobytes()).decode("utf-8")

def frames_to_base64_list(frames_bgr, png_compression: int = 3):
    return [cv_png_to_base64(frame, png_compression=png_compression) for frame in frames_bgr]

#prompts are here
def build_multiframe_message_png(frames_bgr: object, prompt_text: str, max_frames: int = None,
                                 png_compression: int = 3) -> list[dict[str, str | list[dict[str, str | Any] | dict[str, str]]]]:
    if max_frames is not None:
        frames_bgr = frames_bgr[:max_frames]

    b64_list = frames_to_base64_list(frames_bgr, png_compression=png_compression)
    return build_multiframe_message_png_from_b64(b64_list, prompt_text)

def build_multiframe_message_png_from_b64(frames_b64, system_prompt, prompt_text: str):
    content = [{"role": "system", "content": system_prompt}, {"type": "text", "text": prompt_text}]
    for b64 in frames_b64:
        url = f"data:image/png;base64,{b64}"
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    return [{"role": "user", "content": content}]

def run_llm(frames,sys_prompt,usr_prompt,model_name="meta-llama/llama-3.2-11b-vision-instruct"):

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # frames = [...]  # list[np.ndarray], BGR
    messages = build_multiframe_message_png(
        frames_bgr=frames,
        system_prompt=sys_prompt,
        prompt_text=usr_prompt,
        max_frames=16,
        png_compression=3
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    print(completion.choices[0].message.content)
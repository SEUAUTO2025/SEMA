import base64
import os
import json
import math
import random
import re
import sys
from datetime import datetime
from typing import Any, Callable, Optional

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
def build_multiframe_message_png(
    frames_bgr: object,
    prompt_text: str,
    system_prompt: Optional[str] = None,
    max_frames: int = None,
    png_compression: int = 3,
) -> list[dict[str, Any]]:
    if max_frames is not None:
        frames_bgr = frames_bgr[:max_frames]

    b64_list = frames_to_base64_list(frames_bgr, png_compression=png_compression)
    return build_multiframe_message_png_from_b64(
        frames_b64=b64_list,
        prompt_text=prompt_text,
        system_prompt=system_prompt,
    )

def build_multiframe_message_png_from_b64(
    frames_b64,
    prompt_text: str,
    system_prompt: Optional[str] = None,
):
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = [{"type": "text", "text": prompt_text}]
    for b64 in frames_b64:
        url = f"data:image/png;base64,{b64}"
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    messages.append({"role": "user", "content": content})
    return messages

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

    content = completion.choices[0].message.content
    print(content)
    return content


def evaluate_comment(ground_truth: str, generated_comment: str, model_name: str) -> dict:
    """
    Evaluate generated comment using specified model.
    Returns: {"Accuracy": score, "Professionalism": score, "Practicality": score}
    """
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    prompt = f"""You are a professional archery coach evaluation expert. Here are two archery posture evaluation texts:
    [Teacher's Standard Evaluation (Full Score Baseline)]
    {ground_truth}

    [Generated Evaluation to be Assessed]
    {generated_comment}

    Please score the [Generated Evaluation] on the following three dimensions (5-point Likert scale, 1 lowest, 5 highest, decimals allowed):

    1. Accuracy: Are the technical points and action descriptions accurate? How consistent with the standard evaluation?
    2. Professionalism: Are the terms and expressions professional? Does it reflect archery domain expertise?
    3. Practicality: How valuable is the guidance for students? Are improvement suggestions specific and actionable?

    Scoring standards:
    - 5: Excellent, fully meets standard evaluation level
    - 4: Good, mostly meets standard
    - 3: Average, basically meets standard but with obvious deficiencies
    - 2: Poor, significant gap from standard
    - 1: Very poor, seriously deviates from standard

    Output only JSON:
    {{"Accuracy": score, "Professionalism": score, "Practicality": score}}
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional archery coach evaluation expert, responsible for objective and fair scoring of archery posture evaluation texts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        result_text = (response.choices[0].message.content or "").strip()
        json_match = re.search(r"\{[^}]+\}", result_text)
        if json_match:
            scores = json.loads(json_match.group())
            return {
                "Accuracy": float(scores.get("Accuracy", 0)),
                "Professionalism": float(scores.get("Professionalism", 0)),
                "Practicality": float(scores.get("Practicality", 0)),
            }

        print(f"Unable to parse evaluation result: {result_text}")
        return {"Accuracy": 0.0, "Professionalism": 0.0, "Practicality": 0.0}
    except Exception as e:
        print(f"Evaluation API error: {e}")
        return {"Accuracy": 0.0, "Professionalism": 0.0, "Practicality": 0.0}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_eval_db_api():
    try:
        from evaluation.eval_db.eval_db_manager import get_evaluations, list_samples
        return get_evaluations, list_samples
    except ImportError:
        project_root = _project_root()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from evaluation.eval_db.eval_db_manager import get_evaluations, list_samples
        return get_evaluations, list_samples


def _load_dataset_api():
    try:
        from Tools.Exe_dataset.dataset_exe import load_single_csv_with_multipart_labels
        return load_single_csv_with_multipart_labels
    except ImportError:
        project_root = _project_root()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from Tools.Exe_dataset.dataset_exe import load_single_csv_with_multipart_labels
        return load_single_csv_with_multipart_labels


def get_matching_text_from_txt(sample_name: str, txt_dir: Optional[str] = None) -> Optional[str]:
    """
    sample_name -> file stem -> dataset/txt/{stem}.txt
    """
    if txt_dir is None:
        txt_dir = os.path.join(_project_root(), "dataset", "txt")

    file_name = os.path.splitext(os.path.basename(sample_name))[0]
    target_txt_path = os.path.join(txt_dir, f"{file_name}.txt")
    if not os.path.exists(target_txt_path):
        return None

    try:
        with open(target_txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        clean_content = content.replace("\n", "").replace("\r", "")
        return clean_content if clean_content else None
    except Exception:
        return None


def get_total_score_from_dataset_csv(sample_name: str, csv_dir: Optional[str] = None) -> tuple[Optional[float], Optional[str]]:
    """
    Read label_total from dataset/csv/{sample_stem}.csv by load_single_csv_with_multipart_labels.
    """
    if csv_dir is None:
        csv_dir = os.path.join(_project_root(), "dataset", "csv")

    file_name = os.path.splitext(os.path.basename(sample_name))[0]
    csv_path = os.path.join(csv_dir, f"{file_name}.csv")
    if not os.path.exists(csv_path):
        return None, csv_path

    try:
        loader = _load_dataset_api()
        _, _, label_total = loader(csv_path, max_frames=124)
        return float(label_total), csv_path
    except Exception:
        return None, csv_path


def score_to_level(total_score: Optional[float]) -> str:
    """
    >23: 优秀, [21,23]: 良好, [18,21): 中等, [16,18): 及格, <16: 不及格
    """
    if total_score is None:
        return "未知"
    if total_score > 23:
        return "优秀"
    if 21 <= total_score <= 23:
        return "良好"
    if 18 <= total_score < 21:
        return "中等"
    if 16 <= total_score < 18:
        return "及格"
    return "不及格"


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _sas_descriptive_stats(values: list[float]) -> dict[str, Any]:
    import pandas as pd

    s = pd.Series(values, dtype="float64")
    n_total = int(s.shape[0])
    s_valid = s.dropna()
    n = int(s_valid.shape[0])
    nmiss = int(n_total - n)

    if n == 0:
        return {
            "N": 0,
            "NMISS": nmiss,
            "Mean": float("nan"),
            "Std": float("nan"),
            "Variance": float("nan"),
            "Skewness": float("nan"),
            "Kurtosis": float("nan"),
            "CV": float("nan"),
            "StdErr": float("nan"),
            "CI95_L": float("nan"),
            "CI95_U": float("nan"),
            "Min": float("nan"),
            "Q1": float("nan"),
            "Median": float("nan"),
            "Q3": float("nan"),
            "Max": float("nan"),
            "Range": float("nan"),
            "IQR": float("nan"),
            "Sum": float("nan"),
            "USS": float("nan"),
            "CSS": float("nan"),
        }

    mean = float(s_valid.mean())
    std = float(s_valid.std(ddof=1)) if n > 1 else 0.0
    var = float(s_valid.var(ddof=1)) if n > 1 else 0.0
    stderr = std / math.sqrt(n) if n > 0 else float("nan")
    ci95_l = mean - 1.96 * stderr if n > 1 else mean
    ci95_u = mean + 1.96 * stderr if n > 1 else mean

    min_v = float(s_valid.min())
    q1 = float(s_valid.quantile(0.25))
    med = float(s_valid.quantile(0.5))
    q3 = float(s_valid.quantile(0.75))
    max_v = float(s_valid.max())
    range_v = max_v - min_v
    iqr_v = q3 - q1

    sum_v = float(s_valid.sum())
    uss = float((s_valid ** 2).sum())
    css = float(((s_valid - mean) ** 2).sum())

    skew_v = float(s_valid.skew()) if n > 2 else 0.0
    kurt_v = float(s_valid.kurt()) if n > 3 else 0.0
    cv_v = float(std / mean * 100.0) if mean != 0 else float("nan")

    return {
        "N": n,
        "NMISS": nmiss,
        "Mean": mean,
        "Std": std,
        "Variance": var,
        "Skewness": skew_v,
        "Kurtosis": kurt_v,
        "CV": cv_v,
        "StdErr": stderr,
        "CI95_L": ci95_l,
        "CI95_U": ci95_u,
        "Min": min_v,
        "Q1": q1,
        "Median": med,
        "Q3": q3,
        "Max": max_v,
        "Range": range_v,
        "IQR": iqr_v,
        "Sum": sum_v,
        "USS": uss,
        "CSS": css,
    }


def _aqa_metrics_from_pairs(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    import pandas as pd

    if gt.size == 0 or pred.size == 0:
        return {
            "n": 0,
            "spearman_rho": float("nan"),
            "pearson_r": float("nan"),
            "kendall_tau": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "relative_l2": float("nan"),
        }

    s_gt = pd.Series(gt, dtype="float64")
    s_pred = pd.Series(pred, dtype="float64")

    valid = ~(s_gt.isna() | s_pred.isna())
    s_gt = s_gt[valid]
    s_pred = s_pred[valid]

    if s_gt.shape[0] == 0:
        return {
            "n": 0,
            "spearman_rho": float("nan"),
            "pearson_r": float("nan"),
            "kendall_tau": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "relative_l2": float("nan"),
        }

    diff = s_pred - s_gt
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))

    gt_np = s_gt.to_numpy(dtype=np.float64)
    pred_np = s_pred.to_numpy(dtype=np.float64)
    denom = float(np.linalg.norm(gt_np))
    rl2 = float(np.linalg.norm(pred_np - gt_np) / (denom + 1e-12))

    return {
        "n": int(s_gt.shape[0]),
        "spearman_rho": float(s_gt.corr(s_pred, method="spearman")),
        "pearson_r": float(s_gt.corr(s_pred, method="pearson")),
        "kendall_tau": float(s_gt.corr(s_pred, method="kendall")),
        "mae": mae,
        "rmse": rmse,
        "relative_l2": rl2,
    }


def _set_plot_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.facecolor": "#FBFCFF",
            "figure.facecolor": "#FFFFFF",
        }
    )


def _plot_model_mean_bar(by_model_df, out_path: str):
    import matplotlib.pyplot as plt

    if by_model_df.empty:
        return

    _set_plot_style()
    palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    ]

    df = by_model_df.sort_values("mean_avg_score", ascending=False).reset_index(drop=True)
    xs = np.arange(df.shape[0])
    ys = df["mean_avg_score"].to_numpy()
    yerr_low = ys - df["ci95_l"].to_numpy()
    yerr_up = df["ci95_u"].to_numpy() - ys

    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(xs, ys, color=[palette[i % len(palette)] for i in range(len(xs))], alpha=0.92)
    ax.errorbar(xs, ys, yerr=[yerr_low, yerr_up], fmt="none", ecolor="#1f1f1f", capsize=3, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["generated_model"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Mean LLM Score (1-5)")
    ax.set_title("Model-wise Mean Score with 95% CI")
    ax.set_ylim(max(0, float(np.nanmin(ys)) - 0.2), min(5.0, float(np.nanmax(ys)) + 0.4))

    for rect, val in zip(bars, ys):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_level_trend(model_level_df, out_path: str):
    import matplotlib.pyplot as plt

    if model_level_df.empty:
        return

    _set_plot_style()
    level_order = ["不及格", "及格", "中等", "良好", "优秀"]
    x_map = {name: i for i, name in enumerate(level_order)}

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

    for idx, (model_name, part) in enumerate(model_level_df.groupby("generated_model")):
        part = part.copy()
        part = part[part["athlete_level"].isin(level_order)]
        if part.empty:
            continue
        part["x"] = part["athlete_level"].map(x_map)
        part = part.sort_values("x")

        x = part["x"].to_numpy()
        y = part["mean_avg_score"].to_numpy()
        color = palette[idx % len(palette)]

        ax.plot(x, y, marker="o", lw=2.0, color=color, label=model_name)
        ax.fill_between(x, part["ci95_l"].to_numpy(), part["ci95_u"].to_numpy(), color=color, alpha=0.12)

    ax.set_xticks(np.arange(len(level_order)))
    ax.set_xticklabels(level_order)
    ax.set_ylabel("Mean LLM Score (1-5)")
    ax.set_title("Score Trend across Athlete Levels")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_model_level_heatmap(model_level_df, out_path: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    if model_level_df.empty:
        return

    _set_plot_style()
    level_order = ["不及格", "及格", "中等", "良好", "优秀"]
    pivot = (
        model_level_df.pivot_table(
            index="generated_model",
            columns="athlete_level",
            values="mean_avg_score",
            aggfunc="mean",
        )
        .reindex(columns=level_order)
        .sort_index()
    )
    if pivot.empty:
        return

    mat = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    im = ax.imshow(mat, cmap="cividis", aspect="auto", vmin=1.0, vmax=5.0)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist())
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Athlete Level")
    ax.set_ylabel("Generated Model")
    ax.set_title("Model-Level Mean Score Heatmap")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if pd.isna(v):
                txt = "-"
            else:
                txt = f"{v:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if (not pd.isna(v) and v < 3.2) else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Mean LLM Score (1-5)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_total_vs_quality_scatter(records_df, out_path: str):
    import matplotlib.pyplot as plt

    if records_df.empty:
        return

    df = records_df.dropna(subset=["total_score", "avg_score"]).copy()
    if df.empty:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    ]

    for idx, (mname, part) in enumerate(df.groupby("generated_model")):
        color = palette[idx % len(palette)]
        ax.scatter(
            part["total_score"].to_numpy(),
            part["avg_score"].to_numpy(),
            s=26,
            alpha=0.72,
            color=color,
            edgecolors="none",
            label=mname,
        )

    x = df["total_score"].to_numpy(dtype=float)
    y = df["avg_score"].to_numpy(dtype=float)
    if x.shape[0] > 1 and np.nanstd(x) > 1e-9:
        k, b = np.polyfit(x, y, 1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        y_line = k * x_line + b
        ax.plot(x_line, y_line, color="#111111", lw=2.0, linestyle="--", label=f"Overall Trend (k={k:.3f})")

    ax.set_xlabel("Ground-truth Total Score")
    ax.set_ylabel("Generated Comment Quality Score (LLM 1-5)")
    ax.set_title("Quality vs Athlete Real Score")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _make_run_dir(base_results_dir: str, prefix: str = "batch_eval") -> str:
    os.makedirs(base_results_dir, exist_ok=True)
    run_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(base_results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def batch_evaluate_from_eval_db(
    eval_model_name: str,
    ground_truth_map: Optional[dict[str, str]] = None,
    ground_truth_getter: Optional[Callable[[str], Optional[str]]] = None,
    target_models: Optional[list[str]] = None,
    sample_count: Optional[int] = None,
    sample_order: str = "sequential",
    model_count_per_sample: Optional[int] = None,
    model_order: str = "sequential",
    sample_offset: int = 0,
    seed: Optional[int] = None,
    txt_dir: Optional[str] = None,
    csv_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Batch-evaluate generated comments stored in eval DB.
    Ground truth text is loaded by sample_name -> dataset/txt/{sample_name}.txt by default.
    Ground truth total score is loaded from dataset/csv/{sample_name}.csv.

    Returns:
        list[dict]: per-record score rows (also saved to evaluation/results/<run>/records.csv).
    """
    import pandas as pd

    get_evaluations, list_samples = _load_eval_db_api()
    rng = random.Random(seed) if seed is not None else random.Random()

    all_samples = list_samples(limit=None, offset=0)
    sample_names = [str(x.get("sample_name", "")).strip() for x in all_samples if str(x.get("sample_name", "")).strip()]

    if sample_order == "random":
        rng.shuffle(sample_names)
    elif sample_order != "sequential":
        raise ValueError("sample_order must be 'sequential' or 'random'")

    if sample_offset > 0:
        sample_names = sample_names[sample_offset:]
    if sample_count is not None:
        sample_names = sample_names[: int(sample_count)]

    if output_dir is None:
        output_dir = os.path.join(_project_root(), "evaluation", "results")
    run_dir = _make_run_dir(output_dir, prefix="batch_eval")

    records: list[dict[str, Any]] = []
    missing_samples: list[dict[str, Any]] = []

    for sample_name in sample_names:
        gt_text = None
        if ground_truth_map is not None:
            gt_text = ground_truth_map.get(sample_name)
        if gt_text is None and ground_truth_getter is not None:
            gt_text = ground_truth_getter(sample_name)
        if gt_text is None:
            gt_text = get_matching_text_from_txt(sample_name, txt_dir=txt_dir)

        total_score, csv_path = get_total_score_from_dataset_csv(sample_name, csv_dir=csv_dir)
        athlete_level = score_to_level(total_score)

        eval_rows = get_evaluations(sample_name)
        if target_models:
            target_set = set(target_models)
            eval_rows = [x for x in eval_rows if x.get("model_name") in target_set]

        if model_order == "random":
            rng.shuffle(eval_rows)
        elif model_order != "sequential":
            raise ValueError("model_order must be 'sequential' or 'random'")

        if model_count_per_sample is not None:
            eval_rows = eval_rows[: int(model_count_per_sample)]

        if not gt_text:
            missing_samples.append(
                {
                    "sample_name": sample_name,
                    "reason": "missing_ground_truth_txt",
                    "csv_path": csv_path,
                    "athlete_level": athlete_level,
                    "num_candidate_evals": len(eval_rows),
                }
            )
            continue

        if not eval_rows:
            missing_samples.append(
                {
                    "sample_name": sample_name,
                    "reason": "missing_generated_eval_in_db",
                    "csv_path": csv_path,
                    "athlete_level": athlete_level,
                    "num_candidate_evals": 0,
                }
            )
            continue

        for item in eval_rows:
            generated_model = str(item.get("model_name", "")).strip()
            generated_text = item.get("eval_text") or ""
            if not generated_model or not generated_text.strip():
                continue

            scores = evaluate_comment(
                ground_truth=gt_text,
                generated_comment=generated_text,
                model_name=eval_model_name,
            )
            acc = _to_float(scores.get("Accuracy"))
            pro = _to_float(scores.get("Professionalism"))
            pra = _to_float(scores.get("Practicality"))
            avg_score = float(np.nanmean([acc, pro, pra]))

            records.append(
                {
                    "sample_name": sample_name,
                    "generated_model": generated_model,
                    "eval_model_name": eval_model_name,
                    "slot_index": item.get("slot_index"),
                    "total_score": total_score,
                    "athlete_level": athlete_level,
                    "Accuracy": acc,
                    "Professionalism": pro,
                    "Practicality": pra,
                    "avg_score": avg_score,
                    "gt_text_len": len(gt_text),
                    "generated_text_len": len(generated_text),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )

    records_df = pd.DataFrame(records)
    missing_df = pd.DataFrame(missing_samples)

    records_csv = os.path.join(run_dir, "records.csv")
    missing_csv = os.path.join(run_dir, "missing_samples.csv")
    summary_json = os.path.join(run_dir, "summary.json")
    by_model_csv = os.path.join(run_dir, "by_model_stats.csv")
    by_level_csv = os.path.join(run_dir, "by_level_stats.csv")
    model_level_csv = os.path.join(run_dir, "by_model_level_stats.csv")
    aqa_csv = os.path.join(run_dir, "aqa_metrics.csv")
    report_md = os.path.join(run_dir, "report.md")
    manifest_json = os.path.join(run_dir, "manifest.json")

    if records_df.empty:
        records_df.to_csv(records_csv, index=False, encoding="utf-8-sig")
        missing_df.to_csv(missing_csv, index=False, encoding="utf-8-sig")
        summary = {
            "run_dir": run_dir,
            "n_records": 0,
            "n_missing_samples": int(missing_df.shape[0]),
            "message": "No valid records generated.",
        }
        _write_json(summary_json, summary)
        _write_json(
            manifest_json,
            {
                "run_dir": run_dir,
                "records_csv": records_csv,
                "missing_csv": missing_csv,
                "summary_json": summary_json,
                "figures": [],
            },
        )
        return records

    def _grp_stats(df):
        vals = df["avg_score"].astype(float).to_list()
        d = _sas_descriptive_stats(vals)
        return pd.Series(
            {
                "n": d["N"],
                "mean_avg_score": d["Mean"],
                "std_avg_score": d["Std"],
                "ci95_l": d["CI95_L"],
                "ci95_u": d["CI95_U"],
                "median_avg_score": d["Median"],
                "q1_avg_score": d["Q1"],
                "q3_avg_score": d["Q3"],
                "min_avg_score": d["Min"],
                "max_avg_score": d["Max"],
                "mean_accuracy": float(df["Accuracy"].astype(float).mean()),
                "mean_professionalism": float(df["Professionalism"].astype(float).mean()),
                "mean_practicality": float(df["Practicality"].astype(float).mean()),
                "mean_total_score": float(df["total_score"].astype(float).mean()),
            }
        )

    by_model_df = records_df.groupby("generated_model", dropna=False).apply(_grp_stats).reset_index()
    by_level_df = records_df.groupby("athlete_level", dropna=False).apply(_grp_stats).reset_index()
    by_model_level_df = records_df.groupby(["generated_model", "athlete_level"], dropna=False).apply(_grp_stats).reset_index()

    aqa_rows: list[dict[str, Any]] = []
    for model_name, part in records_df.groupby("generated_model"):
        gt = part["total_score"].to_numpy(dtype=float)
        pred = part["avg_score"].to_numpy(dtype=float)
        met = _aqa_metrics_from_pairs(gt, pred)

        slope = float("nan")
        intercept = float("nan")
        if gt.shape[0] > 1 and np.nanstd(gt) > 1e-9:
            slope, intercept = np.polyfit(gt, pred, 1)
            slope = float(slope)
            intercept = float(intercept)

        met.update(
            {
                "generated_model": model_name,
                "trend_slope": slope,
                "trend_intercept": intercept,
            }
        )
        aqa_rows.append(met)

    overall_desc = _sas_descriptive_stats(records_df["avg_score"].astype(float).to_list())
    overall_dim_stats = {
        "Accuracy": _sas_descriptive_stats(records_df["Accuracy"].astype(float).to_list()),
        "Professionalism": _sas_descriptive_stats(records_df["Professionalism"].astype(float).to_list()),
        "Practicality": _sas_descriptive_stats(records_df["Practicality"].astype(float).to_list()),
    }
    overall_aqa = _aqa_metrics_from_pairs(
        records_df["total_score"].to_numpy(dtype=float),
        records_df["avg_score"].to_numpy(dtype=float),
    )

    level_order = ["不及格", "及格", "中等", "良好", "优秀", "未知"]
    by_level_df["level_rank"] = by_level_df["athlete_level"].map({n: i for i, n in enumerate(level_order)})
    by_level_df = by_level_df.sort_values("level_rank").drop(columns=["level_rank"])
    by_model_level_df["level_rank"] = by_model_level_df["athlete_level"].map({n: i for i, n in enumerate(level_order)})
    by_model_level_df = by_model_level_df.sort_values(["generated_model", "level_rank"]).drop(columns=["level_rank"])

    records_df.to_csv(records_csv, index=False, encoding="utf-8-sig")
    missing_df.to_csv(missing_csv, index=False, encoding="utf-8-sig")
    by_model_df.to_csv(by_model_csv, index=False, encoding="utf-8-sig")
    by_level_df.to_csv(by_level_csv, index=False, encoding="utf-8-sig")
    by_model_level_df.to_csv(model_level_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(aqa_rows).to_csv(aqa_csv, index=False, encoding="utf-8-sig")

    fig_paths: list[str] = []
    try:
        fig1 = os.path.join(run_dir, "fig_model_mean_ci.png")
        _plot_model_mean_bar(by_model_df, fig1)
        if os.path.exists(fig1):
            fig_paths.append(fig1)

        fig2 = os.path.join(run_dir, "fig_level_trend.png")
        _plot_level_trend(by_model_level_df, fig2)
        if os.path.exists(fig2):
            fig_paths.append(fig2)

        fig3 = os.path.join(run_dir, "fig_model_level_heatmap.png")
        _plot_model_level_heatmap(by_model_level_df, fig3)
        if os.path.exists(fig3):
            fig_paths.append(fig3)

        fig4 = os.path.join(run_dir, "fig_total_vs_quality_scatter.png")
        _plot_total_vs_quality_scatter(records_df, fig4)
        if os.path.exists(fig4):
            fig_paths.append(fig4)
    except Exception as plot_err:
        print(f"Plot generation failed: {plot_err}")

    summary = {
        "run_dir": run_dir,
        "eval_model_name": eval_model_name,
        "num_samples_requested": len(sample_names),
        "num_records": int(records_df.shape[0]),
        "num_missing_samples": int(missing_df.shape[0]),
        "overall_avg_score_stats": overall_desc,
        "overall_dim_stats": overall_dim_stats,
        "overall_aqa_metrics": overall_aqa,
    }
    _write_json(summary_json, summary)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Batch Evaluation Report\n\n")
        f.write(f"- Run Dir: `{run_dir}`\n")
        f.write(f"- Eval Model: `{eval_model_name}`\n")
        f.write(f"- Samples Processed: `{len(sample_names)}`\n")
        f.write(f"- Scored Records: `{records_df.shape[0]}`\n")
        f.write(f"- Missing Samples: `{missing_df.shape[0]}`\n\n")
        f.write("## Overall Quality (avg_score)\n\n")
        f.write(f"- Mean: `{overall_desc['Mean']:.4f}`\n")
        f.write(f"- Std: `{overall_desc['Std']:.4f}`\n")
        f.write(f"- 95% CI: `[{overall_desc['CI95_L']:.4f}, {overall_desc['CI95_U']:.4f}]`\n")
        f.write(f"- Median: `{overall_desc['Median']:.4f}`\n\n")
        f.write("## Overall AQA-style Metrics (GT total_score vs avg_score)\n\n")
        f.write(f"- Spearman rho: `{overall_aqa['spearman_rho']:.4f}`\n")
        f.write(f"- Pearson r: `{overall_aqa['pearson_r']:.4f}`\n")
        f.write(f"- Kendall tau: `{overall_aqa['kendall_tau']:.4f}`\n")
        f.write(f"- MAE: `{overall_aqa['mae']:.4f}`\n")
        f.write(f"- RMSE: `{overall_aqa['rmse']:.4f}`\n")
        f.write(f"- Relative L2: `{overall_aqa['relative_l2']:.4f}`\n\n")
        f.write("## Output Files\n\n")
        f.write(f"- `records.csv`\n")
        f.write(f"- `missing_samples.csv`\n")
        f.write(f"- `summary.json`\n")
        f.write(f"- `by_model_stats.csv`\n")
        f.write(f"- `by_level_stats.csv`\n")
        f.write(f"- `by_model_level_stats.csv`\n")
        f.write(f"- `aqa_metrics.csv`\n")
        for fp in fig_paths:
            f.write(f"- `{os.path.basename(fp)}`\n")

    _write_json(
        manifest_json,
        {
            "run_dir": run_dir,
            "records_csv": records_csv,
            "missing_csv": missing_csv,
            "summary_json": summary_json,
            "by_model_csv": by_model_csv,
            "by_level_csv": by_level_csv,
            "by_model_level_csv": model_level_csv,
            "aqa_metrics_csv": aqa_csv,
            "report_md": report_md,
            "figures": fig_paths,
        },
    )

    return records

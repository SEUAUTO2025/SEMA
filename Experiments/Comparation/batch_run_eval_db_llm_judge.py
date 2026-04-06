"""
Batch LLM-judge evaluation for assessment texts stored in a wide CSV file.

The script reads `evaluation_texts.csv`, loads the ground-truth txt matched by
`sample_name`, and scores each generated evaluation-text column using the same
judge prompt as the previous DB-based workflow. The output is a new wide CSV
with judge-score columns appended after the original columns.
"""
import argparse
import csv
import json
import os
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DEFAULT_INPUT_CSV = os.path.join(SCRIPT_DIR, "evaluation_texts.csv")
DEFAULT_SAMPLE_NAME_COLUMN = "sample_name"
DEFAULT_DATASET_ROOT = "dataset"
DEFAULT_TXT_SUBDIR = "txt"
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "evaluation", "results")
DEFAULT_OUTPUT_CSV_NAME = "evaluation_texts_scored.csv"
DEFAULT_JUDGE_MODELS = ["qwen3.5-plus", "deepseek-v3.2"]
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "ALI_API_KEY"
DEFAULT_EXCLUDE_COLUMNS = ["merged_model_count", "conflicting_duplicate_models"]
DEFAULT_PREDICTION_BATCH_SIZE = 10
SCORE_SUFFIXES = ("_Accuracy", "_Professionalism", "_Practicality", "_Average")

def _resolve_project_path(path_str: Union[str, Path]) -> Path:
    path_obj = Path(path_str) if path_str else Path()
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(PROJECT_ROOT_DIR) / path_obj).resolve()


def _clean_text(text: Any) -> str:
    return str(text or "").replace("\r", " ").replace("\n", " ").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_run_dir(output_root: Union[str, Path], run_name: Optional[str]) -> Path:
    output_root_path = _resolve_project_path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    suffix = str(run_name or "").strip()
    if suffix:
        run_dir_name = suffix
    else:
        run_dir_name = "csv_llm_judge_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = output_root_path / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(str(csv_path), "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _is_score_column(column_name: str) -> bool:
    text = str(column_name or "")
    return any(text.endswith(suffix) for suffix in SCORE_SUFFIXES)


def _infer_prediction_columns(
    fieldnames: List[str],
    sample_name_column: str,
    explicit_prediction_columns: Optional[List[str]],
    exclude_columns: List[str],
) -> List[str]:
    if explicit_prediction_columns:
        return [str(col) for col in explicit_prediction_columns]

    excluded = set([sample_name_column])
    for column_name in exclude_columns:
        excluded.add(str(column_name))

    prediction_columns = []
    for column_name in fieldnames:
        if column_name in excluded:
            continue
        if _is_score_column(column_name):
            continue
        prediction_columns.append(str(column_name))
    return prediction_columns


def _normalize_scores(scores: Dict[str, Any]) -> Dict[str, float]:
    return {
        "Accuracy": _safe_float(scores.get("Accuracy"), default=0.0),
        "Professionalism": _safe_float(scores.get("Professionalism"), default=0.0),
        "Practicality": _safe_float(scores.get("Practicality"), default=0.0),
    }


def _make_openai_client(
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> OpenAI:
    return OpenAI(
        api_key=os.getenv(api_key_env),
        base_url=base_url,
    )


def _chunk_prediction_items(
    prediction_items: List[Dict[str, str]],
    batch_size: int,
) -> List[List[Dict[str, str]]]:
    if len(prediction_items) == 0:
        return []
    if batch_size <= 0 or batch_size >= len(prediction_items):
        return [list(prediction_items)]
    chunks = []
    for start_idx in range(0, len(prediction_items), batch_size):
        chunks.append(prediction_items[start_idx : start_idx + batch_size])
    return chunks


def _parse_first_json_value(result_text: str) -> Any:
    cleaned_text = str(result_text or "").strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()

    decoder = json.JSONDecoder()
    candidate_positions = []
    for opening_char in ["{", "["]:
        pos = cleaned_text.find(opening_char)
        if pos >= 0:
            candidate_positions.append(pos)

    for start_pos in sorted(set(candidate_positions)):
        try:
            parsed_value, _ = decoder.raw_decode(cleaned_text[start_pos:])
            return parsed_value
        except json.JSONDecodeError:
            continue

    raise ValueError("Unable to parse JSON payload: {0}".format(cleaned_text))


def _normalize_batch_scores(parsed_payload: Any) -> Dict[str, Dict[str, float]]:
    normalized_results = {}

    if isinstance(parsed_payload, dict) and isinstance(parsed_payload.get("results"), list):
        parsed_payload = parsed_payload.get("results")

    if isinstance(parsed_payload, list):
        for item in parsed_payload:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                continue
            normalized_results[item_id] = _normalize_scores(item)
        return normalized_results

    if isinstance(parsed_payload, dict):
        for item_id, score_payload in parsed_payload.items():
            if not isinstance(score_payload, dict):
                continue
            normalized_results[str(item_id)] = _normalize_scores(score_payload)
        return normalized_results

    raise ValueError(
        "Unsupported batch-score JSON structure: {0}".format(type(parsed_payload).__name__)
    )


def call_llm_for_scoring_batch(
    client: OpenAI,
    teacher_comment: str,
    prediction_items: List[Dict[str, str]],
    model_name: str,
    sample_name: str,
) -> Dict[str, Dict[str, float]]:
    """
    Score multiple generated comments for the same sample in one judge request.
    """
    formatted_items = []
    for item in prediction_items:
        formatted_items.append(
            "[{0}]\n{1}".format(
                str(item.get("id") or "").strip(),
                str(item.get("text") or "").strip(),
            )
        )

    prompt = """You are a professional archery coach evaluation expert. Here is the teacher's standard evaluation and multiple generated evaluation texts for the same sample.

[Sample Name]
{sample_name}

[Teacher's Standard Evaluation (Full Score Baseline)]
{teacher_comment}

[Generated Evaluations to be Assessed]
{generated_comments}

Please score each generated evaluation independently on the following three dimensions (5-point Likert scale, 1 lowest, 5 highest, decimals allowed). Do not compare the generated evaluations against each other; compare each one only against the teacher's standard evaluation.

1. **Accuracy**: Are the technical points and action descriptions accurate? How consistent with the standard evaluation?
2. **Professionalism**: Are the terms and expressions professional? Does it reflect archery domain expertise?
3. **Practicality**: How valuable is the guidance for students? Are improvement suggestions specific and actionable?

**Scoring Standards**:
- 5: Excellent, fully meets standard evaluation level
- 4: Good, mostly meets standard
- 3: Average, basically meets standard but with obvious deficiencies
- 2: Poor, significant gap from standard
- 1: Very poor, seriously deviates from standard

**Important**: Output only one JSON object and no other content.
- Use each bracketed id exactly once as the JSON key.
- Do not omit any id.
- Format:
{{
  "candidate_1": {{"Accuracy": score, "Professionalism": score, "Practicality": score}},
  "candidate_2": {{"Accuracy": score, "Professionalism": score, "Practicality": score}}
}}
""".format(
        sample_name=sample_name,
        teacher_comment=teacher_comment,
        generated_comments="\n\n".join(formatted_items),
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional archery coach evaluation expert, "
                    "responsible for objective and fair scoring of archery "
                    "posture evaluation texts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    result_text = _clean_text(response.choices[0].message.content)
    parsed_payload = _parse_first_json_value(result_text)
    normalized_scores = _normalize_batch_scores(parsed_payload)

    expected_ids = [str(item.get("id") or "").strip() for item in prediction_items]
    missing_ids = [item_id for item_id in expected_ids if item_id and item_id not in normalized_scores]
    if missing_ids:
        raise ValueError(
            "Missing batch-score ids from {0}: {1}. Raw output: {2}".format(
                model_name,
                ", ".join(missing_ids),
                result_text,
            )
        )
    return normalized_scores

def _load_gt_text(sample_name: str, txt_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    gt_path = txt_dir / "{0}.txt".format(sample_name)
    if not gt_path.exists():
        return None, "GT text file not found: {0}".format(gt_path)

    try:
        with open(str(gt_path), "r", encoding="utf-8") as handle:
            return _clean_text(handle.read()), None
    except Exception as exc:
        return None, "Failed to read GT text: {0}".format(exc)


def _compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    cleaned = [float(v) for v in values]
    if len(cleaned) == 0:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": float(statistics.mean(cleaned)),
        "std": float(statistics.pstdev(cleaned)) if len(cleaned) > 1 else 0.0,
        "count": int(len(cleaned)),
    }


def _score_field_names(prediction_columns: List[str], judge_models: List[str]) -> List[str]:
    fields = []
    for prediction_column in prediction_columns:
        for judge_model in judge_models:
            fields.extend(
                [
                    "{0}__{1}_Accuracy".format(prediction_column, judge_model),
                    "{0}__{1}_Professionalism".format(prediction_column, judge_model),
                    "{0}__{1}_Practicality".format(prediction_column, judge_model),
                    "{0}__{1}_Average".format(prediction_column, judge_model),
                ]
            )
    return fields


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(str(path), "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _empty_score_row_fields(
    prediction_columns: List[str],
    judge_models: List[str],
) -> Dict[str, str]:
    empty_fields = {}
    for field_name in _score_field_names(prediction_columns, judge_models):
        empty_fields[field_name] = ""
    return empty_fields


def _score_fields_for_target(prediction_column: str, judge_model: str) -> List[str]:
    return [
        "{0}__{1}_Accuracy".format(prediction_column, judge_model),
        "{0}__{1}_Professionalism".format(prediction_column, judge_model),
        "{0}__{1}_Practicality".format(prediction_column, judge_model),
        "{0}__{1}_Average".format(prediction_column, judge_model),
    ]


def _target_has_scores(row: Dict[str, Any], prediction_column: str, judge_model: str) -> bool:
    for field_name in _score_fields_for_target(prediction_column, judge_model):
        if not str(row.get(field_name) or "").strip():
            return False
    return True


def _build_resume_rows(
    input_rows: List[Dict[str, Any]],
    sample_name_column: str,
    score_fieldnames: List[str],
    empty_score_fields: Dict[str, str],
    existing_output_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    existing_by_sample = {}
    for existing_row in existing_output_rows:
        sample_name = str(existing_row.get(sample_name_column) or "").strip()
        if sample_name:
            existing_by_sample[sample_name] = dict(existing_row)

    merged_rows = []
    for input_row in input_rows:
        sample_name = str(input_row.get(sample_name_column) or "").strip()
        merged_row = dict(input_row)
        merged_row.update(dict(empty_score_fields))
        existing_row = existing_by_sample.get(sample_name)
        if existing_row:
            for score_field in score_fieldnames:
                existing_value = existing_row.get(score_field)
                if str(existing_value or "").strip():
                    merged_row[score_field] = existing_value
        merged_rows.append(merged_row)
    return merged_rows


def run_batch(
    input_csv: Union[str, Path],
    sample_name_column: str,
    dataset_root: Union[str, Path],
    txt_subdir: str,
    output_root: Union[str, Path],
    run_name: Optional[str],
    output_csv_name: str,
    judge_models: List[str],
    prediction_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    prediction_batch_size: int = DEFAULT_PREDICTION_BATCH_SIZE,
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> Dict[str, Any]:
    input_csv_path = _resolve_project_path(input_csv)
    dataset_root_path = _resolve_project_path(dataset_root)
    txt_dir = dataset_root_path / txt_subdir
    run_dir = _build_run_dir(output_root, run_name)

    if not input_csv_path.exists():
        raise FileNotFoundError("Input CSV not found: {0}".format(input_csv_path))
    if not txt_dir.exists():
        raise FileNotFoundError("GT txt directory not found: {0}".format(txt_dir))

    fieldnames, rows = _load_csv_rows(input_csv_path)
    if len(fieldnames) == 0:
        raise ValueError("Input CSV has no header: {0}".format(input_csv_path))
    if sample_name_column not in fieldnames:
        raise ValueError(
            "sample_name column '{0}' not found in CSV header. Available columns: {1}".format(
                sample_name_column,
                ", ".join(fieldnames),
            )
        )

    effective_exclude_columns = list(exclude_columns or DEFAULT_EXCLUDE_COLUMNS)
    effective_prediction_columns = _infer_prediction_columns(
        fieldnames=fieldnames,
        sample_name_column=sample_name_column,
        explicit_prediction_columns=prediction_columns,
        exclude_columns=effective_exclude_columns,
    )
    if len(effective_prediction_columns) == 0:
        raise ValueError("No prediction columns were found in input CSV.")

    for column_name in effective_prediction_columns:
        if column_name not in fieldnames:
            raise ValueError("Prediction column not found in CSV: {0}".format(column_name))

    rows = sorted(rows, key=lambda row: str(row.get(sample_name_column) or ""))
    all_rows = list(rows)
    num_rows_before_limit = len(rows)
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]

    if len(rows) > 0 and not os.getenv(api_key_env):
        raise EnvironmentError(
            "Environment variable '{0}' is required for judge-model calls.".format(api_key_env)
        )

    client = None
    if len(rows) > 0:
        client = _make_openai_client(base_url=base_url, api_key_env=api_key_env)

    print("Total CSV rows: {0}".format(num_rows_before_limit))
    print("Rows selected for LLM judge: {0}".format(len(rows)))
    print("Prediction columns: {0}".format(effective_prediction_columns))
    print("Prediction batch size per judge request: {0}".format(prediction_batch_size))

    score_fieldnames = _score_field_names(effective_prediction_columns, judge_models)
    output_fieldnames = list(fieldnames)
    for field_name in score_fieldnames:
        if field_name not in output_fieldnames:
            output_fieldnames.append(field_name)

    scored_csv = run_dir / output_csv_name
    failed_csv = run_dir / "failed_samples.csv"
    summary_json = run_dir / "summary.json"
    manifest_json = run_dir / "manifest.json"
    empty_score_fields = _empty_score_row_fields(
        prediction_columns=effective_prediction_columns,
        judge_models=judge_models,
    )
    existing_output_rows = []
    if scored_csv.exists():
        _, existing_output_rows = _load_csv_rows(scored_csv)
    scored_rows = _build_resume_rows(
        input_rows=all_rows,
        sample_name_column=sample_name_column,
        score_fieldnames=score_fieldnames,
        empty_score_fields=empty_score_fields,
        existing_output_rows=existing_output_rows,
    )
    failures = []
    resumed_rows = 0
    for row in scored_rows:
        if any(str(row.get(field_name) or "").strip() for field_name in score_fieldnames):
            resumed_rows += 1

    print("Rows resumed from existing scored CSV: {0}".format(resumed_rows))

    _write_csv(scored_csv, scored_rows, output_fieldnames)
    _write_csv(
        failed_csv,
        failures,
        [
            "sample_name",
            "prediction_column",
            "judge_model",
            "failure_stage",
            "failure_reason",
        ],
    )

    scored_rows_by_sample = {}
    for scored_row in scored_rows:
        sample_name = str(scored_row.get(sample_name_column) or "").strip()
        if sample_name:
            scored_rows_by_sample[sample_name] = scored_row

    for idx, row in enumerate(rows, start=1):
        sample_name = str(row.get(sample_name_column) or "").strip()
        output_row = scored_rows_by_sample.get(sample_name)
        if output_row is None:
            output_row = dict(row)
            output_row.update(dict(empty_score_fields))

        print("[{0}/{1}] Evaluating sample={2}".format(idx, len(rows), sample_name or "<empty>"))

        if not sample_name:
            failures.append(
                {
                    "sample_name": "",
                    "prediction_column": "",
                    "judge_model": "",
                    "failure_stage": "sample_name",
                    "failure_reason": "empty_sample_name",
                }
            )
            _write_csv(scored_csv, scored_rows, output_fieldnames)
            _write_csv(
                failed_csv,
                failures,
                [
                    "sample_name",
                    "prediction_column",
                    "judge_model",
                    "failure_stage",
                    "failure_reason",
                ],
            )
            continue

        gt_text, gt_error = _load_gt_text(sample_name=sample_name, txt_dir=txt_dir)
        if gt_text is None:
            failures.append(
                {
                    "sample_name": sample_name,
                    "prediction_column": "",
                    "judge_model": "",
                    "failure_stage": "gt_text",
                    "failure_reason": str(gt_error or "unknown_gt_text_error"),
                }
            )
            _write_csv(scored_csv, scored_rows, output_fieldnames)
            _write_csv(
                failed_csv,
                failures,
                [
                    "sample_name",
                    "prediction_column",
                    "judge_model",
                    "failure_stage",
                    "failure_reason",
                ],
            )
            continue

        non_empty_prediction_items = []
        for prediction_index, prediction_column in enumerate(effective_prediction_columns, start=1):
            generated_text = _clean_text(output_row.get(prediction_column))
            if not generated_text:
                failures.append(
                    {
                        "sample_name": sample_name,
                        "prediction_column": prediction_column,
                        "judge_model": "",
                        "failure_stage": "generated_text",
                        "failure_reason": "empty_generated_text",
                    }
                )
                continue

            non_empty_prediction_items.append(
                {
                    "id": "candidate_{0}".format(prediction_index),
                    "prediction_column": prediction_column,
                    "text": generated_text,
                }
            )

        sample_has_pending_scores = False
        for judge_model in judge_models:
            pending_prediction_items = []
            for prediction_item in non_empty_prediction_items:
                prediction_column = str(prediction_item.get("prediction_column") or "").strip()
                if _target_has_scores(output_row, prediction_column, judge_model):
                    continue
                pending_prediction_items.append(dict(prediction_item))

            if len(pending_prediction_items) == 0:
                print("  judge={0} already complete for sample={1}".format(judge_model, sample_name))
                continue

            sample_has_pending_scores = True
            prediction_batches = _chunk_prediction_items(
                prediction_items=pending_prediction_items,
                batch_size=int(prediction_batch_size),
            )
            for batch_index, prediction_batch in enumerate(prediction_batches, start=1):
                batch_columns = [
                    str(item.get("prediction_column") or "").strip()
                    for item in prediction_batch
                ]
                print(
                    "  batch {0}/{1} -> {2} ({3})".format(
                        batch_index,
                        len(prediction_batches),
                        judge_model,
                        ", ".join(batch_columns),
                    )
                )
                try:
                    batch_scores = call_llm_for_scoring_batch(
                        client=client,
                        teacher_comment=gt_text,
                        prediction_items=prediction_batch,
                        model_name=judge_model,
                        sample_name=sample_name,
                    )
                    for batch_item in prediction_batch:
                        prediction_column = str(batch_item.get("prediction_column") or "").strip()
                        batch_item_id = str(batch_item.get("id") or "").strip()
                        scores = batch_scores[batch_item_id]
                        average_score = round(
                            (
                                scores["Accuracy"]
                                + scores["Professionalism"]
                                + scores["Practicality"]
                            )
                            / 3.0,
                            2,
                        )
                        output_row["{0}__{1}_Accuracy".format(prediction_column, judge_model)] = scores["Accuracy"]
                        output_row["{0}__{1}_Professionalism".format(prediction_column, judge_model)] = scores["Professionalism"]
                        output_row["{0}__{1}_Practicality".format(prediction_column, judge_model)] = scores["Practicality"]
                        output_row["{0}__{1}_Average".format(prediction_column, judge_model)] = average_score
                        print(
                            "    sample={0} model={1} judge={2} "
                            "Accuracy={3} Professionalism={4} Practicality={5} Average={6}".format(
                                sample_name,
                                prediction_column,
                                judge_model,
                                scores["Accuracy"],
                                scores["Professionalism"],
                                scores["Practicality"],
                                average_score,
                            )
                        )
                except Exception as exc:
                    for batch_item in prediction_batch:
                        failures.append(
                            {
                                "sample_name": sample_name,
                                "prediction_column": str(batch_item.get("prediction_column") or "").strip(),
                                "judge_model": judge_model,
                                "failure_stage": "llm_scoring_batch",
                                "failure_reason": str(exc),
                            }
                        )
                    print(
                        "    sample={0} judge={1} batch={2} FAILED: {3}".format(
                            sample_name,
                            judge_model,
                            ", ".join(batch_columns),
                            str(exc),
                        )
                    )

                _write_csv(scored_csv, scored_rows, output_fieldnames)
                _write_csv(
                    failed_csv,
                    failures,
                    [
                        "sample_name",
                        "prediction_column",
                        "judge_model",
                        "failure_stage",
                        "failure_reason",
                    ],
                )

        if not sample_has_pending_scores:
            print("  sample={0} already complete, skipped".format(sample_name))

        _write_csv(scored_csv, scored_rows, output_fieldnames)

    judge_summary = {}
    for prediction_column in effective_prediction_columns:
        judge_summary[prediction_column] = {}
        for judge_model in judge_models:
            judge_summary[prediction_column][judge_model] = {
                "Accuracy": _compute_stats(
                    [
                        float(row["{0}__{1}_Accuracy".format(prediction_column, judge_model)])
                        for row in scored_rows
                        if str(row.get("{0}__{1}_Accuracy".format(prediction_column, judge_model)) or "").strip()
                    ]
                ),
                "Professionalism": _compute_stats(
                    [
                        float(row["{0}__{1}_Professionalism".format(prediction_column, judge_model)])
                        for row in scored_rows
                        if str(row.get("{0}__{1}_Professionalism".format(prediction_column, judge_model)) or "").strip()
                    ]
                ),
                "Practicality": _compute_stats(
                    [
                        float(row["{0}__{1}_Practicality".format(prediction_column, judge_model)])
                        for row in scored_rows
                        if str(row.get("{0}__{1}_Practicality".format(prediction_column, judge_model)) or "").strip()
                    ]
                ),
                "Average": _compute_stats(
                    [
                        float(row["{0}__{1}_Average".format(prediction_column, judge_model)])
                        for row in scored_rows
                        if str(row.get("{0}__{1}_Average".format(prediction_column, judge_model)) or "").strip()
                    ]
                ),
            }

    summary = {
        "run_dir": str(run_dir),
        "input_csv": str(input_csv_path),
        "output_csv": str(scored_csv),
        "sample_name_column": sample_name_column,
        "dataset_root": str(dataset_root_path),
        "txt_dir": str(txt_dir),
        "prediction_columns": list(effective_prediction_columns),
        "exclude_columns": list(effective_exclude_columns),
        "judge_models": list(judge_models),
        "prediction_batch_size": int(prediction_batch_size),
        "base_url": base_url,
        "api_key_env": api_key_env,
        "num_rows_before_limit": int(num_rows_before_limit),
        "num_rows": int(len(rows)),
        "num_records": int(len(scored_rows)),
        "num_resumed_rows": int(resumed_rows),
        "num_failed_rows": int(len(failures)),
        "judge_summary": judge_summary,
    }
    with open(str(summary_json), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    manifest = {
        "run_dir": str(run_dir),
        "scored_csv": str(scored_csv),
        "failed_samples_csv": str(failed_csv),
        "summary_json": str(summary_json),
        "config": {
            "input_csv": str(input_csv_path),
            "sample_name_column": sample_name_column,
            "dataset_root": str(dataset_root_path),
            "txt_subdir": txt_subdir,
            "output_csv_name": output_csv_name,
            "prediction_columns": list(effective_prediction_columns),
            "exclude_columns": list(effective_exclude_columns),
            "judge_models": list(judge_models),
            "prediction_batch_size": int(prediction_batch_size),
            "max_samples": max_samples,
            "base_url": base_url,
            "api_key_env": api_key_env,
        },
    }
    with open(str(manifest_json), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch LLM-judge evaluation for CSV-stored assessment texts against GT txt."
        )
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--sample-name-column", default=DEFAULT_SAMPLE_NAME_COLUMN)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--txt-subdir", default=DEFAULT_TXT_SUBDIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="LLM_JUDGE_RESULT")
    parser.add_argument("--output-csv-name", default=DEFAULT_OUTPUT_CSV_NAME)
    parser.add_argument("--prediction-columns", nargs="+", default=None)
    parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_COLUMNS),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=DEFAULT_PREDICTION_BATCH_SIZE,
        help="How many prediction columns to score together in one judge request.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--judge-models",
        nargs="+",
        default=list(DEFAULT_JUDGE_MODELS),
        help="Judge models used with the existing scoring prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_batch(
        input_csv=args.input_csv,
        sample_name_column=args.sample_name_column,
        dataset_root=args.dataset_root,
        txt_subdir=args.txt_subdir,
        output_root=args.output_root,
        run_name=args.run_name,
        output_csv_name=args.output_csv_name,
        judge_models=list(args.judge_models),
        prediction_columns=args.prediction_columns,
        exclude_columns=args.exclude_columns,
        max_samples=args.max_samples,
        prediction_batch_size=args.prediction_batch_size,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
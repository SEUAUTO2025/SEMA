"""
Batch LLM choose-judge experiment for assessment texts stored in a wide CSV file.

The script reads `evaluation_texts.csv`, loads the ground-truth txt matched by
`sample_name`, and asks the judge model to choose the single best generated
evaluation text for each sample among all candidate columns using the same
Accuracy / Professionalism / Practicality standards as the score-based judge.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from batch_run_eval_db_llm_judge import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXCLUDE_COLUMNS,
    DEFAULT_INPUT_CSV,
    DEFAULT_JUDGE_MODELS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLE_NAME_COLUMN,
    DEFAULT_TXT_SUBDIR,
    _build_run_dir,
    _clean_text,
    _infer_prediction_columns,
    _load_csv_rows,
    _load_gt_text,
    _make_openai_client,
    _parse_first_json_value,
    _resolve_project_path,
)


DEFAULT_OUTPUT_CSV_NAME = "evaluation_texts_choose_records.csv"
DEFAULT_PERCENTAGE_CSV_NAME = "selection_percentage.csv"


def _selected_prediction_field_name(judge_model: str) -> str:
    return "{0}__selected_prediction_column".format(judge_model)


def _selected_candidate_field_name(judge_model: str) -> str:
    return "{0}__selected_candidate_id".format(judge_model)


def _selection_field_names(judge_models: List[str]) -> List[str]:
    fields = []
    for judge_model in judge_models:
        fields.extend(
            [
                _selected_prediction_field_name(judge_model),
                _selected_candidate_field_name(judge_model),
            ]
        )
    return fields


def _empty_selection_fields(judge_models: List[str]) -> Dict[str, str]:
    empty_fields = {}
    for field_name in _selection_field_names(judge_models):
        empty_fields[field_name] = ""
    return empty_fields


def _build_resume_rows(
    input_rows: List[Dict[str, Any]],
    sample_name_column: str,
    selection_fieldnames: List[str],
    empty_selection_fields: Dict[str, str],
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
        merged_row.update(dict(empty_selection_fields))
        existing_row = existing_by_sample.get(sample_name)
        if existing_row:
            for field_name in selection_fieldnames:
                existing_value = existing_row.get(field_name)
                if str(existing_value or "").strip():
                    merged_row[field_name] = existing_value
        merged_rows.append(merged_row)
    return merged_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(str(path), "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def call_llm_for_choice(
    client: Any,
    teacher_comment: str,
    prediction_items: List[Dict[str, str]],
    model_name: str,
    sample_name: str,
) -> Dict[str, str]:
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

Please choose the single best generated evaluation that is closest to the teacher's standard evaluation. Use the same three dimensions and standards below:

1. **Accuracy**: Are the technical points and action descriptions accurate? How consistent with the standard evaluation?
2. **Professionalism**: Are the terms and expressions professional? Does it reflect archery domain expertise?
3. **Practicality**: How valuable is the guidance for students? Are improvement suggestions specific and actionable?

**Scoring Standards**:
- 5: Excellent, fully meets standard evaluation level
- 4: Good, mostly meets standard
- 3: Average, basically meets standard but with obvious deficiencies
- 2: Poor, significant gap from standard
- 1: Very poor, seriously deviates from standard

Select the overall best candidate across the three dimensions. If several candidates are very close, break ties by Accuracy first, then Professionalism, then Practicality.

**Important**: Output only one JSON object and no other content. Format:
{{"best_candidate_id": "candidate_1"}}
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
                    "responsible for objective and fair selection of the best "
                    "archery posture evaluation text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    result_text = _clean_text(response.choices[0].message.content)
    parsed_payload = _parse_first_json_value(result_text)
    if not isinstance(parsed_payload, dict):
        raise ValueError(
            "Unexpected choose-result JSON structure from {0}: {1}".format(
                model_name,
                result_text,
            )
        )
    best_candidate_id = str(parsed_payload.get("best_candidate_id") or "").strip()
    if not best_candidate_id:
        raise ValueError(
            "Missing best_candidate_id from {0}: {1}".format(model_name, result_text)
        )

    candidate_lookup = {}
    for item in prediction_items:
        candidate_id = str(item.get("id") or "").strip()
        if candidate_id:
            candidate_lookup[candidate_id] = str(item.get("prediction_column") or "").strip()

    if best_candidate_id not in candidate_lookup:
        raise ValueError(
            "Unknown best_candidate_id from {0}: {1}. Raw output: {2}".format(
                model_name,
                best_candidate_id,
                result_text,
            )
        )

    return {
        "best_candidate_id": best_candidate_id,
        "selected_prediction_column": candidate_lookup[best_candidate_id],
    }


def _compute_selection_summary(
    rows: List[Dict[str, Any]],
    prediction_columns: List[str],
    judge_models: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = {}
    percentage_rows = []
    for judge_model in judge_models:
        selected_field = _selected_prediction_field_name(judge_model)
        valid_rows = [
            row for row in rows if str(row.get(selected_field) or "").strip()
        ]
        total_valid = len(valid_rows)
        judge_summary = {
            "num_valid_samples": int(total_valid),
            "selection_percentage": {},
        }
        for prediction_column in prediction_columns:
            selected_count = sum(
                1
                for row in valid_rows
                if str(row.get(selected_field) or "").strip() == prediction_column
            )
            selected_percentage = (
                round((100.0 * float(selected_count)) / float(total_valid), 4)
                if total_valid > 0
                else None
            )
            judge_summary["selection_percentage"][prediction_column] = {
                "selected_count": int(selected_count),
                "selected_percentage": selected_percentage,
            }
            percentage_rows.append(
                {
                    "judge_model": judge_model,
                    "prediction_column": prediction_column,
                    "selected_count": int(selected_count),
                    "selected_percentage": selected_percentage,
                    "num_valid_samples": int(total_valid),
                }
            )
        summary[judge_model] = judge_summary
    return summary, percentage_rows


def run_batch(
    input_csv: Union[str, Path],
    sample_name_column: str,
    dataset_root: Union[str, Path],
    txt_subdir: str,
    output_root: Union[str, Path],
    run_name: Optional[str],
    output_csv_name: str,
    percentage_csv_name: str,
    judge_models: List[str],
    prediction_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
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
            "Environment variable '{0}' is required for choose-model calls.".format(api_key_env)
        )

    client = None
    if len(rows) > 0:
        client = _make_openai_client(base_url=base_url, api_key_env=api_key_env)

    selection_fieldnames = _selection_field_names(judge_models)
    output_fieldnames = list(fieldnames)
    for field_name in selection_fieldnames:
        if field_name not in output_fieldnames:
            output_fieldnames.append(field_name)

    records_csv = run_dir / output_csv_name
    percentage_csv = run_dir / percentage_csv_name
    failed_csv = run_dir / "failed_samples.csv"
    summary_json = run_dir / "summary.json"
    manifest_json = run_dir / "manifest.json"

    empty_selection_fields = _empty_selection_fields(judge_models)
    existing_output_rows = []
    if records_csv.exists():
        _, existing_output_rows = _load_csv_rows(records_csv)
    output_rows = _build_resume_rows(
        input_rows=all_rows,
        sample_name_column=sample_name_column,
        selection_fieldnames=selection_fieldnames,
        empty_selection_fields=empty_selection_fields,
        existing_output_rows=existing_output_rows,
    )
    resumed_rows = 0
    for row in output_rows:
        if any(str(row.get(field_name) or "").strip() for field_name in selection_fieldnames):
            resumed_rows += 1

    output_rows_by_sample = {}
    for output_row in output_rows:
        sample_name = str(output_row.get(sample_name_column) or "").strip()
        if sample_name:
            output_rows_by_sample[sample_name] = output_row

    failures = []

    print("Total CSV rows: {0}".format(num_rows_before_limit))
    print("Rows selected for choose judge: {0}".format(len(rows)))
    print("Prediction columns: {0}".format(effective_prediction_columns))
    print("Rows resumed from existing choose CSV: {0}".format(resumed_rows))

    _write_csv(records_csv, output_rows, output_fieldnames)
    _write_csv(
        failed_csv,
        failures,
        ["sample_name", "judge_model", "failure_stage", "failure_reason"],
    )

    for idx, row in enumerate(rows, start=1):
        sample_name = str(row.get(sample_name_column) or "").strip()
        output_row = output_rows_by_sample.get(sample_name)
        if output_row is None:
            output_row = dict(row)
            output_row.update(dict(empty_selection_fields))

        print("[{0}/{1}] Choosing sample={2}".format(idx, len(rows), sample_name or "<empty>"))

        if not sample_name:
            failures.append(
                {
                    "sample_name": "",
                    "judge_model": "",
                    "failure_stage": "sample_name",
                    "failure_reason": "empty_sample_name",
                }
            )
            _write_csv(records_csv, output_rows, output_fieldnames)
            _write_csv(
                failed_csv,
                failures,
                ["sample_name", "judge_model", "failure_stage", "failure_reason"],
            )
            continue

        gt_text, gt_error = _load_gt_text(sample_name=sample_name, txt_dir=txt_dir)
        if gt_text is None:
            failures.append(
                {
                    "sample_name": sample_name,
                    "judge_model": "",
                    "failure_stage": "gt_text",
                    "failure_reason": str(gt_error or "unknown_gt_text_error"),
                }
            )
            _write_csv(records_csv, output_rows, output_fieldnames)
            _write_csv(
                failed_csv,
                failures,
                ["sample_name", "judge_model", "failure_stage", "failure_reason"],
            )
            continue

        prediction_items = []
        for prediction_index, prediction_column in enumerate(effective_prediction_columns, start=1):
            generated_text = _clean_text(output_row.get(prediction_column))
            if not generated_text:
                continue
            prediction_items.append(
                {
                    "id": "candidate_{0}".format(prediction_index),
                    "prediction_column": prediction_column,
                    "text": generated_text,
                }
            )

        if len(prediction_items) == 0:
            failures.append(
                {
                    "sample_name": sample_name,
                    "judge_model": "",
                    "failure_stage": "prediction_items",
                    "failure_reason": "no_non_empty_prediction_text",
                }
            )
            _write_csv(records_csv, output_rows, output_fieldnames)
            _write_csv(
                failed_csv,
                failures,
                ["sample_name", "judge_model", "failure_stage", "failure_reason"],
            )
            continue

        sample_has_pending = False
        for judge_model in judge_models:
            selected_prediction_field = _selected_prediction_field_name(judge_model)
            selected_candidate_field = _selected_candidate_field_name(judge_model)
            if str(output_row.get(selected_prediction_field) or "").strip():
                print("  judge={0} already complete for sample={1}".format(judge_model, sample_name))
                continue

            sample_has_pending = True
            print("  choosing with judge={0}".format(judge_model))
            try:
                selection = call_llm_for_choice(
                    client=client,
                    teacher_comment=gt_text,
                    prediction_items=prediction_items,
                    model_name=judge_model,
                    sample_name=sample_name,
                )
                output_row[selected_prediction_field] = selection["selected_prediction_column"]
                output_row[selected_candidate_field] = selection["best_candidate_id"]
                print(
                    "    sample={0} judge={1} selected={2}".format(
                        sample_name,
                        judge_model,
                        selection["selected_prediction_column"],
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "sample_name": sample_name,
                        "judge_model": judge_model,
                        "failure_stage": "llm_choose",
                        "failure_reason": str(exc),
                    }
                )
                print(
                    "    sample={0} judge={1} FAILED: {2}".format(
                        sample_name,
                        judge_model,
                        str(exc),
                    )
                )

            _write_csv(records_csv, output_rows, output_fieldnames)
            _write_csv(
                failed_csv,
                failures,
                ["sample_name", "judge_model", "failure_stage", "failure_reason"],
            )

        if not sample_has_pending:
            print("  sample={0} already complete, skipped".format(sample_name))

        _write_csv(records_csv, output_rows, output_fieldnames)

    selection_summary, percentage_rows = _compute_selection_summary(
        rows=output_rows,
        prediction_columns=effective_prediction_columns,
        judge_models=judge_models,
    )
    _write_csv(
        percentage_csv,
        percentage_rows,
        [
            "judge_model",
            "prediction_column",
            "selected_count",
            "selected_percentage",
            "num_valid_samples",
        ],
    )

    summary = {
        "run_dir": str(run_dir),
        "input_csv": str(input_csv_path),
        "records_csv": str(records_csv),
        "percentage_csv": str(percentage_csv),
        "sample_name_column": sample_name_column,
        "dataset_root": str(dataset_root_path),
        "txt_dir": str(txt_dir),
        "prediction_columns": list(effective_prediction_columns),
        "exclude_columns": list(effective_exclude_columns),
        "judge_models": list(judge_models),
        "base_url": base_url,
        "api_key_env": api_key_env,
        "num_rows_before_limit": int(num_rows_before_limit),
        "num_rows": int(len(rows)),
        "num_records": int(len(output_rows)),
        "num_resumed_rows": int(resumed_rows),
        "num_failed_rows": int(len(failures)),
        "selection_summary": selection_summary,
    }
    with open(str(summary_json), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    manifest = {
        "run_dir": str(run_dir),
        "records_csv": str(records_csv),
        "percentage_csv": str(percentage_csv),
        "failed_samples_csv": str(failed_csv),
        "summary_json": str(summary_json),
        "config": {
            "input_csv": str(input_csv_path),
            "sample_name_column": sample_name_column,
            "dataset_root": str(dataset_root_path),
            "txt_subdir": txt_subdir,
            "output_csv_name": output_csv_name,
            "percentage_csv_name": percentage_csv_name,
            "prediction_columns": list(effective_prediction_columns),
            "exclude_columns": list(effective_exclude_columns),
            "judge_models": list(judge_models),
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
            "Batch choose-judge evaluation for CSV-stored assessment texts against GT txt."
        )
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--sample-name-column", default=DEFAULT_SAMPLE_NAME_COLUMN)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--txt-subdir", default=DEFAULT_TXT_SUBDIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-csv-name", default=DEFAULT_OUTPUT_CSV_NAME)
    parser.add_argument("--percentage-csv-name", default=DEFAULT_PERCENTAGE_CSV_NAME)
    parser.add_argument("--prediction-columns", nargs="+", default=None)
    parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_COLUMNS),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--judge-models",
        nargs="+",
        default=list(DEFAULT_JUDGE_MODELS),
        help="Judge models used with the existing choose prompt.",
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
        percentage_csv_name=args.percentage_csv_name,
        judge_models=list(args.judge_models),
        prediction_columns=args.prediction_columns,
        exclude_columns=args.exclude_columns,
        max_samples=args.max_samples,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

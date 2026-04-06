---
title: SEMA Web Demo
sdk: docker
app_port: 7860
pinned: false
---

# SEMA: Modular Automated Archery Coaching with LLMs

The code repository for our work **"A Modular Approach to Automated Archery Coaching for Action Quality Assessment and Feedback Generation Using Large Language Models"**.

This repository contains the main SEMA inference pipeline, the RTMPose-based biomechanical analysis modules, the RAG knowledge-base runtime, and the experiment scripts used to reproduce the paper-facing results under `Experiments/`.

## 1. Project Overview

- `main.py`  
  Single-sample entry point for the default SEMA pipeline. It runs multimodal assessment, knowledge retrieval, response generation, metric scoring against the matched GT text, and optional follow-up QA.
- `RTMPose/`  
  Pose extraction, keyframe selection, and biomechanical feature computation.
- `RAG/`  
  Knowledge-base runtime, retrieval logic, and bundled SQLite knowledge databases in `RAG/db_files/`.
- `Tools/`  
  Dataset helpers, text metrics, and evaluation utilities.
- `evaluation/eval_db/`  
  SQLite storage used by the comparison batch pipelines.
- `Experiments/BIO_FEATURE_CAL/`  
  Biomechanical feature validation, threshold fitting, and paper figure scripts.
- `Experiments/Keyframe/`  
  Keyframe IoU / hit-ratio robustness experiments and plotting scripts.
- `Experiments/RAG/`  
  GT-keyword RAG evaluation.
- `Experiments/Comparation/`  
  Batch evaluation pipelines for SEMA, DashScope/Zhipu, OpenRouter, and LLM-judge comparison.
- `output_keyframes/`  
  Reference template keyframes used by the FastDTW baseline and paper figure scripts.
- `PAPER_WORKLOG.md`  
  Ongoing reproducibility / paper-writing worklog for the SCI version of the project.

## 2. Environment

- Recommended Python version: `3.9`
- Recommended platform: Windows + PowerShell (the commands below follow that setup)
- Install dependencies with:

```powershell
pip install -r requirements.txt
```

- Optional but recommended NLTK resource download for text metrics:

```powershell
python -m nltk.downloader punkt wordnet omw-1.4
```

Notes:

- RTMPose ONNX checkpoints are downloaded automatically by the bundled `rtmlib` code on first use.
- Some LanguageBind assets may also be downloaded automatically the first time the corresponding embedding path is executed.
- The default paper pipeline uses `onnxruntime` as the RTMPose backend.
- `pytorchvideo` is kept out of the default Windows install path in `requirements.txt` because its upstream `av` dependency is often unavailable on Python 3.9 Windows wheels. If you need the full LanguageBind video stack on Windows, install the missing dependency manually or use WSL / Linux.

## 3. External Runtime Setup

### 3.1 API Keys

Set the environment variables required by the scripts you want to run:

```powershell
$env:ALI_API_KEY="YOUR_DASHSCOPE_KEY"
$env:OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
$env:ZHIPU_API_KEY="YOUR_ZHIPU_KEY"
$env:DEEPL_API_KEY="YOUR_DEEPL_KEY"
```

Which key is used where:

- `ALI_API_KEY`  
  Required by the default SEMA pipeline in `main.py`, the DashScope comparison scripts, and the CSV LLM-judge scripts.
- `OPENROUTER_API_KEY`  
  Required by `Experiments/Comparation/batch_run_openrouter_multimodal_eval.py`.
- `ZHIPU_API_KEY`  
  Required only if you keep `glm-4.6v` in the DashScope comparison model list.
- `DEEPL_API_KEY`  
  Required by the RAG experiment when English translation is enabled for semantic metrics.

### 3.2 Bundled Databases

The repository already includes the current runtime knowledge bases:

- `RAG/db_files/LLM_Knowledge_New_ali_zh.db`
- `RAG/db_files/LLM_Knowledge_New_En.db`
- `RAG/db_files/LLM_Knowledge_Pure.db`

The comparison pipelines also use SQLite evaluation databases under `evaluation/eval_db/`.

To switch the active evaluation database without editing code, set:

```powershell
$env:SEMA_EVAL_DB_PATH="evaluation.db"
```

The path is resolved relative to `evaluation/eval_db/` unless you provide an absolute path.

## 4. Prepare the Datasets

The datasets are **not distributed in this repository**. Users must download them separately and create the required folders manually.

Google Drive download links to be added before release:

- Main / validation set (`dataset`): `<ADD_GOOGLE_DRIVE_LINK_FOR_DATASET>`
- Train subset for biomechanical threshold fitting (`dataset_choose`): `<ADD_GOOGLE_DRIVE_LINK_FOR_DATASET_CHOOSE>`
- GT-keyword RAG subset (`dataset_ex`): `<ADD_GOOGLE_DRIVE_LINK_FOR_DATASET_EX>`

### 4.1 Minimum setup for `main.py`

If your repo does not contain a dataset folder yet, create one manually at the repository root:

```text
SEMA/
  dataset/
    csv/
    keyframes/
    txt/
    video/
```

Then download the main dataset archive from Google Drive and extract it into `dataset/`.

The default `main.py` pipeline assumes:

- `dataset/video/<sample>.mp4`
- `dataset/txt/<sample>.txt`
- `dataset/csv/<sample>.csv`

For example, running `main.py` on `136.mp4` expects:

- `dataset/video/136.mp4`
- `dataset/txt/136.txt`
- `dataset/csv/136.csv`

### 4.2 Additional folders for full experiment reproduction

Some experiments require extra dataset roots:

```text
SEMA/
  dataset_choose/
    csv/
    keyframes/
    txt/
    video/
  dataset_ex/
    csv/
    keywords/
    txt/
    video/
```

Use:

- `dataset_choose/` for biomechanical threshold fitting in `Experiments/BIO_FEATURE_CAL/`
- `dataset_ex/` for GT-keyword RAG evaluation in `Experiments/RAG/`

## 5. How to Run

This repository supports two common workflows:

- Run the default SEMA pipeline on one labeled sample with `main.py`
- Reproduce the experiment scripts under `Experiments/`

### 5.1 Run `main.py`

Recommended command:

```powershell
python main.py --dataset-root dataset --video-name 136.mp4 --language zh --assessment-model-name qwen3-vl-plus --response-model-name qwen-plus
```

Useful variants:

```powershell
python main.py --dataset-root dataset --video-name 136.mp4 --skip-qa
python main.py --dataset-root dataset --video-name 136.mp4 --hide-progress
python main.py --dataset-root dataset --video-name 136.mp4 --language en
```

What `main.py` does:

- loads the target video from `dataset/video/`
- runs `Tokenize_SearchKeyword(...)`
- retrieves knowledge snippets from the bundled SQLite KB
- generates the final coaching response with `get_response(...)`
- matches the GT text / scores from `dataset/txt` and `dataset/csv`
- computes BLEU and BERTScore against the GT evaluation text
- optionally enters the interactive QA loop via `answer_archery_question(...)`

Important limitation:

- `main.py` is currently designed for labeled dataset samples, not arbitrary unlabeled user videos. It expects the matching GT `txt` and `csv` files to exist.

### 5.2 Reproduce `Experiments/BIO_FEATURE_CAL`

Main validation run:

```powershell
python Experiments/BIO_FEATURE_CAL/run_bio_feature_guidance_validation.py --train-dataset-root dataset_choose --val-dataset-root dataset
```

Default outputs:

- `Experiments/BIO_FEATURE_CAL/evaluation/results/<run_name>/train_sample_features.csv`
- `Experiments/BIO_FEATURE_CAL/evaluation/results/<run_name>/val_sample_features.csv`
- `Experiments/BIO_FEATURE_CAL/evaluation/results/<run_name>/thresholds_train.json`
- `Experiments/BIO_FEATURE_CAL/evaluation/results/<run_name>/sample_estimates_validation.csv`
- `Experiments/BIO_FEATURE_CAL/evaluation/results/<run_name>/summary.json`
- `Experiments/BIO_FEATURE_CAL/evaluation/results/<run_name>/manifest.json`

Paper figure scripts:

```powershell
python Experiments/BIO_FEATURE_CAL/plot_paper_bio_threshold_landscape.py --run-dir Experiments/BIO_FEATURE_CAL/evaluation/results/TRAIN_VAL_EX_RESULTS
python Experiments/BIO_FEATURE_CAL/plot_paper_bio_threshold_hist_panels.py --run-dir Experiments/BIO_FEATURE_CAL/evaluation/results/TRAIN_VAL_EX_RESULTS
python Experiments/BIO_FEATURE_CAL/plot_paper_bio_fourclass_panels_and_mae.py --run-dir Experiments/BIO_FEATURE_CAL/evaluation/results/TRAIN_VAL_EX_RESULTS
python Experiments/BIO_FEATURE_CAL/plot_paper_bio_confusion_and_thresholds.py --run-dir Experiments/BIO_FEATURE_CAL/evaluation/results/TRAIN_VAL_EX_RESULTS
python Experiments/BIO_FEATURE_CAL/plot_paper_pose_topology_and_biomech_points.py
python Experiments/BIO_FEATURE_CAL/render_rtmpose_keyframes_for_paper.py --input-dir output_keyframes --manifest-path output_keyframes/manifest.json
```

### 5.3 Reproduce `Experiments/Keyframe`

Single-sample sanity check:

```powershell
python Experiments/Keyframe/keyframe_test.py --dataset-root dataset --sample-name 136
```

Angle-curve export:

```powershell
python Experiments/Keyframe/plot_angle_curve_from_csv.py --dataset-root dataset --sample-name 136 --output-root Experiments/Keyframe/evaluation/angle_curve
```

IoU robustness evaluation:

```powershell
python Experiments/Keyframe/run_keyframe_interval_iou_robustness.py --dataset-root dataset --output-root Experiments/Keyframe/evaluation --target-k 6
```

Dynamic hit-ratio evaluation:

```powershell
python Experiments/Keyframe/run_keyframe_dynamic_hit_ratio_robustness.py --dataset-root dataset --output-root Experiments/Keyframe/evaluation --target-k 6
```

Paper-style boxplot from a finished run directory:

```powershell
python Experiments/Keyframe/evaluation/515_FINAL_RESULTS/plot_paper_iou_boxplot.py --run-dir Experiments/Keyframe/evaluation/515_FINAL_RESULTS
```

Key dataset requirements:

- `dataset/csv/*.csv`
- `dataset/video/*.mp4`
- `dataset/keyframes/*.txt`
- `dataset/keyframes/keyframes.txt` for the dynamic-hit experiment

### 5.4 Reproduce `Experiments/RAG`

GT-keyword RAG evaluation:

```powershell
python Experiments/RAG/run_gt_keywords_rag_evaluation.py --dataset-root dataset_ex --run-name RAG_RESULTS
```

This experiment expects:

- `dataset_ex/csv/`
- `dataset_ex/keywords/`
- `dataset_ex/txt/`
- `dataset_ex/video/`
- `ALI_API_KEY`
- `DEEPL_API_KEY` for the default English semantic-metric path

Default outputs:

- `Experiments/RAG/evaluation/results/<run_name>/sample_manifest.csv`
- `Experiments/RAG/evaluation/results/<run_name>/records.csv`
- `Experiments/RAG/evaluation/results/<run_name>/metric_summary.csv`
- `Experiments/RAG/evaluation/results/<run_name>/summary.json`
- `Experiments/RAG/evaluation/results/<run_name>/manifest.json`

### 5.5 Reproduce `Experiments/Comparation`

These scripts are used to build cross-model assessment-text comparisons.

Important note:

- `batch_run_main_pipeline.py`, `batch_run_dashscope_multimodal_eval.py`, and `batch_run_openrouter_multimodal_eval.py` use a fixed configuration block at the top of each file.
- For fresh reruns, review `MODEL_LIST`, `MAX_SAMPLES`, and other fixed settings before execution.
- The active evaluation database can now be switched with `SEMA_EVAL_DB_PATH`.

Recommended run order:

1. Run SEMA batch generation:

```powershell
$env:SEMA_EVAL_DB_PATH="evaluation.db"
python Experiments/Comparation/batch_run_main_pipeline.py
```

2. Run DashScope / Zhipu multimodal baselines:

```powershell
$env:SEMA_EVAL_DB_PATH="evaluation2.db"
python Experiments/Comparation/batch_run_dashscope_multimodal_eval.py
```

If you do not have the Zhipu SDK that provides `from zai import ZhipuAiClient`, remove `glm-4.6v` from `MODEL_LIST` before running.

3. Run OpenRouter multimodal baselines:

```powershell
$env:SEMA_EVAL_DB_PATH="evaluation3.db"
python Experiments/Comparation/batch_run_openrouter_multimodal_eval.py
```

4. Re-run the score-based LLM judge on the bundled wide CSV:

```powershell
python Experiments/Comparation/batch_run_eval_db_llm_judge.py --dataset-root dataset --run-name LLM_JUDGE_RESULT
```

5. Re-run the choose-style LLM judge:

```powershell
python Experiments/Comparation/batch_run_eval_text_choose_judge.py --dataset-root dataset
```

Current comparison artifacts already included in the repo:

- `Experiments/Comparation/evaluation_texts.csv`
- `evaluation/eval_db/evaluation.db`
- `evaluation/eval_db/evaluation2.db`
- `evaluation/eval_db/evaluation3.db`

This means you can re-run the judge scripts directly even if you do not want to regenerate all candidate model outputs from scratch.

## 6. Output Summary

The repository already contains several paper-facing result folders under `Experiments/*/evaluation/results/` and `Experiments/*/paper_figures/`.

In practice:

- Use the experiment entry scripts if you want to regenerate raw records, summaries, and manifests from data.
- Use the figure scripts if you only want to regenerate the paper figures from an existing run directory.

## 7. Citation

If you use this repository, please cite our paper:

- **A Modular Approach to Automated Archery Coaching for Action Quality Assessment and Feedback Generation Using Large Language Models**

The BibTeX entry can be added here after publication.

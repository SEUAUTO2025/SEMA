# AI Agent Coding Guidelines for the SEMA Codebase

## Overview
This codebase supports multimodal research and development, focusing on extending video-language pretraining to multiple modalities using the LanguageBind framework. The project includes components for dataset processing, model training, evaluation, and deployment.

### Key Components
- **Dataset Handling**: Located in `dataset/` and `RAG/Knowledge_Database/LanguageBind_main/data/`, these modules handle various modalities like audio, video, depth, and thermal data.
- **Model Training**: Training scripts and configurations are in `train.py` and `training/`.
- **Evaluation**: Evaluation scripts are in `evaluate_blind_comments.py` and `evaluation_set/`.
- **Demo and Inference**: The `gradio_app.py` script provides a local demo, while `inference.py` handles inference tasks.
- **Utilities**: Shared utilities for data processing and evaluation are in `RAG/Knowledge_Database/LanguageBind_main/a_cls/`, `d_cls/`, and `i_cls/`.

## Developer Workflows

### Setting Up the Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/PKU-YuanGroup/LanguageBind
   ```
2. Install dependencies:
   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   pip install -r requirements.txt
   ```

### Running the Demo
- **Local Demo**: Start the Gradio app to test multimodal alignment.
  ```bash
  python gradio_app.py
  ```

### Training and Evaluation
- **Training**: Use `train.py` for training models. Ensure datasets are preprocessed and available in the `dataset/` directory.
- **Evaluation**: Run `evaluate_blind_comments.py` for evaluation tasks. Results are stored in `report_output/`.

## Project-Specific Conventions
- **Dataset Structure**: Organize datasets under `dataset/` with subfolders for `csv/`, `txt/`, and `video/`.
- **Modular Design**: Follow the modular structure in `RAG/Knowledge_Database/LanguageBind_main/` for adding new modalities or utilities.
- **Logging**: Use the logging utilities in `RTMPose/rtmlib/tools/` for consistent logging across modules.

## Integration Points
- **External Dependencies**: The project relies on PyTorch and Hugging Face libraries for model training and deployment.
- **Cross-Component Communication**: Shared utilities in `RAG/Knowledge_Database/LanguageBind_main/` ensure seamless integration between dataset processing, training, and evaluation.

## Examples
- **Adding a New Dataset**: Place the dataset in `dataset/`, then update `RAG/Knowledge_Database/LanguageBind_main/data/build_datasets.py` to include preprocessing steps.
- **Custom Evaluation**: Modify `evaluate_blind_comments.py` to include new metrics or evaluation criteria.

## Notes
- Refer to `README.md` in `RAG/Knowledge_Database/LanguageBind_main/` for detailed project highlights and updates.
- For issues, consult the `report_output/` directory for logs and debugging information.

---
This document is auto-generated. Please review and update as necessary to ensure accuracy.
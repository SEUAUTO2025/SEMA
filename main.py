import argparse
import os
import sys
from typing import Any, List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RTMPose"))  # Solve the import problem of rtmlib.

from RTMPose.Bone_Feature_Extract import *  # noqa: F401,F403
from Tools.LLMTools.performance_test_tools import *  # noqa: F401,F403
from RAG.Knowledge_Database.AIdbconfig import session, session_en, session_pure
from RAG.Knowledge_Database.RAGFunc import *  # noqa: F401,F403
from RAG.tokenize_search import *  # noqa: F401,F403


DEFAULT_DATASET_ROOT = "dataset"
DEFAULT_VIDEO_NAME = "136.mp4"
DEFAULT_LANGUAGE = "zh"
DEFAULT_PIPELINE = 1
DEFAULT_SUBPIPELINE = 4
DEFAULT_ASSESSMENT_MODEL_NAME = "qwen3-vl-plus"
DEFAULT_RESPONSE_MODEL_NAME = "qwen-plus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the default SEMA single-sample pipeline on one dataset video, "
            "then optionally enter the interactive QA loop."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing video/, txt/, and csv/ subfolders.",
    )
    parser.add_argument(
        "--video-name",
        default=DEFAULT_VIDEO_NAME,
        help="Video file name under <dataset-root>/video, for example 136.mp4.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        choices=["zh", "en"],
        help="Language used by the main pipeline and QA stage.",
    )
    parser.add_argument(
        "--pipeline",
        type=int,
        default=DEFAULT_PIPELINE,
        help="Tokenize_SearchKeyword pipeline id.",
    )
    parser.add_argument(
        "--subpipeline",
        type=int,
        default=DEFAULT_SUBPIPELINE,
        help="Tokenize_SearchKeyword subpipeline id.",
    )
    parser.add_argument(
        "--assessment-model-name",
        default=DEFAULT_ASSESSMENT_MODEL_NAME,
        help="Multimodal model used in the initial action assessment stage.",
    )
    parser.add_argument(
        "--response-model-name",
        default=DEFAULT_RESPONSE_MODEL_NAME,
        help="Text model used for the final coaching response stage.",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip the follow-up interactive QA loop.",
    )
    parser.add_argument(
        "--hide-progress",
        action="store_true",
        help="Disable verbose progress printing inside the pipeline.",
    )
    return parser.parse_args()


def _resolve_project_path(path_text: str) -> str:
    if os.path.isabs(path_text):
        return os.path.abspath(path_text)
    return os.path.abspath(os.path.join(ROOT, path_text))


def _comment_to_keywords(comment: Any) -> List[str]:
    if isinstance(comment, list):
        return [str(item) for item in comment]
    return [item for item in str(comment).split("&") if str(item).strip()]


def main() -> None:
    args = parse_args()

    dataset_path = _resolve_project_path(str(args.dataset_root))
    video_name = str(args.video_name or "").strip()
    if not video_name:
        raise ValueError("--video-name must be a non-empty file name.")

    video_path = os.path.join(dataset_path, "video", video_name)
    txt_path = os.path.join(dataset_path, "txt")
    csv_path = os.path.join(dataset_path, "csv")

    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found: {0}".format(video_path))
    if not os.path.isdir(txt_path):
        raise FileNotFoundError("Ground-truth txt directory not found: {0}".format(txt_path))
    if not os.path.isdir(csv_path):
        raise FileNotFoundError("Ground-truth csv directory not found: {0}".format(csv_path))

    show_progress = not bool(args.hide_progress)
    language = str(args.language).strip()
    assessment_model_name = str(args.assessment_model_name).strip()
    response_model_name = str(args.response_model_name).strip()

    print("dataset_root:", dataset_path)
    print("video_path:", video_path)
    print("language:", language)
    print("assessment_model_name:", assessment_model_name)
    print("response_model_name:", response_model_name)

    scores, comment, retrieved_result, keyword_polarity = Tokenize_SearchKeyword(
        video_path=video_path,
        pipeline=int(args.pipeline),
        subpipeline=int(args.subpipeline),
        language=language,
        show=show_progress,
        return_keyword_polarity=True,
        assessment_model_name=assessment_model_name,
    )
    print(scores)
    print(comment)
    print("Retrieved {0} snippets from knowledge database.".format(len(retrieved_result)))

    keywords = _comment_to_keywords(comment)
    answer = get_response(
        keywords=keywords,
        score_dict=scores,
        retrieved_snippets=retrieved_result,
        keyword_polarity=keyword_polarity,
        language=language,
        model_name=response_model_name,
    )

    txt_label, label, label_total = get_matching_text(video_path, txt_dir=txt_path, csv_dir=csv_path)
    bleu_scores = calculate_bleu(answer, txt_label)
    bert_scores = calculate_BF1_score(answer, txt_label, lang="en")

    print("{0:<15} | {1}".format("Metric", "Score"))
    print("-" * 25)
    for key, value in bleu_scores.items():
        print("BLEU:{0:<15} | {1:.4f}".format(key, value))
    for key, value in bert_scores.items():
        print("BERT:{0:<20} | {1:.4f}".format(key, value))

    print("Scores:")
    print(scores)
    print(label)
    print("Ground Truth Total: {0}, Model Total: {1}".format(label_total, scores["total_score"]))
    print("\nFinal Evaluation:")
    print(answer)

    if args.skip_qa:
        return

    qt_session = session_en if str(language).lower().startswith("en") else session
    qk_session = session_pure

    print("\n" + "=" * 20 + " QA Chat " + "=" * 20)
    print("Ask a question about the current sample. Enter q / quit / exit to stop.")
    while True:
        user_question = input("Question: ").strip()
        if not user_question:
            print("Question is empty. Please try again.")
            continue
        if user_question.lower() in {"q", "quit", "exit"}:
            print("QA finished.")
            break
        qt_answer = answer_archery_question(
            keywords=keywords,
            evaluation_text=answer,
            question=user_question,
            tech_session=qt_session,
            knowledge_session=qk_session,
            language=language,
            top_k_knowledge_title=5,
            top_k_knowledge_chunks=8,
            show=show_progress,
        )
        print("\nAnswer:")
        print(qt_answer)

        follow_up = input("\nAsk another question? (y/n): ").strip().lower()
        if follow_up not in {"y", "yes"}:
            print("QA finished.")
            break


if __name__ == "__main__":
    main()

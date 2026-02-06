import json
from DIR import project_root
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import jieba
from bert_score import score
import logging
from pycocoevalcap.cider.cider import Cider
from deep_translator import DeeplTranslator
import os
from Tools.Exe_dataset.dataset_exe import *

#TODO 接着做实验，大笔记本拉代码，挑数据，compareLLM，统计分布（有待优化）的逻辑重写成函数，论文里没做的实验接着做

def calculate_BF1_score(reference_text: str, candidate_text: str, lang="zh"):
    """
    Calculate BERTScore (P, R, F1) for two texts
    :param reference_text: Reference text
    :param candidate_text: Generated text
    :param lang: Language setting, "zh" for Chinese, "en" for English
    :return: Dictionary containing F1, Precision, Recall
    """
    cands = [candidate_text]
    refs = [reference_text]

    P, R, F1 = score(cands, refs, lang=lang, verbose=False)

    return {
        "BF1 (Semantic Similarity)": F1.item(),
        "Precision": P.item(),
        "Recall": R.item()
    }


def calculate_bleu(reference_text: str, candidate_text: str):
    """
    Calculate BLEU scores for two pieces of English text
    :param reference_text: Reference text (Ground Truth)
    :param candidate_text: Model-generated text
    :return: Dictionary containing various BLEU metrics
    """
    ref_tokens = nltk.word_tokenize(reference_text.lower())
    cand_tokens = nltk.word_tokenize(candidate_text.lower())

    references = [ref_tokens]

    smooth = SmoothingFunction().method1

    weights = {
        'BLEU-Total': (0.25, 0.25, 0.25, 0.25),
        'BLEU-1 (Word-level)': (1, 0, 0, 0),
        'BLEU-2 (Phrase-level)': (0.5, 0.5, 0, 0),
        'BLEU-4 (Sentence-level)': (0.25, 0.25, 0.25, 0.25)
    }

    results = {}
    for name, weight in weights.items():
        score = sentence_bleu(references, cand_tokens, weights=weight, smoothing_function=smooth)
        results[name] = score

    return results


def compute_cider_score(reference_text: str, candidate_text: str):
    """
    Compute CIDEr-D score for two English text strings
    :param reference_text: Ground Truth (Reference)
    :param candidate_text: Model Prediction (Candidate)
    :return: CIDEr-D mean score
    """
    ref_processed = reference_text.lower()
    cand_processed = candidate_text.lower()

    gts = {'0': [ref_processed]}
    res = {'0': [cand_processed]}

    cider_scorer = Cider()

    mean_score, individual_scores = cider_scorer.compute_score(gts, res)

    return mean_score


def compute_meteor_score(reference_text: str, candidate_text: str):
    """
    Compute METEOR score for two English text strings
    :param reference_text: Ground Truth (Reference)
    :param candidate_text: Model Prediction (Candidate)
    :return: METEOR score (float)
    """
    ref_tokens = word_tokenize(reference_text.lower())
    cand_tokens = word_tokenize(candidate_text.lower())

    references = [ref_tokens]

    score = meteor_score(references, cand_tokens)

    return score


def batch_translate_txt(input_dir, output_dir, auth_key):
    """
    Batch translate txt files to English using DeepL

    :param input_dir: Path to folder containing original txt files
    :param output_dir: Path to folder where translated files will be saved
    :param auth_key: Your DeepL API Authentication Key
    """
    translator = DeeplTranslator(api_key=auth_key, source="zh", target="en", use_free_api=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    if not files:
        print("No text files found in the input directory.")
        return

    print(f"Starting translation for {len(files)} files...")

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                merged_content = content.replace('\n', '').replace('\r', '')
                translated_text = translator.translate(merged_content)
                translated_text = translated_text.replace('\n', '').replace('\r', '')
            if not content:
                print(f"Skipping empty file: {file_name}")
                continue

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)

            print(f"Successfully translated: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("Batch translation completed.")


def get_matching_text(video_path, txt_dir, csv_dir):
    """
    Find matching txt file based on video path and extract content
    :param video_path: Full path to video file
    :param txt_dir: Directory containing text files
    :param csv_dir: Directory containing CSV files
    :return: Processed text content (newlines removed), returns None if file doesn't exist
    """
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    target_txt_path = os.path.join(txt_dir, f"{file_name}.txt")
    csv_path = os.path.join(csv_dir, f"{file_name}.csv")
    _, label, label_total = load_single_csv_with_multipart_labels(csv_path, max_frames=124)

    if os.path.exists(target_txt_path):
        try:
            with open(target_txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                clean_content = content.replace('\n', '').replace('\r', '')
                return clean_content, label, label_total
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    else:
        print(f"Text file not found: {target_txt_path}")
        return None
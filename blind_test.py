# -*- coding: utf-8 -*-
"""
Teacher evaluation script - Generate blind evaluation CSV
Batch process videos, calculate MAE, save results to txt, generate blind evaluation CSV
"""
import sys, os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RTMPose"))
sys.path.insert(0, os.path.join(ROOT, "RAG", "Knowledge_Database"))

import numpy as np
import torch
from RAG.tokenize_search import *
from RAG.Knowledge_Database.RAGFunc import *
from Tools.Gen_dataset.create_dummy_dataset import load_single_csv_with_multipart_labels
import json
from tqdm import tqdm
import datetime
import csv
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_dir = r"D:\\pythonWorks\\SpatialTemporalAttentionGCN-master\\SpatialTemporalAttentionGCN-master\\whole_dataset_txt!!!!!!!!"
video_dir = os.path.join(base_dir, "video")
txt_dir = os.path.join(base_dir, "txt")
csv_dir = os.path.join(base_dir, "csv")

parts = ['head', 'hand', 'feet', 'arm', 'body']

def get_matching_text(video_path, txt_dir, csv_dir):
    """Get corresponding txt content and csv labels based on video path"""
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    target_txt_path = os.path.join(txt_dir, f"{file_name}.txt")
    csv_path_file = os.path.join(csv_dir, f"{file_name}.csv")
    
    if not os.path.exists(csv_path_file):
        print(f"CSV not found: {csv_path_file}")
        return None, None, None

    try:
        _, label, label_total = load_single_csv_with_multipart_labels(csv_path_file, max_frames=124)
    except Exception as e:
        print(f"Error loading CSV {csv_path_file}: {e}")
        return None, None, None
        
    if os.path.exists(target_txt_path):
        try:
            with open(target_txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                clean_content = content.replace('\n', '').replace('\r', '')
                return clean_content, label, label_total
        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None, None
    else:
        print(f"Text file not found: {target_txt_path}")
        return None, None, None

if __name__ == '__main__':
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    print(f"Found {len(video_files)} videos in {video_dir}")

    score_mae_accumulator = {
        'total': [], 'head': [], 'hand': [], 'body': [], 'feet': [], 'arm': []
    }
    
    blind_eval_data = []

    output_txt_path = os.path.join(base_dir, "evaluation_results_teacher.txt")
    print(f"Saving results to: {output_txt_path}")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"Evaluation Started: {datetime.datetime.now()}\n")
        f_out.write("=" * 50 + "\n\n")

        for video_file in tqdm(video_files, desc="Processing Videos"):
            video_path_full = os.path.join(video_dir, video_file)
            
            try:
                keywords = Tokenize_SearchKeyword(video_path=video_path_full, pipeline=1, language='zh')
                
                answer_json = get_response_ali(keywords, pipeline=1, math_feature=None)
                
                try:
                    data = json.loads(answer_json)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for {video_file}")
                    f_out.write(f"ERROR: Failed to parse JSON for {video_file}\n")
                    continue
                    
                txt_label, label_dict_gt, label_total_gt = get_matching_text(video_path_full, txt_dir, csv_dir)
                
                if txt_label is None or label_dict_gt is None:
                    print(f"Skipping {video_file} due to missing data.")
                    f_out.write(f"SKIPPED: {video_file} (Missing Data)\n")
                    continue

                generated_comment = data.get("comment", "")
                
                scores_pred = {
                    "total": float(data.get("total_score", 0)),
                    "head": float(data.get("head_score", 0)),
                    "hand": float(data.get("hand_score", 0)),
                    "body": float(data.get("torso_score", 0)),
                    "feet": float(data.get("foot_score", 0)),
                    "arm": float(data.get("arm_score", 0))
                }

                scores_gt = {"total": float(label_total_gt)}
                for part in ['head', 'hand', 'body', 'feet', 'arm']:
                    scores_gt[part] = float(label_dict_gt.get(part, 0))

                score_mae_accumulator['total'].append(abs(scores_pred['total'] - scores_gt['total']))
                for part in ['head', 'hand', 'body', 'feet', 'arm']:
                    score_mae_accumulator[part].append(abs(scores_pred[part] - scores_gt[part]))

                # 写入txt
                f_out.write(f"--- Sample: {video_file} ---\n")
                f_out.write(f"Ground Truth Comment: {txt_label}\n")
                f_out.write(f"Predicted Comment: {generated_comment}\n")
                f_out.write("Scores (Pred vs GT):\n")
                f_out.write(f"  Total: {scores_pred['total']} / {scores_gt['total']}\n")
                for part in parts:
                    f_out.write(f"  {part.capitalize()}: {scores_pred[part]} / {scores_gt.get(part, 'N/A')}\n")
                f_out.write("\n")
                f_out.flush()
                
                # 收集盲评数据
                blind_eval_data.append((video_file, txt_label, generated_comment))

            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                f_out.write(f"ERROR processing {video_file}: {str(e)}\n")
                continue

        f_out.write("=" * 50 + "\n")
        f_out.write("FINAL SUMMARY - Score MAE\n")
        f_out.write("=" * 50 + "\n")
        
        if score_mae_accumulator['total']:
            f_out.write(f"Samples processed: {len(score_mae_accumulator['total'])}\n\n")
            for part, errors in score_mae_accumulator.items():
                if errors:
                    f_out.write(f"{part}: {np.mean(errors):.4f}\n")
                    print(f"MAE {part}: {np.mean(errors):.4f}")
        else:
            f_out.write("No samples were successfully processed.\n")

    if blind_eval_data:
        blind_csv_path = os.path.join(base_dir, "blind_evaluation.csv")
        answer_csv_path = os.path.join(base_dir, "blind_evaluation_answer.csv")
        
        with open(blind_csv_path, 'w', encoding='utf-8-sig', newline='') as f_blind, \
             open(answer_csv_path, 'w', encoding='utf-8-sig', newline='') as f_answer:
            
            blind_writer = csv.writer(f_blind)
            answer_writer = csv.writer(f_answer)
            
            blind_writer.writerow(['Video Name', 'Evaluation A', 'Evaluation B', 'Teacher Choice (A/B)'])
            answer_writer.writerow(['Video Name', 'Evaluation A Source', 'Evaluation B Source', 'Swapped'])
            
            for video_name, gt_comment, pred_comment in blind_eval_data:
                swap = random.choice([True, False])
                
                if swap:
                    text_a, text_b = pred_comment, gt_comment
                    source_a, source_b = 'Model Prediction', 'Ground Truth'
                else:
                    text_a, text_b = gt_comment, pred_comment
                    source_a, source_b = 'Ground Truth', 'Model Prediction'
                
                blind_writer.writerow([video_name, text_a, text_b, ''])
                answer_writer.writerow([video_name, source_a, source_b, 'Yes' if swap else 'No'])
        
        print(f"\nBlind evaluation CSV generated: {blind_csv_path}")
        print(f"Answer file generated: {answer_csv_path}")

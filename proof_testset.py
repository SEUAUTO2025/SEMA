"""
Batch generate evaluation texts for all samples (with biomechanical features)
Add new column to existing blind_evaluation.csv: Evaluation SEAM_RTMPOSE
Calculate MAE and save to txt file
"""
import sys, os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RTMPose"))
sys.path.insert(0, os.path.join(ROOT, "RAG", "Knowledge_Database"))

import torch
import numpy as np
import json
import csv
from tqdm import tqdm
from RTMPose.Bone_Feature_Extract import Keypoint_Extract, cal_math_features, extract_action_features
from RAG.tokenize_search import Tokenize_SearchKeyword
from RAG.Knowledge_Database.RAGFunc import get_response_ali
from Tools.Gen_dataset.create_dummy_dataset import load_single_csv_with_multipart_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = r"D:\pythonWorks\SpatialTemporalAttentionGCN-master\SpatialTemporalAttentionGCN-master\whole_dataset_txt!!!!!!!!"
VIDEO_DIR = os.path.join(BASE_DIR, "video")
TXT_DIR = os.path.join(BASE_DIR, "txt")
CSV_DIR = os.path.join(BASE_DIR, "csv")
INPUT_CSV = os.path.join(BASE_DIR, "blind_evaluation.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "blind_evaluation.csv")
MAE_TXT = os.path.join(BASE_DIR, "mae_statistics_SEAM_RTMPOSE.txt")

def get_matching_labels(video_name, csv_dir):
    """Get score labels based on video name"""
    file_name = os.path.splitext(video_name)[0]
    csv_path = os.path.join(csv_dir, f"{file_name}.csv")
    
    if os.path.exists(csv_path):
        try:
            _, label_dict, label_total = load_single_csv_with_multipart_labels(csv_path, max_frames=124)
            return label_dict, label_total
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None, None
    return None, None


def process_single_video(video_path):
    """Process single video: extract features, generate evaluation"""
    try:
        all_data = Keypoint_Extract(video_path)
        math_feature = cal_math_features(all_data)
        math_feature_input = extract_action_features(math_feature)
        
        keywords = Tokenize_SearchKeyword(video_path=video_path, pipeline=1, language='zh')
        
        answer = get_response_ali(keywords, pipeline=2, math_feature=math_feature_input)
        
        data = json.loads(answer)
        
        scores_pred = {
            "total": float(data.get("total_score", 0)),
            "head": float(data.get("head_score", 0)),
            "hand": float(data.get("hand_score", 0)),
            "body": float(data.get("torso_score", 0)),
            "feet": float(data.get("foot_score", 0)),
            "arm": float(data.get("arm_score", 0))
        }
        
        generated_comment = data.get("comment", "")
        
        return {
            'pred_comment': generated_comment,
            'scores_pred': scores_pred
        }
        
    except Exception as e:
        print(f"Error processing video {os.path.basename(video_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("Batch Generate Evaluation Texts (SEAM+RTMPOSE)")
    print("="*80)
    
    print(f"\nReading existing CSV file: {INPUT_CSV}")
    existing_data = []
    with open(INPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_data.append(row)
    
    print(f"Existing CSV contains {len(existing_data)} records")
    
    video_to_result = {}
    mae_accumulator = {
        'total': [], 'head': [], 'hand': [], 'body': [], 'feet': [], 'arm': []
    }
    mae_details = []
    
    print("\nStarting video processing...")
    for row in tqdm(existing_data, desc="Generating evaluations"):
        video_name = row['Video Name']
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        if not os.path.exists(video_path):
            print(f"\nWarning: Video file not found {video_name}")
            video_to_result[video_name] = {'pred_comment': '', 'scores_pred': None}
            continue
        
        result = process_single_video(video_path)
        
        if result is not None:
            video_to_result[video_name] = result
            
            label_dict, label_total = get_matching_labels(video_name, CSV_DIR)
            
            if label_dict is not None:
                scores_gt = {
                    "total": float(label_total),
                    "head": float(label_dict.get('head', 0)),
                    "hand": float(label_dict.get('hand', 0)),
                    "body": float(label_dict.get('body', 0)),
                    "feet": float(label_dict.get('feet', 0)),
                    "arm": float(label_dict.get('arm', 0))
                }
                
                mae = {}
                for part in ['total', 'head', 'hand', 'body', 'feet', 'arm']:
                    mae[part] = abs(result['scores_pred'][part] - scores_gt[part])
                    mae_accumulator[part].append(mae[part])
                
                mae_details.append({
                    'video_name': video_name,
                    'mae': mae,
                    'scores_pred': result['scores_pred'],
                    'scores_gt': scores_gt
                })
        else:
            video_to_result[video_name] = {'pred_comment': '', 'scores_pred': None}
    
    print(f"\nAdding new column to CSV...")
    for row in existing_data:
        video_name = row['Video Name']
        result = video_to_result.get(video_name, {'pred_comment': ''})
        row['Evaluation SEAM_RTMPOSE'] = result['pred_comment']
    
    print(f"\nSaving updated CSV to: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = list(existing_data[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)
    
    print(f"✓ CSV file updated, new column added: Evaluation SEAM_RTMPOSE")
    
    print("\n" + "="*80)
    print("MAE Statistics (SEAM+RTMPOSE with biomechanical features)")
    print("="*80)
    
    mae_stats = {}
    for part, errors in mae_accumulator.items():
        if errors:
            mae_stats[part] = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors)
            }
            print(f"{part:8s}: MAE = {mae_stats[part]['mean']:.4f} ± {mae_stats[part]['std']:.4f}")
    
    print(f"\nSaving MAE statistics to: {MAE_TXT}")
    with open(MAE_TXT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MAE Statistics (SEAM+RTMPOSE with biomechanical features)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Successfully processed samples: {len(mae_details)}\n")
        f.write(f"Total samples: {len(existing_data)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Average MAE Statistics\n")
        f.write("-"*80 + "\n")
        for part, stats in mae_stats.items():
            f.write(f"\n{part}:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n")
            f.write(f"  Min: {stats['min']:.4f}\n")
            f.write(f"  Max: {stats['max']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Detailed MAE for Each Sample\n")
        f.write("="*80 + "\n\n")
        
        for detail in mae_details:
            f.write(f"Video: {detail['video_name']}\n")
            f.write(f"  Predicted scores: total={detail['scores_pred']['total']:.2f}, "
                   f"head={detail['scores_pred']['head']:.2f}, "
                   f"hand={detail['scores_pred']['hand']:.2f}, "
                   f"body={detail['scores_pred']['body']:.2f}, "
                   f"feet={detail['scores_pred']['feet']:.2f}, "
                   f"arm={detail['scores_pred']['arm']:.2f}\n")
            f.write(f"  Ground truth scores: total={detail['scores_gt']['total']:.2f}, "
                   f"head={detail['scores_gt']['head']:.2f}, "
                   f"hand={detail['scores_gt']['hand']:.2f}, "
                   f"body={detail['scores_gt']['body']:.2f}, "
                   f"feet={detail['scores_gt']['feet']:.2f}, "
                   f"arm={detail['scores_gt']['arm']:.2f}\n")
            f.write(f"  MAE: total={detail['mae']['total']:.4f}, "
                   f"head={detail['mae']['head']:.4f}, "
                   f"hand={detail['mae']['hand']:.4f}, "
                   f"body={detail['mae']['body']:.4f}, "
                   f"feet={detail['mae']['feet']:.4f}, "
                   f"arm={detail['mae']['arm']:.4f}\n\n")
    
    print(f"\n✓ MAE statistics saved to: {MAE_TXT}")
    print("\n" + "="*80)
    print("Completed!")
    print("="*80)
    print(f"✓ Updated CSV: {OUTPUT_CSV}")
    print(f"  - New column: Evaluation SEAM_RTMPOSE")
    print(f"✓ MAE statistics file: {MAE_TXT}")
    print(f"  - Contains average MAE and detailed information for each sample")


if __name__ == '__main__':
    main()

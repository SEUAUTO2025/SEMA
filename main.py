import sys,os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RTMPose"))
sys.path.insert(0, os.path.join(ROOT, "RAG","Knowledge_Database"))

import matplotlib as mpl
mpl.use('Agg')
import torch
from RTMPose.Bone_Feature_Extract import *
from RAG.tokenize_search import *
from RAG.Knowledge_Database.RAGFunc import *
from Tools.Exe_dataset.dataset_exe import load_single_csv_with_multipart_labels
from Tools.LLMTools.performance_test_tools import *
import json
from DIR import project_root

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
video_path = r""
txt_path = r""
csv_path = r""
parts = ['head','hand','foot','arm','torso']

gt_score_dict = {
    'total': [],
    'head': [],
    'hand': [],
    'torso': [],
    'foot': [],
    'arm': []
}
pred_score_dict = {
    'total': [],
    'head': [],
    'hand': [],
    'torso': [],
    'foot': [],
    'arm': []
}

if __name__ == '__main__':
    all_data = Keypoint_Extract(video_path)
    _, math_feature = Extract_Bodypart_Data(all_data, 2560, 1440)
    math_feature_input = extract_action_features(math_feature)

    keywords = Tokenize_SearchKeyword(video_path=video_path, pipeline=1, language='en')
    answer = get_response_ali(keywords, pipeline=1, math_feature=None)

    data = json.loads(answer)
    
    scores = {
        "total": data.get("total_score"),
        "head": data.get("head_score"),
        "hand": data.get("hand_score"),
        "torso": data.get("torso_score"),
        "foot": data.get("foot_score"),
        "arm": data.get("arm_score")
    }
    
    txt_label, label, label_total = get_matching_text(video_path, txt_dir=txt_path, csv_dir=csv_path)
    bleu_scores = calculate_bleu(data.get("comment"), txt_label)
    bert_scores = calculate_BF1_score(data.get("comment"), txt_label, lang="en")
    CIDEr_score = compute_cider_score(txt_label, data.get("comment"))
    meteors_score = compute_meteor_score(txt_label, data.get("comment"))
    
    print(f"{'Metric':<15} | {'Score'}")
    print("-" * 25)
    for k, v in bleu_scores.items():
        print(f"BLEU:{k:<15} | {v:.4f}")
    for k, v in bert_scores.items():
        print(f"BERT:{k:<20} | {v:.4f}")
    print(f"{'CIDEr-D':<15} | {CIDEr_score:.4f}")
    print(f"{'METEOR':<15} | {meteors_score:.4f}")

    print("Scores:")
    print(scores)
    print(label)
    print(f'Ground Truth Total: {label_total}, Model Total: {scores["total"]}')
    

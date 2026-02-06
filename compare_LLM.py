"""
Compare LLM-generated comments with framework-generated comments
Generate comments using ChatGPT-5, Qwen-vl-plus, and the framework
Evaluate all three using Qwen and DeepSeek models
"""
import sys
import os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RTMPose"))
sys.path.insert(0, os.path.join(ROOT, "RAG", "Knowledge_Database"))

import torch
import json
import re
import cv2
import base64
import numpy as np
from openai import OpenAI
from RTMPose.Bone_Feature_Extract import *
from RAG.tokenize_search import *
from RAG.Knowledge_Database.RAGFunc import *
from Tools.Exe_dataset.dataset_exe import load_single_csv_with_multipart_labels

# Configuration
VIDEO_PATH = r""  # TODO: Set video path
TXT_PATH = r""    # TODO: Set txt directory path
CSV_PATH = r""    # TODO: Set csv directory path

# Evaluation models
EVAL_MODEL_1 = "qwen-vl-plus"
EVAL_MODEL_2 = "deepseek-v3.2"

# Generation models
CHATGPT_MODEL = "gpt-5"

# Prompts
CHATGPT_PROMPT = """
Please evaluate the quality of the archer's movements based on the extracted video frames. The evaluation text should follow this format: “The student's correct movements are as follows... The student's incorrect movements are as follows...”
"""

def extract_frames_from_video(video_path, num_frames=124):
    """
    Extract frames from video and convert to base64 format for ChatGPT
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 124)
    
    Returns:
        List of base64 encoded frame strings
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames_base64 = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Failed to read frame {idx}")
            continue
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        frames_base64.append(frame_base64)
    
    cap.release()
    
    print(f"  Extracted {len(frames_base64)} frames from video")
    return frames_base64


def get_matching_text(video_path, txt_dir, csv_dir):
    """
    Find matching txt and csv files based on video path
    
    Args:
        video_path: Full path to video file
        txt_dir: Directory containing text files
        csv_dir: Directory containing CSV files
    
    Returns:
        Tuple of (ground_truth_text, labels, label_total)
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
            return None, label, label_total
    else:
        print(f"Text file not found: {target_txt_path}")
        return None, label, label_total


def generate_comment_chatgpt(video_path, prompt):
    """
    Generate archery evaluation comment using ChatGPT with video frames
    
    Args:
        video_path: Path to video file
        prompt: Prompt template for ChatGPT
    
    Returns:
        Generated comment text
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
    )
    
    # Extract frames from video
    print("  Extracting frames from video...")
    frames_base64 = extract_frames_from_video(video_path, num_frames=124)
    
    if not frames_base64:
        print("  Error: No frames extracted from video")
        return ""
    
    # Build message content with text prompt and frames
    message_content = [
        {
            "type": "text",
            "text": prompt
        }
    ]
    
    # Add frames as images (sample every N frames to avoid token limit)
    # For 124 frames, sample every 4th frame to get ~31 frames
    sample_interval = max(1, len(frames_base64) // 31)
    sampled_frames = frames_base64[::sample_interval]
    
    print(f"  Sending {len(sampled_frames)} frames to ChatGPT...")
    
    for frame_base64 in sampled_frames:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_base64}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=[
                {"role": "user", "content": message_content}
            ],
            temperature=0.7,
        )
        
        comment = response.choices[0].message.content.strip()
        return comment
        
    except Exception as e:
        print(f"  ChatGPT API error: {e}")
        return ""


def generate_comment_qwen(video_path):
    """
    Generate archery evaluation comment using Qwen-plus
    
    Args:
        video_path: Path to video file
        csv_path: Path to CSV file
        txt_path: Path to txt file (ground truth)
        prompt: Prompt template for Qwen
    
    Returns:
        Generated comment text
    """
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    comment = get_video_ori_keywords_ali(video_path)

    return comment

def generate_comment_framework(video_path):
    """
    Generate archery evaluation comment using the framework (RAG + LLM)
    
    Args:
        video_path: Path to video file
    
    Returns:
        Generated comment text
    """
    try:
        # Extract features
        all_data = Keypoint_Extract(video_path)
        _, math_feature = Extract_Bodypart_Data(all_data, 2560, 1440)
        math_feature_input = extract_action_features(math_feature)
        
        # Get keywords and generate response
        keywords = Tokenize_SearchKeyword(video_path=video_path, pipeline=1, language='en')
        answer = get_response_ali(keywords, pipeline=2, math_feature=math_feature_input)
        
        # Parse JSON response
        data = json.loads(answer)
        comment = data.get("comment", "")
        
        return comment
        
    except Exception as e:
        print(f"Framework generation error: {e}")
        return ""


def evaluate_comment(ground_truth: str, generated_comment: str, model_name: str) -> dict:
    """
    Evaluate generated comment using specified model
    
    Args:
        ground_truth: Teacher's standard evaluation
        generated_comment: Generated evaluation to be assessed
        model_name: Model name for evaluation (qwen-plus or deepseek-v3.2)
    
    Returns:
        Dictionary with scores: {"Accuracy": score, "Professionalism": score, "Practicality": score}
    """
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    prompt = f"""You are a professional archery coach evaluation expert. Here are two archery posture evaluation texts:
    【Teacher's Standard Evaluation (Full Score Baseline)】:
    {ground_truth}
    
    【Generated Evaluation to be Assessed】:
    {generated_comment}
    
    Please score the 【Generated Evaluation】 on the following three dimensions (5-point Likert scale, 1 lowest, 5 highest, decimals allowed):
    
    1. **Accuracy**: Are the technical points and action descriptions accurate? How consistent with the standard evaluation?
    2. **Professionalism**: Are the terms and expressions professional? Does it reflect archery domain expertise?
    3. **Practicality**: How valuable is the guidance for students? Are improvement suggestions specific and actionable?
    
    **Scoring Standards**:
    - 5: Excellent, fully meets standard evaluation level
    - 4: Good, mostly meets standard
    - 3: Average, basically meets standard but with obvious deficiencies
    - 2: Poor, significant gap from standard
    - 1: Very poor, seriously deviates from standard
    
    **Important**: Output only JSON format scoring results, no other content. Format:
    {{"Accuracy": score, "Professionalism": score, "Practicality": score}}
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional archery coach evaluation expert, responsible for objective and fair scoring of archery posture evaluation texts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
        else:
            print(f"Unable to parse evaluation result: {result_text}")
            return {"Accuracy": 0, "Professionalism": 0, "Practicality": 0}
            
    except Exception as e:
        print(f"Evaluation API error: {e}")
        return {"Accuracy": 0, "Professionalism": 0, "Practicality": 0}


def main():
    """
    Main function to compare three generation methods
    """
    print("="*80)
    print("LLM Comment Generation Comparison")
    print("="*80)
    
    # Validate paths
    if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
        print("Error: Please set valid VIDEO_PATH")
        return
    
    if not TXT_PATH or not os.path.exists(TXT_PATH):
        print("Error: Please set valid TXT_PATH")
        return
    
    if not CSV_PATH or not os.path.exists(CSV_PATH):
        print("Error: Please set valid CSV_PATH")
        return
    
    # Get ground truth
    print("\n[1/7] Loading ground truth...")
    gt_text, labels, label_total = get_matching_text(VIDEO_PATH, TXT_PATH, CSV_PATH)
    if not gt_text:
        print("Error: Ground truth not found")
        return
    print(f"Ground truth loaded: {len(gt_text)} characters")
    
    # Generate comments using three methods
    print("\n[2/7] Generating comment using ChatGPT-4o-thinking...")
    chatgpt_comment = generate_comment_chatgpt(VIDEO_PATH, CHATGPT_PROMPT)
    print(f"ChatGPT comment generated: {len(chatgpt_comment)} characters")
    
    print("\n[3/7] Generating comment using Qwen-plus...")
    qwen_comment = generate_comment_qwen(VIDEO_PATH)
    print(f"Qwen comment generated: {len(qwen_comment)} characters")
    
    print("\n[4/7] Generating comment using Framework (RAG + LLM)...")
    framework_comment = generate_comment_framework(VIDEO_PATH)
    print(f"Framework comment generated: {len(framework_comment)} characters")
    
    # Evaluate ChatGPT comment
    print("\n[5/7] Evaluating ChatGPT comment...")
    print(f"  Using {EVAL_MODEL_1}...")
    chatgpt_scores_m1 = evaluate_comment(gt_text, chatgpt_comment, EVAL_MODEL_1)
    print(f"    Accuracy: {chatgpt_scores_m1['Accuracy']}, Professionalism: {chatgpt_scores_m1['Professionalism']}, Practicality: {chatgpt_scores_m1['Practicality']}")
    
    print(f"  Using {EVAL_MODEL_2}...")
    chatgpt_scores_m2 = evaluate_comment(gt_text, chatgpt_comment, EVAL_MODEL_2)
    print(f"    Accuracy: {chatgpt_scores_m2['Accuracy']}, Professionalism: {chatgpt_scores_m2['Professionalism']}, Practicality: {chatgpt_scores_m2['Practicality']}")
    
    # Evaluate Qwen comment
    print("\n[6/7] Evaluating Qwen comment...")
    print(f"  Using {EVAL_MODEL_1}...")
    qwen_scores_m1 = evaluate_comment(gt_text, qwen_comment, EVAL_MODEL_1)
    print(f"    Accuracy: {qwen_scores_m1['Accuracy']}, Professionalism: {qwen_scores_m1['Professionalism']}, Practicality: {qwen_scores_m1['Practicality']}")
    
    print(f"  Using {EVAL_MODEL_2}...")
    qwen_scores_m2 = evaluate_comment(gt_text, qwen_comment, EVAL_MODEL_2)
    print(f"    Accuracy: {qwen_scores_m2['Accuracy']}, Professionalism: {qwen_scores_m2['Professionalism']}, Practicality: {qwen_scores_m2['Practicality']}")
    
    # Evaluate Framework comment
    print("\n[7/7] Evaluating Framework comment...")
    print(f"  Using {EVAL_MODEL_1}...")
    framework_scores_m1 = evaluate_comment(gt_text, framework_comment, EVAL_MODEL_1)
    print(f"    Accuracy: {framework_scores_m1['Accuracy']}, Professionalism: {framework_scores_m1['Professionalism']}, Practicality: {framework_scores_m1['Practicality']}")
    
    print(f"  Using {EVAL_MODEL_2}...")
    framework_scores_m2 = evaluate_comment(gt_text, framework_comment, EVAL_MODEL_2)
    print(f"    Accuracy: {framework_scores_m2['Accuracy']}, Professionalism: {framework_scores_m2['Professionalism']}, Practicality: {framework_scores_m2['Practicality']}")
    
    # Calculate averages
    chatgpt_avg_m1 = (chatgpt_scores_m1['Accuracy'] + chatgpt_scores_m1['Professionalism'] + chatgpt_scores_m1['Practicality']) / 3
    chatgpt_avg_m2 = (chatgpt_scores_m2['Accuracy'] + chatgpt_scores_m2['Professionalism'] + chatgpt_scores_m2['Practicality']) / 3
    
    qwen_avg_m1 = (qwen_scores_m1['Accuracy'] + qwen_scores_m1['Professionalism'] + qwen_scores_m1['Practicality']) / 3
    qwen_avg_m2 = (qwen_scores_m2['Accuracy'] + qwen_scores_m2['Professionalism'] + qwen_scores_m2['Practicality']) / 3
    
    framework_avg_m1 = (framework_scores_m1['Accuracy'] + framework_scores_m1['Professionalism'] + framework_scores_m1['Practicality']) / 3
    framework_avg_m2 = (framework_scores_m2['Accuracy'] + framework_scores_m2['Professionalism'] + framework_scores_m2['Practicality']) / 3
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n【ChatGPT-4o-thinking】")
    print(f"  {EVAL_MODEL_1} Scores:")
    print(f"    Accuracy: {chatgpt_scores_m1['Accuracy']:.2f}")
    print(f"    Professionalism: {chatgpt_scores_m1['Professionalism']:.2f}")
    print(f"    Practicality: {chatgpt_scores_m1['Practicality']:.2f}")
    print(f"    Average: {chatgpt_avg_m1:.2f}")
    print(f"  {EVAL_MODEL_2} Scores:")
    print(f"    Accuracy: {chatgpt_scores_m2['Accuracy']:.2f}")
    print(f"    Professionalism: {chatgpt_scores_m2['Professionalism']:.2f}")
    print(f"    Practicality: {chatgpt_scores_m2['Practicality']:.2f}")
    print(f"    Average: {chatgpt_avg_m2:.2f}")
    
    print(f"\n【Qwen-plus】")
    print(f"  {EVAL_MODEL_1} Scores:")
    print(f"    Accuracy: {qwen_scores_m1['Accuracy']:.2f}")
    print(f"    Professionalism: {qwen_scores_m1['Professionalism']:.2f}")
    print(f"    Practicality: {qwen_scores_m1['Practicality']:.2f}")
    print(f"    Average: {qwen_avg_m1:.2f}")
    print(f"  {EVAL_MODEL_2} Scores:")
    print(f"    Accuracy: {qwen_scores_m2['Accuracy']:.2f}")
    print(f"    Professionalism: {qwen_scores_m2['Professionalism']:.2f}")
    print(f"    Practicality: {qwen_scores_m2['Practicality']:.2f}")
    print(f"    Average: {qwen_avg_m2:.2f}")
    
    print(f"\n【Framework (RAG + LLM)】")
    print(f"  {EVAL_MODEL_1} Scores:")
    print(f"    Accuracy: {framework_scores_m1['Accuracy']:.2f}")
    print(f"    Professionalism: {framework_scores_m1['Professionalism']:.2f}")
    print(f"    Practicality: {framework_scores_m1['Practicality']:.2f}")
    print(f"    Average: {framework_avg_m1:.2f}")
    print(f"  {EVAL_MODEL_2} Scores:")
    print(f"    Accuracy: {framework_scores_m2['Accuracy']:.2f}")
    print(f"    Professionalism: {framework_scores_m2['Professionalism']:.2f}")
    print(f"    Practicality: {framework_scores_m2['Practicality']:.2f}")
    print(f"    Average: {framework_avg_m2:.2f}")
    
    # Save results to file
    output_file = "llm_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LLM Comment Generation Comparison Results\n")
        f.write("="*80 + "\n\n")
        
        f.write("Ground Truth:\n")
        f.write(gt_text + "\n\n")
        
        f.write("-"*80 + "\n")
        f.write("ChatGPT-5 Generated Comment:\n")
        f.write(chatgpt_comment + "\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Qwen-vl-plus Generated Comment:\n")
        f.write(qwen_comment + "\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Framework Generated Comment:\n")
        f.write(framework_comment + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("EVALUATION SCORES\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"ChatGPT-5:\n")
        f.write(f"  {EVAL_MODEL_1}: Accuracy={chatgpt_scores_m1['Accuracy']:.2f}, Professionalism={chatgpt_scores_m1['Professionalism']:.2f}, Practicality={chatgpt_scores_m1['Practicality']:.2f}, Avg={chatgpt_avg_m1:.2f}\n")
        f.write(f"  {EVAL_MODEL_2}: Accuracy={chatgpt_scores_m2['Accuracy']:.2f}, Professionalism={chatgpt_scores_m2['Professionalism']:.2f}, Practicality={chatgpt_scores_m2['Practicality']:.2f}, Avg={chatgpt_avg_m2:.2f}\n\n")
        
        f.write(f"Qwen-vl-plus:\n")
        f.write(f"  {EVAL_MODEL_1}: Accuracy={qwen_scores_m1['Accuracy']:.2f}, Professionalism={qwen_scores_m1['Professionalism']:.2f}, Practicality={qwen_scores_m1['Practicality']:.2f}, Avg={qwen_avg_m1:.2f}\n")
        f.write(f"  {EVAL_MODEL_2}: Accuracy={qwen_scores_m2['Accuracy']:.2f}, Professionalism={qwen_scores_m2['Professionalism']:.2f}, Practicality={qwen_scores_m2['Practicality']:.2f}, Avg={qwen_avg_m2:.2f}\n\n")
        
        f.write(f"Framework (RAG + LLM):\n")
        f.write(f"  {EVAL_MODEL_1}: Accuracy={framework_scores_m1['Accuracy']:.2f}, Professionalism={framework_scores_m1['Professionalism']:.2f}, Practicality={framework_scores_m1['Practicality']:.2f}, Avg={framework_avg_m1:.2f}\n")
        f.write(f"  {EVAL_MODEL_2}: Accuracy={framework_scores_m2['Accuracy']:.2f}, Professionalism={framework_scores_m2['Professionalism']:.2f}, Practicality={framework_scores_m2['Practicality']:.2f}, Avg={framework_avg_m2:.2f}\n")
    
    print(f"\nResults saved to: {output_file}")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

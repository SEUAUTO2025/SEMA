# -*- coding: utf-8 -*-
"""
Automated evaluation of blind evaluation CSV comments
Uses two AI models for scoring, generates two sets of results
Uses 5-point Likert scale to evaluate: accuracy, professionalism, practicality
"""
import sys, os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv
import json
import re
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"D:\pythonWorks\SpatialTemporalAttentionGCN-master\SpatialTemporalAttentionGCN-master\whole_dataset_txt!!!!!!!!"
BLIND_CSV = os.path.join(BASE_DIR, "blind_evaluation.csv")
ANSWER_CSV = os.path.join(BASE_DIR, "blind_evaluation_answer.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "blind_evaluation_scores.csv")

MODEL_1 = "qwen-plus"
MODEL_2 = "deepseek-v3.2"

def call_llm_for_scoring(teacher_comment: str, generated_comment: str, model_name: str) -> dict:
    """
    Call Alibaba Cloud LLM to score generated evaluation
    Args:
        teacher_comment: Teacher's standard evaluation
        generated_comment: Generated evaluation to be assessed
        model_name: Model name to use
    Returns: {"Accuracy": score, "Professionalism": score, "Practicality": score}
    """
    client = OpenAI(
        api_key=os.getenv("ALI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    prompt = f"""You are a professional archery coach evaluation expert. Here are two archery posture evaluation texts:

【Teacher's Standard Evaluation (Full Score Baseline)】:
{teacher_comment}

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
        
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
        else:
            print(f"Unable to parse LLM result: {result_text}")
            return {"Accuracy": 0, "Professionalism": 0, "Practicality": 0}
            
    except Exception as e:
        print(f"LLM call error: {e}")
        return {"Accuracy": 0, "Professionalism": 0, "Practicality": 0}

def main():
    print("Reading CSV files...")
    blind_data = []
    with open(BLIND_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            blind_data.append(row)
    
    answer_data = {}
    with open(ANSWER_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer_data[row['Video Name']] = row
    
    print(f"Total {len(blind_data)} evaluations to assess")
    print(f"Using Model 1: {MODEL_1}")
    print(f"Using Model 2: {MODEL_2}")
    
    results = []
    for i, row in enumerate(blind_data, 1):
        video_name = row['Video Name']
        text_a = row['Evaluation A']
        text_b = row['Evaluation B']
        
        answer = answer_data.get(video_name)
        if not answer:
            print(f"Warning: Answer not found for {video_name}")
            continue
        
        if answer['Evaluation A Source'] == 'Ground Truth':
            teacher_comment = text_a
            generated_comment = text_b
        else:
            teacher_comment = text_b
            generated_comment = text_a
        
        print(f"\n[{i}/{len(blind_data)}] Evaluating: {video_name}")
        
        print(f"  {MODEL_1} scoring...")
        scores_model1 = call_llm_for_scoring(teacher_comment, generated_comment, MODEL_1)
        print(f"    Accuracy: {scores_model1['Accuracy']}, Professionalism: {scores_model1['Professionalism']}, Practicality: {scores_model1['Practicality']}")
        
        print(f"  {MODEL_2} scoring...")
        scores_model2 = call_llm_for_scoring(teacher_comment, generated_comment, MODEL_2)
        print(f"    Accuracy: {scores_model2['Accuracy']}, Professionalism: {scores_model2['Professionalism']}, Practicality: {scores_model2['Practicality']}")
        
        results.append({
            'Video Name': video_name,
            f'{MODEL_1}_Accuracy': scores_model1['Accuracy'],
            f'{MODEL_1}_Professionalism': scores_model1['Professionalism'],
            f'{MODEL_1}_Practicality': scores_model1['Practicality'],
            f'{MODEL_1}_Average': round((scores_model1['Accuracy'] + scores_model1['Professionalism'] + scores_model1['Practicality']) / 3, 2),
            f'{MODEL_2}_Accuracy': scores_model2['Accuracy'],
            f'{MODEL_2}_Professionalism': scores_model2['Professionalism'],
            f'{MODEL_2}_Practicality': scores_model2['Practicality'],
            f'{MODEL_2}_Average': round((scores_model2['Accuracy'] + scores_model2['Professionalism'] + scores_model2['Practicality']) / 3, 2),
        })
    
    print(f"\nSaving results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['Video Name', 
                      f'{MODEL_1}_Accuracy', f'{MODEL_1}_Professionalism', f'{MODEL_1}_Practicality', f'{MODEL_1}_Average',
                      f'{MODEL_2}_Accuracy', f'{MODEL_2}_Professionalism', f'{MODEL_2}_Practicality', f'{MODEL_2}_Average']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "="*60)
    print("Evaluation completed! Statistics:")
    print("="*60)
    
    model1_accuracy = [r[f'{MODEL_1}_Accuracy'] for r in results]
    model1_professionalism = [r[f'{MODEL_1}_Professionalism'] for r in results]
    model1_practicality = [r[f'{MODEL_1}_Practicality'] for r in results]
    model1_avg = [r[f'{MODEL_1}_Average'] for r in results]
    
    model2_accuracy = [r[f'{MODEL_2}_Accuracy'] for r in results]
    model2_professionalism = [r[f'{MODEL_2}_Professionalism'] for r in results]
    model2_practicality = [r[f'{MODEL_2}_Practicality'] for r in results]
    model2_avg = [r[f'{MODEL_2}_Average'] for r in results]
    
    print(f"\n【{MODEL_1} Scoring Statistics】")
    print(f"Accuracy - Mean: {np.mean(model1_accuracy):.2f}, Std: {np.std(model1_accuracy):.2f}")
    print(f"专业度 - 均值: {np.mean(model1_professionalism):.2f}, 标准差: {np.std(model1_professionalism):.2f}")
    print(f"实用性 - 均值: {np.mean(model1_practicality):.2f}, 标准差: {np.std(model1_practicality):.2f}")
    print(f"总体平均分: {np.mean(model1_avg):.2f}")
    print(f"Professionalism - Mean: {np.mean(model1_professionalism):.2f}, Std: {np.std(model1_professionalism):.2f}")
    print(f"Practicality - Mean: {np.mean(model1_practicality):.2f}, Std: {np.std(model1_practicality):.2f}")
    print(f"Overall Average: {np.mean(model1_avg):.2f}")
    
    print(f"\n【{MODEL_2} Scoring Statistics】")
    print(f"Accuracy - Mean: {np.mean(model2_accuracy):.2f}, Std: {np.std(model2_accuracy):.2f}")
    print(f"Professionalism - Mean: {np.mean(model2_professionalism):.2f}, Std: {np.std(model2_professionalism):.2f}")
    print(f"Practicality - Mean: {np.mean(model2_practicality):.2f}, Std: {np.std(model2_practicality):.2f}")
    print(f"Overall Average: {np.mean(model2_avg):.2f}")
    
    plot_dir = os.path.join(BASE_DIR, "evaluation_plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    dimensions = ['Accuracy', 'Professionalism', 'Practicality']
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    model1_means = [np.mean(model1_accuracy), np.mean(model1_professionalism), np.mean(model1_practicality)]
    model2_means = [np.mean(model2_accuracy), np.mean(model2_professionalism), np.mean(model2_practicality)]
    
    model1_stds = [np.std(model1_accuracy), np.std(model1_professionalism), np.std(model1_practicality)]
    model2_stds = [np.std(model2_accuracy), np.std(model2_professionalism), np.std(model2_practicality)]
    
    bars1 = ax.bar(x - width/2, model1_means, width, yerr=model1_stds, capsize=5, 
                   label=MODEL_1, color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, model2_means, width, yerr=model2_stds, capsize=5,
                   label=MODEL_2, color='#e74c3c', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score (5-point scale)', fontsize=12, fontweight='bold')
    ax.set_title('Two AI Models Three-Dimension Scoring Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "two_models_comparison.png"))
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    data1 = [model1_accuracy, model1_professionalism, model1_practicality]
    bp1 = axes[0].boxplot(data1, labels=dimensions, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Score (5-point scale)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{MODEL_1} Score Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    data2 = [model2_accuracy, model2_professionalism, model2_practicality]
    bp2 = axes[1].boxplot(data2, labels=dimensions, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Score (5-point scale)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{MODEL_2} Score Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "two_models_distribution.png"))
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    x = range(1, len(model1_avg) + 1)
    
    ax.plot(x, model1_avg, marker='o', linestyle='-', color='#3498db', 
            linewidth=2, markersize=5, alpha=0.7, label=f'{MODEL_1} (Mean: {np.mean(model1_avg):.2f})')
    ax.plot(x, model2_avg, marker='s', linestyle='-', color='#e74c3c', 
            linewidth=2, markersize=5, alpha=0.7, label=f'{MODEL_2} (Mean: {np.mean(model2_avg):.2f})')
    
    ax.axhline(y=np.mean(model1_avg), color='#3498db', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(y=np.mean(model2_avg), color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Video Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Two AI Models Average Score Trend Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "two_models_trend.png"))
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    scatter_data = [
        (model1_accuracy, model2_accuracy, 'Accuracy'),
        (model1_professionalism, model2_professionalism, 'Professionalism'),
        (model1_practicality, model2_practicality, 'Practicality')
    ]
    
    for idx, (data1, data2, title) in enumerate(scatter_data):
        axes[idx].scatter(data1, data2, alpha=0.6, s=50, color='#9467bd', edgecolors='black')
        axes[idx].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Agreement')
        
        corr = np.corrcoef(data1, data2)[0, 1]
        axes[idx].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[idx].transAxes, fontsize=10, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[idx].set_xlabel(f'{MODEL_1}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(f'{MODEL_2}', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{title} Consistency', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, linestyle='--', alpha=0.5)
        axes[idx].set_xlim(0.5, 5.5)
        axes[idx].set_ylim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "two_models_consistency.png"))
    plt.close()
    
    print(f"\nVisualization charts saved to: {plot_dir}")
    print(f"  - two_models_comparison.png (Three-dimension comparison)")
    print(f"  - two_models_distribution.png (Score distribution)")
    print(f"  - two_models_trend.png (Average score trend)")
    print(f"  - two_models_consistency.png (Model consistency analysis)")
    print(f"\nScoring results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

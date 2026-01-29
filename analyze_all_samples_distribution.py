"""
Analyze biomechanical feature distribution of all samples
Determine grading standards based on distribution
Exclude first 180 samples and generate pie charts
"""
import sys
import os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from Tools.Gen_dataset.dataset_exe import load_single_csv_with_multipart_labels
from Tools.Gen_dataset.dataset_exe import NUM_COORDS, MAX_FRAMES
from RTMPose.Bone_Feature_Extract import cal_math_features, extract_action_features

def plot_grade_distribution(stats, grade_standards, output_folder='output_charts'):
    """
    Plot pie charts for grade distribution of each feature
    
    Args:
        stats: Statistics dictionary containing feature values
        grade_standards: Grading standards for each feature
        output_folder: Folder to save the charts
    """
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Feature display names
    feature_names = {
        'max_angle_avg': 'Hand-Shoulder-Hand Angle',
        'min_dist_avg': 'Left Hand to Chin Distance',
        'min_x_diff_avg': 'Shoulder-Foot X Difference'
    }
    
    # Grade colors
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Excellent, Good, Average, Poor
    
    for feature_key, feature_stat in stats.items():
        if feature_key not in grade_standards:
            continue
            
        values = feature_stat['values']
        standard = grade_standards[feature_key]
        
        # Count samples in each grade
        excellent_count = 0
        good_count = 0
        average_count = 0
        poor_count = 0
        
        if standard['direction'] == 'higher_is_better':
            # Higher is better (e.g., angle)
            for val in values:
                if val >= standard['excellent']:
                    excellent_count += 1
                elif val >= standard['good']:
                    good_count += 1
                elif val >= standard['average']:
                    average_count += 1
                else:
                    poor_count += 1
        else:
            # Lower is better (e.g., distance)
            for val in values:
                if val <= standard['excellent']:
                    excellent_count += 1
                elif val <= standard['good']:
                    good_count += 1
                elif val <= standard['average']:
                    average_count += 1
                else:
                    poor_count += 1
        
        # Create pie chart
        counts = [excellent_count, good_count, average_count, poor_count]
        labels = ['Excellent', 'Good', 'Average', 'Poor']
        
        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        
        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        plt.title(f'{feature_names[feature_key]} - Grade Distribution\n(Total: {len(values)} samples, Excluding first 180)',
                  fontsize=14, weight='bold', pad=20)
        
        # Add legend with counts
        legend_labels = [f'{labels[i]}: {counts[i]} ({counts[i]/len(values)*100:.1f}%)' 
                        for i in range(len(labels))]
        plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(output_folder, f'{feature_key}_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved chart: {output_path}")
        plt.close()

def analyze_all_samples(csv_folder_path):
    """
    Analyze biomechanical feature distribution and determine grading standards
    
    Args:
        csv_folder_path: Path to folder containing all sample CSV files
    
    Returns:
        Statistics dictionary and grading standards
    """
    csv_folder = Path(csv_folder_path)
    csv_files = sorted(list(csv_folder.glob("*.csv")))  # Sort for consistent ordering
    
    print(f"Found {len(csv_files)} sample files")
    
    # Skip first N samples
    csv_files = csv_files[180:]
    print(f"Analyzing {len(csv_files)} samples (after skipping)")
    
    all_features = {
        'max_angle_avg': [],
        'min_dist_avg': [],
        'min_x_diff_avg': []
    }
    
    for i, csv_file in enumerate(csv_files):
        try:
            print(f"\rProcessing: {i+1}/{len(csv_files)} - {csv_file.name}", end='')
            
            data, labels, label_total = load_single_csv_with_multipart_labels(str(csv_file), max_frames=MAX_FRAMES)
            
            all_data = torch.from_numpy(data).float()
            all_data = all_data[:, :, :, 0]
            all_data = all_data.permute(1, 2, 0)
            
            math_feature = cal_math_features(all_data)
            action_features = extract_action_features(math_feature)
            
            for key in all_features.keys():
                if key in action_features:
                    all_features[key].append(action_features[key])
            
        except Exception as e:
            print(f"\nError processing {csv_file.name}: {e}")
            continue
    
    print("\n\n" + "="*80)
    print("Biomechanical Feature Distribution Analysis")
    print("="*80)
    
    stats = {}
    grade_standards = {}
    
    for feature_name, values in all_features.items():
        if len(values) > 0:
            values_array = np.array(values)
            stats[feature_name] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'median': np.median(values_array),
                'q25': np.percentile(values_array, 25),
                'q75': np.percentile(values_array, 75),
                'q10': np.percentile(values_array, 10),
                'q90': np.percentile(values_array, 90),
                'values': values_array
            }
    
    print("\n1. Hand-Shoulder-Hand Angle (max_angle_avg) - Higher is better")
    print("-" * 80)
    if 'max_angle_avg' in stats:
        s = stats['max_angle_avg']
        print(f"   Sample count: {len(s['values'])}")
        print(f"   Mean: {s['mean']:.2f}°")
        print(f"   Std: {s['std']:.2f}°")
        print(f"   Min: {s['min']:.2f}°")
        print(f"   10th percentile: {s['q10']:.2f}°")
        print(f"   25th percentile: {s['q25']:.2f}°")
        print(f"   Median: {s['median']:.2f}°")
        print(f"   75th percentile: {s['q75']:.2f}°")
        print(f"   90th percentile: {s['q90']:.2f}°")
        print(f"   Max: {s['max']:.2f}°")
        print(f"\n   Grading standards:")
        print(f"   - Excellent: ≥ {s['q75']:.2f}° (Top 25%)")
        print(f"   - Good: {s['median']:.2f}° - {s['q75']:.2f}° (25%-50%)")
        print(f"   - Average: {s['q25']:.2f}° - {s['median']:.2f}° (50%-75%)")
        print(f"   - Poor: < {s['q25']:.2f}° (Bottom 25%)")
        
        grade_standards['max_angle_avg'] = {
            'excellent': s['q75'],
            'good': s['median'],
            'average': s['q25'],
            'direction': 'higher_is_better'
        }
    
    print("\n2. Left Hand to Chin Distance (min_dist_avg) - Lower is better")
    print("-" * 80)
    if 'min_dist_avg' in stats:
        s = stats['min_dist_avg']
        print(f"   Sample count: {len(s['values'])}")
        print(f"   Mean: {s['mean']:.4f}")
        print(f"   Std: {s['std']:.4f}")
        print(f"   Min: {s['min']:.4f}")
        print(f"   10th percentile: {s['q10']:.4f}")
        print(f"   25th percentile: {s['q25']:.4f}")
        print(f"   Median: {s['median']:.4f}")
        print(f"   75th percentile: {s['q75']:.4f}")
        print(f"   90th percentile: {s['q90']:.4f}")
        print(f"   Max: {s['max']:.4f}")
        print(f"\n   Grading standards:")
        print(f"   - Excellent: ≤ {s['q25']:.4f} (Top 25%)")
        print(f"   - Good: {s['q25']:.4f} - {s['median']:.4f} (25%-50%)")
        print(f"   - Average: {s['median']:.4f} - {s['q75']:.4f} (50%-75%)")
        print(f"   - Poor: > {s['q75']:.4f} (Bottom 25%)")
        
        grade_standards['min_dist_avg'] = {
            'excellent': s['q25'],
            'good': s['median'],
            'average': s['q75'],
            'direction': 'lower_is_better'
        }
    
    print("\n3. Shoulder-Foot Midpoint X Coordinate Difference (min_x_diff_avg) - Lower is better")
    print("-" * 80)
    if 'min_x_diff_avg' in stats:
        s = stats['min_x_diff_avg']
        print(f"   Sample count: {len(s['values'])}")
        print(f"   Mean: {s['mean']:.4f}")
        print(f"   Std: {s['std']:.4f}")
        print(f"   Min: {s['min']:.4f}")
        print(f"   10th percentile: {s['q10']:.4f}")
        print(f"   25th percentile: {s['q25']:.4f}")
        print(f"   Median: {s['median']:.4f}")
        print(f"   75th percentile: {s['q75']:.4f}")
        print(f"   90th percentile: {s['q90']:.4f}")
        print(f"   Max: {s['max']:.4f}")
        print(f"\n   Grading standards:")
        print(f"   - Excellent: ≤ {s['q25']:.4f} (Top 25%)")
        print(f"   - Good: {s['q25']:.4f} - {s['median']:.4f} (25%-50%)")
        print(f"   - Average: {s['median']:.4f} - {s['q75']:.4f} (50%-75%)")
        print(f"   - Poor: > {s['q75']:.4f} (Bottom 25%)")
        
        grade_standards['min_x_diff_avg'] = {
            'excellent': s['q25'],
            'good': s['median'],
            'average': s['q75'],
            'direction': 'lower_is_better'
        }
    
    print("\n" + "="*80)
    print("Interpretability Standards")
    print("="*80)
    
    if all(key in stats for key in ['max_angle_avg', 'min_dist_avg', 'min_x_diff_avg']):
        angle_s = stats['max_angle_avg']
        dist_s = stats['min_dist_avg']
        x_diff_s = stats['min_x_diff_avg']
        
        print(f"""
        1. Hand-Shoulder-Hand Angle (Draw Angle) - Higher is better
        • Excellent (≥{angle_s['q75']:.2f}°): Top 25%
        • Good ({angle_s['median']:.2f}°-{angle_s['q75']:.2f}°): 25%-50%
        • Average ({angle_s['q25']:.2f}°-{angle_s['median']:.2f}°): 50%-75%
        • Poor (<{angle_s['q25']:.2f}°): Bottom 25%

        2. Left Hand to Chin Distance (Anchor Stability) - Lower is better
        • Excellent (≤{dist_s['q25']:.4f}): Top 25%
        • Good ({dist_s['q25']:.4f}-{dist_s['median']:.4f}): 25%-50%
        • Average ({dist_s['median']:.4f}-{dist_s['q75']:.4f}): 50%-75%
        • Poor (>{dist_s['q75']:.4f}): Bottom 25%

        3. Body Center Alignment (Shoulder-Foot X Difference) - Lower is better
        • Excellent (≤{x_diff_s['q25']:.4f}): Top 25%
        • Good ({x_diff_s['q25']:.4f}-{x_diff_s['median']:.4f}): 25%-50%
        • Average ({x_diff_s['median']:.4f}-{x_diff_s['q75']:.4f}): 50%-75%
        • Poor (>{x_diff_s['q75']:.4f}): Bottom 25%
        """)
    
    print("\n" + "="*80)
    
    return stats, grade_standards


if __name__ == "__main__":
    csv_folder = r"dataset\csv"
    stats, grade_standards = analyze_all_samples(csv_folder)
    
    # Generate pie charts
    print("\n" + "="*80)
    print("Generating pie charts...")
    print("="*80)
    plot_grade_distribution(stats, grade_standards)
    
    # Save text report
    output_file = "report_output/biomechanics_grade_standards.txt"
    with open(output_file, 'w', encoding='utf-8') as f:        
        for feature_name, standard in grade_standards.items():
            f.write(f"\n{feature_name}:\n")
            f.write(f"  Direction: {standard['direction']}\n")
            f.write(f"  Excellent threshold: {standard['excellent']:.4f}\n")
            f.write(f"  Good threshold: {standard['good']:.4f}\n")
            f.write(f"  Average threshold: {standard['average']:.4f}\n")
        
        f.write("\n\nDetailed Statistics:\n")
        for feature_name, s in stats.items():
            f.write(f"\n{feature_name}:\n")
            f.write(f"  Sample count: {len(s['values'])}\n")
            f.write(f"  Mean: {s['mean']:.4f}\n")
            f.write(f"  Std: {s['std']:.4f}\n")
            f.write(f"  Min: {s['min']:.4f}\n")
            f.write(f"  25th percentile: {s['q25']:.4f}\n")
            f.write(f"  Median: {s['median']:.4f}\n")
            f.write(f"  75th percentile: {s['q75']:.4f}\n")
            f.write(f"  Max: {s['max']:.4f}\n")
    
    print(f"\nGrading standards saved to: {output_file}")
    print("Analysis complete!")

"""
Infer biomechanics feature thresholds with GT supervision.

Logic:
1) Compute three biomechanics scalars for each sample.
2) Build per-feature GT quality labels using mapped part-score GT only (fixed bins).
3) Infer thresholds with mixed one-sided/two-sided strategy.
"""
import json
import os

from Tools.Exe_dataset.dataset_test_tools import (
    DEFAULT_FEATURE_GT_PART_MAP,
    DEFAULT_GT_FIXED_BINS,
    DEFAULT_THRESHOLD_SHAPE_MAP,
    run_biomech_threshold_experiment,
)


def main():
    dataset_path = r"D:\Pythonworks\datasets\SEMA"
    skip_first_n = 0
    sort_files = True
    max_search_candidates = 101
    gt_label_strategy = "fixed_bins"
    gt_fixed_bins = DEFAULT_GT_FIXED_BINS
    threshold_shape_map = dict(DEFAULT_THRESHOLD_SHAPE_MAP)
    enable_candidate_features = True

    result = run_biomech_threshold_experiment(
        dataset_path=dataset_path,
        skip_first_n=skip_first_n,
        sort_files=sort_files,
        gt_label_strategy=gt_label_strategy,
        gt_fixed_bins=gt_fixed_bins,
        threshold_shape_map=threshold_shape_map,
        max_search_candidates=max_search_candidates,
        enable_candidate_features=enable_candidate_features,
    )

    print("=" * 80)
    print("GT-Supervised Biomechanics Threshold Inference Finished")
    print("=" * 80)
    print(f"Dataset root: {os.path.abspath(dataset_path)}")
    print(f"CSV folder: {os.path.abspath(result.get('csv_folder_path', os.path.join(dataset_path, 'csv')))}")
    print(f"Processed samples: {len(result.get('sample_feature_rows', []))}")
    print(f"Failed samples: {len(result.get('failed_rows', []))}")
    print(f"GT mapping: {json.dumps(DEFAULT_FEATURE_GT_PART_MAP, ensure_ascii=False)}")
    print("quality_method: part_only")
    print(f"gt_label_strategy: {gt_label_strategy}")
    print(f"gt_fixed_bins: {gt_fixed_bins}")
    print(f"threshold_shape_map: {json.dumps(threshold_shape_map, ensure_ascii=False)}")
    print(f"max_search_candidates: {max_search_candidates}")

    print("\nInferred Thresholds:")
    standards = result.get("grade_standards", {})
    fit_metrics = result.get("fit_metrics", {})
    for feature_key, standard in standards.items():
        metric = fit_metrics.get(feature_key, {})
        print(f"\n[{feature_key}]")
        print(f"  mode={standard.get('mode', 'one_sided')}")
        print(f"  direction={standard.get('direction')}")
        if str(standard.get("mode", "one_sided")).strip().lower() == "two_sided":
            print(f"  center={float(standard.get('center', float('nan'))):.6f}")
            print(f"  excellent_dev={float(standard.get('excellent_dev', float('nan'))):.6f}")
            print(f"  good_dev={float(standard.get('good_dev', float('nan'))):.6f}")
            print(f"  average_dev={float(standard.get('average_dev', float('nan'))):.6f}")
        else:
            print(f"  excellent={float(standard.get('excellent', float('nan'))):.6f}")
            print(f"  good={float(standard.get('good', float('nan'))):.6f}")
            print(f"  average={float(standard.get('average', float('nan'))):.6f}")
        print(
            f"  threshold_fit_accuracy="
            f"{float(metric.get('match_accuracy', 0.0)):.4f} "
            f"({int(metric.get('match_count', 0))}/{int(metric.get('valid_count', 0))})"
        )
        print(
            f"  threshold_fit_accuracy_rule="
            f"{metric.get('match_accuracy_rule', 'binary_grouped')} "
            "(Excellent/Good vs Average/Poor)"
        )
        print(
            f"  strict_threshold_fit_accuracy="
            f"{float(metric.get('strict_match_accuracy', 0.0)):.4f} "
            f"({int(metric.get('strict_match_count', 0))}/{int(metric.get('valid_count', 0))})"
        )
        print(
            f"  threshold_search_balanced_accuracy="
            f"{float(metric.get('threshold_search_balanced_accuracy', 0.0)):.4f}"
        )

    print("\nFeature Diagnostics:")
    diagnostics = result.get("feature_diagnostics", {})
    for feature_key, item in diagnostics.items():
        corr = item.get("correlations", {})
        dir_cmp = item.get("direction_comparison", {})
        center_search = item.get("two_sided_center_search", {})
        print(f"\n[{feature_key}]")
        print(f"  mode={item.get('mode')}, part_key={item.get('part_key')}, valid_count={item.get('valid_count')}")
        print(f"  gt_grade_counts={json.dumps(item.get('gt_grade_counts', {}), ensure_ascii=False)}")
        print(
            "  corr: "
            f"pearson={float(corr.get('pearson_raw', float('nan'))):.4f}, "
            f"spearman={float(corr.get('spearman_raw', float('nan'))):.4f}, "
            f"eta2={float(item.get('eta_squared', float('nan'))):.4f}"
        )
        print(
            "  direction_compare: "
            f"higher={float(dir_cmp.get('higher_spearman', float('nan'))):.4f}, "
            f"lower={float(dir_cmp.get('lower_spearman', float('nan'))):.4f}, "
            f"best={dir_cmp.get('best_direction')}"
        )
        if str(item.get("mode", "")).strip().lower() == "two_sided":
            print(
                "  two_sided_center: "
                f"center={float(center_search.get('best_center', float('nan'))):.6f}, "
                f"best_s={float(center_search.get('best_center_spearman', float('nan'))):.4f}, "
                f"improve={float(center_search.get('improvement_vs_monotonic', float('nan'))):.4f}"
            )

    if enable_candidate_features:
        relevance = result.get("candidate_feature_relevance", {})
        ranking = relevance.get("ranking", [])
        print("\nTop Candidate Features by |Spearman|:")
        for row in ranking[:10]:
            print(
                f"  - {row.get('feature')}: "
                f"{float(row.get('best_spearman', 0.0)):.4f} "
                f"({row.get('best_direction')} -> {row.get('part_key')})"
            )

    print("\nArtifacts:")
    for key, value in result.get("artifacts", {}).items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
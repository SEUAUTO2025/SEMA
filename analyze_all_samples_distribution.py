"""
Dataset-level sample distribution analysis entry script.
Thin wrapper around Tools.Exe_dataset.dataset_test_tools.
"""
import os
import sys

from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Tools.Exe_dataset.dataset_test_tools import run_dataset_level_distribution_analysis


def main():
    csv_folder = r"dataset\csv"
    output_folder = "report_output"
    chart_output_folder = "output_charts"
    skip_first_n = 180

    result = run_dataset_level_distribution_analysis(
        csv_folder_path=csv_folder,
        skip_first_n=skip_first_n,
        output_folder=output_folder,
        chart_output_folder=chart_output_folder,
    )

    sample_count = len(result["sample_feature_rows"])
    failed_count = len(result["failed_rows"])
    feature_count = len(result["stats"])

    print("=" * 80)
    print("Dataset Level Distribution Analysis Finished")
    print("=" * 80)
    print(f"CSV folder: {os.path.abspath(csv_folder)}")
    print(f"Skip first N samples: {skip_first_n}")
    print(f"Processed samples: {sample_count}")
    print(f"Failed samples: {failed_count}")
    print(f"Features analyzed: {feature_count}")
    print("\nArtifacts:")
    for key, value in result["artifacts"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()

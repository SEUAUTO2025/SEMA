"""
Dataset-level sample distribution analysis entry script.
Thin wrapper around Tools.Exe_dataset.dataset_test_tools.
"""
import os

from Tools.Exe_dataset.dataset_test_tools import run_biomech_threshold_experiment


def main():
    dataset_path = r"dataset"
    skip_first_n = 180

    result = run_biomech_threshold_experiment(
        dataset_path=dataset_path,
        skip_first_n=skip_first_n,
    )

    sample_count = len(result["sample_feature_rows"])
    failed_count = len(result["failed_rows"])
    feature_count = len(result["stats"])

    print("=" * 80)
    print("Dataset Level Distribution Analysis Finished")
    print("=" * 80)
    print(f"Dataset root: {os.path.abspath(dataset_path)}")
    print(f"CSV folder: {os.path.abspath(result.get('csv_folder_path', os.path.join(dataset_path, 'csv')))}")
    print(f"Skip first N samples: {skip_first_n}")
    print(f"Processed samples: {sample_count}")
    print(f"Failed samples: {failed_count}")
    print(f"Features analyzed: {feature_count}")
    print("\nArtifacts:")
    for key, value in result["artifacts"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()

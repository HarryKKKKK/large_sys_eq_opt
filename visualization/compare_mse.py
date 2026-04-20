import os
import glob
import argparse
import numpy as np
import pandas as pd

FIELDS = ["rho", "rhou", "rhov", "E"]


def load_snapshot(path):
    df = pd.read_csv(path)
    required = ["i", "j"] + FIELDS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def compare_one_snapshot(cpu_path, gpu_path):
    cpu = load_snapshot(cpu_path).sort_values(["j", "i"]).reset_index(drop=True)
    gpu = load_snapshot(gpu_path).sort_values(["j", "i"]).reset_index(drop=True)

    if len(cpu) != len(gpu):
        raise ValueError(
            f"Row count mismatch:\n  {cpu_path}: {len(cpu)}\n  {gpu_path}: {len(gpu)}"
        )

    if not np.array_equal(cpu[["i", "j"]].values, gpu[["i", "j"]].values):
        raise ValueError(f"Grid index mismatch between:\n  {cpu_path}\n  {gpu_path}")

    result = {}
    total_mse_sum = 0.0

    for field in FIELDS:
        diff = cpu[field].to_numpy() - gpu[field].to_numpy()
        mse = float(np.mean(diff ** 2))
        result[field] = mse
        total_mse_sum += mse

    result["total_mse_sum"] = total_mse_sum
    return result


def get_input_dir():
    parser = argparse.ArgumentParser(
        description="Compare CPU and GPU snapshot CSV files."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Directory containing cpu_snapshot_*.csv and gpu_snapshot_*.csv"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir:
        output_dir = input("Please enter the snapshot directory: ").strip()

    if not output_dir:
        raise ValueError("No directory was provided.")

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Directory not found: {output_dir}")

    return output_dir


def main():
    output_dir = get_input_dir()

    cpu_files = sorted(glob.glob(os.path.join(output_dir, "cpu_snapshot_*.csv")))
    gpu_files = sorted(glob.glob(os.path.join(output_dir, "gpu_snapshot_*.csv")))

    if not cpu_files:
        raise FileNotFoundError(f"No cpu_snapshot_*.csv files found in {output_dir}")
    if not gpu_files:
        raise FileNotFoundError(f"No gpu_snapshot_*.csv files found in {output_dir}")

    if len(cpu_files) != len(gpu_files):
        raise ValueError(
            f"Snapshot count mismatch: {len(cpu_files)} CPU files vs {len(gpu_files)} GPU files"
        )

    rows = []

    for cpu_path, gpu_path in zip(cpu_files, gpu_files):
        cpu_name = os.path.basename(cpu_path)
        gpu_name = os.path.basename(gpu_path)

        snap_cpu = cpu_name.replace("cpu_", "").replace(".csv", "")
        snap_gpu = gpu_name.replace("gpu_", "").replace(".csv", "")

        if snap_cpu != snap_gpu:
            raise ValueError(f"Snapshot name mismatch: {cpu_name} vs {gpu_name}")

        stats = compare_one_snapshot(cpu_path, gpu_path)

        print(f"\n=== {snap_cpu} ===")
        for field in FIELDS:
            print(f"{field}_mse = {stats[field]:.16e}")
        print(f"total_mse_sum = {stats['total_mse_sum']:.16e}")

        rows.append({
            "snapshot": snap_cpu,
            "rho_mse": stats["rho"],
            "rhou_mse": stats["rhou"],
            "rhov_mse": stats["rhov"],
            "E_mse": stats["E"],
            "total_mse_sum": stats["total_mse_sum"],
        })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
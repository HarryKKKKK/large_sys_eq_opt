import pandas as pd
from pathlib import Path
import glob
import re


# =========================
# Configuration
# =========================

OUTPUT_DIR = Path("outputs")

CASES = [
    "shock_bubble",
    "blast_wave",
    # add more cases here if needed
]

SOLVERS = [
    "hll",
    "hllc",
    "exact",
    "force"
]


FIELDS = ["rho", "rhou", "rhov", "E"]


# =========================
# Helper functions
# =========================

def snapshot_id_from_filename(filename):
    """
    Extract snapshot number from filenames like:
    gpu_shock_bubble_hll_n1_snapshot_3.csv
    cpu_shock_bubble_hll_n1_snapshot_3.csv
    """
    match = re.search(r"snapshot_(\d+)\.csv$", str(filename))
    if match is None:
        raise ValueError(f"Cannot extract snapshot id from filename: {filename}")
    return int(match.group(1))


def load_snapshots(prefix, case, solver):
    """
    Load all snapshots for either cpu or gpu.

    Example pattern:
    outputs/gpu_shock_bubble_hll_n1_snapshot_*.csv
    """
    pattern = OUTPUT_DIR / f"{prefix}_{case}_{solver}_n1_snapshot_*.csv"
    files = sorted(glob.glob(str(pattern)), key=snapshot_id_from_filename)

    snapshots = {}
    for file in files:
        sid = snapshot_id_from_filename(file)
        snapshots[sid] = Path(file)

    return snapshots


def compute_mse_for_pair(cpu_file, gpu_file):
    """
    Compute MSE between one CPU snapshot and one GPU snapshot.

    MSE is averaged over:
    - all cells
    - all conserved variables: rho, rhou, rhov, E
    """
    cpu_df = pd.read_csv(cpu_file)
    gpu_df = pd.read_csv(gpu_file)

    if len(cpu_df) != len(gpu_df):
        raise ValueError(
            f"Different number of rows:\n"
            f"CPU: {cpu_file}, rows={len(cpu_df)}\n"
            f"GPU: {gpu_file}, rows={len(gpu_df)}"
        )

    # Make sure cells are compared in the same order
    key_cols = ["i", "j"]

    cpu_df = cpu_df.sort_values(key_cols).reset_index(drop=True)
    gpu_df = gpu_df.sort_values(key_cols).reset_index(drop=True)

    if not cpu_df[key_cols].equals(gpu_df[key_cols]):
        raise ValueError(
            f"CPU and GPU grids do not match:\n"
            f"CPU: {cpu_file}\n"
            f"GPU: {gpu_file}"
        )

    diff = cpu_df[FIELDS] - gpu_df[FIELDS]

    mse_all = (diff.to_numpy() ** 2).mean()

    mse_by_field = {
        field: ((cpu_df[field] - gpu_df[field]) ** 2).mean()
        for field in FIELDS
    }

    return mse_all, mse_by_field


def compute_case_solver_error(case, solver):
    cpu_snapshots = load_snapshots("cpu", case, solver)
    gpu_snapshots = load_snapshots("gpu", case, solver)

    common_ids = sorted(set(cpu_snapshots.keys()) & set(gpu_snapshots.keys()))

    if not common_ids:
        print(f"[SKIP] {case}, {solver}: no matching CPU/GPU snapshots found")
        return None

    missing_cpu = sorted(set(gpu_snapshots.keys()) - set(cpu_snapshots.keys()))
    missing_gpu = sorted(set(cpu_snapshots.keys()) - set(gpu_snapshots.keys()))

    if missing_cpu:
        print(f"[WARNING] {case}, {solver}: missing CPU snapshots {missing_cpu}")

    if missing_gpu:
        print(f"[WARNING] {case}, {solver}: missing GPU snapshots {missing_gpu}")

    snapshot_results = []

    for sid in common_ids:
        cpu_file = cpu_snapshots[sid]
        gpu_file = gpu_snapshots[sid]

        mse_all, mse_by_field = compute_mse_for_pair(cpu_file, gpu_file)

        row = {
            "case": case,
            "solver": solver,
            "snapshot": sid,
            "mse_all_fields": mse_all,
        }

        for field, value in mse_by_field.items():
            row[f"mse_{field}"] = value

        snapshot_results.append(row)

    result_df = pd.DataFrame(snapshot_results)

    # Average across all matched snapshots
    summary = {
        "case": case,
        "solver": solver,
        "num_snapshots": len(common_ids),
        "mean_mse_all_fields": result_df["mse_all_fields"].mean(),
    }

    for field in FIELDS:
        summary[f"mean_mse_{field}"] = result_df[f"mse_{field}"].mean()

    return summary, result_df


# =========================
# Main
# =========================

def main():
    summaries = []

    detailed_output_dir = Path("mse_results")
    detailed_output_dir.mkdir(exist_ok=True)

    for case in CASES:
        for solver in SOLVERS:
            result = compute_case_solver_error(case, solver)

            if result is None:
                continue

            summary, detail_df = result
            summaries.append(summary)

            detail_file = detailed_output_dir / f"mse_detail_{case}_{solver}.csv"
            detail_df.to_csv(detail_file, index=False)

            print("=" * 70)
            print(f"Case   : {case}")
            print(f"Solver : {solver}")
            print(f"Snapshots compared : {summary['num_snapshots']}")
            print(f"Mean MSE over all cells and all fields: {summary['mean_mse_all_fields']:.12e}")

            for field in FIELDS:
                print(f"Mean MSE {field:>4}: {summary[f'mean_mse_{field}']:.12e}")

            print(f"Detailed snapshot errors saved to: {detail_file}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_file = detailed_output_dir / "mse_summary_all_cases.csv"
        summary_df.to_csv(summary_file, index=False)

        print("=" * 70)
        print(f"Summary saved to: {summary_file}")
        print()
        print(summary_df.to_string(index=False))
    else:
        print("No valid CPU/GPU snapshot pairs found.")


if __name__ == "__main__":
    main()
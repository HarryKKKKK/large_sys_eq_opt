import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CPU_PATTERN = "cpu_snapshot_*.csv"
GPU_PATTERN = "gpu_snapshot_*.csv"
OUT_NAME = "cpu_gpu_density_contours.png"

DENSITY_LEVELS = np.linspace(0.1, 2.8, 45)

DRAW_BUBBLE = True
R_BUBBLE = 0.025
BUBBLE_CX = 0.035
BUBBLE_CY = 0.0445
BUBBLE_LINEWIDTH = 0.8


def load_snapshot(path):
    df = pd.read_csv(path)
    required = ["i", "j", "x", "y", "rho"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def build_grid(df):
    df = df.sort_values(["j", "i"]).reset_index(drop=True)

    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())

    nx = len(x_vals)
    ny = len(y_vals)

    if len(df) != nx * ny:
        raise ValueError(
            f"Grid size mismatch: len(df)={len(df)}, nx={nx}, ny={ny}, nx*ny={nx * ny}"
        )

    rho = df["rho"].to_numpy(dtype=float).reshape((ny, nx))
    X, Y = np.meshgrid(x_vals, y_vals)

    return X, Y, rho, nx, ny


def match_snapshot_name(cpu_file, gpu_file):
    cpu_name = os.path.basename(cpu_file).replace("cpu_", "").replace(".csv", "")
    gpu_name = os.path.basename(gpu_file).replace("gpu_", "").replace(".csv", "")
    return cpu_name == gpu_name, cpu_name, gpu_name


def get_input_dir():
    parser = argparse.ArgumentParser(
        description="Plot CPU and GPU density contours from snapshot CSV files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        help="Directory containing cpu_snapshot_*.csv and gpu_snapshot_*.csv"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image filename (default: saved inside input_dir)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir:
        input_dir = input("Please enter the snapshot directory: ").strip()

    if not input_dir:
        raise ValueError("No input directory provided.")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    return input_dir, args.output


def main():
    input_dir, output_name = get_input_dir()

    cpu_files = sorted(glob.glob(os.path.join(input_dir, CPU_PATTERN)))
    gpu_files = sorted(glob.glob(os.path.join(input_dir, GPU_PATTERN)))

    if not cpu_files:
        raise FileNotFoundError(f"No files found: {os.path.join(input_dir, CPU_PATTERN)}")
    if not gpu_files:
        raise FileNotFoundError(f"No files found: {os.path.join(input_dir, GPU_PATTERN)}")

    if len(cpu_files) != len(gpu_files):
        raise ValueError(
            f"Snapshot count mismatch: {len(cpu_files)} CPU files vs {len(gpu_files)} GPU files"
        )

    pairs = []
    for cpu_file, gpu_file in zip(cpu_files, gpu_files):
        ok, cpu_snap, gpu_snap = match_snapshot_name(cpu_file, gpu_file)
        if not ok:
            raise ValueError(f"Snapshot name mismatch: {cpu_snap} vs {gpu_snap}")
        pairs.append((cpu_file, gpu_file, cpu_snap))

    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows), squeeze=False)

    if DRAW_BUBBLE:
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        xb = BUBBLE_CX + R_BUBBLE * np.cos(theta)
        yb = BUBBLE_CY + R_BUBBLE * np.sin(theta)

    for row, (cpu_file, gpu_file, snap_name) in enumerate(pairs):
        cpu_df = load_snapshot(cpu_file)
        gpu_df = load_snapshot(gpu_file)

        cpu_df = cpu_df.sort_values(["j", "i"]).reset_index(drop=True)
        gpu_df = gpu_df.sort_values(["j", "i"]).reset_index(drop=True)

        if len(cpu_df) != len(gpu_df):
            raise ValueError(
                f"Row count mismatch:\n  {cpu_file}: {len(cpu_df)}\n  {gpu_file}: {len(gpu_df)}"
            )

        if not np.array_equal(cpu_df[["i", "j"]].values, gpu_df[["i", "j"]].values):
            raise ValueError(f"Grid index mismatch between:\n  {cpu_file}\n  {gpu_file}")

        Xc, Yc, rho_cpu, nx_cpu, ny_cpu = build_grid(cpu_df)
        Xg, Yg, rho_gpu, nx_gpu, ny_gpu = build_grid(gpu_df)

        if nx_cpu != nx_gpu or ny_cpu != ny_gpu:
            raise ValueError(
                f"Grid shape mismatch in {snap_name}: "
                f"CPU=({ny_cpu}, {nx_cpu}) vs GPU=({ny_gpu}, {nx_gpu})"
            )

        if not np.allclose(Xc, Xg) or not np.allclose(Yc, Yg):
            raise ValueError(f"x/y grid mismatch between CPU and GPU for {snap_name}")

        ax_cpu = axes[row, 0]
        ax_gpu = axes[row, 1]

        ax_cpu.contour(Xc, Yc, rho_cpu, levels=DENSITY_LEVELS, colors="k", linewidths=0.5)
        ax_gpu.contour(Xg, Yg, rho_gpu, levels=DENSITY_LEVELS, colors="k", linewidths=0.5)

        if DRAW_BUBBLE:
            ax_cpu.plot(xb, yb, "k--", linewidth=BUBBLE_LINEWIDTH)
            ax_gpu.plot(xb, yb, "k--", linewidth=BUBBLE_LINEWIDTH)

        ax_cpu.set_title(f"{snap_name} - CPU", fontsize=11)
        ax_gpu.set_title(f"{snap_name} - GPU", fontsize=11)

        for ax in (ax_cpu, ax_gpu):
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color("black")

        print(f"[{row}] plotted {snap_name}")

    plt.tight_layout()

    if output_name is None:
        output_path = os.path.join(input_dir, OUT_NAME)
    else:
        output_path = output_name if os.path.isabs(output_name) else os.path.join(input_dir, output_name)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    main()
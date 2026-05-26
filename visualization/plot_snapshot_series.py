#!/usr/bin/env python3
'''
python3 visualization/plot_snapshot_series.py \
  --input "outputs/gpu_shock_bubble_hll_n1_snapshot_*.csv" \
  --field rho \
  --t-end 0.0011741 \
  --output figs/gpu_shock_bubble_hll_density_series.png

python3 visualization/plot_snapshot_series.py \
  --input "outputs/gpu_shock_bubble_hllc_n1_snapshot_*.csv" \
  --field rho \
  --t-end 0.0011741 \
  --output figs/gpu_shock_bubble_hllc_density_series.png

python3 visualization/plot_snapshot_series.py \
  --input "outputs/gpu_blast_wave_hll_n1_snapshot_*.csv" \
  --field rho \
  --t-end 0.2 \
  --output figs/gpu_blast_wave_hll_density_series.png

python3 visualization/plot_snapshot_series.py \
  --input "outputs/gpu_blast_wave_hllc_n1_snapshot_*.csv" \
  --field rho \
  --t-end 0.2 \
  --output figs/gpu_blast_wave_hllc_density_series.png

python3 visualization/plot_snapshot_series.py \
  --input "outputs/gpu_blast_wave_hll_n1_snapshot_*.csv" \
  --field p \
  --t-end 0.2 \
  --filled \
  --levels 60 \
  --output figs/gpu_blast_wave_hll_pressure_series.png

python3 visualization/plot_snapshot_series.py \
  --input "outputs/gpu_blast_wave_hllc_n1_snapshot_*.csv" \
  --field p \
  --t-end 0.2 \
  --filled \
  --levels 60 \
  --output figs/gpu_blast_wave_hllc_pressure_series.png
'''
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


GAMMA = 1.4


def natural_key(path: str):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path)]


def infer_case_from_filename(path: str) -> str:
    name = os.path.basename(path)

    if "shock_bubble" in name:
        return "shock_bubble"
    if "blast_wave" in name:
        return "blast_wave"

    return "unknown"


def default_levels(case_name: str, field: str):
    if field == "rho":
        if case_name == "shock_bubble":
            return np.linspace(0.1, 2.8, 45)
        if case_name == "blast_wave":
            return np.linspace(0.8, 6.5, 60)

    if field == "p":
        if case_name == "shock_bubble":
            return np.linspace(0.8e5, 1.8e5, 50)
        if case_name == "blast_wave":
            return np.linspace(1.0, 100.0, 60)

    return 60


def snapshot_id(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"snapshot_(\d+)", name)
    if m:
        return int(m.group(1))
    return -1


def load_snapshot(csv_path: str, field: str):
    usecols = ["x", "y", "rho", "rhou", "rhov", "E"]
    df = pd.read_csv(csv_path, usecols=usecols)

    df = df.sort_values(["y", "x"])

    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())

    nx = len(x_unique)
    ny = len(y_unique)

    X = df["x"].to_numpy().reshape(ny, nx)
    Y = df["y"].to_numpy().reshape(ny, nx)

    rho = df["rho"].to_numpy()
    rhou = df["rhou"].to_numpy()
    rhov = df["rhov"].to_numpy()
    E = df["E"].to_numpy()

    if field == "rho":
        Z = rho
    elif field == "rhou":
        Z = rhou
    elif field == "rhov":
        Z = rhov
    elif field == "E":
        Z = E
    elif field in {"u", "v", "p", "speed"}:
        u = rhou / rho
        v = rhov / rho
        kinetic = 0.5 * rho * (u * u + v * v)
        p = (GAMMA - 1.0) * (E - kinetic)

        if field == "u":
            Z = u
        elif field == "v":
            Z = v
        elif field == "p":
            Z = p
        elif field == "speed":
            Z = np.sqrt(u * u + v * v)
    else:
        raise ValueError(f"Unknown field: {field}")

    return X, Y, Z.reshape(ny, nx)


def get_snapshot_times(num_files: int, t_end: float | None):
    if t_end is None:
        return [None] * num_files

    return [
        t_end * float(k + 1) / float(num_files)
        for k in range(num_files)
    ]


def plot_series(
    input_pattern: str,
    field: str,
    output: str,
    levels,
    t_end: float | None,
    draw_bubble: bool,
    contourf: bool,
    dpi: int,
):
    files = sorted(glob.glob(input_pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No files matched: {input_pattern}")

    case_name = infer_case_from_filename(files[0])
    times = get_snapshot_times(len(files), t_end)

    ncols = len(files)
    fig_width = max(3.0 * ncols, 8.0)
    fig_height = 2.2

    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height))

    if ncols == 1:
        axes = [axes]

    if levels is None:
        levels = default_levels(case_name, field)

    # shock-bubble initial bubble outline
    if draw_bubble:
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        bubble_r = 0.025
        bubble_cx = 0.035
        bubble_cy = 0.0445
        xb = bubble_cx + bubble_r * np.cos(theta)
        yb = bubble_cy + bubble_r * np.sin(theta)

    last_contour = None

    for ax, csv_path, time_value in zip(axes, files, times):
        X, Y, Z = load_snapshot(csv_path, field)

        if contourf:
            last_contour = ax.contourf(X, Y, Z, levels=levels)
        else:
            last_contour = ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.45)

        if draw_bubble and case_name == "shock_bubble":
            ax.plot(xb, yb, "k--", linewidth=0.8)

        sid = snapshot_id(csv_path)

        if time_value is None:
            title = f"snapshot {sid}"
        else:
            title = f"snapshot {sid}, t = {time_value:.4g}"

        ax.set_title(title, fontsize=10, pad=2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("black")

        print(f"Plotted {os.path.basename(csv_path)}")
    plt.tight_layout(w_pad=0.25, rect=[0, 0.12, 1, 0.88])

    if contourf and last_contour is not None:
        cax = fig.add_axes([0.30, 0.035, 0.40, 0.015])
        cbar = fig.colorbar(
            last_contour,
            cax=cax,
            orientation="horizontal",
        )
        cbar.set_label(field, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot multiple Euler snapshot CSV files in one vertical contour figure."
    )

    parser.add_argument(
        "--input",
        required=True,
        help='Glob pattern, e.g. "outputs/gpu_blast_wave_hll_n1_snapshot_*.csv"',
    )

    parser.add_argument(
        "--field",
        default="rho",
        choices=["rho", "rhou", "rhov", "E", "p", "u", "v", "speed"],
        help="Field to plot.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )

    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Physical final time. If provided, snapshot times are labelled as t_end*k/N.",
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help="Number of contour levels. If omitted, case-specific defaults are used.",
    )

    parser.add_argument(
        "--filled",
        action="store_true",
        help="Use filled contours instead of black contour lines.",
    )

    parser.add_argument(
        "--no-bubble",
        action="store_true",
        help="Do not draw initial bubble outline for shock_bubble.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )

    args = parser.parse_args()

    if args.levels is None:
        levels = None
    else:
        levels = args.levels

    plot_series(
        input_pattern=args.input,
        field=args.field,
        output=args.output,
        levels=levels,
        t_end=args.t_end,
        draw_bubble=not args.no_bubble,
        contourf=args.filled,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
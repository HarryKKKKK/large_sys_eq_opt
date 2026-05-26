#!/usr/bin/env python3
"""
Plot blast-wave snapshot series using black contour lines.

Example:

python3 visualization/plot_blast_wave_series.py \
  --input "outputs/gpu_blast_wave_hll_n1_snapshot_*.csv" \
  --field p \
  --t-end 0.2 \
  --levels 45 \
  --output figures/validation/blast_wave_hll_pressure.png

python3 visualization/plot_blast_wave_series.py \
  --input "outputs/gpu_blast_wave_hllc_n1_snapshot_*.csv" \
  --field rho \
  --t-end 0.2 \
  --levels 45 \
  --output figures/validation/blast_wave_hllc_density.png
"""

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


def snapshot_id(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"snapshot_(\d+)", name)
    if m:
        return int(m.group(1))
    return -1


def get_snapshot_times(num_files: int, t_end: float | None):
    if t_end is None:
        return [None] * num_files

    return [
        t_end * float(k + 1) / float(num_files)
        for k in range(num_files)
    ]


def default_levels(field: str):
    if field == "rho":
        return np.linspace(0.8, 6.5, 45)

    if field == "p":
        return np.linspace(1.0, 100.0, 45)

    if field == "speed":
        return 45

    return 45


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


def plot_blast_series(
    input_pattern: str,
    field: str,
    output: str,
    levels,
    t_end: float | None,
    dpi: int,
):
    files = sorted(glob.glob(input_pattern), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No files matched: {input_pattern}")

    times = get_snapshot_times(len(files), t_end)

    if levels is None:
        levels = default_levels(field)

    # One-row compact layout for square blast-wave snapshots.
    ncols = len(files)
    fig_width = 1.65 * ncols
    fig_height = 1.95

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.03},
    )

    axes = np.asarray(axes).reshape(-1)

    for ax, csv_path, time_value in zip(axes, files, times):
        X, Y, Z = load_snapshot(csv_path, field)

        ax.contour(
            X,
            Y,
            Z,
            levels=levels,
            colors="k",
            linewidths=0.45,
        )

        sid = snapshot_id(csv_path)

        if time_value is None:
            title = f"snapshot {sid}"
        else:
            title = f"snapshot {sid}, t = {time_value:.3g}"

        ax.set_title(title, fontsize=8, pad=1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
            spine.set_color("black")

        print(f"Plotted {os.path.basename(csv_path)}")

    # Small margins, no suptitle, no colour bar.
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.03,
        top=0.86,
        wspace=0.03,
    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Saved figure: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot blast-wave snapshots in one compact row using black contour lines."
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
        help="Number of contour levels. If omitted, field-specific defaults are used.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )

    args = parser.parse_args()

    plot_blast_series(
        input_pattern=args.input,
        field=args.field,
        output=args.output,
        levels=args.levels,
        t_end=args.t_end,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
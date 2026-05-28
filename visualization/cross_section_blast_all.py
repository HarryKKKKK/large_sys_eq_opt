#!/usr/bin/env python3
"""
Plot CPU-only half cross-sections of density for the blast_wave test case.

This script plots one solver across all available CPU snapshots. For each
snapshot, it takes a centre-line cut and keeps only half of the section, so the
horizontal coordinate becomes a radial distance from the domain centre.

Expected CSV columns:
    i,j,x,y,rho,rhou,rhov,E

Typical usage:
    python visualization/cross_section_blast_all.py outputs \
      --solver exact \
      --final-time 0.2

Assumed filename pattern:
    cpu_blast_wave_exact_snapshot_1.csv
    cpu_blast_wave_exact_snapshot_2.csv
    ...
"""

import os
import re
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FINAL_TIME = 0.2
SNAPSHOT_RE = re.compile(r"_snapshot_(\d+)\.csv$")


def load_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["i", "j", "x", "y", "rho"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def build_grid(df: pd.DataFrame):
    """Return sorted x/y coordinates and rho[j, i]."""
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
    return x_vals, y_vals, rho


def snapshot_number(path: str) -> int:
    m = SNAPSHOT_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse snapshot number from filename: {path}")
    return int(m.group(1))


def find_cpu_files(input_dir: str, solver: str, snapshots: str):
    """
    Strictly find CPU files only.

    This intentionally does not support GPU or both, so it can never pick up
    gpu_blast_wave_* files by accident.
    """
    solver = solver.lower()
    pattern = os.path.join(input_dir, f"cpu_blast_wave_{solver}_n1_snapshot_*.csv")
    files = sorted(glob.glob(pattern), key=snapshot_number)

    if snapshots.lower() != "all":
        requested = {int(s.strip()) for s in snapshots.split(",") if s.strip()}
        files = [p for p in files if snapshot_number(p) in requested]

    if not files:
        raise FileNotFoundError(
            f"No CPU files found matching pattern:\n  {pattern}\n"
            f"Expected names like:\n  cpu_blast_wave_{solver}_n1_snapshot_1.csv"
        )

    return files, pattern


def get_half_cross_section(x_vals, y_vals, rho, axis: str, coord, half: str):
    """
    axis='x': take rho(x) at fixed y, then keep left/right half from x-centre.
    axis='y': take rho(y) at fixed x, then keep bottom/top half from y-centre.

    The returned position is radial distance from the domain centre, not absolute x/y.
    """
    if axis == "x":
        if coord is None:
            coord = 0.5 * (float(y_vals.min()) + float(y_vals.max()))
        line_idx = int(np.argmin(np.abs(y_vals - coord)))
        actual_cut = float(y_vals[line_idx])

        centre = 0.5 * (float(x_vals.min()) + float(x_vals.max()))
        position = x_vals.astype(float)
        density = rho[line_idx, :]

        if half == "right":
            mask = position >= centre
            radial = position[mask] - centre
            density = density[mask]
            half_desc = "right half"
        elif half == "left":
            mask = position <= centre
            radial = centre - position[mask]
            density = density[mask]
            order = np.argsort(radial)
            radial = radial[order]
            density = density[order]
            half_desc = "left half"
        else:
            raise ValueError("For --axis x, --half must be left or right")

        xlabel = "radial position from centre, r"
        cut_desc = f"horizontal centre-line cut: y = {actual_cut:.8g}, {half_desc}"

    elif axis == "y":
        if coord is None:
            coord = 0.5 * (float(x_vals.min()) + float(x_vals.max()))
        line_idx = int(np.argmin(np.abs(x_vals - coord)))
        actual_cut = float(x_vals[line_idx])

        centre = 0.5 * (float(y_vals.min()) + float(y_vals.max()))
        position = y_vals.astype(float)
        density = rho[:, line_idx]

        if half == "top":
            mask = position >= centre
            radial = position[mask] - centre
            density = density[mask]
            half_desc = "top half"
        elif half == "bottom":
            mask = position <= centre
            radial = centre - position[mask]
            density = density[mask]
            order = np.argsort(radial)
            radial = radial[order]
            density = density[order]
            half_desc = "bottom half"
        else:
            raise ValueError("For --axis y, --half must be bottom or top")

        xlabel = "radial position from centre, r"
        cut_desc = f"vertical centre-line cut: x = {actual_cut:.8g}, {half_desc}"

    else:
        raise ValueError("axis must be either 'x' or 'y'")

    return radial, density, xlabel, cut_desc


def physical_time(snapshot_id: int, final_time: float, num_snapshots: int) -> float:
    """
    The code normally writes equally spaced snapshots ending at final_time.
    Therefore snapshot k corresponds to k * final_time / num_snapshots.
    """
    return snapshot_id * final_time / num_snapshots


def auto_subplot_shape(n: int):
    if n <= 3:
        return n, 1
    ncols = 2
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot CPU-only half density cross-sections for one blast_wave solver across snapshots."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        help="Directory containing CPU snapshot CSV files, e.g. outputs/"
    )
    parser.add_argument(
        "--solver",
        required=True,
        help="Solver to plot, e.g. hll, hllc, exact, force."
    )
    parser.add_argument(
        "--snapshots",
        default="all",
        help="Snapshots to plot, e.g. all or 1,2,3,4,5. Default: all."
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=DEFAULT_FINAL_TIME,
        help="Physical final time. Snapshot times are assumed equally spaced. Default: 0.2."
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=None,
        help="Total number of equally spaced snapshots. Default: inferred from max snapshot number found."
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y"],
        default="x",
        help="Cross-section direction. axis=x plots rho along x at fixed y. Default: x."
    )
    parser.add_argument(
        "--coord",
        type=float,
        default=None,
        help="Fixed coordinate of the cut. For --axis x this is y; for --axis y this is x. Default: domain midline."
    )
    parser.add_argument(
        "--half",
        choices=["right", "left", "top", "bottom"],
        default="right",
        help="Which half-section to plot. For --axis x use left/right; for --axis y use bottom/top. Default: right."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image filename. Default: saved inside input_dir."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution. Default: 300."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir:
        input_dir = input("Please enter the snapshot directory: ").strip()

    if not input_dir:
        raise ValueError("No input directory provided.")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    if args.axis == "x" and args.half not in ("left", "right"):
        raise ValueError("For --axis x, please use --half left or --half right")
    if args.axis == "y" and args.half not in ("bottom", "top"):
        raise ValueError("For --axis y, please use --half bottom or --half top")

    return args, input_dir


def main():
    args, input_dir = get_args()

    files, pattern = find_cpu_files(input_dir, args.solver, args.snapshots)
    snapshot_ids = [snapshot_number(p) for p in files]
    num_snapshots = args.num_snapshots if args.num_snapshots is not None else max(snapshot_ids)

    nplots = len(files)
    nrows, ncols = auto_subplot_shape(nplots)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    global_xlabel = None
    cut_desc_used = None

    for ax_idx, path in enumerate(files):
        ax = axes_flat[ax_idx]
        snapshot_id = snapshot_number(path)
        t = physical_time(snapshot_id, args.final_time, num_snapshots)

        df = load_snapshot(path)
        x_vals, y_vals, rho = build_grid(df)
        radial, density, xlabel, cut_desc = get_half_cross_section(
            x_vals, y_vals, rho, axis=args.axis, coord=args.coord, half=args.half
        )

        global_xlabel = xlabel
        cut_desc_used = cut_desc

        ax.plot(radial, density, marker="o", markersize=2.0, linewidth=1.1)
        ax.set_title(f"t = {t:.6f}", fontsize=11)
        ax.set_ylabel("density")
        ax.grid(True, linewidth=0.4, alpha=0.5)

        print(f"Plotted CPU file {os.path.basename(path)} at t={t:.8g} using {cut_desc}")

    for k in range(nplots, len(axes_flat)):
        axes_flat[k].set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible() and global_xlabel:
            ax.set_xlabel(global_xlabel)

    fig.suptitle(
        f"Blast wave CPU density half cross-sections, solver = {args.solver.upper()}\n"
        f"{cut_desc_used}; final_time = {args.final_time:g}, equally spaced snapshots",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if args.output is None:
        output_name = f"blast_wave_cpu_{args.solver.lower()}_density_half_cross_sections.png"
        output_path = os.path.join(input_dir, output_name)
    else:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(input_dir, args.output)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    main()

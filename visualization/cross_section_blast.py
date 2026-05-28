#!/usr/bin/env python3
"""
Plot CPU-only 45-degree radial density cuts for the blast_wave test case,
comparing different Riemann solvers at one selected snapshot.

This version makes one subplot per solver. With the default four solvers
(hll, hllc, force, exact), the figure is arranged as a 2 x 2 grid.

The default cut is a ray starting from the domain centre and going in the
45-degree direction. The horizontal coordinate is the distance from the
centre along that ray.

The y-axis scale and y-axis limits are configured inside this script using
Y_SCALE, Y_LIMITS, and SHARE_AUTO_Y_LIMITS near the top of the file.

Expected CSV columns:
    i,j,x,y,rho,rhou,rhov,E

Expected filename pattern used by this script:
    cpu_blast_wave_hll_n1_snapshot_5.csv
    cpu_blast_wave_hllc_n1_snapshot_5.csv
    cpu_blast_wave_force_n1_snapshot_5.csv
    cpu_blast_wave_exact_n1_snapshot_5.csv

Typical usage:
    python visualization/cross_section_blast_solvers_snapshot_cpu_2x2_45deg.py outputs \
        --snapshot 5 \
        --final-time 0.2 \
        --shared-limits

    # Use another angle, e.g. 30 degrees from the positive x-axis
    python visualization/cross_section_blast_solvers_snapshot_cpu_2x2_45deg.py outputs \
        --snapshot 5 \
        --angle-deg 30 \
        --final-time 0.2
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
DEFAULT_NUM_SNAPSHOTS = 5
DEFAULT_SOLVERS = ["hll", "hllc", "force", "exact"]
SNAPSHOT_RE = re.compile(r"_snapshot_(\d+)\.csv$")

# ============================================================
# Plot scale configuration
# Modify these values directly in this script instead of using
# command-line arguments.
# ============================================================

# Options supported by matplotlib: "linear", "log", "symlog", "logit".
# For blast-wave density, "linear" is usually safest.
Y_SCALE = "linear"

# Manually set y-axis limits for every subplot.
# Use None for automatic limits.
#
# Examples:
#   Y_LIMITS = None
#   Y_LIMITS = (0.0, 1.2)     # zoom into low-density region
#   Y_LIMITS = (0.8, 4.2)     # focus on shock/peak region
Y_LIMITS = None

# If Y_LIMITS is None and this is True, all subplots use the same
# automatically detected y-limits.
# This is useful for solver-to-solver comparison.
SHARE_AUTO_Y_LIMITS = True

# Padding used for shared automatic y-limits.
AUTO_Y_PADDING_FRAC = 0.05


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


def parse_solver_from_filename(path: str) -> str:
    """Parse solver from cpu_blast_wave_<solver>_n1_snapshot_<k>.csv."""
    name = os.path.basename(path)
    snap = snapshot_number(path)

    prefix = "cpu_blast_wave_"
    suffix = f"_n1_snapshot_{snap}.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]

    # fallback for names without n1: cpu_blast_wave_<solver>_snapshot_<k>.csv
    suffix_no_n = f"_snapshot_{snap}.csv"
    if name.startswith(prefix) and name.endswith(suffix_no_n):
        return name[len(prefix):-len(suffix_no_n)]

    return name.replace(".csv", "")


def parse_solvers(solvers_arg: str):
    if solvers_arg.lower() == "all":
        return DEFAULT_SOLVERS
    return [s.strip().lower() for s in solvers_arg.split(",") if s.strip()]


def find_solver_files(input_dir: str, snapshot: int, solvers):
    """
    CPU-only file search: one file per requested solver if available.
    Prefer filenames with _n1_, but also accept old names without _n1_.
    """
    files = []
    missing = []

    for solver in solvers:
        preferred = os.path.join(input_dir, f"cpu_blast_wave_{solver}_n1_snapshot_{snapshot}.csv")
        fallback = os.path.join(input_dir, f"cpu_blast_wave_{solver}_snapshot_{snapshot}.csv")

        matches = sorted(glob.glob(preferred))
        if not matches:
            matches = sorted(glob.glob(fallback))

        if matches:
            files.append(matches[0])
        else:
            missing.append(f"{preferred}  OR  {fallback}")

    return files, missing


def max_ray_distance_to_domain(cx: float, cy: float, dx: float, dy: float,
                               xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    """Maximum positive distance s such that (cx+s*dx, cy+s*dy) remains in the domain."""
    candidates = []

    if dx > 0:
        candidates.append((xmax - cx) / dx)
    elif dx < 0:
        candidates.append((xmin - cx) / dx)

    if dy > 0:
        candidates.append((ymax - cy) / dy)
    elif dy < 0:
        candidates.append((ymin - cy) / dy)

    candidates = [s for s in candidates if s >= 0]
    if not candidates:
        raise ValueError("Cannot determine ray length. Check the angle and domain bounds.")

    return float(min(candidates))


def bilinear_interpolate(x_vals, y_vals, field, xq, yq):
    """
    Bilinear interpolation on a structured Cartesian grid.
    field is indexed as field[j, i], with x along columns and y along rows.

    The query points are allowed to be only a tiny floating-point tolerance
    outside the cell-centre coordinate range. This avoids failing at the last
    sample of a diagonal ray, where round-off can produce e.g. yq = ymax + 1e-17.
    """
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)

    xmin, xmax = float(x_vals[0]), float(x_vals[-1])
    ymin, ymax = float(y_vals[0]), float(y_vals[-1])
    tol = 1.0e-10 * max(1.0, abs(xmax - xmin), abs(ymax - ymin))

    if (
        np.any(xq < xmin - tol) or np.any(xq > xmax + tol)
        or np.any(yq < ymin - tol) or np.any(yq > ymax + tol)
    ):
        raise ValueError(
            "Some interpolation points are outside the grid domain. "
            "Try using a smaller --samples value or check the cut angle/domain."
        )

    # Clip only tiny round-off excursions back onto the valid cell-centre domain.
    xq = np.clip(xq, xmin, xmax)
    yq = np.clip(yq, ymin, ymax)

    ix = np.searchsorted(x_vals, xq, side="right") - 1
    iy = np.searchsorted(y_vals, yq, side="right") - 1

    ix = np.clip(ix, 0, len(x_vals) - 2)
    iy = np.clip(iy, 0, len(y_vals) - 2)

    x0 = x_vals[ix]
    x1 = x_vals[ix + 1]
    y0 = y_vals[iy]
    y1 = y_vals[iy + 1]

    # Avoid division by zero, although x/y values should be unique.
    tx = np.where(x1 != x0, (xq - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 != y0, (yq - y0) / (y1 - y0), 0.0)

    f00 = field[iy, ix]
    f10 = field[iy, ix + 1]
    f01 = field[iy + 1, ix]
    f11 = field[iy + 1, ix + 1]

    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def get_angle_radial_section(x_vals, y_vals, rho, angle_deg: float, samples: int):
    """
    Take a ray from the domain centre at angle_deg from the positive x-axis.
    Return distance from centre and interpolated density along the ray.
    """
    xmin, xmax = float(x_vals.min()), float(x_vals.max())
    ymin, ymax = float(y_vals.min()), float(y_vals.max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    smax = max_ray_distance_to_domain(cx, cy, dx, dy, xmin, xmax, ymin, ymax)

    # Do not put the final point exactly on the limiting boundary. On a diagonal
    # cut, floating-point round-off can otherwise make the final x/y query just
    # outside the grid by ~1e-17 and trigger a false interpolation error.
    smax_safe = smax * (1.0 - 1.0e-12)
    radial = np.linspace(0.0, smax_safe, samples)
    xq = cx + radial * dx
    yq = cy + radial * dy

    density = bilinear_interpolate(x_vals, y_vals, rho, xq, yq)

    xlabel = "distance from centre along cut, r"
    cut_desc = (
        f"{angle_deg:g} degree radial cut from centre "
        f"(x_c={cx:.8g}, y_c={cy:.8g})"
    )
    return radial, density, xlabel, cut_desc


def physical_time(snapshot_id: int, final_time: float, num_snapshots: int) -> float:
    """
    If snapshots are equally spaced and snapshot num_snapshots is final_time,
    snapshot k corresponds to k * final_time / num_snapshots.
    """
    return snapshot_id * final_time / num_snapshots


def subplot_shape(n: int):
    """Use two plots per row by default."""
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def apply_y_axis_config(ax, shared_ylim=None):
    """
    Apply y-axis scale and limits from the global script-level configuration.
    The user can modify Y_SCALE, Y_LIMITS, and SHARE_AUTO_Y_LIMITS at the top
    of this file without changing command-line arguments.
    """
    ax.set_yscale(Y_SCALE)

    if Y_LIMITS is not None:
        ax.set_ylim(Y_LIMITS[0], Y_LIMITS[1])
    elif shared_ylim is not None:
        ax.set_ylim(shared_ylim[0], shared_ylim[1])


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot CPU-only blast_wave density radial cuts for different solvers at one snapshot."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        help="Directory containing snapshot CSV files, e.g. outputs/"
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        default=5,
        help="Snapshot number to plot. Default: 5."
    )
    parser.add_argument(
        "--solvers",
        default="all",
        help="Comma-separated solvers, e.g. hll,hllc,force,exact, or all. Default: all."
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=DEFAULT_FINAL_TIME,
        help="Physical final time. Default: 0.2."
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=DEFAULT_NUM_SNAPSHOTS,
        help="Total number of equally spaced snapshots. Default: 5."
    )
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=45.0,
        help="Angle of the radial cut measured counter-clockwise from the positive x-axis. Default: 45."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of interpolation points along the radial cut. Default: 1000."
    )
    parser.add_argument(
        "--shared-limits",
        action="store_true",
        help="Use the same x/y limits for all subplots. Recommended for comparing solvers."
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

    if args.snapshot < 1:
        raise ValueError("--snapshot must be >= 1")
    if args.num_snapshots < args.snapshot:
        raise ValueError("--num-snapshots must be >= --snapshot")
    if args.samples < 2:
        raise ValueError("--samples must be >= 2")

    return args, input_dir


def main():
    args, input_dir = get_args()
    solvers = parse_solvers(args.solvers)

    files, missing = find_solver_files(input_dir, args.snapshot, solvers)
    if not files:
        tried = "\n".join(missing)
        raise FileNotFoundError(
            "No CPU files found for the requested snapshot/solvers.\n"
            f"Tried patterns:\n{tried}\n"
            "Expected names like: cpu_blast_wave_hllc_n1_snapshot_5.csv"
        )

    if missing:
        print("Warning: some requested solver files were not found:")
        for p in missing:
            print(f"  missing: {p}")

    t = physical_time(args.snapshot, args.final_time, args.num_snapshots)

    # Load all curves first so that optional shared axis limits can be applied.
    curves = []
    xlabel_used = "distance from centre along cut, r"
    cut_desc_used = None

    for path in files:
        solver = parse_solver_from_filename(path)
        df = load_snapshot(path)
        x_vals, y_vals, rho = build_grid(df)
        radial, density, xlabel, cut_desc = get_angle_radial_section(
            x_vals,
            y_vals,
            rho,
            angle_deg=args.angle_deg,
            samples=args.samples,
        )

        xlabel_used = xlabel
        cut_desc_used = cut_desc
        curves.append((solver, path, radial, density))
        print(f"Loaded CPU file {os.path.basename(path)} at t={t:.8g} using {cut_desc}")

    nplots = len(curves)
    nrows, ncols = subplot_shape(nplots)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    if args.shared_limits:
        xmin = min(float(np.min(radial)) for _, _, radial, _ in curves)
        xmax = max(float(np.max(radial)) for _, _, radial, _ in curves)
    else:
        xmin = xmax = None

    if Y_LIMITS is None and (args.shared_limits or SHARE_AUTO_Y_LIMITS):
        ymin = min(float(np.min(density)) for _, _, _, density in curves)
        ymax = max(float(np.max(density)) for _, _, _, density in curves)
        ypad = AUTO_Y_PADDING_FRAC * (ymax - ymin) if ymax > ymin else 0.05
        shared_ylim = (ymin - ypad, ymax + ypad)
    else:
        shared_ylim = None

    for ax_idx, (solver, path, radial, density) in enumerate(curves):
        ax = axes_flat[ax_idx]

        ax.plot(
            radial,
            density,
            marker="o",
            linestyle="None",
            markersize=1.6,
            linewidth=1.2,
        )

        ax.set_title(f"{solver.upper()}, t = {t:.6f}", fontsize=12)
        ax.set_ylabel("density")
        ax.grid(True, linewidth=0.4, alpha=0.5)

        if args.shared_limits:
            ax.set_xlim(xmin, xmax)

        apply_y_axis_config(ax, shared_ylim=shared_ylim)

        row = ax_idx // ncols
        if row == nrows - 1:
            ax.set_xlabel(xlabel_used)

    for k in range(nplots, len(axes_flat)):
        axes_flat[k].set_visible(False)

    # fig.suptitle(
    #     f"Blast wave CPU density 45-degree radial cuts, snapshot {args.snapshot}/{args.num_snapshots}\n"
    #     f"{cut_desc_used}; final_time = {args.final_time:g}, physical time t = {t:.6f}",
    #     fontsize=14,
    #     y=0.995,
    # )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if args.output is None:
        solver_part = "all_solvers" if args.solvers.lower() == "all" else "_".join(solvers)
        angle_part = str(args.angle_deg).replace(".", "p").replace("-", "m")
        output_name = (
            f"blast_wave_cpu_{solver_part}_snapshot_{args.snapshot}"
            f"_t_{t:.6f}_density_{angle_part}deg_radial_cut_2x2.png"
        )
        output_path = os.path.join(input_dir, output_name)
    else:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(input_dir, args.output)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    main()

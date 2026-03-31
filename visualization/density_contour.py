import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    filename = "shock_bubble_cpu_final.csv"

    # Read CSV
    df = pd.read_csv(filename)

    # Extract columns
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    rho = df["rho"].to_numpy()

    # Recover structured grid dimensions
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    nx = len(x_unique)
    ny = len(y_unique)

    # Reshape to 2D arrays
    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    RHO = rho.reshape(ny, nx)

    # Plot filled contour
    plt.figure(figsize=(10, 4))
    contour = plt.contourf(X, Y, RHO, levels=50)
    plt.colorbar(contour, label="Density")

    # Optional contour lines on top
    plt.contour(X, Y, RHO, levels=20, linewidths=0.4)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Density Contour: Shock-Bubble CPU Result")
    plt.tight_layout()
    plt.savefig("density_contour")


if __name__ == "__main__":
    main()
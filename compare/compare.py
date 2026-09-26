#!/usr/bin/env python3

import sys
import numpy as np


def compare_files(file1, file2):
    # Load only the first 3 columns
    data1 = np.loadtxt(file1, usecols=(0, 1, 2))
    data2 = np.loadtxt(file2, usecols=(0, 1, 2))

    if data1.shape != data2.shape:
        raise ValueError(
            f"Files have different shapes: "
            f"{data1.shape} vs {data2.shape}"
        )

    # Absolute differences
    diff = np.abs(data1 - data2)

    print(f"Comparing:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print(f"  Rows:   {data1.shape[0]}")
    print()

    print("Column        Avg difference        Max difference")
    print("--------------------------------------------------")

    for i in range(3):
        avg_diff = np.mean(diff[:, i])
        max_diff = np.max(diff[:, i])

        print(
            f"{i + 1:>6}        "
            f"{avg_diff:>15.10f}        "
            f"{max_diff:>15.10f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} file1 file2")
        sys.exit(1)

    compare_files(sys.argv[1], sys.argv[2])

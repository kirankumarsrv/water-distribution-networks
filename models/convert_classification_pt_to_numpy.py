"""Convert a saved PyTorch classification dataset into NumPy files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def convert(pt_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
        data = torch.load(pt_path, weights_only=False)

    X = data["X"]
    y = data["y"]
    np.save(out_dir / "X_classification.npy", X)
    np.save(out_dir / "y_classification.npy", y)
    np.savez(out_dir / "classification_B.npz", X=X, y=y)

    print("Converted:")
    print(" -", pt_path)
    print("To:")
    print(" -", out_dir / "X_classification.npy")
    print(" -", out_dir / "y_classification.npy")
    print(" -", out_dir / "classification_B.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert classification_B.pt to plain NumPy files.")
    parser.add_argument("--pt-path", default="DATASETS/classification_B.pt", help="Path to the .pt dataset file")
    parser.add_argument("--out-dir", default="DATASETS", help="Output directory for NumPy files")
    args = parser.parse_args()

    convert(Path(args.pt_path), Path(args.out_dir))


if __name__ == "__main__":
    main()

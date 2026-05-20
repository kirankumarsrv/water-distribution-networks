"""Create a cleaned leak-detection dataset by removing any features derived
from simulated leak flow (`Q_leak`).

Saves:
- DATASETS/X_classification_no_leak.npy
- DATASETS/feature_names_no_leak.json

Also prints a short summary of removed features.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATASETS"


def main() -> int:
    feat_path = DATA_DIR / "feature_names.json"
    x_path = DATA_DIR / "X_classification.npy"

    if not feat_path.exists() or not x_path.exists():
        print("Missing feature_names.json or X_classification.npy in DATASETS/", file=sys.stderr)
        return 2

    feature_names = json.loads(feat_path.read_text())
    X = np.load(x_path)

    if X.ndim != 2:
        print("Unexpected X_classification shape", X.shape, file=sys.stderr)
        return 3

    if X.shape[1] != len(feature_names):
        print(f"Warning: feature count mismatch: X has {X.shape[1]} cols, names list has {len(feature_names)}", file=sys.stderr)

    # Identify leak-derived features by substring match
    leak_inds = [i for i, n in enumerate(feature_names) if "leak" in n.lower() or "q_leak" in n.lower()]
    leak_names = [feature_names[i] for i in leak_inds]

    keep_inds = [i for i in range(len(feature_names)) if i not in set(leak_inds)]
    cleaned_names = [feature_names[i] for i in keep_inds]

    X_clean = X[:, keep_inds]

    out_x = DATA_DIR / "X_classification_no_leak.npy"
    out_names = DATA_DIR / "feature_names_no_leak.json"

    np.save(out_x, X_clean)
    out_names.write_text(json.dumps(cleaned_names, indent=2))

    print("Saved cleaned dataset:")
    print(" -", out_x.relative_to(ROOT), "shape=", X_clean.shape)
    print(" -", out_names.relative_to(ROOT), "count=", len(cleaned_names))
    print()
    print("Removed features (count=", len(leak_names), "):")
    for n in leak_names:
        print(" -", n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

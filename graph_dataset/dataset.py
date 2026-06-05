import json
import os
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT_DIR))

from integration.EPANET_Integration import EPANETIntegrator
from physics.FeatureExtractor import FeatureExtractor
from physics.LocalizationFeatureExtractor import LocalizationFeatureExtractor

INP_FILE = ROOT_DIR / "EPANETINPUTFILESFOR7NEWORKS" / "2_Extended Hanoi.inp"
N_SAMPLES = 3600
SAVE_DIR = ROOT_DIR / "DATASETS"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)


def df_to_graph(df: pd.DataFrame, wn):
    try:
        from torch_geometric.data import Data
    except ImportError:
        return None

    pipe_names = df["pipe"].tolist()
    if len(pipe_names) == 0:
        return None

    pipe_to_idx = {name: idx for idx, name in enumerate(pipe_names)}
    features = []
    for _, row in df.iterrows():
        features.append([
            float(row.get("Q1", 0.0)),
            float(row.get("Q2", 0.0)),
            float(row.get("Q_leak", 0.0)),
            float(row.get("Hm", 0.0)),
            float(row.get("f", 0.0)),
            float(row.get("Q_EPANET", 0.0)),
            float(row.get("H_in", 0.0)),
            float(row.get("H_out", 0.0)),
        ])

    x = torch.tensor(features, dtype=torch.float32)
    links = dict(wn.links())
    edges = []

    for i, p1 in enumerate(pipe_names):
        pipe1 = links.get(p1)
        if pipe1 is None:
            continue
        nodes1 = {pipe1.start_node_name, pipe1.end_node_name}
        for j, p2 in enumerate(pipe_names):
            if i >= j:
                continue
            pipe2 = links.get(p2)
            if pipe2 is None:
                continue
            nodes2 = {pipe2.start_node_name, pipe2.end_node_name}
            if nodes1.intersection(nodes2):
                edges.append([i, j])
                edges.append([j, i])

    if len(edges) == 0:
        edges.append([0, 0])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    cracked_rows = df[df["cracked"] == True]
    if len(cracked_rows) == 0:
        return None

    leak_pipe = cracked_rows.iloc[0]["pipe"]
    y = torch.tensor([pipe_to_idx[leak_pipe]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = x.shape[0]
    return data


def build_baseline(sample_records):
    totals = {}
    counts = {}
    for record in sample_records:
        if record["scenario"] != "normal":
            continue
        for _, row in record["df"].iterrows():
            pipe = row["pipe"]
            totals[pipe] = totals.get(pipe, 0.0) + float(row.get("Hm", 0.0))
            counts[pipe] = counts.get(pipe, 0) + 1
    return {pipe: totals[pipe] / max(1, counts[pipe]) for pipe in totals}


def stratified_split(X, y, seed=SEED):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO + TEST_RATIO, random_state=seed)
    train_idx, hold_idx = next(sss.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_hold, y_hold = X[hold_idx], y[hold_idx]

    ratio = TEST_RATIO / (TEST_RATIO + VAL_RATIO)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    val_idx, test_idx = next(sss2.split(X_hold, y_hold))
    X_val, y_val = X_hold[val_idx], y_hold[val_idx]
    X_test, y_test = X_hold[test_idx], y_hold[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_npy_arrays(prefix: str, arrays: dict[str, np.ndarray]) -> None:
    for name, array in arrays.items():
        filename = f"{prefix}_{name}.npy" if prefix else f"{name}.npy"
        path = SAVE_DIR / filename
        np.save(path, array)


def save_classification_splits(X, y):
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    save_npy_arrays("", {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    })


def save_localization_splits(X, y):
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    save_npy_arrays("", {
        "X_loc_train": X_train,
        "X_loc_val": X_val,
        "X_loc_test": X_test,
        "y_loc_train": y_train,
        "y_loc_val": y_val,
        "y_loc_test": y_test,
    })


def build_dataset():
    integrator = EPANETIntegrator(str(INP_FILE))
    scenario_types = ["normal", "leak", "burst", "blockage"]
    sample_records = []
    all_pipes = [p for p, pipe in integrator.wn.links() if pipe.link_type == "Pipe"]
    if len(all_pipes) == 0:
        raise RuntimeError("No pipe objects found in the network")

    zone_map = {pipe_name: idx + 1 for idx, pipe_name in enumerate(sorted(all_pipes))}

    per_class = N_SAMPLES // len(scenario_types)
    for scen in scenario_types:
        for sample_idx in range(per_class):
            target_pipe = None
            if scen != "normal":
                target_pipe = random.choice(all_pipes)
            try:
                integrator.run_simulation()
                heads = integrator.results.node["head"]
                flows = integrator.results.link["flowrate"]
                t = heads.index[0]
                rows = []
                for pipe_name, pipe in integrator.wn.links():
                    if pipe.link_type != "Pipe":
                        continue
                    L = pipe.length
                    D = pipe.diameter
                    if L <= 0 or D <= 0:
                        continue
                    start = pipe.start_node_name
                    end = pipe.end_node_name
                    H_in = float(heads.loc[t, start])
                    H_out = float(heads.loc[t, end])
                    Q_epanet = float(flows.loc[t, pipe_name])
                    if H_out > H_in:
                        H_in, H_out = H_out, H_in
                    if abs(Q_epanet) < 1e-12:
                        Q_epanet = 0.0
                    hf = max(abs(H_in - H_out), 1e-8)
                    R = hf / (Q_epanet ** 2) if Q_epanet != 0 else 1e8
                    g = 9.81
                    f = (R * np.pi**2 * g * D**5) / (8 * L)
                    if not np.isfinite(f) or f <= 0:
                        f = 0.02
                    if scen == "normal":
                        A_leak = 0.0
                        Cd = 0.0
                        x_leak = L / 2
                        blocked = False
                    elif scen == "leak":
                        if pipe_name == target_pipe:
                            A_leak = np.random.uniform(1e-8, 5e-7) * (D ** 2)
                            Cd = np.random.uniform(0.60, 0.80)
                            x_leak = np.random.uniform(0.05 * L, 0.95 * L)
                        else:
                            A_leak = 0.0
                            Cd = 0.0
                            x_leak = L / 2
                        blocked = False
                    elif scen == "burst":
                        if pipe_name == target_pipe:
                            A_leak = np.random.uniform(1e-5, 5e-4) * (D ** 2)
                            Cd = np.random.uniform(0.8, 1.0)
                            x_leak = np.random.uniform(0.05 * L, 0.95 * L)
                        else:
                            A_leak = 0.0
                            Cd = 0.0
                            x_leak = L / 2
                        blocked = False
                    else:
                        if pipe_name == target_pipe:
                            f = f * np.random.uniform(5.0, 50.0)
                            blocked = True
                        else:
                            blocked = False
                        A_leak = 0.0
                        Cd = 0.0
                        x_leak = L / 2

                    from physics.PipeModel import PipeLeakModel
                    pipe_model = PipeLeakModel(L=L, D=D, f=f, A_leak=A_leak, Cd=Cd, x_leak=x_leak)
                    Q1, Q2, Q_leak, Hm = pipe_model.solve_model(H_in, H_out)
                    sign = np.sign(Q_epanet)
                    Q1 *= sign
                    Q2 *= sign
                    rows.append({
                        "pipe": pipe_name,
                        "cracked": bool(A_leak > 0),
                        "H_in": H_in,
                        "H_out": H_out,
                        "Q_EPANET": Q_epanet,
                        "Q1": Q1,
                        "Q2": Q2,
                        "Q_leak": Q_leak,
                        "Hm": Hm,
                        "A_leak": A_leak,
                        "Cd": Cd,
                        "f": f,
                        "x_leak": x_leak,
                        "blocked": blocked,
                    })
                sample_records.append({
                    "scenario": scen,
                    "target_pipe": target_pipe,
                    "df": pd.DataFrame(rows),
                })
            except Exception as error:
                print("Scenario failed:", error)
                continue

    baseline_hm = build_baseline(sample_records)
    zone_definitions = {pipe: idx + 1 for idx, pipe in enumerate(sorted(all_pipes))}
    classification_rows = []
    localization_rows = []
    localization_graphs = []

    for record in sample_records:
        label = {"normal": 0, "leak": 1, "burst": 2, "blockage": 3}[record["scenario"]]
        df = record["df"]
        classification_rows.append((FeatureExtractor.build_detection_features(df), label))

        loc_label = 0
        if record["scenario"] != "normal" and record["target_pipe"]:
            loc_label = zone_definitions.get(record["target_pipe"], 0)
        localization_rows.append((LocalizationFeatureExtractor.build_localization_features(df, baseline_hm=baseline_hm), loc_label))

        if record["scenario"] != "normal":
            graph = df_to_graph(df, integrator.wn)
            if graph is not None:
                localization_graphs.append(graph)

    X = np.asarray([row[0] for row in classification_rows], dtype=np.float32)
    y = np.asarray([row[1] for row in classification_rows], dtype=np.int64)
    X_loc = np.asarray([row[0] for row in localization_rows], dtype=np.float32)
    y_loc = np.asarray([row[1] for row in localization_rows], dtype=np.int64)

    torch.save({"X": X, "y": y}, SAVE_DIR / "classification_B.pt")
    np.save(SAVE_DIR / "X_classification.npy", X)
    np.save(SAVE_DIR / "y_classification.npy", y)
    np.save(SAVE_DIR / "X_localization.npy", X_loc)
    np.save(SAVE_DIR / "y_localization.npy", y_loc)
    with open(SAVE_DIR / "zone_definitions.json", "w", encoding="utf-8") as handle:
        json.dump(zone_definitions, handle, indent=2)
    with open(SAVE_DIR / "baseline_pressure_reference.json", "w", encoding="utf-8") as handle:
        json.dump(baseline_hm, handle, indent=2)
    with open(SAVE_DIR / "feature_names.json", "w", encoding="utf-8") as handle:
        json.dump(FeatureExtractor.get_feature_names(sensor_count=len(all_pipes)), handle, indent=2)
    with open(SAVE_DIR / "localization_feature_names.json", "w", encoding="utf-8") as handle:
        json.dump(LocalizationFeatureExtractor.get_feature_names(sensor_count=len(all_pipes)), handle, indent=2)

    if localization_graphs:
        save_dataset(localization_graphs)

    save_classification_splits(X, y)
    save_localization_splits(X_loc, y_loc)

    print("Generated classification dataset:", X.shape, y.shape)
    print("Generated localization dataset:", X_loc.shape, y_loc.shape)
    print("Saved baseline and zone definitions to", SAVE_DIR)


def split_dataset(dataset):
    random.shuffle(dataset)
    n = len(dataset)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    train_set = dataset[:n_train]
    val_set = dataset[n_train : n_train + n_val]
    test_set = dataset[n_train + n_val :]
    return train_set, val_set, test_set


def save_dataset(dataset, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(dataset, os.path.join(save_dir, "water_dataset.pt"))
    train_set, val_set, test_set = split_dataset(dataset)
    torch.save(train_set, os.path.join(save_dir, "train.pt"))
    torch.save(val_set, os.path.join(save_dir, "val.pt"))
    torch.save(test_set, os.path.join(save_dir, "test.pt"))
    return train_set, val_set, test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training datasets from an EPANET input file")
    parser.add_argument("--inp-file", default=str(INP_FILE), help="Path to the EPANET .inp file")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Number of samples to generate")
    parser.add_argument("--save-dir", default=str(SAVE_DIR), help="Directory to save dataset files")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    INP_FILE = Path(args.inp_file)
    N_SAMPLES = args.samples
    SAVE_DIR = Path(args.save_dir)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Generating dataset from", INP_FILE)
    build_dataset()

# dataset.py
# Full corrected dataset generator using your existing:
#   - integration/EPANET_Integration.py
#   - physics/PipeModel.py
#
# Output:
#   DATASETS/
#       water_dataset.pt
#       train.pt
#       val.pt
#       test.pt

import os
import sys
import argparse
import torch
import random
import numpy as np
import pandas as pd
import math

# =====================================================
# SETTINGS
# =====================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from torch_geometric.data import Data
from integration.EPANET_Integration import EPANETIntegrator
from physics.PipeModel import PipeLeakModel

INP_FILE = os.path.join(ROOT_DIR, "EPANETINPUTFILESFOR7NEWORKS", "2_Extended Hanoi.inp")
N_SAMPLES = 2500               # number of scenarios
SAVE_DIR = os.path.join(ROOT_DIR, "DATASETS")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# CONVERT ONE DATAFRAME TO GRAPH
# =====================================================

def df_to_graph(df, wn):
    """
    Pipe rows -> graph nodes
    Pipes connected if share junction
    """

    # ------------------------------------------
    # pipe names
    # ------------------------------------------
    pipe_names = df["pipe"].tolist()

    if len(pipe_names) == 0:
        return None

    pipe_to_idx = {p: i for i, p in enumerate(pipe_names)}

    # ------------------------------------------
    # NODE FEATURES
    # use ONLY non-cheating features
    # ------------------------------------------
    feats = []

    for _, row in df.iterrows():

        feats.append([
            float(row["Q1"]),
            float(row["Q2"]),
            float(row["Q_leak"]),
            float(row["Hm"]),
            float(row["f"]),
            float(row["Q_EPANET"]),
            float(row["H_in"]),
            float(row["H_out"])
        ])

    x = torch.tensor(feats, dtype=torch.float32)

    # ------------------------------------------
    # CONNECTIVITY
    # ------------------------------------------
    links = dict(wn.links())

    edges = []

    for i, p1 in enumerate(pipe_names):

        pipe1 = links[p1]

        nodes1 = {
            pipe1.start_node_name,
            pipe1.end_node_name
        }

        for j, p2 in enumerate(pipe_names):

            if i >= j:
                continue

            pipe2 = links[p2]

            nodes2 = {
                pipe2.start_node_name,
                pipe2.end_node_name
            }

            # if share node => connect
            if len(nodes1.intersection(nodes2)) > 0:
                edges.append([i, j])
                edges.append([j, i])

    # fallback if isolated
    if len(edges) == 0:
        edges.append([0, 0])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    # ------------------------------------------
    # LABEL = cracked pipe index
    # ------------------------------------------
    cracked_rows = df[df["cracked"] == True]

    if len(cracked_rows) == 0:
        return None

    # choose first cracked pipe
    leak_pipe = cracked_rows.iloc[0]["pipe"]

    y = torch.tensor(
        [pipe_to_idx[leak_pipe]],
        dtype=torch.long
    )

    # ------------------------------------------
    # RETURN GRAPH
    # ------------------------------------------
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    data.num_nodes = x.shape[0]

    return data


def extract_classification_features(df):
    """
    Build a tabular feature vector for classification from the per-pipe dataframe.
    We intentionally DO NOT expose `Q_leak` to the classifier features.
    Returns: feature_list (python list), where caller will assemble X/y.
    """
    # pressures: use H_in and H_out and Hm
    H_in = df["H_in"].astype(float)
    H_out = df["H_out"].astype(float)
    Hm = df["Hm"].astype(float)

    Q_epanet = df["Q_EPANET"].abs().astype(float)
    Q1 = df["Q1"].abs().astype(float)
    Q2 = df["Q2"].abs().astype(float)
    f = df["f"].astype(float)

    feats = []
    # basic statistics
    feats.extend([
        H_in.mean(), H_in.std(), H_in.min(), H_in.max(),
        H_out.mean(), H_out.std(),
        Hm.mean(), Hm.std(),
        Q_epanet.mean(), Q_epanet.std(),
        Q1.mean(), Q1.std(),
        Q2.mean(), Q2.std(),
        f.mean()
    ])

    return feats


# =====================================================
# BUILD DATASET
# =====================================================

def build_dataset():

    # We will generate two datasets:
    # - localization_graphs: list of Data objects for Dataset C (graph localization)
    # - classification_rows: list of (feature_vector, label) for Dataset B (tabular classification)
    localization_graphs = []
    classification_rows = []

    integrator = EPANETIntegrator(INP_FILE)

    scenario_types = ["normal", "leak", "burst", "blockage"]

    for k in range(N_SAMPLES):

        scen_idx = k + 1
        print(f"Generating sample {scen_idx}/{N_SAMPLES}")

        try:
            # choose scenario type uniformly for now
            scen = np.random.choice(scenario_types)

            # run base EPANET simulation and collect steady-state results
            integrator.run_simulation()
            heads = integrator.results.node["head"]
            flows = integrator.results.link["flowrate"]
            t = heads.index[0]

            # create per-pipe results similar to EPANET_Integration.simulate_leak
            results_list = []

            all_pipes = [p for p, pipe in integrator.wn.links() if pipe.link_type == "Pipe"]
            if len(all_pipes) == 0:
                print("No pipes found, skipping")
                continue

            # select one affected pipe for scenarios other than normal
            target_pipe = None
            if scen != "normal":
                target_pipe = np.random.choice(all_pipes)

            for pipe_name, pipe in integrator.wn.links():
                if pipe.link_type != "Pipe":
                    continue

                try:
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
                        # keep but mark tiny flow
                        pass

                    hf = abs(H_in - H_out)
                    hf = max(hf, 1e-8)
                    R = hf / (Q_epanet ** 2) if Q_epanet != 0 else 1e8
                    g = 9.81
                    f = (R * math.pi**2 * g * D**5) / (8 * L)
                    if (not np.isfinite(f)) or f <= 0:
                        f = 0.02

                    # scenario-specific modifications
                    if scen == "normal":
                        A_leak = 0.0
                        Cd = 0.0
                        x_leak = L / 2
                        blocked = False
                    elif scen == "leak":
                        # small leak on target pipe only
                        if pipe_name == target_pipe:
                            A_leak = np.random.uniform(1e-8,5e-7)*(D**2)
                            Cd = np.random.uniform(0.60, 0.80)
                            x_leak = np.random.uniform(0.05*L, 0.95*L)
                        else:
                            A_leak = 0.0; Cd = 0.0; x_leak = L/2
                        blocked = False
                    elif scen == "burst":
                        if pipe_name == target_pipe:
                            A_leak = np.random.uniform(1e-5,5e-4)*(D**2)
                            Cd = np.random.uniform(0.8, 1.0)
                            x_leak = np.random.uniform(0.05*L, 0.95*L)
                        else:
                            A_leak = 0.0; Cd = 0.0; x_leak = L/2
                        blocked = False
                    elif scen == "blockage":
                        # model blockage by increasing friction for target pipe
                        if pipe_name == target_pipe:
                            f = f * np.random.uniform(5.0, 50.0)
                            blocked = True
                        else:
                            blocked = False
                        A_leak = 0.0; Cd = 0.0; x_leak = L/2

                    # solve pipe model using existing PipeLeakModel
                    pipe_model = PipeLeakModel(
                        L=L,
                        D=D,
                        f=f,
                        A_leak=A_leak,
                        Cd=Cd,
                        x_leak=x_leak
                    )

                    Q1, Q2, Q_leak, Hm = pipe_model.solve_model(H_in, H_out)

                    sign = np.sign(Q_epanet)
                    Q1 *= sign
                    Q2 *= sign

                    results_list.append({
                        "pipe": pipe_name,
                        "cracked": (A_leak > 0),
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
                        "blocked": blocked
                    })

                except Exception as e:
                    print(f"Skipping pipe {pipe_name}: {e}")
                    continue

            df = pd.DataFrame(results_list)

            # build classification features and label
            # label mapping: normal=0, leak=1, burst=2, blockage=3
            label_map = {"normal":0, "leak":1, "burst":2, "blockage":3}
            features = extract_classification_features(df)
            classification_rows.append((features, label_map[scen]))

            # build localization graph only for fault scenarios (not normal)
            if scen != "normal":
                graph = df_to_graph(df, integrator.wn)
                if graph is not None:
                    localization_graphs.append(graph)

        except Exception as e:
            print("Scenario failed:", e)
            continue

    return localization_graphs, classification_rows


# =====================================================
# TRAIN / VAL / TEST SPLIT
# =====================================================

def split_dataset(dataset):

    random.shuffle(dataset)

    n = len(dataset)

    n_train = int(TRAIN_RATIO * n)
    n_val   = int(VAL_RATIO * n)

    train_set = dataset[:n_train]
    val_set   = dataset[n_train:n_train+n_val]
    test_set  = dataset[n_train+n_val:]

    return train_set, val_set, test_set


# =====================================================
# HELPERS
# =====================================================

def save_dataset(dataset, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(dataset, os.path.join(save_dir, "water_dataset.pt"))
    train_set, val_set, test_set = split_dataset(dataset)
    torch.save(train_set, os.path.join(save_dir, "train.pt"))
    torch.save(val_set, os.path.join(save_dir, "val.pt"))
    torch.save(test_set, os.path.join(save_dir, "test.pt"))
    return train_set, val_set, test_set


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graph dataset from an EPANET input file")
    parser.add_argument("--inp-file", default=INP_FILE, help="Path to the EPANET .inp file")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="Number of samples to generate")
    parser.add_argument("--save-dir", default=SAVE_DIR, help="Directory to save dataset files")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    INP_FILE = args.inp_file
    N_SAMPLES = args.samples
    SAVE_DIR = args.save_dir
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Using input file:", INP_FILE)
    print("Saving datasets to:", SAVE_DIR)
    print("Generating", N_SAMPLES, "samples")

    localization_graphs, classification_rows = build_dataset()

    print("\nTotal localization graphs:", len(localization_graphs))
    print("Total classification rows:", len(classification_rows))

    if len(localization_graphs) == 0 and len(classification_rows) == 0:
        raise RuntimeError("No valid samples generated")

    # Save localization graphs (Dataset C)
    if len(localization_graphs) > 0:
        save_dataset(localization_graphs, SAVE_DIR)

    # Save classification dataset (Dataset B)
    if len(classification_rows) > 0:
        X = np.array([r[0] for r in classification_rows], dtype=np.float32)
        y = np.array([r[1] for r in classification_rows], dtype=np.int64)
        torch.save({"X": X, "y": y}, os.path.join(SAVE_DIR, "classification_B.pt"))

    print("\nSaved:")
    if len(localization_graphs) > 0:
        print("water_dataset.pt, train.pt, val.pt, test.pt")
    if len(classification_rows) > 0:
        print("classification_B.pt")

    # Example sample inspection for localization graphs
    if len(localization_graphs) > 0:
        sample = localization_graphs[0]
        print("\nSample graph:")
        print("Nodes:", sample.x.shape[0])
        print("Features:", sample.x.shape[1])
        print("Edges:", sample.edge_index.shape[1])
        print("Leak label:", sample.y.item())
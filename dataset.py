# dataset.py
# Full corrected dataset generator using your existing:
#   - EPANET_Integration.py
#   - PipeModel.py
#
# Output:
#   DATASETS/
#       water_dataset.pt
#       train.pt
#       val.pt
#       test.pt

import os
import torch
import random
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from EPANET_Integration import EPANETIntegrator


# =====================================================
# SETTINGS
# =====================================================

INP_FILE = "Test.inp"          # your EPANET network
N_SAMPLES = 2500               # number of scenarios
SAVE_DIR = "DATASETS"

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


# =====================================================
# BUILD DATASET
# =====================================================

def build_dataset():

    dataset = []

    integrator = EPANETIntegrator(INP_FILE)

    for k in range(N_SAMPLES):

        print(f"Generating sample {k+1}/{N_SAMPLES}")

        try:
            df = integrator.simulate_leak()

            if df.empty:
                print("Skipped empty scenario")
                continue

            graph = df_to_graph(df, integrator.wn)

            if graph is None:
                print("Skipped invalid graph")
                continue

            dataset.append(graph)

        except Exception as e:
            print("Scenario failed:", e)
            continue

    return dataset


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
# MAIN
# =====================================================

if __name__ == "__main__":

    dataset = build_dataset()

    print("\nTotal valid samples:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError("No valid samples generated")

    # save full dataset
    torch.save(
        dataset,
        os.path.join(SAVE_DIR, "water_dataset.pt")
    )

    # split
    train_set, val_set, test_set = split_dataset(dataset)

    torch.save(
        train_set,
        os.path.join(SAVE_DIR, "train.pt")
    )

    torch.save(
        val_set,
        os.path.join(SAVE_DIR, "val.pt")
    )

    torch.save(
        test_set,
        os.path.join(SAVE_DIR, "test.pt")
    )

    print("\nSaved:")
    print("water_dataset.pt")
    print("train.pt")
    print("val.pt")
    print("test.pt")

    # sample inspection
    sample = dataset[0]

    print("\nSample graph:")
    print("Nodes:", sample.x.shape[0])
    print("Features:", sample.x.shape[1])
    print("Edges:", sample.edge_index.shape[1])
    print("Leak label:", sample.y.item())
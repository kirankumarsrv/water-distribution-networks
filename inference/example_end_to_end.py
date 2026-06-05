"""End-to-end example for leak detection and localisation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from integration.EPANET_Integration import EPANETIntegrator
from physics.PipeModel import PipeLeakModel
from inference.real_time_detector import RealTimeLeakDetector


def build_sample_df(inp_file: str, scenario: str, target_pipe: str | list[str] | list[dict[str, str]] | None = None) -> pd.DataFrame:
    integrator = EPANETIntegrator(inp_file)
    integrator.run_simulation()
    heads = integrator.results.node["head"]
    flows = integrator.results.link["flowrate"]
    t = heads.index[0]

    all_pipes = [p for p, link in integrator.wn.links() if link.link_type == "Pipe"]
    if len(all_pipes) == 0:
        raise RuntimeError("No pipe links found in the network")

    target_pipes: list[dict[str, str]] | None = None
    if isinstance(target_pipe, str):
        target_pipes = [{"pipe": target_pipe, "scenario": scenario}]
    elif isinstance(target_pipe, list):
        if all(isinstance(pipe, str) for pipe in target_pipe):
            target_pipes = [{"pipe": str(pipe), "scenario": scenario} for pipe in target_pipe]
        else:
            target_pipes = []
            for entry in target_pipe:
                if not isinstance(entry, dict):
                    raise ValueError("Each target_pipe entry must be a string or a dict with pipe and scenario")
                pipe_id = entry.get("pipe")
                if pipe_id is None:
                    raise ValueError("Each fault entry must include a 'pipe' key")
                target_pipes.append({"pipe": str(pipe_id), "scenario": str(entry.get("scenario", scenario))})
    else:
        target_pipes = None

    if scenario != "normal" and (target_pipes is None or not target_pipes):
        target_pipes = [{"pipe": random.choice(all_pipes), "scenario": scenario}]

    if target_pipes:
        for entry in target_pipes:
            if entry["pipe"] not in all_pipes:
                raise ValueError(f"Target pipe '{entry['pipe']}' not found in network")

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
        f = (R * 3.141592653589793**2 * g * D**5) / (8 * L)
        if not (f > 0 and float(f) != float("inf")):
            f = 0.02

        fault = None
        if target_pipes is not None:
            for entry in target_pipes:
                if entry["pipe"] == pipe_name:
                    fault = entry
                    break

        if fault is None:
            A_leak = 0.0
            Cd = 0.0
            x_leak = L / 2
            blocked = False
        elif fault["scenario"] == "leak":
            A_leak = random.uniform(1e-8, 5e-7) * (D ** 2)
            Cd = random.uniform(0.60, 0.80)
            x_leak = random.uniform(0.05 * L, 0.95 * L)
            blocked = False
        elif fault["scenario"] == "burst":
            A_leak = random.uniform(1e-5, 5e-4) * (D ** 2)
            Cd = random.uniform(0.8, 1.0)
            x_leak = random.uniform(0.05 * L, 0.95 * L)
            blocked = False
        elif fault["scenario"] == "blockage":
            A_leak = 0.0
            Cd = 0.0
            x_leak = L / 2
            f = f * random.uniform(5.0, 50.0)
            blocked = True
        else:
            A_leak = 0.0
            Cd = 0.0
            x_leak = L / 2
            blocked = False

        pipe_model = PipeLeakModel(L=L, D=D, f=f, A_leak=A_leak, Cd=Cd, x_leak=x_leak)
        Q1, Q2, Q_leak, Hm = pipe_model.solve_model(H_in, H_out)
        sign = 0.0 if Q_epanet == 0.0 else (1.0 if Q_epanet >= 0 else -1.0)
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

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Example pipeline: generate one sample and run detection + localisation.")
    parser.add_argument("--inp-file", default="EPANETINPUTFILESFOR7NEWORKS/2_Extended Hanoi.inp", help="Path to the EPANET .inp file")
    parser.add_argument("--scenario", default="leak", choices=["normal", "leak", "burst", "blockage"], help="Scenario type for the sample")
    parser.add_argument("--target-pipe", default=None, help="Specific pipe to apply a fault to")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained model artifacts")
    parser.add_argument("--output", default=None, help="Optional output JSON file for results")
    args = parser.parse_args()

    sample_df = build_sample_df(args.inp_file, args.scenario, args.target_pipe)
    detector = RealTimeLeakDetector(models_dir=Path(args.models_dir))
    result = detector.infer(sample_df)

    print("Sample scenario:", args.scenario)
    print("Detection result:")
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"scenario": args.scenario, "result": result}, handle, indent=2)
        print(f"Saved end-to-end output to {args.output}")


if __name__ == "__main__":
    main()

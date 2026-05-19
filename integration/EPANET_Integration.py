from scipy import *
import math
import pandas
import numpy as np
from scipy.optimize import root
from physics.PipeModel import PipeLeakModel
import wntr

class EPANETIntegrator:
    def __init__(self,inp_file):
        self.inp_file = inp_file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.epsilon = 1e-6
    
    def run_simulation(self):
        sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = sim.run_sim()

    def simulate_leak(self):
        """
        Final robust version
        --------------------
        - Runs EPANET
        - Randomly cracks 10% of pipes
        - Avoids empty dataframe from over-filtering
        - Uses fallback friction factor if needed
        - Handles tiny headloss / tiny flow
        """

        # -------------------------------------------------
        # Run EPANET
        # -------------------------------------------------
        self.run_simulation()

        heads = self.results.node["head"]
        flows = self.results.link["flowrate"]

        # first timestep
        t = heads.index[0]

        results_list = []

        # -------------------------------------------------
        # Collect all real pipes
        # -------------------------------------------------
        all_pipes = []

        for pipe_name, pipe in self.wn.links():
            if pipe.link_type == "Pipe":
                all_pipes.append(pipe_name)

        if len(all_pipes) == 0:
            return pandas.DataFrame()

        # -------------------------------------------------
        # Randomly crack 10%
        # -------------------------------------------------
        crack_prob = np.random.uniform(0.1, 0.3)
        n_total = len(all_pipes)
        n_cracked = max(1, int(crack_prob * n_total))

        cracked_pipes = set(
            np.random.choice(all_pipes, n_cracked, replace=False)
        )

        print("Total pipes    :", n_total)
        print("Cracked pipes  :", n_cracked)
        print("Selected       :", cracked_pipes)

        valid_count = 0

        # -------------------------------------------------
        # Process each pipe
        # -------------------------------------------------
        for pipe_name, pipe in self.wn.links():

            if pipe.link_type != "Pipe":
                continue

            try:
                # -----------------------------
                # Geometry
                # -----------------------------
                L = pipe.length
                D = pipe.diameter

                if L <= 0 or D <= 0:
                    continue

                # -----------------------------
                # Connected nodes
                # -----------------------------
                start = pipe.start_node_name
                end   = pipe.end_node_name

                # -----------------------------
                # EPANET heads + flow
                # -----------------------------
                H_in  = float(heads.loc[t, start])
                H_out = float(heads.loc[t, end])
                Q_epanet = float(flows.loc[t, pipe_name])

                # -----------------------------
                # Use positive head direction
                # -----------------------------
                if H_out > H_in:
                    H_in, H_out = H_out, H_in

                # -----------------------------
                # Tiny flow skip only
                # -----------------------------
                if abs(Q_epanet) < 1e-12:
                    continue

                # -----------------------------
                # Headloss regularization
                # -----------------------------
                hf = abs(H_in - H_out)
                hf = max(hf, 1e-8)

                # -----------------------------
                # Resistance
                # -----------------------------
                R = hf / (Q_epanet ** 2)

                # -----------------------------
                # Friction factor
                # -----------------------------
                g = 9.81

                f = (R * math.pi**2 * g * D**5) / (8 * L)

                # fallback if unstable
                if (not np.isfinite(f)) or f <= 0:
                    f = 0.02

                # -----------------------------
                # Crack selection
                # -----------------------------
                is_cracked = pipe_name in cracked_pipes

                if is_cracked:
                    A_leak = np.random.uniform(1e-8,5e-7)*(D**2)
                    Cd = np.random.uniform(0.60, 0.80)

                    margin = 0.05 * L
                    x_leak = np.random.uniform(margin, L - margin)

                else:
                    A_leak = 0.0
                    Cd = 0.0
                    x_leak = L / 2

                # -----------------------------
                # Solve your pipe model
                # -----------------------------
                pipe_model = PipeLeakModel(
                    L=L,
                    D=D,
                    f=f,
                    A_leak=A_leak,
                    Cd=Cd,
                    x_leak=x_leak
                )

                Q1, Q2, Q_leak, Hm = pipe_model.solve_model(H_in, H_out)

                # preserve EPANET flow sign
                sign = np.sign(Q_epanet)

                Q1 *= sign
                Q2 *= sign

                # -----------------------------
                # Save
                # -----------------------------
                results_list.append({
                    "pipe": pipe_name,
                    "cracked": is_cracked,
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
                    "x_leak": x_leak
                })

                valid_count += 1

            except Exception as e:
                print(f"Skipping pipe {pipe_name}: {e}")
                continue

        print("Valid solved pipes:", valid_count)

        # -------------------------------------------------
        # Always return dataframe with columns
        # -------------------------------------------------
        columns = [
            "pipe", "cracked", "H_in", "H_out", "Q_EPANET",
            "Q1", "Q2", "Q_leak", "Hm", "A_leak", "Cd", "f","x_leak"
        ]

        df = pandas.DataFrame(results_list, columns=columns)

        return df

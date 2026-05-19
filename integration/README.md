# Integration Module: EPANET Integration

This module connects EPANET hydraulic simulation with the custom pipe leak physics model and dataset generation.

## What Happens Here

1. Run the EPANET simulation using `wntr`.
2. Collect node heads and pipe flows from the first timestep.
3. Randomly crack 10–30% of pipes in each scenario.
4. Generate stochastic leak properties:
   - leak area
   - discharge coefficient
   - leak location
5. Solve local leak hydraulics with `PipeLeakModel`.
6. Produce per-pipe values for:
   - `Q1`, `Q2`, `Q_leak`, `Hm`
   - `H_in`, `H_out`, `Q_EPANET`

## Why This Is Better Than Raw EPANET

Raw EPANET gives network-level hydraulics but does not model local leak physics.
This integration adds:

- local leak mechanics
- nonlinear solver behavior
- stochastic crack parameterization
- physically-based leakage

The result is a more realistic dataset for leak localization tasks.

# Physics Module: PipeModel

This module contains the physical leak model for a single pipe.
It is the most important physics file and implements nonlinear hydraulic equations using Newton-Raphson iteration.

## What This File Does

`PipeModel.py` solves core pipe hydraulics with leak physics:

- Flow conservation: incoming flow = outgoing flow + leak flow
- Pressure head losses along pipe segments
- Leak discharge using real fluid mechanics
- Friction loss with quadratic flow terms
- Newton-Raphson solver for nonlinear equations

## Important Variables

| Variable | Meaning |
| --- | --- |
| `L` | Pipe length |
| `D` | Pipe diameter |
| `f` | Friction factor |
| `A_leak` | Leak/crack area |
| `Cd` | Discharge coefficient |
| `x_leak` | Leak location along pipe |
| `H_in` | Inlet pressure head |
| `H_out` | Outlet pressure head |
| `Q1` | Flow before leak |
| `Q2` | Flow after leak |
| `Hm` | Pressure head at leak location |

## Leakage Equation

The leakage equation used is:

`Q = Cd × A × sqrt(2 g H)`

This is a physically meaningful fluid mechanics expression and is the reason the dataset is nonlinear and realistic.

## Why the Dataset Is Nonlinear

- Flow equations are quadratic
- Pressure losses depend on flow²
- Leak discharge depends on sqrt(H)
- Newton iterations solve the coupled nonlinear system

Real water distribution systems behave nonlinearly, so this model supports physically meaningful graph ML data.

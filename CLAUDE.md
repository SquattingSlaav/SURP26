# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SURP26 is a quantum machine learning research project (SUNY Poly, 2026) exploring quantum neuron simulation via Qiskit. The two active research areas are:
- **Part 2**: Quantum teleportation with noise analysis
- **Part 3**: Quantum neuron architectures and parameter space exploration

## Commands

```bash
# Install dependencies
uv sync

# Run any experiment script directly
uv run python src/Part_2_Quantum_Teleportation/QuantumTeleportation.py
uv run python src/Part_3_Quantum_Neuron/QuantumNeuron.py
uv run python src/Part_3_Quantum_Neuron/QuantumNeuronFix.py
# (same pattern for all scripts)
```

There is no test suite or lint configuration. Python version is 3.13 (see `.python-version`).

## Architecture

Each file in `src/` is a standalone script: run it to execute an experiment, save results to `results/*.npz`, and write figures to `figures/`. No shared library code exists — utilities are defined per-file.

### Core quantum building block (Part 3)

All Part 3 neurons share the same gate sequence (RY–CRy–RZ–CRy–RY), called "RZR" informally:

```
ry(alpha, q_input)           # encode input
ry(param, q_input)           # trainable weight
cry(theta, q_ctrl, q_tgt)    # controlled-RY
rz(π/2, q_tgt)               # fixed phase
cry(-theta, q_ctrl, q_tgt)   # inverse controlled-RY
ry(bias, q_tgt)              # output bias
```

The analytic expectation value for the single-neuron case is `y = a·sin²(θ)·cos²(θ)`.

### Part 3 file map

| File | What it does |
|---|---|
| `QuantumNeuron.py` | Single neuron, 2-param sweep (RZ variant), generates circuit diagram |
| `QuantumNeuronH.py` | Same but with Hadamard on ancilla |
| `QuantumNeuronSurvey.py` | 3-param systematic grid (10×10 params × 20×20 inputs) |
| `QuantumNeuron5Param.py` | 5-param neuron, 400 random models surveyed |
| `QuantumNeuronLayered.py` | 2→1 two-layer network, 9 params, 400 random models |
| `QuantumNeuronChain.py` | Sequential 3-neuron chain, 3D parameter sweep (101³ circuits) |
| `QuantumNeuronFix.py` | 2D and 3D sweeps with fixed architecture |
| `QuantumNeuron9D.py` | 9-param MCMC exploration (Metropolis–Hastings, 10k samples) |

### Simulation backend

All scripts use **Qiskit-Aer** (`StatevectorSimulator` or `AerSimulator`) locally. IBM cloud runtime (`qiskit-ibm-runtime`) is a dependency but not currently used. Noise is modeled with depolarizing errors via `NoiseModel`.

### Data flow

1. Build a parameterized `QuantumCircuit` with `ParameterVector`
2. Bind parameters in batch via `assign_parameters`
3. Run via `backend.run(circuits, shots=N)` and extract counts
4. Compute expectation value as `P(|0⟩) - P(|1⟩)` or similar
5. Save arrays with `np.savez(results/...)` and plot with matplotlib

The 3D sweeps (101³ = ~1M circuits) and the MCMC script (10k × full circuit) are computationally heavy — expect multi-minute runtimes.


## Project structure
- Part 3: Quantum neuron parameter survey
- All code lives in `/home/jah/SURP26/src/Part_3_Quantum_Neuron/`
- Results saved to `results/`, figures to `figures/`

## Circuit conventions
- Never use transpile — use `sim.run(qc, shots=SHOTS)` directly
- CRy pairs must always be symmetric: cry(theta) ... cry(-theta)
- Rz(pi/2) always sits between the CRy pair

## Neuron architectures
- **3-param**: qreg_0 (input) → qreg_1 (CRy, Rz, CRy, Ry bias), measure q1
- **5-param**: qreg_0, qreg_1 (inputs) → qreg_2 (CRy from q0, CRy from q1, Rz, CRy from q1, CRy from q0, Ry bias), measure q2
- **9-param**: two 3-param neurons feeding into a third, measure q3
- **15-param**: two 5-param neurons feeding into a third, measure q5

## Survey conventions
- Input axes: alpha (qreg_0), beta (qreg_1), both swept linspace(0, pi/2, 20)
- Neuron params randomly drawn from linspace(0, pi/2, 40)
- SHOTS = 1024

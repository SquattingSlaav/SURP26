from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import numpy as np
import os

SHOTS = 1024

os.makedirs("results", exist_ok=True)
theta = Parameter('θ')

def build_circuit(a):
    inp = QuantumRegister(1, "input")
    anc = QuantumRegister(1, "ancilla")
    out = QuantumRegister(1, "output")
    cr = ClassicalRegister(1, "cr")

    qc = QuantumCircuit(inp, anc, out, cr)
    qc.ry(2 * np.arcsin(np.sqrt(a)), inp)
    qc.cry(2 * theta, inp, anc)
    qc.cy(anc, out)
    qc.rz(-np.pi/2, anc)
    qc.cry(-2 * theta, inp, anc)
    qc.measure(anc, cr)

    return qc

def sweep():
    a_vals = np.linspace(0, 1, 101)
    theta_vals = np.linspace(0, np.pi, 101)
    grid = np.zeros((len(theta_vals), len(a_vals)))

    sim = AerSimulator()

    for i, theta_v in enumerate(theta_vals):
        for j, a_v in enumerate(a_vals):
            qc = build_circuit(a_v)
            bound = qc.assign_parameters({theta: theta_v})
            compiled  = transpile(bound, sim, optimization_level=0)
            counts = sim.run(compiled, shots=SHOTS).result().get_counts()
            grid[i, j] = counts.get('1', 0) / SHOTS
        print(f"theta = {theta_v:.4f}  ({i+1}/{len(theta_vals)})")
    np.savez("results/sweep.npz", a_vals=a_vals, theta_vals=theta_vals, grid=grid)
    return a_vals, theta_vals, grid

def plot_sweep(a_vals, theta_vals, grid):
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(a_vals, theta_vals, grid, shading='auto', cmap='viridis')
    plt.colorbar(label='y = P(ancilla = 1)')
    plt.xlabel('a')
    plt.ylabel('θ')
    plt.title('Quantum Neuron Output  y = f(a, θ)')
    plt.tight_layout()
    plt.savefig('figures/neuron_phase_diagram.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/neuron_phase_diagram.png")

if __name__ == "__main__":
    #a_vals, theta_vals, grid = sweep()
    data = np.load("results/sweep.npz")
    a_vals, theta_vals, grid = data["a_vals"], data["theta_vals"], data["grid"]
    print("Saved → results/sweep.npz")
    plot_sweep(a_vals, theta_vals, grid)


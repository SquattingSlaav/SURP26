from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import numpy as np
import matplotlib.pyplot as plt
import os

SHOTS = 1024
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

theta = Parameter('θ')

def build_circuit(a):
    inp = QuantumRegister(1, "input")
    anc = QuantumRegister(1, "ancilla")
    out = QuantumRegister(1, "output")
    cr  = ClassicalRegister(1, "cr")
    qc  = QuantumCircuit(inp, anc, out, cr)
    qc.ry(2 * np.arcsin(np.sqrt(a)), inp)
    qc.cry( 2 * theta, inp, anc)
    qc.cy(anc, out)
    qc.h(anc)
    qc.cry(-2 * theta, inp, anc)
    qc.measure(anc, cr)
    return qc

def sweep():
    a_vals     = np.linspace(0, 1,      101)
    theta_vals = np.linspace(0, np.pi,  101)
    grid       = np.zeros((len(theta_vals), len(a_vals)))
    sim        = AerSimulator()

    for i, theta_v in enumerate(theta_vals):
        for j, a_v in enumerate(a_vals):
            qc       = build_circuit(a_v)
            bound    = qc.assign_parameters({theta: theta_v})
            compiled = transpile(bound, sim, optimization_level=0)
            counts   = sim.run(compiled, shots=SHOTS).result().get_counts()
            grid[i, j] = counts.get('1', 0) / SHOTS
        print(f"theta = {theta_v:.4f}  ({i+1}/{len(theta_vals)})")

    np.savez("results/sweep_h.npz", a_vals=a_vals, theta_vals=theta_vals, grid=grid)
    return a_vals, theta_vals, grid

def plot_sweep(a_vals, theta_vals, grid):
    A, T          = np.meshgrid(a_vals, theta_vals)
    analytic_grid = A * np.sin(T)**2 * np.cos(T)**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.pcolormesh(a_vals, theta_vals, grid, shading='auto', cmap='viridis')
    fig.colorbar(im1, ax=ax1, label='y = P(ancilla = 1)')
    ax1.set_xlabel('a')
    ax1.set_ylabel('θ')
    ax1.set_title('Simulation')

    im2 = ax2.pcolormesh(a_vals, theta_vals, analytic_grid, shading='auto', cmap='viridis')
    fig.colorbar(im2, ax=ax2, label='y = P(ancilla = 1)')
    ax2.set_xlabel('a')
    ax2.set_ylabel('θ')
    ax2.set_title('Analytic:  a·sin²(θ)·cos²(θ)')

    plt.tight_layout()
    plt.savefig('figures/neuron_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/neuron_comparison.png")

if __name__ == "__main__":
    a_vals, theta_vals, grid = sweep()
    data = np.load("results/sweep.npz")
    a_vals, theta_vals, grid = data["a_vals"], data["theta_vals"], data["grid"]

    plot_sweep(a_vals, theta_vals, grid, filename="figures/comparison_h.png", title="H Circuit")

    build_circuit(0.5).assign_parameters({theta: np.pi / 4}).draw("mpl").savefig(
        "figures/neuron_circuit.png", dpi=150, bbox_inches="tight"
    )
    print("Saved → figures/neuron_circuit.png")

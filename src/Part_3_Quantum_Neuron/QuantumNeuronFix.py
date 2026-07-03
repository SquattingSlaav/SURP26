from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import numpy as np
import matplotlib.pyplot as plt
import os

SHOTS = 1024
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

theta1 = Parameter('θ_1')
theta2 = Parameter('θ_2')

def build_circuit():
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, cr)

    qc.ry(np.pi / 2, q0)
    qc.cry(2 * theta1, q0, q1)
    qc.rz(np.pi / 2, q1)
    qc.cry(-2 * theta1, q0, q1)
    qc.ry(theta2, q1)

    qc.measure(q1, cr)
    return qc

def sweep():
    theta_vals = np.linspace(0, np.pi, 101)
    grid       = np.zeros((len(theta_vals), len(theta_vals)))
    sim        = AerSimulator()

    qc       = build_circuit()
    compiled = transpile(qc, sim, optimization_level=0)

    total = len(theta_vals) ** 2
    count = 0
    for i, t1 in enumerate(theta_vals):
        for j, t2 in enumerate(theta_vals):
            bound  = compiled.assign_parameters({theta1: t1, theta2: t2})
            counts = sim.run(bound, shots=SHOTS).result().get_counts()
            grid[i, j] = counts.get('1', 0) / SHOTS
            count += 1
        print(f"theta1 = {t1:.4f}  ({count}/{total})")

    np.savez("results/sweep_2d.npz", theta_vals=theta_vals, grid=grid)
    return theta_vals, grid

def plot_sweep(theta_vals, grid):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(theta_vals, theta_vals, grid, shading='auto', cmap='RdBu_r')
    fig.colorbar(im, ax=ax, label='expectation')
    ax.set_xlabel('θ_1')
    ax.set_ylabel('θ_2')
    ax.set_title('Single neuron sweep')
    plt.tight_layout()
    plt.savefig('figures/sweep_2d.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/sweep_2d.png")

if __name__ == "__main__":
    theta_vals, grid = sweep()
    data = np.load("results/sweep_2d.npz")
    plot_sweep(data["theta_vals"], data["grid"])
    build_circuit().assign_parameters({theta1: np.pi/4, theta2: np.pi/6}).draw("mpl").savefig(
        "figures/neuron_circuit.png", dpi=150, bbox_inches="tight")
    print("Done.")

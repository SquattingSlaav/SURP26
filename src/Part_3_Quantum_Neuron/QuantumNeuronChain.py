from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import numpy as np
import matplotlib.pyplot as plt
import os
import time

SHOTS = 1024
A     = 1
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

theta1 = Parameter('θ_1')
theta2 = Parameter('θ_2')
theta3 = Parameter('θ_3')

def rzr_block(qc, control, target, theta):
    qc.cry(2 * theta, control, target)
    qc.rz(np.pi / 2, target)
    qc.cry(-2 * theta, control, target)

def build_circuit_2():
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, cr)

    qc.ry(np.pi / 2, q0)       # a = 1
    rzr_block(qc, q0, q1, theta1)
    qc.ry(theta2, q1)           # bias

    qc.measure(q1, cr)
    return qc

def build_circuit_3():
    q0 = QuantumRegister(1, 'qreg_0')
    q1 = QuantumRegister(1, 'qreg_1')
    q2 = QuantumRegister(1, 'qreg_2')
    q3 = QuantumRegister(1, 'qreg_3')
    cr = ClassicalRegister(1, 'creg')
    qc = QuantumCircuit(q0, q1, q2, q3, cr)

    qc.ry(np.pi / 2, q0)       # a = 1
    rzr_block(qc, q0, q1, theta1)
    rzr_block(qc, q2, q3, theta2)
    rzr_block(qc, q1, q3, theta3)

    qc.measure(q3, cr)
    return qc

def sweep_2d():
    theta_vals = np.linspace(0, np.pi, 101)
    grid       = np.zeros((len(theta_vals), len(theta_vals)))
    sim        = AerSimulator()

    qc       = build_circuit_2()
    compiled = transpile(qc, sim, optimization_level=0)

    total = len(theta_vals) ** 2
    count = 0
    t_start = time.time()
    for i, t1 in enumerate(theta_vals):
        for j, t2 in enumerate(theta_vals):
            bound  = compiled.assign_parameters({theta1: t1, theta2: t2})
            counts = sim.run(bound, shots=SHOTS).result().get_counts()
            grid[i, j] = counts.get('1', 0) / SHOTS
            count += 1
        print(f"theta1 = {t1:.4f}  ({count}/{total})")

    elapsed = time.time() - t_start
    print(f"2D sweep done in {elapsed:.1f}s ({elapsed/60:.2f} min)")

    np.savez("results/sweep_2d.npz", theta_vals=theta_vals, grid=grid)
    return theta_vals, grid

def sweep_3d():
    theta_vals = np.linspace(0, np.pi, 101)
    grid       = np.zeros((len(theta_vals), len(theta_vals), len(theta_vals)))
    sim        = AerSimulator()

    qc       = build_circuit_3()
    compiled = transpile(qc, sim, optimization_level=0)

    total = len(theta_vals) ** 3
    count = 0
    t_start = time.time()
    for i, t1 in enumerate(theta_vals):
        for j, t2 in enumerate(theta_vals):
            for k, t3 in enumerate(theta_vals):
                bound  = compiled.assign_parameters({theta1: t1, theta2: t2, theta3: t3})
                counts = sim.run(bound, shots=SHOTS).result().get_counts()
                grid[i, j, k] = counts.get('1', 0) / SHOTS
                count += 1
            print(f"({i},{j}) — {count}/{total}")

    elapsed = time.time() - t_start
    print(f"3D sweep done in {elapsed:.1f}s ({elapsed/3600:.2f} hr)")

    np.savez("results/sweep_3d.npz", theta_vals=theta_vals, grid=grid)
    return theta_vals, grid

def plot_2d(theta_vals, grid):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(theta_vals, theta_vals, grid, shading='auto', cmap='RdBu_r')
    fig.colorbar(im, ax=ax, label='expectation')
    ax.set_xlabel('θ_1')
    ax.set_ylabel('θ_2')
    ax.set_title('2-neuron — a = 1')
    plt.tight_layout()
    plt.savefig('figures/sweep_2d.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/sweep_2d.png")

def plot_3d(theta_vals, grid):
    T1, T2, T3 = np.meshgrid(theta_vals, theta_vals, theta_vals, indexing='ij')
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(T1.ravel(), T2.ravel(), T3.ravel(),
                     c=grid.ravel(), cmap='RdBu_r', alpha=0.1, s=1)
    fig.colorbar(sc, label='expectation')
    ax.set_xlabel('θ_1')
    ax.set_ylabel('θ_2')
    ax.set_zlabel('θ_3')
    ax.set_title('3-neuron tree — a = 1')
    plt.tight_layout()
    plt.savefig('figures/sweep_3d.png', dpi=150, bbox_inches='tight')
    print("Saved → figures/sweep_3d.png")

if __name__ == "__main__":
    print("Running 2D sweep...")
    theta_vals, grid_2d = sweep_2d()
    plot_2d(theta_vals, grid_2d)

    print("Running 3D sweep...")
    theta_vals, grid_3d = sweep_3d()
    plot_3d(theta_vals, grid_3d)

    build_circuit_2().assign_parameters({theta1: np.pi/4, theta2: np.pi/6}).draw("mpl").savefig(
        "figures/circuit_2neuron.png", dpi=150, bbox_inches="tight")
    build_circuit_3().assign_parameters({theta1: np.pi/4, theta2: np.pi/4, theta3: np.pi/6}).draw("mpl").savefig(
        "figures/circuit_3neuron.png", dpi=150, bbox_inches="tight")

    print("Done.")
